# © copyright 2017 Ghent University – Universiteit Gent, all rights reserved; this Derivative work is
# made available for non-commercial academic research purposes and subject to an UGent Academic
# License (see LICENSE and NOTICE).
# Changed files notice: Python front end (2026) of the Rust port of the cochlear core of
# Verhulstetal2018Model v1.2 (`cochlea_model.init_model` + `solve` of cochlear_model2018.py).
"""Fast Verhulst et al. (2018) transmission-line cochlear core (Rust, PyO3).

``simulate`` mirrors ``cochlea_model().init_model(stim, ...); solve()`` of the original
``cochlear_model2018.py`` but only returns the otoacoustic emission (and optional probe
channels). The Puria middle-ear filter and the 600–4000 Hz emission filter are applied with
the same SciPy calls as the original; the transmission line, its nonlinearity and the dopri5
integrator run in Rust.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import signal

from ._core import NOTICE, CochleaCore, cubic_interpolate, delay_line, solve_tridiagonal

__all__ = [
    "NOTICE",
    "CochleaCore",
    "SimulationResult",
    "simulate",
    "simulate_pressure",
    "make_core",
    "puria_filter",
    "emission_filter",
    "cubic_interpolate",
    "delay_line",
    "solve_tridiagonal",
]

PURIA_GAIN = 10 ** (18.0 / 20.0)


def _bandpass(samplerate: float):
    return signal.butter(1, [600 / (samplerate / 2.0), 4000.0 / (samplerate / 2)], "bandpass")


def puria_filter(stim: np.ndarray, samplerate: float) -> np.ndarray:
    """Middle-ear (Puria) filter exactly as in ``init_model``."""
    b, a = _bandpass(samplerate)
    return signal.lfilter(b * PURIA_GAIN, a, stim)


def emission_filter(qsol0: np.ndarray, samplerate: float, q0_factor: float) -> np.ndarray:
    """Reverse-path filter of ``solve()``: ``lfilter(b * q0_factor, a, oto_emission)``."""
    b, a = _bandpass(samplerate)
    return signal.lfilter(b * q0_factor, a, qsol0)


def _rth(subject: int, n1: int) -> np.ndarray:
    # init_model: np.random.seed(subject); Rth = 2 * (np.random.random(n + 1) - 0.5)
    # RandomState(seed).random_sample is the same MT19937 stream, without touching the
    # global NumPy generator.
    return 2 * (np.random.RandomState(subject).random_sample(n1) - 0.5)


def make_core(
    *,
    samplerate: float,
    sections: int,
    sheraPo,
    compression_slope: float = 0.4,
    Zweig_irregularities: int = 1,
    non_linearity_type: str = "vel",
    KneeVar: float = 1.0,
    low_freq_irregularities: int = 1,
    subject: int = 1,
    IrrPct: float = 0.05,
    rtol: float = 1e-2,
    atol: float = 1e-13,
    nsteps: int = 500,
    pole_update: str = "faithful",
) -> CochleaCore:
    """Initialised core (state after ``init_model`` and the ``solve()`` prologue)."""
    n1 = int(sections) + 1
    shera_po = np.zeros(n1) + np.asarray(sheraPo, dtype=np.float64)
    return CochleaCore(
        samplerate=float(samplerate),
        sections=int(sections),
        shera_po=np.ascontiguousarray(shera_po, dtype=np.float64),
        rth=np.ascontiguousarray(_rth(subject, n1), dtype=np.float64),
        compression_slope=float(compression_slope),
        zweig_irregularities=Zweig_irregularities != 0,
        non_linearity_type=str(non_linearity_type),
        knee_var=float(KneeVar),
        low_freq_irregularities=bool(low_freq_irregularities),
        irr_pct=float(IrrPct),
        rtol=float(rtol),
        atol=float(atol),
        nsteps=int(nsteps),
        pole_update=pole_update,
    )


@dataclass
class SimulationResult:
    oto_emission: np.ndarray
    """Filtered ear-canal pressure, length ``len(stim)`` (last two samples come from the
    unintegrated padding, as in the original)."""
    qsol0: np.ndarray
    """Unfiltered ``Qsol[0]`` per integrated sample, length ``len(stim) - 2``."""
    probe_channels: np.ndarray
    velocity: np.ndarray
    """BM velocity at the probe channels, shape (len(stim) - 2, n_probes)."""
    displacement: np.ndarray
    stim_filtered: np.ndarray
    q0_factor: float
    stats: dict = field(default_factory=dict)


def _probe_channels(core: CochleaCore, probe_freq, probe_channels) -> np.ndarray:
    if probe_channels is not None and probe_freq is not None:
        raise ValueError("give probe_freq or probe_channels, not both")
    if probe_channels is not None:
        return np.asarray(probe_channels, dtype=np.int64).reshape(-1)
    if probe_freq is None:
        return np.zeros(0, dtype=np.int64)
    f_res = core.vector("f_resonance")
    if isinstance(probe_freq, str):
        n1 = len(f_res)
        if probe_freq == "all":
            return np.arange(1, n1, dtype=np.int64)
        if probe_freq == "abr":
            return np.arange(110, 911, 2, dtype=np.int64)
        if probe_freq == "half":
            return np.arange(1, n1, 2, dtype=np.int64)
        raise ValueError(f"unknown probe_freq {probe_freq!r}")
    freqs = np.atleast_1d(np.asarray(probe_freq, dtype=np.float64))
    return np.array([int(np.argmin(np.abs(f_res - f))) for f in freqs], dtype=np.int64)


def simulate(
    stim: np.ndarray,
    samplerate: float,
    sections: int,
    sheraPo,
    *,
    probe_freq=None,
    probe_channels: Sequence[int] | None = None,
    compression_slope: float = 0.4,
    Zweig_irregularities: int = 1,
    non_linearity_type: str = "vel",
    KneeVar: float = 1.0,
    low_freq_irregularities: int = 1,
    subject: int = 1,
    IrrPct: float = 0.05,
    rtol: float = 1e-2,
    atol: float = 1e-13,
    nsteps: int = 500,
    pole_update: str = "faithful",
) -> SimulationResult:
    """Equivalent of ``m = cochlea_model(); m.init_model(stim, ...); m.solve()``.

    Like the original, ``len(stim) - 2`` samples are integrated. ``probe_freq`` follows the
    original semantics (``"all"``, ``"half"``, ``"abr"`` or frequencies in Hz → nearest
    section); ``probe_channels`` selects section indices directly (0 = stapes/middle ear
    node, 1..sections = BM sections). By default no probe channel is stored.

    ``pole_update="faithful"`` (default) reproduces the original right-hand side, which
    re-evaluates the poles in every dopri5 stage. ``"accepted_step"`` is a model variant that
    keeps the right-hand side pure and applies the 1 % pole-update rule only after accepted
    steps; it is *not* numerically equivalent to the original.
    """
    stim = np.asarray(stim, dtype=np.float64)
    if stim.ndim != 1 or stim.size < 3:
        raise ValueError("stim must be one-dimensional with at least 3 samples")
    if not np.isfinite(stim).all():
        raise ValueError("stim must be finite")
    core = make_core(
        samplerate=samplerate,
        sections=sections,
        sheraPo=sheraPo,
        compression_slope=compression_slope,
        Zweig_irregularities=Zweig_irregularities,
        non_linearity_type=non_linearity_type,
        KneeVar=KneeVar,
        low_freq_irregularities=low_freq_irregularities,
        subject=subject,
        IrrPct=IrrPct,
        rtol=rtol,
        atol=atol,
        nsteps=nsteps,
        pole_update=pole_update,
    )
    probes = _probe_channels(core, probe_freq, probe_channels)
    filtered = puria_filter(stim, samplerate)
    out = core.run(np.ascontiguousarray(filtered), [int(p) for p in probes])
    q0 = core.scalar("q0_factor")
    oto = np.zeros(stim.size)
    oto[: stim.size - 2] = out["qsol0"]
    oto = emission_filter(oto, samplerate, q0)
    stats = {k: out[k] for k in ("rhs_calls", "nfcn", "nstep", "naccept", "nreject", "pole_updates")}
    return SimulationResult(
        oto_emission=oto,
        qsol0=out["qsol0"],
        probe_channels=probes,
        velocity=out["v_probe"],
        displacement=out["y_probe"],
        stim_filtered=filtered,
        q0_factor=q0,
        stats=stats,
    )


def simulate_pressure(raw: np.ndarray, sheraPo, **kwargs) -> SimulationResult:
    """Worker convention: pad two zeros, simulate at 100 kHz with 1000 sections, and crop
    every output to ``len(raw)`` samples (output sample j is the state at (j+1)/fs)."""
    raw = np.asarray(raw, dtype=np.float64)
    kwargs.setdefault("samplerate", 100_000)
    kwargs.setdefault("sections", 1_000)
    padded = np.concatenate((raw, np.zeros(2, dtype=np.float64)))
    res = simulate(padded, sheraPo=sheraPo, **kwargs)
    res.oto_emission = res.oto_emission[: raw.size].copy()
    res.stim_filtered = res.stim_filtered[: raw.size].copy()
    return res
