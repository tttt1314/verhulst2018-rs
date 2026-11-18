# © copyright 2017 Ghent University – Universiteit Gent, all rights reserved; this Derivative work is
# made available for non-commercial academic research purposes and subject to an UGent Academic
# License (see LICENSE and NOTICE).
"""Helpers that run the *original* Verhulst 2018 Python model (read-only source directory)."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

DEFAULT_SOURCE = (
    Path(__file__).resolve().parents[2] / "Verhulstetal2018Model"
)
SOURCE_DIR = Path(os.environ.get("VERHULST_SOURCE_DIR", DEFAULT_SOURCE))

MODEL_CONFIG = {
    "samplerate": 100_000,
    "sections": 1_000,
    "compression_slope": 0.4,
    "Zweig_irregularities": 1,
    "non_linearity_type": "vel",
    "KneeVar": 1.0,
    "low_freq_irregularities": 1,
    "subject": 1,
    "IrrPct": 0.05,
}


def load_original(source_dir: Path = SOURCE_DIR):
    path = Path(source_dir) / "cochlear_model2018.py"
    spec = importlib.util.spec_from_file_location("_original_verhulst_cochlear_model", path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def starting_poles(source_dir: Path = SOURCE_DIR) -> np.ndarray:
    return np.loadtxt(Path(source_dir) / "StartingPoles.dat", dtype=np.float64)


def run_original(module, stim, poles, *, rtol=1e-2, probe_freq=None, config=None, record=False):
    """``init_model(stim, ...)`` + ``solve()`` with an optional rtol override.

    ``probe_freq`` defaults to a single probe (section 500 via the integer-array path) so that
    the original does not allocate all-channel histories. Returns the model and, if
    ``record``, the list of (t, y-copy) of every TLsolver call.
    """
    cfg = dict(MODEL_CONFIG)
    if config:
        cfg.update(config)
    base_ode = module.ode
    calls = []

    class Ode(base_ode):
        def set_integrator(self, name, **kw):
            kw["rtol"] = rtol
            return super().set_integrator(name, **kw)

    original_solver = module.TLsolver

    def recording_solver(t, y, model):
        calls.append(t)
        return original_solver(t, y, model)

    module.ode = Ode
    if record:
        module.TLsolver = recording_solver
    try:
        model = module.cochlea_model()
        if probe_freq is None:
            # Single probe channel (section 500) without the all-channel histories. The
            # frequency-list path of init_model uses np.float (removed from NumPy), so the
            # probe list is set after init; solve() then takes its "else" branch.
            model.init_model(np.asarray(stim, float), sheraPo=poles, probe_freq="abr", **cfg)
            model.probe_freq = "single"
            model.probe_points = np.array([500])
        else:
            model.init_model(np.asarray(stim, float), sheraPo=poles, probe_freq=probe_freq, **cfg)
        model.solve()
    finally:
        module.ode = base_ode
        module.TLsolver = original_solver
    return model, calls
