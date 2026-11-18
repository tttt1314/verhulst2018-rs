# © copyright 2017 Ghent University – Universiteit Gent, all rights reserved; this Derivative work is
# made available for non-commercial academic research purposes and subject to an UGent Academic
# License (see LICENSE and NOTICE).
# Changed files notice: drop-in replacement (2026) for carfac_ephys/tl_reference_worker.py that
# runs the Rust port of the Verhulstetal2018Model v1.2 cochlear core.
"""Drop-in replacement for ``carfac_ephys.tl_reference_worker`` backed by verhulst2018-rs.

Same command line, same ``.npz`` keys and the same time convention as the original worker
(output sample j is the integrated state at (j + 1) / fs; two guard samples are appended and
removed). Differences: nothing is imported from the Verhulst source directory (it is only
hashed for provenance and used for the default poles file), only the requested probe channel
is stored, and the integrator tolerances, the irregularities and the pole-update variant are
options. With the defaults the ear-canal pressure is bit-identical to the original worker on
Apple arm64 (see README).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from . import NOTICE, simulate_pressure

SAMPLE_RATE_HZ = 100_000
MODEL_CONFIG = {
    "samplerate": SAMPLE_RATE_HZ,
    "sections": 1_000,
    "probe_freq": "all",
    "compression_slope": 0.4,
    "Zweig_irregularities": 1,
    "non_linearity_type": "vel",
    "KneeVar": 1.0,
    "low_freq_irregularities": 1,
    "subject": 1,
    "IrrPct": 0.05,
}
SOURCE_NAMES = ("cochlear_model2018.py", "cochlea_utils.c", "StartingPoles.dat")
REPO = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(path: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    except OSError:
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _atomic_write(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            tmp = Path(stream.name)
            writer(stream)
        tmp.replace(path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def _save_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    _atomic_write(path, lambda s: np.savez_compressed(s, allow_pickle=False, **arrays))


def _save_metadata(path: Path, metadata: dict[str, Any]) -> None:
    _atomic_write(path, lambda s: s.write((json.dumps(metadata, indent=2, allow_nan=False) + "\n").encode()))


def _check_destinations(output_path: Path, metadata_path: Path, input_path: Path | None, source_dir: Path | None) -> None:
    if output_path == metadata_path or input_path in (output_path, metadata_path):
        raise ValueError("Input, output, and metadata must be distinct files.")
    for destination in (output_path, metadata_path):
        if source_dir is not None and destination.is_relative_to(source_dir):
            raise ValueError("TL worker outputs must be outside the external source directory.")
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite an existing TL worker artifact: {destination}")


def _load_poles(path: Path) -> np.ndarray:
    poles = np.loadtxt(path, dtype=np.float64)
    expected = int(MODEL_CONFIG["sections"]) + 1
    if poles.shape != (expected,) or not np.isfinite(poles).all():
        raise ValueError(f"Expected {expected} finite real starting poles in {path}.")
    return poles


def _options(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "rtol": float(args.rtol),
        "atol": float(args.atol),
        "pole_update": args.pole_update,
        "Zweig_irregularities": 0 if args.smooth else 1,
    }


def _simulate(raw: np.ndarray, poles: np.ndarray, opts: dict[str, Any], probes: list[int]):
    cfg = {k: v for k, v in MODEL_CONFIG.items() if k not in ("probe_freq", "samplerate", "sections")}
    cfg["Zweig_irregularities"] = opts["Zweig_irregularities"]
    started = time.perf_counter()
    res = simulate_pressure(
        raw,
        poles,
        samplerate=SAMPLE_RATE_HZ,
        sections=MODEL_CONFIG["sections"],
        probe_channels=probes,
        rtol=opts["rtol"],
        atol=opts["atol"],
        pole_update=opts["pole_update"],
        **{k: v for k, v in cfg.items() if k != "Zweig_irregularities"},
        Zweig_irregularities=cfg["Zweig_irregularities"],
    )
    seconds = time.perf_counter() - started
    pressure = np.asarray(res.oto_emission, dtype=np.float64)
    finite = {
        "input_pa": bool(np.isfinite(raw).all()),
        "ear_canal_pressure_pa": bool(np.isfinite(pressure).all()),
        "filtered_drive": bool(np.isfinite(res.stim_filtered).all()),
        "velocity_m_s": bool(np.isfinite(res.velocity).all()),
        "displacement_m": bool(np.isfinite(res.displacement).all()),
    }
    if not all(finite.values()):
        raise FloatingPointError(f"Nonfinite TL output: {finite}")
    checks = {
        "samples": int(raw.size),
        "all_finite": finite,
        "checked_integration_steps": int(raw.size),
        "failed_integration_steps": 0,
        "integrator_stats": res.stats,
        "warnings": [],
        "compute_seconds": seconds,
    }
    return res, checks


def _base_metadata(source_dir: Path | None, poles_path: Path, opts: dict[str, Any]) -> dict[str, Any]:
    source_hashes = None
    if source_dir is not None:
        source_hashes = {n: _sha256(source_dir / n) for n in SOURCE_NAMES if (source_dir / n).is_file()}
    from importlib.metadata import PackageNotFoundError, version

    try:
        pkg_version = version("verhulst2018-rs")
    except PackageNotFoundError:
        pkg_version = None
    return {
        "schema_version": 1,
        "model": "Verhulst 2018 v1.2 cochlear core (verhulst2018-rs Rust port)",
        "license_notice": NOTICE,
        "implementation": {
            "package": "verhulst2018-rs",
            "version": pkg_version,
            "repository": str(REPO),
            "commit": _git(REPO),
            "core_library_sha256": _sha256(Path(sys.modules["verhulst2018_rs._core"].__file__)),
        },
        "source_dir": str(source_dir) if source_dir is not None else None,
        "upstream_commit": _git(source_dir) if source_dir is not None else None,
        "source_sha256": source_hashes,
        "poles_path": str(poles_path),
        "poles_sha256": _sha256(poles_path),
        "configuration": {
            "fs_hz": SAMPLE_RATE_HZ,
            "dtype": "float64",
            **MODEL_CONFIG,
            "probe_freq": "not stored (verhulst2018-rs stores only requested probe channels)",
            "Zweig_irregularities": opts["Zweig_irregularities"],
            "input_guard_samples": 2,
            "integrator": "verhulst2018-rs dopri5 (control flow and arithmetic of scipy 1.18.1 dop.c)",
            "integrator_rtol": opts["rtol"],
            "integrator_atol": opts["atol"],
            "pole_update": opts["pole_update"],
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "process_id": os.getpid(),
        },
        "time_convention": {
            "input_sample_j_seconds": "j / fs_hz",
            "output_sample_j_seconds": "(j + 1) / fs_hz",
            "output_sample_zero_seconds": 1 / SAMPLE_RATE_HZ,
            "unused_output_padding_removed": True,
        },
    }


def _publish(output_path: Path, metadata_path: Path, arrays, metadata, started: float) -> dict[str, Any]:
    _save_arrays(output_path, arrays)
    metadata["output_sha256"] = _sha256(output_path)
    metadata["worker_seconds_before_metadata_write"] = time.perf_counter() - started
    metadata["completed"] = True
    _save_metadata(metadata_path, metadata)
    return metadata


def _resolve_poles(source_dir: Path | None, poles_path: Path | None) -> Path:
    if poles_path is not None:
        return poles_path.resolve()
    if source_dir is None:
        raise ValueError("--poles is required when --source-dir is not given")
    return (source_dir / "StartingPoles.dat").resolve()


def run_reference(
    source_dir: Path | None,
    input_path: Path,
    output_path: Path,
    metadata_path: Path,
    *,
    poles_path: Path | None = None,
    options: dict[str, Any] | None = None,
    probe_channels: list[int] | None = None,
) -> dict[str, Any]:
    """Pressure mode of the original worker."""
    started = time.perf_counter()
    opts = options or {"rtol": 1e-2, "atol": 1e-13, "pole_update": "faithful", "Zweig_irregularities": 1}
    source_dir = source_dir.resolve() if source_dir is not None else None
    input_path, output_path, metadata_path = (p.resolve() for p in (input_path, output_path, metadata_path))
    _check_destinations(output_path, metadata_path, input_path, source_dir)
    poles_path = _resolve_poles(source_dir, poles_path)
    input_hash = _sha256(input_path)
    loaded = np.load(input_path, allow_pickle=False)
    if not isinstance(loaded, np.ndarray):
        loaded.close()
        raise ValueError("Expected one pressure array in a .npy file, not an archive.")
    if loaded.ndim != 1 or not loaded.size or loaded.dtype.kind not in "fiu":
        raise ValueError("Expected a nonempty, real one-dimensional pressure array in Pa at 100000 Hz.")
    raw = np.asarray(loaded, dtype=np.float64)
    if not np.isfinite(raw).all():
        raise ValueError("TL input pressure must contain only finite values.")
    poles = _load_poles(poles_path)
    probes = list(probe_channels or [])
    res, checks = _simulate(raw, poles, opts, probes)
    if input_hash != _sha256(input_path):
        raise RuntimeError("TL input file changed during simulation.")
    samples = raw.size
    pressure = np.asarray(res.oto_emission[:samples], dtype=np.float64).copy()
    arrays = {
        "input_pa": raw,
        "ear_canal_pressure_pa": pressure,
        "nominal_input_time_s": np.arange(samples, dtype=np.float64) / SAMPLE_RATE_HZ,
        "integrated_state_time_s": (np.arange(samples, dtype=np.float64) + 1) / SAMPLE_RATE_HZ,
        "fs_hz": np.asarray(SAMPLE_RATE_HZ, dtype=np.int64),
    }
    if probes:
        arrays["probe_channels"] = np.asarray(probes, dtype=np.int64)
        arrays["velocity_m_s"] = res.velocity
        arrays["displacement_m"] = res.displacement
    metadata = _base_metadata(source_dir, poles_path, opts)
    metadata.update(
        **checks,
        mode="pressure",
        input_sha256=input_hash,
        input_array_sha256=hashlib.sha256(raw.tobytes(order="C")).hexdigest(),
        maximum_abs_pressure_pa=float(np.max(np.abs(pressure))),
        observable=(
            "Upstream oto_emission, including its native reverse filtering, in Pa. "
            "The response can contain driven components; it is not isolated R-type OAE."
        ),
    )
    return _publish(output_path, metadata_path, arrays, metadata, started)


def run_tone_probes(
    source_dir: Path | None,
    output_path: Path,
    metadata_path: Path,
    *,
    poles_path: Path | None = None,
    protocol: Any = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Tone-probes mode of the original worker (needs ``carfac_ephys`` for the protocol)."""
    from carfac_ephys.pure_tone_protocol import PureToneProtocol, make_tones, measurement_slice

    from . import make_core

    started = time.perf_counter()
    opts = options or {"rtol": 1e-2, "atol": 1e-13, "pole_update": "faithful", "Zweig_irregularities": 1}
    protocol = protocol or PureToneProtocol()
    source_dir = source_dir.resolve() if source_dir is not None else None
    output_path, metadata_path = output_path.resolve(), metadata_path.resolve()
    _check_destinations(output_path, metadata_path, None, source_dir)
    poles_path = _resolve_poles(source_dir, poles_path)
    poles = _load_poles(poles_path)
    tones = make_tones(protocol, SAMPLE_RATE_HZ)
    window = measurement_slice(protocol, SAMPLE_RATE_HZ, time_offset_samples=1)
    f_res = make_core(samplerate=SAMPLE_RATE_HZ, sections=MODEL_CONFIG["sections"], sheraPo=poles).vector("f_resonance")
    cfs = np.asarray(f_res[1:], dtype=np.float64)  # probe_freq="all": cf = f_resonance[1:]
    actual_cfs, channel_indices, rms_values, integration_checks = [], [], [], []
    for frequency, tone in zip(protocol.frequencies_hz, tones, strict=True):
        channel = int(np.argmin(np.abs(cfs - frequency)))
        res, checks = _simulate(np.asarray(tone, dtype=np.float64), poles, opts, [channel + 1])
        measured = np.asarray(res.displacement[: len(tone), 0], dtype=np.float64)[window]
        rms = float(np.sqrt(np.mean(np.square(measured), dtype=np.float64)))
        if not np.isfinite(rms) or rms <= 0:
            raise FloatingPointError(f"Expected finite nonzero TL displacement RMS at {frequency} Hz.")
        actual_cfs.append(float(cfs[channel]))
        channel_indices.append(channel)
        rms_values.append(rms)
        integration_checks.append({"frequency_hz": frequency, **checks})
    arrays = {
        "frequency_hz": np.asarray(protocol.frequencies_hz, dtype=np.float64),
        "actual_cf_hz": np.asarray(actual_cfs, dtype=np.float64),
        "channel_index": np.asarray(channel_indices, dtype=np.int64),
        "bm_rms": np.asarray(rms_values, dtype=np.float64),
        "input_pa": tones,
        "fs_hz": np.asarray(SAMPLE_RATE_HZ, dtype=np.int64),
    }
    metadata = _base_metadata(source_dir, poles_path, opts)
    metadata.update(
        mode="tone-probes",
        protocol=protocol.to_dict(),
        samples=tones.shape[1],
        probes=len(tones),
        input_array_sha256=hashlib.sha256(tones.tobytes(order="C")).hexdigest(),
        integration_checks=integration_checks,
        all_finite={name: bool(np.isfinite(a).all()) for name, a in arrays.items()},
        compute_seconds=sum(c["compute_seconds"] for c in integration_checks),
        warnings=[],
        measurement_sample_slice=[window.start, window.stop],
        cf_mapping="nearest nominal CF by absolute difference in Hz; numpy.argmin first tie",
        bm_observable="displacement",
        bm_rms_unit="m",
        observable="Unwindowed RMS of upstream Ysolution, without mean subtraction.",
    )
    return _publish(output_path, metadata_path, arrays, metadata, started)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, help="Verhulst source directory (provenance and default poles only)")
    parser.add_argument("--input", type=Path, dest="input_path")
    parser.add_argument("--output", type=Path, required=True, dest="output_path")
    parser.add_argument("--metadata", type=Path, required=True, dest="metadata_path")
    parser.add_argument("--mode", choices=("pressure", "tone-probes"), default="pressure")
    parser.add_argument("--poles", type=Path, dest="poles_path")
    parser.add_argument("--protocol", type=Path, help="Pure-tone protocol JSON; tone-probes mode only.")
    parser.add_argument("--rtol", type=float, default=1e-2, help="dopri5 rtol (original: 1e-2)")
    parser.add_argument("--atol", type=float, default=1e-13, help="dopri5 atol (original: 1e-13)")
    parser.add_argument("--smooth", action="store_true", help="Zweig_irregularities=0 (smooth cochlea)")
    parser.add_argument(
        "--pole-update",
        choices=("faithful", "accepted_step"),
        default="faithful",
        help="'accepted_step' is a model variant (pure right-hand side), not the original",
    )
    parser.add_argument("--probe-channels", type=int, nargs="*", default=[], help="section indices to store (pressure mode)")
    args = parser.parse_args(argv)
    if args.mode == "pressure" and (args.input_path is None or args.protocol is not None):
        parser.error("pressure mode requires --input and does not accept --protocol")
    if args.mode == "tone-probes" and (args.input_path is not None or args.protocol is None):
        parser.error("tone-probes mode requires --protocol and does not accept --input")
    opts = _options(args)
    try:
        if args.mode == "pressure":
            metadata = run_reference(
                args.source_dir,
                args.input_path,
                args.output_path,
                args.metadata_path,
                poles_path=args.poles_path,
                options=opts,
                probe_channels=args.probe_channels,
            )
        else:
            from carfac_ephys.pure_tone_protocol import PureToneProtocol

            protocol = PureToneProtocol(**json.loads(args.protocol.read_text()))
            metadata = run_tone_probes(
                args.source_dir, args.output_path, args.metadata_path, poles_path=args.poles_path, protocol=protocol, options=opts
            )
    except (OSError, ValueError, TypeError, RuntimeError, FloatingPointError, ImportError) as error:
        parser.exit(1, f"TL worker failed: {error}\n")
    print(
        json.dumps({k: metadata[k] for k in ("samples", "compute_seconds", "output_sha256", "completed")}, allow_nan=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
