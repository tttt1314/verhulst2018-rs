# © copyright 2017 Ghent University – Universiteit Gent, all rights reserved; this Derivative work is
# made available for non-commercial academic research purposes and subject to an UGent Academic
# License (see LICENSE and NOTICE).
"""Nonlinear acceptance metrics, computed with the r_stability pipeline itself.

Run with the tdlm-snl interpreter (needs jax and the r_stability modules):

    ../tdlm-snl/.venv/bin/python validation/analyze_rs.py

Click: r-s (irregular - smooth) after the stored 100 -> 48 kHz conversion, 0.8-5 kHz zero-phase
filter, 2-60 ms after onset (integration_precision_check.scored). Swept: r-s rwCR crop of the
4-phase sum (swept_precision_check pipeline). For every case the difference between
verhulst2018-rs and the original at rtol 1e-2 is compared with the original's own variation
across rtol 1e-2 / 1e-4 / 1e-6 (stored runs). Thresholds are not tuned: the criterion is
"difference <= max original rtol variation" for each case, reported with all numbers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RS = HERE.parents[1] / "tdlm-snl" / "experiments" / "r_stability"
sys.path.insert(0, str(RS))
NATIVE = HERE / "results" / "native"


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def load(path):
    with np.load(path) as z:
        return np.asarray(z["ear_canal_pressure_pa"], float)


def click_metrics():
    import stage2d_decomp as sd
    from integration_precision_check import scored

    manifest = sd.load_manifest()
    cases = {(c["condition"], c["case_id"]): c for c in manifest["cases"] if c["family"] == "click"}
    with np.load(RS / "data/click.npz") as z:
        time_s, onset_s = z["time_s"].copy(), float(z["click_onset_s"])

    def orig_path(cond, level, rtol, variant):
        for root in ("integration-precision", "integration-precision-posthoc"):
            p = RS / "results" / root / cond / f"click-L{level}" / f"rtol-{rtol:g}" / variant / "tl-native.npz"
            if p.exists():
                return p
        return None

    def ours_path(cond, level, rtol, variant, mode):
        for root in ("integration-precision", "integration-precision-posthoc"):
            p = NATIVE / root / cond / f"click-L{level}" / f"rtol-{rtol:g}" / variant / f"{mode}.npz"
            if p.exists():
                return p
        return None

    out = {}
    for cond in ("root-nh", "upstream-slope20"):
        for level in (40, 45, 50, 55, 60):
            case = cases[(cond, f"click-L{level}")]

            def rs(paths):
                irr, smo = (sd.convert_native(load(p), case) for p in paths)
                return scored(irr - smo, time_s, onset_s)

            orig = {}
            for rtol in (1e-2, 1e-4, 1e-6, 1e-8):
                pi, ps = orig_path(cond, level, rtol, "irregular"), orig_path(cond, level, rtol, "smooth")
                if pi and ps:
                    orig[rtol] = rs((pi, ps))
            ref = orig[1e-2]
            variation = {f"{r:g}": rel(orig[r], ref) for r in orig if r != 1e-2}
            pair = {f"{a:g}-{b:g}": rel(orig[a], orig[b]) for a in orig for b in orig if a > b}
            rec = {"orig_rtol_vs_1e-2": variation, "orig_pairwise": pair,
                   "orig_max_variation_vs_1e-2": max(variation.values())}
            for mode in ("faithful", "accepted_step"):
                pi = ours_path(cond, level, 1e-2, "irregular", mode)
                ps = ours_path(cond, level, 1e-2, "smooth", mode)
                if pi and ps:
                    rec[f"rust_{mode}_vs_orig_1e-2"] = rel(rs((pi, ps)), ref)
            # faithful mode at the other stored tolerances
            rec["rust_faithful_vs_orig_same_rtol"] = {}
            for rtol in orig:
                pi = ours_path(cond, level, rtol, "irregular", "faithful")
                ps = ours_path(cond, level, rtol, "smooth", "faithful")
                if pi and ps:
                    rec["rust_faithful_vs_orig_same_rtol"][f"{rtol:g}"] = rel(rs((pi, ps)), orig[rtol])
            for mode in ("faithful", "accepted_step"):
                k = f"rust_{mode}_vs_orig_1e-2"
                if k in rec:
                    rec[f"{mode}_within_orig_variation"] = rec[k] <= rec["orig_max_variation_vs_1e-2"]
            out[f"{cond}/L{level}"] = rec
    return out


def swept_metrics():
    import jax
    import swept_basis
    from prepare_stage_b_swept_targets import resample_100k_to_48k

    with jax.enable_x64(True):
        plan = swept_basis.prepare_split_wavelet_plan(signal_length=100_095, dtype=np.float64)

    def extract(x):
        with jax.enable_x64(True):
            return swept_basis.extract_wavelet_columns(x, plan, "cr", crop=True)

    def rs_cr(path_of):
        sums = {}
        for v in ("irregular", "smooth"):
            phases = [resample_100k_to_48k(load(path_of(v, p)), beta=14.0, prepend_zero=True) for p in range(4)]
            sums[v] = np.sum(phases, axis=0)
        return extract(sums["irregular"] - sums["smooth"])

    out = {}
    for cond in ("root-nh", "upstream-slope20"):
        orig = {r: rs_cr(lambda v, p, r=r: RS / "results/swept-precision/native" / f"rtol-{r:g}" / cond / v / f"phase-{p}" / "tl-native.npz")
                for r in (1e-2, 1e-4, 1e-6)}
        ref = orig[1e-2]
        variation = {f"{r:g}": rel(orig[r], ref) for r in (1e-4, 1e-6)}
        rec = {"orig_rtol_vs_1e-2": variation, "orig_1e-4_vs_1e-6": rel(orig[1e-4], orig[1e-6]),
               "orig_max_variation_vs_1e-2": max(variation.values())}
        for mode, rtol in (("faithful", 1e-2), ("accepted_step", 1e-2), ("faithful", 1e-4)):
            paths = lambda v, p, mode=mode, rtol=rtol: NATIVE / "swept" / cond / f"phase-{p}" / f"rtol-{rtol:g}" / v / f"{mode}.npz"
            if all(paths(v, p).exists() for v in ("irregular", "smooth") for p in range(4)):
                ours = rs_cr(paths)
                rec[f"rust_{mode}_rtol{rtol:g}_vs_orig_same_rtol"] = rel(ours, orig[rtol])
                if rtol == 1e-2:
                    rec[f"rust_{mode}_vs_orig_1e-2"] = rel(ours, ref)
                    rec[f"{mode}_within_orig_variation"] = rel(ours, ref) <= rec["orig_max_variation_vs_1e-2"]
                    rec[f"rust_{mode}_corr_vs_orig_1e-2"] = float(np.corrcoef(ours, ref)[0, 1])
        out[cond] = rec
    return out


def main():
    res = {"click_rs_scored": click_metrics(), "swept_rs_rwcr": swept_metrics()}
    (HERE / "results" / "acceptance.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
