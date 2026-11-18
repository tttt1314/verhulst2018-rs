# © copyright 2017 Ghent University – Universiteit Gent, all rights reserved; this Derivative work is
# made available for non-commercial academic research purposes and subject to an UGent Academic
# License (see LICENSE and NOTICE).
"""Re-simulate every stored original Verhulst run of the r_stability precision checks.

For each stored native output of the original model (click: integration-precision and its
post-hoc runs; swept: swept-precision), run verhulst2018-rs with the same stimulus, poles,
irregularity setting and rtol, and record (i) bitwise parity of the ear-canal pressure in the
default (faithful) mode and (ii) the native outputs of the accepted-step variant at rtol 1e-2.
Outputs go to validation/results/ (native arrays) and validation/results/parity.json.

    python validation/run_validation.py --workers 6 [--families click swept]
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RS = HERE.parents[1] / "tdlm-snl" / "experiments" / "r_stability"
OUT = HERE / "results"
CLICK_MANIFEST = RS / "data" / "stimulus-manifest.json"
SWEPT_MANIFEST = RS / "results" / "stage-b-verhulst-swept" / "targets" / "stimulus-manifest.json"


def tasks(families):
    out = []
    if "click" in families:
        poles = {c: v["poles_path"] for c, v in json.loads(CLICK_MANIFEST.read_text())["conditions"].items()}
        for root in ("integration-precision", "integration-precision-posthoc"):
            for ref in sorted((RS / "results" / root).glob("*/click-L*/rtol-*/*/tl-native.npz")):
                cond, case, rtol_dir, variant = ref.parts[-5:-1]
                rtol = float(rtol_dir.removeprefix("rtol-"))
                for mode in ("faithful",) + (("accepted_step",) if rtol == 1e-2 and root == "integration-precision" else ()):
                    out.append(dict(family="click", cond=cond, case=case, rtol=rtol, variant=variant, mode=mode,
                                    ref=str(ref), poles=poles[cond], key=f"{root}/{cond}/{case}/rtol-{rtol:g}/{variant}/{mode}"))
    if "swept" in families:
        poles = {c: v["poles_path"] for c, v in json.loads(SWEPT_MANIFEST.read_text())["conditions"].items()}
        for ref in sorted((RS / "results" / "swept-precision" / "native").glob("rtol-*/*/*/phase-*/tl-native.npz")):
            rtol_dir, cond, variant, phase = ref.parts[-5:-1]
            rtol = float(rtol_dir.removeprefix("rtol-"))
            if rtol < 1e-4:
                continue  # rtol 1e-6 swept runs are long; parity is shown at 1e-2 and 1e-4
            for mode in ("faithful",) + (("accepted_step",) if rtol == 1e-2 else ()):
                out.append(dict(family="swept", cond=cond, case=phase, rtol=rtol, variant=variant, mode=mode,
                                ref=str(ref), poles=poles[cond], key=f"swept/{cond}/{phase}/rtol-{rtol:g}/{variant}/{mode}"))
    return out


def run(task):
    import verhulst2018_rs as vr

    ref = np.load(task["ref"])
    raw = ref["input_pa"]
    poles = np.loadtxt(task["poles"])
    started = time.perf_counter()
    res = vr.simulate_pressure(raw, poles, rtol=task["rtol"], pole_update=task["mode"],
                               Zweig_irregularities=0 if task["variant"] == "smooth" else 1)
    seconds = time.perf_counter() - started
    a = np.asarray(ref["ear_canal_pressure_pa"])
    b = res.oto_emission
    dest = OUT / "native" / (task["key"] + ".npz")
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, ear_canal_pressure_pa=b)
    return {
        **{k: task[k] for k in ("family", "cond", "case", "rtol", "variant", "mode", "key")},
        "seconds": seconds,
        "stats": res.stats,
        "bitwise_equal": bool(np.array_equal(a, b)),
        "max_abs_diff": float(np.max(np.abs(a - b))),
        "rel_l2_diff": float(np.linalg.norm(a - b) / np.linalg.norm(a)),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--families", nargs="+", default=["click", "swept"])
    args = p.parse_args()
    todo = tasks(args.families)
    todo.sort(key=lambda t: (t["family"] != "swept", -t["rtol"] if t["family"] == "swept" else 0))  # long ones first
    OUT.mkdir(exist_ok=True)
    results = {}
    path = OUT / "parity.json"
    if path.exists():
        results = json.loads(path.read_text())
    todo = [t for t in todo if t["key"] not in results]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run, t): t for t in todo}
        for f in as_completed(futs):
            r = f.result()
            results[r["key"]] = r
            path.write_text(json.dumps(results, indent=1, sort_keys=True))
            print(f"[{time.time() - t0:6.0f}s] {r['key']}: equal={r['bitwise_equal']} rel={r['rel_l2_diff']:.2e} ({r['seconds']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
