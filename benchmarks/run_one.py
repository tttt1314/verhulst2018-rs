# © copyright 2017 Ghent University – Universiteit Gent, all rights reserved; this Derivative work is
# made available for non-commercial academic research purposes and subject to an UGent Academic
# License (see LICENSE and NOTICE).
"""One timed simulation (used by bench.py; one process per measurement for clean peak RSS)."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tests"))
RS = HERE.parents[1] / "tdlm-snl" / "experiments" / "r_stability"
STIM = {
    "click": RS / "results/integration-precision/root-nh/click-L60/rtol-0.01/irregular/tl-native.npz",
    "swept": RS / "results/swept-precision/native/rtol-0.01/root-nh/irregular/phase-0/tl-native.npz",
}
POLES = {
    "click": json.loads((RS / "data/stimulus-manifest.json").read_text())["conditions"]["root-nh"]["poles_path"],
    "swept": json.loads((RS / "results/stage-b-verhulst-swept/targets/stimulus-manifest.json").read_text())["conditions"]["root-nh"]["poles_path"],
}

p = argparse.ArgumentParser()
p.add_argument("--impl", choices=["python-all", "python-single", "rust", "rust-accepted"], required=True)
p.add_argument("--stim", choices=list(STIM), required=True)
a = p.parse_args()
with np.load(STIM[a.stim]) as z:
    raw, ref = z["input_pa"].copy(), z["ear_canal_pressure_pa"].copy()
poles = np.loadtxt(POLES[a.stim])
padded = np.concatenate((raw, np.zeros(2)))
# imports (scipy, the extension, the original module) happen before the timer
from scipy import signal  # noqa: F401
if a.impl.startswith("python"):
    from reference import MODEL_CONFIG, load_original
    mod = load_original()
else:
    import verhulst2018_rs as vr
t0 = time.perf_counter()
if a.impl.startswith("python"):
    m = mod.cochlea_model()
    if a.impl == "python-all":
        m.init_model(padded, sheraPo=poles, probe_freq="all", **MODEL_CONFIG)
    else:
        m.init_model(padded, sheraPo=poles, probe_freq="abr", **MODEL_CONFIG)
        m.probe_freq, m.probe_points = "single", np.array([500])
    m.solve()
    out = m.oto_emission[: raw.size]
else:
    r = vr.simulate_pressure(raw, poles, pole_update="faithful" if a.impl == "rust" else "accepted_step")
    out = r.oto_emission
el = time.perf_counter() - t0
print(json.dumps({"impl": a.impl, "stim": a.stim, "samples": int(raw.size), "seconds": el,
                  "bitwise_equal_to_stored_original": bool(np.array_equal(out, ref))}))
