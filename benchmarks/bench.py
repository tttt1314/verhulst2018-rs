# © copyright 2017 Ghent University – Universiteit Gent, all rights reserved; this Derivative work is
# made available for non-commercial academic research purposes and subject to an UGent Academic
# License (see LICENSE and NOTICE).
"""Benchmark the original Python model against verhulst2018-rs on the same machine.

Click: NH L60 click, 8,200 samples. Swept: L60 4-phase sweep, phase 0, 208,732 samples.
Each measurement is a fresh process run under /usr/bin/time -l (wall time inside the
process, peak resident memory from the OS). Results: benchmarks/results.json.

    python benchmarks/bench.py [--repeats 3] [--skip-python-swept]
"""
import argparse
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def measure(impl, stim):
    cmd = ["/usr/bin/time", "-l", sys.executable, str(HERE / "run_one.py"), "--impl", impl, "--stim", stim]
    p = subprocess.run(cmd, capture_output=True, text=True, check=True)
    rec = json.loads(p.stdout.strip().splitlines()[-1])
    rss = re.search(r"(\d+)\s+maximum resident set size", p.stderr)
    rec["peak_rss_mb"] = int(rss.group(1)) / 2**20 if rss else None
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--skip-python-swept", action="store_true")
    args = ap.parse_args()
    out = {"machine": platform.platform(), "cpu": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
           capture_output=True, text=True).stdout.strip(), "python": platform.python_version(),
           "started": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": []}
    plan = [(i, "click") for i in ("python-all", "python-single", "rust", "rust-accepted") for _ in range(args.repeats)]
    plan += [("rust", "swept"), ("rust-accepted", "swept")]
    if not args.skip_python_swept:
        plan += [("python-single", "swept"), ("python-all", "swept")]
    for impl, stim in plan:
        rec = measure(impl, stim)
        out["runs"].append(rec)
        print(json.dumps(rec), flush=True)
        (HERE / "results.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
