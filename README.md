# verhulst2018-rs

A Rust rewrite of the cochlear core of the Verhulst et al. (2018) transmission-line cochlear model, with Python bindings (PyO3 + maturin + rust-numpy).

> © copyright 2017 Ghent University – Universiteit Gent, all rights reserved; this Derivative work is made available for non-commercial academic research purposes and subject to an UGent Academic License.
>
> This repository is a derivative work of `cochlear_model2018.py` and `cochlea_utils.c` from Verhulstetal2018Model v1.2 (upstream commit `ed8d576`). It may be used **for non-commercial academic research only**. The full license text is in [LICENSE](LICENSE). The changes are listed in [NOTICE](NOTICE).

## Highlights

- **On Apple arm64, the default faithful mode produces output identical, bit for bit, to the original.** This is exact equality, not a small error. Across 82 stored runs of the original (click at rtol 1e-2 to 1e-8; four-phase sweep at rtol 1e-2 and 1e-4), the output `ear_canal_pressure_pa` is exactly equal (max |diff| = 0), and the number of RHS calls is the same.
- **Low memory.** It stores only `oto_emission` and the probe channels you ask for, and the delay line uses a per-channel ring buffer. Peak memory for the whole process is about 130 MB. The original, storing all channels, needs about 2.7–3.3 GB for the sweep.
- **Configurable tolerances.** `rtol` and `atol` default to 1e-2 and 1e-13, the values hard-coded in the original.
- **Optional "pure RHS" variant.** `pole_update="accepted_step"` is clearly labelled as a different model and is **not** the default.
- **Drop-in worker.** The command line and output `.npz` of `python -m verhulst2018_rs.worker` match `carfac_ephys/tl_reference_worker.py`.
- **Speed: a single run is 4.8× faster (click 0.70 s vs 3.33 s; sweep 39.6 s vs 189.9 s). This misses the 10× target.** [Speed](#speed) explains why.

## Installation

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install maturin numpy scipy pytest
maturin develop --release
```

When building inside Google Drive, keep the Rust build directory out of the synced folder: `export CARGO_TARGET_DIR=$HOME/.cache/verhulst2018-rs-target`.

## Usage

```python
import numpy as np, verhulst2018_rs as vr
poles = np.loadtxt(".../StartingPoles.dat")
r = vr.simulate_pressure(raw_pa_100k, poles)            # worker convention: pads 2 zeros, output length = len(raw)
r.oto_emission                                           # equals the original's model.oto_emission[:len(raw)]
r = vr.simulate(stim, 100_000, 1000, poles, probe_freq=[1000.0], rtol=1e-4,
                Zweig_irregularities=0)                  # counterpart of init_model(...) + solve()
```

**Arguments.** `simulate` takes the same argument names as the original `init_model`: `compression_slope`, `Zweig_irregularities`, `non_linearity_type`, `KneeVar`, `low_freq_irregularities`, `subject` and `IrrPct`. It also accepts `rtol`, `atol`, `nsteps`, `pole_update`, `probe_channels` and `compact_delay`.

**Filters.** The Puria middle-ear filter and the 600–4000 Hz emission filter still use SciPy and are called the same way as in the original.

**Randomness.** `Rth` comes from NumPy's `RandomState(subject)`, the same random stream as the original's `np.random.seed`, but this package leaves the global random generator untouched.

Worker:

```bash
python -m verhulst2018_rs.worker --source-dir <Verhulstetal2018Model> --input stim.npy \
  --output out.npz --metadata out.json --poles poles.dat [--rtol 1e-4] [--smooth] \
  [--pole-update accepted_step] [--probe-channels 500]
```

- **`--source-dir`** is used only to hash the source files (for provenance) and to find the default poles file. The original code is never loaded.
- **Metadata** records the `rtol` actually used, package versions, the git commit and a hash of the `_core` library.
- **`tone-probes` mode** needs `carfac_ephys`, which reads the protocol.

## Fidelity: method

The original's numbers depend on how each of its three components was compiled. This port matches each one:

| Original component | How it is handled here |
|---|---|
| NumPy code (`TLsolver`, `SheraParameters`, `ZweigImpedance`, init) | Written in the same operation order as the original, without FMA (NumPy ufuncs do not fuse multiply-add). `exp`, `pow`, `atan`, `cos` etc. call the same libm. |
| `tridiag.so` (`gcc -O3 -ffast-math`, Apple clang arm64) | Actual instruction sequences confirmed by `otool` disassembly: `cubic_interpolate` takes **float32** arguments; `a0` is re-associated as `(y1-(y0+y2))+y3`; the cubic is fused into 3 FMAs; `b - a*c` in the Thomas algorithm uses FMA. |
| SciPy 1.18.1 `dopri5` (a **C translation** of `scipy/integrate/src/dop.c`) | Its control flow is reproduced: safety 0.9, fac1 0.2, fac2 10, beta 0.04, nmax 500, HINIT, stiffness detection. Clang's FP-contraction rule (left operand checked first) was also confirmed from disassembly. Note: the C wrapper writes back `work[6]=h` using the **input** h (0), so **HINIT re-runs on every output sample**. This is where the ~9 RHS calls per sample come from. |
| RHS side effects | The 1% pole-update rule is applied on every evaluation, including the HINIT call and trial stages. The floating-point round-trip error of `Dev += frac` / `-= frac` stays in `Dev`. The delay line and `ZweigImpedance` are updated after every output sample. `self.V1[0]=self.Vtmp[0]` writes back into `r.y` through a view, and this port does the same. |

These optimizations do not change results; each was verified bit for bit:

- the tridiagonal factorization is cached (the matrix is constant);
- `delay_line`, `calculate_g`, `calculate_right` and the forward sweep are fused into one loop;
- the delay line stores float32 (the original also converts to float32 before computing);
- per-channel ring buffers, with a runtime check of the required delay length that raises an error if it is exceeded;
- the 5-point window is gathered once per sample, with the cubic coefficients precomputed;
- the 1% rule is screened with multiplications first, falling back to the original's division test only when the result is near the threshold.

The screen's error is about 1e-13, and the threshold keeps a relative margin of 1e-7.

## Fidelity: validation results

### Unit tests (`tests/test_against_original.py`, all 26 pass)

These compare directly against the original and all require **bitwise equality**. `cubic_interpolate`, `delay_line` and `solve_tridiagonal` are called in the original `tridiag.so` through ctypes.

- **Kernels:** cubic interpolation (5000 random sets), the delay line (both branches covered) and the tridiagonal solve.
- **State after `init_model`:** all 34 vectors and 22 scalars, across 6 configurations (smooth, `low_freq_irregularities=0`, disp, linear, and other subject/IrrPct/KneeVar/slope values).
- **Impedance:** `SheraParameters` and `ZweigImpedance` with random poles, including the wrap-around of `Zrp*`.
- **A single `TLsolver` evaluation:** the output derivative, then `SheraP`, `Dev`, `YZweig`, `g`, `Qsol` and `Zrp1`. Each is tested at three intensities: no update, update, and saturation.
- **Short end-to-end simulations:**
  - linear (with an extra check of relative error ≤ 1e-6);
  - nonlinear click at three intensities;
  - smooth, disp, and `low_freq_irregularities=0`;
  - rtol 1e-4 and 1e-6;
  - a ramped pure tone, which triggers pole updates.

  These also compare the number of RHS calls and the probe-channel V/Y.

On the Rust side, `cargo test` covers the Thomas algorithm, cubic, DOPRI5 (harmonic oscillator) and the remainder equivalence.

### Comparison with stored original runs (`validation/run_validation.py`, results in `validation/results/parity.json`)

| Data | Runs | Faithful mode vs original |
|---|---|---|
| click, 2 conditions × L40–L60 × irregular/smooth, rtol 1e-2 and 1e-4 (L45 also at 1e-6; post-hoc 1e-6 and 1e-8) | 54 | **54/54 bit-identical** |
| four-phase L60 sweep (208,732 samples), 2 conditions × irregular/smooth × 4 phases, rtol 1e-2 and 1e-4 | 32 | **32/32 bit-identical** |

(The sweep at rtol 1e-6 was not re-run because it takes too long.)

### Nonlinear acceptance criterion (`validation/analyze_rs.py`, scored with the r_stability pipeline, results in `validation/results/acceptance.json`)

- **Click scoring:** r−s, using the stored 48 kHz conversion, a 0.8–5 kHz filter and the window 2–60 ms after onset.
- **Sweep scoring:** the four-phase rwCR crop of r−s.
- **Numbers:** relative L2 differences against the original at rtol 1e-2.
- **Criterion:** difference ≤ the original's own maximum variation across rtol 1e-2 / 1e-4 / 1e-6. Thresholds were not tuned to the results.

| Case | Original's rtol variation | Faithful mode | accepted_step variant |
|---|---|---|---|
| NH click L40 | 0.010 | **0** | 0.000 ✓ |
| NH click L45 | 0.120 | **0** | 0.244 ✗ |
| NH click L50 | 0.050 | **0** | 0.058 ✗ |
| NH click L55 | 0.018 | **0** | 0.033 ✗ |
| NH click L60 | 0.012 | **0** | 0.035 ✗ |
| Slope20 click L40 | 0.003 | **0** | 0.000 ✓ |
| Slope20 click L45 | 0.004 | **0** | 0.000 ✓ |
| Slope20 click L50 | 0.396 | **0** | 0.406 ✗ |
| Slope20 click L55 | 0.246 | **0** | 0.028 ✓ |
| Slope20 click L60 | 0.049 | **0** | 0.117 ✗ |
| NH sweep r−s rwCR | 0.006 | **0** | 0.117 ✗ (corr 0.993) |
| Slope20 sweep r−s rwCR | 0.012 | **0** | 0.084 ✗ (corr 0.996) |

Conclusions:

- **Faithful mode passes every case** with a difference of 0.
- **The accepted_step variant is a genuinely different model.** Its sweep rwCR differs from the original by 8–12%, roughly 10× the original's numerical noise floor. It **cannot** replace the original. It is useful only for studying the effect of pole-update timing itself.
- **L40 shows 0 for the accepted_step variant** because no pole updates happen at that intensity.

## Speed

**Setup:**

- **Machine:** Apple M1 Pro (10 cores, 32 GB), Python 3.12, NumPy 2.5.3, SciPy 1.18.1.
- **Procedure:** the original and the Rust version were run one after the other on the same machine with `benchmarks/bench.py`. Raw data is in `benchmarks/results.json`.
- **Caveat:** other programs were running in the background, so absolute times vary by about ±10%.

| Stimulus | Implementation | Time (median) | vs original | Peak memory (whole process) | Bit-identical to stored original |
|---|---|---|---|---|---|
| click, 8,200 samples | original, `probe_freq="all"` (worker setting) | 3.33 s | 1× | 381 MB | ✓ |
| | original, single probe | 3.36 s | 1× | 318 MB | ✓ |
| | **Rust faithful** | **0.70 s** | **4.8×** | 127 MB | ✓ |
| | Rust accepted_step (different model) | 0.60 s | 5.6× | 127 MB | ✗ |
| L60 sweep, 208,732 samples | original, `probe_freq="all"` | 189.9 s | 1× | **2,677 MB** | ✓ |
| | original, single probe | 187.1 s | 1× | 325 MB | ✓ |
| | **Rust faithful** | **39.6 s** | **4.8×** | 136 MB | ✓ |
| | Rust accepted_step (different model) | 26.3 s | 7.2× | 136 MB | ✗ |

Click was repeated 3 times and the sweep run once each. Times exclude imports. Most of Rust's peak memory is Python, NumPy and SciPy themselves; the model state is about 10 MB.

**Why 10× was not reached:**

To stay bit-identical, every RHS call must run the roughly 1000 dependent steps of the Thomas algorithm in order. Each forward step is one FMA plus one multiply, and each backward step is one FMA. On the M1 this dependency chain takes about 12k cycles, or about 4 µs.

On top of that:

- the DOPRI5 error norm must add its terms one by one in a fixed order;
- HINIT adds about 6000 sequential additions per sample.

Together these set a floor of about 0.39 s for the click. Against the measured 3.33 s, that allows at most about 8.5×, and only if everything else took no time. The current cost is about 11 µs per RHS (including the integrator), roughly half of it this sequential work that cannot be reordered.

**The fast-arithmetic attempt.** I also tried a version that drops bit-exactness: a SPIKE partitioned tridiagonal solver, a polynomial exp, and multi-accumulator norms. Replacing only the Thomas algorithm gave no speedup, because:

1. the fused loop already hides the other work inside the latency of the Thomas dependency chain;
2. at rtol = 1e-2, any rounding difference changes step-size decisions, which pushes the linear-case difference to 3e-5, failing the ≤ 1e-6 requirement.

That version was therefore removed.

**Memory may matter more in practice than single-run speed.** Each run needs about 130 MB (vs about 2.7–3.3 GB for the original), so all 10 cores can run in parallel without swapping. A batch of 48 sweeps that took 39 minutes on the original because of swapping should take an estimated 5–8 minutes with the same 6–8 parallel workers.

## Known differences and limitations

- **Platform.** Bit-identity has been verified only on Apple arm64, with this machine's `tridiag.so` and the SciPy 1.18.1 wheel. On other platforms `fma` falls back to a plain multiply-then-add. The original's compiled code may also differ there (e.g. different fast-math reordering), so expect the same algorithm with tiny numerical differences.
- **Global random state.** The original `init_model` calls `np.random.seed(subject)`, which changes the global random generator state; this port does not.
- **No all-channel histories.** There is no all-channel `Vsolution`/`Ysolution` as in the original. Request the channels you need with `probe_channels` or `probe_freq`. `probe_freq` also accepts `"all"`, at a corresponding memory cost.
- **Delay buffers.** With `compact_delay=True` (the default), a simulation raises an error if the poles require a delay longer than that channel's buffer. In practice this does not happen, because SheraMu ≤ 1.9 < SheraMuMax = 3. `compact_delay=False` uses the original buffer layout.
- **`np.float`.** The original's `probe_freq` path for a frequency list uses `np.float`, which no longer exists in NumPy 2. This port uses `argmin`, with the same semantics.

## License

Under Section 4 of the UGent Academic License, this derivative work:

- may be distributed only for non-commercial academic research purposes;
- must be distributed with the same license ([LICENSE](LICENSE));
- must mark every modified file as modified (every derived file starts with this notice);
- must keep the copyright notice quoted above.

Note Section 2: **the original Work itself** may not be redistributed without UGent's written consent. This repository does not contain the original code. The tests load it at runtime from your local copy of the original (`VERHULST_SOURCE_DIR`).
