// © copyright 2017 Ghent University – Universiteit Gent, all rights reserved; this Derivative work is
// made available for non-commercial academic research purposes and subject to an UGent Academic
// License (see LICENSE and NOTICE).
// Changed files notice: this file replaces the `scipy.integrate.ode('dopri5')` call of
// cochlear_model2018.py (Verhulstetal2018Model v1.2) with a Rust implementation (2026).
//
// DOPRI5 of E. Hairer & G. Wanner, following the control flow of SciPy 1.18.1
// (scipy/integrate/src/dop.c, a C translation of the Fortran code) as called by
// `scipy.integrate.ode(...).set_integrator('dopri5', rtol, atol)` with all other options at their
// defaults: nsteps = 500, safety = 0.9, fac1 (dfactor) = 0.2, fac2 (ifactor) = 10, beta = 0 → 0.04,
// first_step = 0, max_step = 0 (→ xend − x), no dense output, scalar tolerances (itol = 0).
//
// Two behaviours of that call path matter for faithfulness:
// * Every `ode.integrate(t + dt)` is an independent `dopri5()` call. SciPy 1.18's C wrapper
//   writes `work[6] = h` with the *input* step (0), so each output sample recomputes the initial
//   step with HINIT, which costs one extra right-hand-side evaluation with side effects.
// * The SciPy wheel is compiled with FP contraction, so `a + b*c` patterns are fused
//   multiply-adds. The order of the fused operations follows clang's rule (left operand
//   checked first) and was verified against the arm64 disassembly of `_dop.cpython-312-darwin.so`.

use crate::kernels::fma;

pub trait OdeSystem {
    /// Right-hand side. May mutate the system (the Verhulst model does).
    fn rhs(&mut self, t: f64, y: &[f64], dy: &mut [f64]);
    /// Called after every accepted step with the new state. Returns `true` if the system changed
    /// in a way that invalidates the FSAL derivative (it is then re-evaluated).
    fn accepted(&mut self, _t: f64, _y: &[f64]) -> bool {
        false
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub rtol: f64,
    pub atol: f64,
    pub nmax: i64,
    pub safe: f64,
    pub fac1: f64,
    pub fac2: f64,
    pub beta: f64,
    pub uround: f64,
    pub nstiff: i64,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            rtol: 1e-2,
            atol: 1e-13,
            nmax: 500,
            safe: 0.9,
            fac1: 0.2,
            fac2: 10.0,
            beta: 0.04,
            uround: 2.220446049250313e-16,
            nstiff: 1000,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub nfcn: u64,
    pub nstep: u64,
    pub naccept: u64,
    pub nreject: u64,
    pub rhs_calls: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DopriError {
    TooManySteps,
    StepTooSmall,
    Stiff,
    /// Not a dopri5 error: a compact delay ring was too short (see `Config::compact_delay`).
    RingTooShort { channel: usize, age: usize },
}

impl DopriError {
    pub fn code(&self) -> i32 {
        match self {
            DopriError::TooManySteps => -2,
            DopriError::StepTooSmall => -3,
            DopriError::Stiff => -4,
            DopriError::RingTooShort { .. } => -100,
        }
    }
    pub fn message(&self) -> &'static str {
        match self {
            DopriError::TooManySteps => "larger nsteps is needed",
            DopriError::StepTooSmall => "step size becomes too small",
            DopriError::Stiff => "problem is probably stiff (interrupted)",
            DopriError::RingTooShort { .. } => "a compact delay ring is too short for the current poles; rerun with compact_delay=False",
        }
    }
}

pub struct Workspace {
    y1: Vec<f64>,
    k1: Vec<f64>,
    k2: Vec<f64>,
    k3: Vec<f64>,
    k4: Vec<f64>,
    k5: Vec<f64>,
    k6: Vec<f64>,
    ysti: Vec<f64>,
    q1: Vec<f64>,
    q2: Vec<f64>,
}

impl Workspace {
    pub fn new(n: usize) -> Self {
        let z = || vec![0.0; n];
        Workspace { y1: z(), k1: z(), k2: z(), k3: z(), k4: z(), k5: z(), k6: z(), ysti: z(), q1: z(), q2: z() }
    }
}



const C2: f64 = 0.2;
const C3: f64 = 0.3;
const C4: f64 = 0.8;
const C5: f64 = 8.0 / 9.0;
const A21: f64 = 0.2;
const A31: f64 = 3.0 / 40.0;
const A32: f64 = 9.0 / 40.0;
const A41: f64 = 44.0 / 45.0;
const A42: f64 = -56.0 / 15.0;
const A43: f64 = 32.0 / 9.0;
const A51: f64 = 19372.0 / 6561.0;
const A52: f64 = -25360.0 / 2187.0;
const A53: f64 = 64448.0 / 6561.0;
const A54: f64 = -212.0 / 729.0;
const A61: f64 = 9017.0 / 3168.0;
const A62: f64 = -355.0 / 33.0;
const A63: f64 = 46732.0 / 5247.0;
const A64: f64 = 49.0 / 176.0;
const A65: f64 = -5103.0 / 18656.0;
const A71: f64 = 35.0 / 384.0;
const A73: f64 = 500.0 / 1113.0;
const A74: f64 = 125.0 / 192.0;
const A75: f64 = -2187.0 / 6784.0;
const A76: f64 = 11.0 / 84.0;
const E1: f64 = 71.0 / 57600.0;
const E3: f64 = -71.0 / 16695.0;
const E4: f64 = 71.0 / 1920.0;
const E5: f64 = -17253.0 / 339200.0;
const E6: f64 = 22.0 / 525.0;
const E7: f64 = -1.0 / 40.0;

extern "C" {
    #[link_name = "pow"]
    fn c_pow(x: f64, y: f64) -> f64;
}

/// libm `pow`, opaque to LLVM's algebraic rewrites (e.g. `pow(x, 0.5) → sqrt`).
#[inline(always)]
pub fn libm_pow(x: f64, y: f64) -> f64 {
    unsafe { c_pow(std::hint::black_box(x), std::hint::black_box(y)) }
}

#[inline(always)]
fn sq(x: f64) -> f64 {
    x * x
}

#[allow(clippy::too_many_arguments)]
fn hinit<S: OdeSystem>(
    sys: &mut S,
    x: f64,
    y: &[f64],
    posneg: f64,
    f0: &[f64],
    f1: &mut [f64],
    y1: &mut [f64],
    iord: i32,
    hmax: f64,
    opt: &Options,
    q1: &mut [f64],
    q2: &mut [f64],
) -> f64 {
    let (atoli, rtoli) = (opt.atol, opt.rtol);
    let _ = (&q1, &q2);
    let n = y.len();
    // Sequential sums in the C loop's order (latency bound: the divisions overlap the chain).
    let mut dnf = 0.0;
    let mut dny = 0.0;
    for i in 0..n {
        let sk = fma(rtoli, y[i].abs(), atoli);
        dnf += sq(f0[i] / sk);
        dny += sq(y[i] / sk);
    }
    let mut h = if dnf <= 1.0e-10 || dny <= 1.0e-10 { 1.0e-6 } else { (dny / dnf).sqrt() * 0.01 };
    h = h.min(hmax);
    h = h.copysign(posneg);
    for i in 0..y.len() {
        y1[i] = fma(h, f0[i], y[i]);
    }
    sys.rhs(x + h, y1, f1);
    let mut der2 = 0.0;
    for i in 0..n {
        let sk = fma(rtoli, y[i].abs(), atoli);
        der2 += sq((f1[i] - f0[i]) / sk);
    }
    der2 = der2.sqrt() / h;
    let der12 = der2.abs().max(dnf.sqrt());
    let h1 = if der12 <= 1.0e-15 {
        (1.0e-6f64).max(h.abs() * 1.0e-3)
    } else {
        libm_pow(0.01 / der12, 1.0 / iord as f64)
    };
    h = (100.0 * h.abs()).min(h1.min(hmax));
    h.copysign(posneg)
}

/// One `ode.integrate(xend)` call: a fresh `dopri5(fcn, x, y, xend, ...)` with `h = 0`.
/// On success `x == xend` (bitwise, as in SciPy) and `y` holds the solution.
pub fn integrate<S: OdeSystem>(
    sys: &mut S,
    x: &mut f64,
    y: &mut [f64],
    xend: f64,
    opt: &Options,
    ws: &mut Workspace,
    stats: &mut Stats,
) -> Result<(), DopriError> {
    let n = y.len();
    let Workspace { y1, k1, k2, k3, k4, k5, k6, ysti, q1, q2 } = ws;
    let (q1, q2) = (&mut q1[..n], &mut q2[..n]);
    let (y1, k1, k2, k3, k4, k5, k6, ysti) = (
        &mut y1[..n],
        &mut k1[..n],
        &mut k2[..n],
        &mut k3[..n],
        &mut k4[..n],
        &mut k5[..n],
        &mut k6[..n],
        &mut ysti[..n],
    );

    let beta = opt.beta;
    let mut hmax = xend - *x;
    let mut h = 0.0f64;
    let (atoli, rtoli) = (opt.atol, opt.rtol);

    // ---- dopcor ----
    let mut facold = 1.0e-4;
    let expo1 = fma(beta, -0.75, 0.2);
    let facc1 = 1.0 / opt.fac1;
    let facc2 = 1.0 / opt.fac2;
    let posneg = 1.0f64.copysign(xend - *x);
    let mut last = false;
    let mut hlamb = 0.0;
    let mut iasti = 0;
    let mut nonsti = 0;
    let mut naccept: i64 = 0;
    let mut nstep: i64 = 0;

    sys.rhs(*x, y, k1);
    stats.rhs_calls += 1;
    hmax = hmax.abs();
    let iord = 5;
    if h == 0.0 {
        h = hinit(sys, *x, y, posneg, k1, k2, k3, iord, hmax, opt, q1, q2);
        stats.rhs_calls += 1;
    }
    stats.nfcn += 2;
    let mut reject = false;

    loop {
        if nstep > opt.nmax {
            return Err(DopriError::TooManySteps);
        }
        if 0.1 * h.abs() <= x.abs() * opt.uround {
            return Err(DopriError::StepTooSmall);
        }
        if (fma(1.01, h, *x) - xend) * posneg > 0.0 {
            h = xend - *x;
            last = true;
        }
        nstep += 1;
        stats.nstep += 1;

        let ha21 = h * A21;
        for i in 0..n {
            y1[i] = fma(ha21, k1[i], y[i]);
        }
        sys.rhs(fma(C2, h, *x), y1, k2);

        for i in 0..n {
            y1[i] = fma(h, fma(A31, k1[i], A32 * k2[i]), y[i]);
        }
        sys.rhs(fma(C3, h, *x), y1, k3);

        for i in 0..n {
            y1[i] = fma(h, fma(A43, k3[i], fma(A41, k1[i], A42 * k2[i])), y[i]);
        }
        sys.rhs(fma(C4, h, *x), y1, k4);

        for i in 0..n {
            let s = fma(A54, k4[i], fma(A53, k3[i], fma(A51, k1[i], A52 * k2[i])));
            y1[i] = fma(h, s, y[i]);
        }
        sys.rhs(fma(C5, h, *x), y1, k5);

        for i in 0..n {
            let s = fma(A65, k5[i], fma(A64, k4[i], fma(A63, k3[i], fma(A61, k1[i], A62 * k2[i]))));
            ysti[i] = fma(h, s, y[i]);
        }
        let xph = *x + h;
        sys.rhs(xph, ysti, k6);

        for i in 0..n {
            let s = fma(A76, k6[i], fma(A75, k5[i], fma(A74, k4[i], fma(A71, k1[i], A73 * k3[i]))));
            y1[i] = fma(h, s, y[i]);
        }
        sys.rhs(xph, y1, k2);
        stats.rhs_calls += 6;

        for i in 0..n {
            let s = fma(E7, k2[i], fma(E6, k6[i], fma(E5, k5[i], fma(E4, k4[i], fma(E1, k1[i], E3 * k3[i])))));
            k4[i] = s * h;
        }
        stats.nfcn += 6;
        let mut err = 0.0;
        for i in 0..n {
            let sk = fma(rtoli, y[i].abs().max(y1[i].abs()), atoli);
            err += sq(k4[i] / sk);
        }
        err = (err / n as f64).sqrt();

        let fac11 = libm_pow(err, expo1);
        let mut fac = fac11 / libm_pow(facold, beta);
        fac = facc2.max(facc1.min(fac / opt.safe));
        let mut hnew = h / fac;

        if err <= 1.0 {
            facold = err.max(1.0e-4);
            naccept += 1;
            stats.naccept += 1;
            if naccept % opt.nstiff == 0 || iasti > 0 {
                let mut stnum = 0.0;
                let mut stden = 0.0;
                for i in 0..n {
                    stnum += sq(k2[i] - k6[i]);
                    stden += sq(y1[i] - ysti[i]);
                }
                if stden > 0.0 {
                    hlamb = h * (stnum / stden).sqrt();
                }
                if hlamb > 3.25 {
                    nonsti = 0;
                    iasti += 1;
                    if iasti == 15 {
                        return Err(DopriError::Stiff);
                    }
                } else {
                    nonsti += 1;
                    if nonsti == 6 {
                        iasti = 0;
                    }
                }
            }
            k1.copy_from_slice(k2);
            y.copy_from_slice(y1);
            *x = xph;

            let changed = sys.accepted(*x, y);
            if last {
                return Ok(());
            }
            if changed {
                sys.rhs(*x, y, k1);
                stats.rhs_calls += 1;
                stats.nfcn += 1;
            }
            if hnew.abs() > hmax {
                hnew = posneg * hmax;
            }
            if reject {
                hnew = posneg * hnew.abs().min(h.abs());
            }
            reject = false;
        } else {
            hnew = h / facc1.min(fac11 / opt.safe);
            reject = true;
            if naccept >= 1 {
                stats.nreject += 1;
            }
            last = false;
        }
        h = hnew;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Harmonic;
    impl OdeSystem for Harmonic {
        fn rhs(&mut self, _t: f64, y: &[f64], dy: &mut [f64]) {
            dy[0] = y[1];
            dy[1] = -y[0];
        }
    }

    #[test]
    fn harmonic_oscillator_converges() {
        let opt = Options { rtol: 1e-10, atol: 1e-12, ..Default::default() };
        let mut ws = Workspace::new(2);
        let mut st = Stats::default();
        let mut y = [1.0, 0.0];
        let mut x = 0.0;
        let mut sys = Harmonic;
        let dt = 0.01;
        for _ in 0..100 {
            let xend = x + dt;
            integrate(&mut sys, &mut x, &mut y, xend, &opt, &mut ws, &mut st).unwrap();
        }
        assert!((y[0] - x.cos()).abs() < 1e-8, "{} {}", y[0], x.cos());
        assert!((y[1] + x.sin()).abs() < 1e-8);
    }
}
