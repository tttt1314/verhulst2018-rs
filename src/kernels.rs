// © copyright 2017 Ghent University – Universiteit Gent, all rights reserved; this Derivative work is
// made available for non-commercial academic research purposes and subject to an UGent Academic
// License (see LICENSE and NOTICE).
// Changed files notice: Rust port (2026) of `cubic_interpolate`, `delay_line` and
// `solve_tridiagonal` from cochlea_utils.c of Verhulstetal2018Model v1.2.
//
// The upstream library is built with `gcc -O3 -ffast-math` (Apple clang on arm64). That build
// (1) evaluates the delay-line interpolation in float32 because `cubic_interpolate` takes `float`
// arguments, (2) re-associates `a0 = y3 - y2 - y0 + y1` as `(y1 - (y0 + y2)) + y3`, (3) factors
// the cubic as `(a0*mu + a1)*mu2 + (a2*mu + y1)` with fused multiply-adds, and (4) fuses
// `b - a*c` in the Thomas algorithm. The `fused` helpers reproduce exactly that instruction
// sequence on aarch64 (verified against `otool -tv tridiag.so`); on other targets they fall back
// to separate multiply and add, which is what a non-FMA build of the C code produces.

#[inline(always)]
pub fn fma(a: f64, b: f64, c: f64) -> f64 {
    #[cfg(target_arch = "aarch64")]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        a * b + c
    }
}

/// `cubic_interpolate(float y0, float y1, float y2, float y3, float mu)` as compiled upstream.
#[inline(always)]
pub fn cubic_interpolate(y0: f32, y1: f32, y2: f32, y3: f32, mu: f32) -> f64 {
    let mu2 = (mu * mu) as f64;
    let a0 = ((y1 - (y0 + y2)) + y3) as f64;
    let a1 = ((y0 - y1) as f64) - a0;
    let a2 = (y2 - y0) as f64;
    let mud = mu as f64;
    let t1 = fma(a0, mud, a1);
    let t2 = fma(a2, mud, y1 as f64);
    fma(t1, mu2, t2)
}

/// Zweig delay line. `ybuf` is the channel-major ring buffer (`n` rows of length `m`).
///
/// Upstream stores the buffer in float64 and converts each tap to float32 inside
/// `cubic_interpolate`; storing the taps already rounded to float32 is therefore bit-identical
/// and halves the memory traffic.
#[inline(always)]
pub fn delay_tap(row: &[f32], d0: usize, d1: usize, d2: usize, d3: usize, dev: f64, m: usize) -> f64 {
    if dev < 1.0 {
        cubic_interpolate(row[d0], row[d1], row[d2], row[d3], dev as f32)
    } else {
        cubic_interpolate(
            row[(d0 + 1) % m],
            row[(d1 + 1) % m],
            row[(d2 + 1) % m],
            row[(d3 + 1) % m],
            (dev + -1.0) as f32,
        )
    }
}

/// `delay_tap` on the five consecutive samples `w = Y[Zrp1-1 .. Zrp1+3]` of one channel:
/// the unshifted taps are `w[0..4]`, the shifted ones (`dev >= 1`, or NaN, where upstream's
/// `dev < 1` test is false) are `w[1..5]` with `mu = dev - 1`. Branch free.
#[inline(always)]
pub fn window_tap(w: &[f32; 5], dev: f64) -> f64 {
    let shift = !(dev < 1.0);
    let mu = if shift { dev + -1.0 } else { dev } as f32;
    let (a, b, c, d) = if shift { (w[1], w[2], w[3], w[4]) } else { (w[0], w[1], w[2], w[3]) };
    cubic_interpolate(a, b, c, d, mu)
}

#[allow(clippy::too_many_arguments)]
pub fn delay_line(
    ybuf: &[f32],
    zrp: &[usize],
    zrp1: &[usize],
    zrp2: &[usize],
    zrp3: &[usize],
    dev: &[f64],
    out: &mut [f64],
    m: usize,
) {
    for i in 0..out.len() {
        let row = &ybuf[m * i..m * (i + 1)];
        out[i] = delay_tap(row, zrp[i], zrp1[i], zrp2[i], zrp3[i], dev[i], m);
    }
}

/// Thomas algorithm with the factorisation of the (constant) matrix cached.
///
/// Upstream recomputes `cprime[i] = c[i] * (1/(b[i] - a[i]*cprime[i-1]))` on every call; those
/// values depend only on the matrix, so caching `cprime` and the reciprocal pivots gives the
/// same bits while removing 1000 divisions from every right-hand-side evaluation.
#[derive(Clone, Debug)]
pub struct Tridiag {
    pub a: Vec<f64>,
    pub inv_b0: f64,
    pub b0: f64,
    pub cprime: Vec<f64>,
    pub m: Vec<f64>,
}

impl Tridiag {
    pub fn new(a: &[f64], b: &[f64], c: &[f64]) -> Self {
        let n = b.len();
        let mut cprime = vec![0.0; n];
        let mut m = vec![0.0; n];
        cprime[0] = c[0] / b[0];
        for i in 1..n {
            let mi = 1.0 / fma(-cprime[i - 1], a[i], b[i]);
            m[i] = mi;
            cprime[i] = mi * c[i];
        }
        Tridiag { a: a.to_vec(), inv_b0: 1.0 / b[0], b0: b[0], cprime, m }
    }

    /// Solve `T x = r` (upstream `solve_tridiagonal`).
    pub fn solve(&self, r: &[f64], x: &mut [f64]) {
        let n = r.len();
        assert!(n >= 1 && x.len() == n && self.a.len() == n && self.m.len() == n && self.cprime.len() == n);
        // The recurrences are latency bound; keep the carried value in a register (no
        // store-to-load round trip) and avoid bounds checks on the critical path.
        let mut prev = r[0] / self.b0;
        x[0] = prev;
        for i in 1..n {
            // SAFETY: i < n and all slices have length n (asserted above).
            unsafe {
                prev = fma(-prev, *self.a.get_unchecked(i), *r.get_unchecked(i)) * *self.m.get_unchecked(i);
                *x.get_unchecked_mut(i) = prev;
            }
        }
        let mut next = prev;
        for i in (0..n - 1).rev() {
            unsafe {
                next = fma(-next, *self.cprime.get_unchecked(i), *x.get_unchecked(i));
                *x.get_unchecked_mut(i) = next;
            }
        }
    }
}

/// Reference (uncached) transcription of upstream `solve_tridiagonal`, used by tests.
pub fn solve_tridiagonal_reference(a: &[f64], b: &[f64], c: &[f64], r: &[f64], x: &mut [f64]) {
    let n = r.len();
    let mut cprime = vec![0.0; n];
    cprime[0] = c[0] / b[0];
    x[0] = r[0] / b[0];
    for i in 1..n {
        let m = 1.0 / fma(-cprime[i - 1], a[i], b[i]);
        cprime[i] = m * c[i];
        x[i] = fma(-x[i - 1], a[i], r[i]) * m;
    }
    for i in (0..n - 1).rev() {
        x[i] = fma(-x[i + 1], cprime[i], x[i]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_tridiag_matches_reference() {
        let n = 50;
        let a: Vec<f64> = (0..n).map(|i| if i == 0 { 0.0 } else { -0.3 - 0.01 * i as f64 }).collect();
        let b: Vec<f64> = (0..n).map(|i| 2.0 + 0.1 * (i as f64).sin()).collect();
        let c: Vec<f64> = (0..n).map(|i| if i == n - 1 { 0.0 } else { -0.4 + 0.002 * i as f64 }).collect();
        let r: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).cos()).collect();
        let mut x1 = vec![0.0; n];
        let mut x2 = vec![0.0; n];
        solve_tridiagonal_reference(&a, &b, &c, &r, &mut x1);
        Tridiag::new(&a, &b, &c).solve(&r, &mut x2);
        assert_eq!(x1, x2);
        // residual check
        for i in 0..n {
            let mut s = b[i] * x1[i];
            if i > 0 {
                s += a[i] * x1[i - 1];
            }
            if i + 1 < n {
                s += c[i] * x1[i + 1];
            }
            assert!((s - r[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn cubic_hits_nodes() {
        assert_eq!(cubic_interpolate(1.0, 2.0, 3.0, 4.0, 0.0), 2.0);
        assert_eq!(cubic_interpolate(1.0, 2.0, 3.0, 5.0, 1.0), 3.0);
    }
}
