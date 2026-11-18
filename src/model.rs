// © copyright 2017 Ghent University – Universiteit Gent, all rights reserved; this Derivative work is
// made available for non-commercial academic research purposes and subject to an UGent Academic
// License (see LICENSE and NOTICE).
// Changed files notice: Rust port (2026) of the `cochlea_model` class and `TLsolver` of
// cochlear_model2018.py (Verhulstetal2018Model v1.2): initialisation (`init_model`,
// `initCochlea`, `initMiddleEar`, `SetDampingAndStiffness`, `initZweig`,
// `initGaussianElimination`, `compression_slope_param`), `SheraParameters`, `ZweigImpedance`,
// `polecalculation`, `calculate_g`, `calculate_right`, `TLsolver` and `solve`. Changes: only the
// otoacoustic emission and selected probe channels are stored; tolerances are parameters; an
// optional accepted-step pole-update variant was added (not the default).
//
// Arithmetic is written in the evaluation order of the NumPy expressions upstream (no fused
// operations: NumPy ufuncs never contract), so that the default mode reproduces the original
// floating-point results. The random level-dependent irregularities (`Rth`) come from NumPy's
// legacy MT19937 generator and are passed in by the caller.

use crate::dopri5::{self, libm_pow, DopriError, OdeSystem, Options, Stats, Workspace};
use crate::kernels::{fma as kfma, window_tap, Tridiag};
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NonLinearity {
    Linear,
    Disp,
    Vel,
}

impl NonLinearity {
    pub fn code(&self) -> i32 {
        match self {
            NonLinearity::Linear => 0,
            NonLinearity::Disp => 1,
            NonLinearity::Vel => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoleUpdate {
    /// Upstream behaviour: poles are re-evaluated inside every right-hand-side evaluation,
    /// including rejected and trial stages, and updated when any pole moves by more than 1 %.
    Faithful,
    /// Model variant: the right-hand side is a pure function of (t, y); the 1 % rule is applied
    /// only to accepted states (after every accepted step).
    AcceptedStep,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub samplerate: f64,
    pub sections: usize,
    pub shera_po: Vec<f64>,
    pub compression_slope: f64,
    pub zweig_irregularities: bool,
    pub non_linearity: NonLinearity,
    pub knee_var: f64,
    pub low_freq_irregularities: bool,
    pub irr_pct: f64,
    /// `2 * (np.random.random(sections + 1) - 0.5)` after `np.random.seed(subject)`.
    pub rth: Vec<f64>,
    pub options: Options,
    pub pole_update: PoleUpdate,
    /// Per-channel delay rings sized to the channel's reachable delay (bit-identical; see
    /// `Model::ring_len`). `false` reproduces the upstream buffer layout exactly.
    pub compact_delay: bool,
}

pub struct Model {
    // ---- scalar constants (names follow the Python attributes) ----
    pub n: usize,  // sections
    pub n1: usize, // sections + 1
    pub fs: f64,
    pub dt: f64,
    pub bm_length: f64,
    pub bm_width: f64,
    pub zweig_mso: f64,
    pub zweig_l: f64,
    pub zweig_omega_co: f64,
    pub zweig_mpo: f64,
    pub ko: f64,
    pub dx: f64,
    pub q0_factor: f64,
    pub p0x: f64,
    pub d_m_factor: f64,
    pub rk4_0: f64,
    pub rk4g_0: f64,
    pub c: f64,
    pub ctr: usize,
    pub onek: usize,
    pub ybuffer_lgt: usize,
    pub yknee1: f64,
    pub yknee2: f64,
    pub vknee1: f64,
    pub vknee2: f64,
    pub non_linearity: NonLinearity,
    pub pole_update: PoleUpdate,
    pub options: Options,

    // ---- vectors ----
    pub x: Vec<f64>,
    pub f_resonance: Vec<f64>,
    pub omega: Vec<f64>,
    pub omega2: Vec<f64>,
    pub sherad_factor: Vec<f64>,
    pub shera_po: Vec<f64>,
    pub shera_p: Vec<f64>,
    pub shera_d: Vec<f64>,
    pub shera_rho: Vec<f64>,
    pub shera_mu: Vec<f64>,
    pub pole_s: Vec<f64>,
    pub pole_e: Vec<f64>,
    pub rth: Vec<f64>,
    pub rthy1: Vec<f64>,
    pub rthy2: Vec<f64>,
    pub rthv1: Vec<f64>,
    pub rthv2: Vec<f64>,
    pub sa: Vec<f64>,
    pub sb: Vec<f64>,
    pub const_nl1: Vec<f64>,
    pub cos_theta: Vec<f64>,
    pub sin_theta: Vec<f64>,
    pub exact_delay: Vec<f64>,
    pub zweig_ms: Vec<f64>,
    pub zweig_mp: Vec<f64>,
    pub zasq: Vec<f64>,
    pub zasc: Vec<f64>,
    pub zal: Vec<f64>,
    pub zah: Vec<f64>,
    pub tridiag: Tridiag,

    // ---- Zweig delay line state ----
    /// Delay-line taps in float32 (upstream converts every tap to float32 before
    /// interpolating, so storing them rounded is bit-identical). Channel `i` owns the ring
    /// `ybuffer[ring_off[i] .. ring_off[i] + ring_len[i]]`.
    pub ybuffer: Vec<f32>,
    pub ring_off: Vec<usize>,
    pub ring_len: Vec<usize>,
    /// Ring slot of the most recent write per channel (upstream: column `Zwp`).
    pub wpos: Vec<usize>,
    /// The five consecutive delay-line samples that `delay_line` can touch for each channel
    /// (upstream columns Zrp1-1 .. Zrp1+3), gathered whenever Zwp or the delays change.
    pub taps: Vec<[f32; 5]>,
    /// First (channel, age) for which a compact ring would be too short (never observed with
    /// physiological poles; the run then fails rather than silently differing).
    pub ring_violation: Option<(usize, usize)>,
    pub zwp: usize,
    pub dev: Vec<f64>,
    pub yzweig: Vec<f64>,
    pub zrp1: Vec<usize>,

    // ---- solver scratch / state ----
    pub g: Vec<f64>,
    pub right: Vec<f64>,
    pub qsol: Vec<f64>,
    pole_scratch: Vec<f64>,
    pole_scratch2: Vec<f64>,
    inv_rthv1: Vec<f64>,
    inv_sa: Vec<f64>,
    inv_abs_p: Vec<f64>,
    screen_ok: bool,
    pub exact_pole_checks: u64,
    pub interp: [f64; 4],
    pub last_t: f64,
    /// `model.Vtmp[0]` of the most recent right-hand-side call (upstream copies it into `r.y`).
    pub last_rhs_v0: f64,
    pub pole_updates: u64,
}

/// `np.remainder` for float64 with a positive integer divisor (exact for integral inputs).
#[cfg_attr(not(test), allow(dead_code))]
fn py_mod_f64(a: f64, b: f64) -> f64 {
    let mut m = a % b; // fmod
    if m != 0.0 && ((b < 0.0) != (m < 0.0)) {
        m += b;
    }
    m
}

fn argmin_abs_diff(v: &[f64], target: f64) -> usize {
    // np.argmin: first occurrence of the minimum; NaN would win (not expected here).
    let mut best = 0;
    let mut bv = (v[0] - target).abs();
    for (i, x) in v.iter().enumerate().skip(1) {
        let d = (x - target).abs();
        if d < bv {
            bv = d;
            best = i;
        }
    }
    best
}

/// `np.linspace(0, stop, num)` (endpoint=True).
fn linspace0(stop: f64, num: usize) -> Vec<f64> {
    let div = (num - 1) as f64;
    let step = stop / div;
    let mut y: Vec<f64> = (0..num).map(|i| i as f64 * step + 0.0).collect();
    y[num - 1] = stop;
    y
}

impl Model {
    pub fn new(cfg: &Config) -> Result<Model, String> {
        let n = cfg.sections;
        let n1 = n + 1;
        if cfg.shera_po.len() != n1 && cfg.shera_po.len() != 1 {
            return Err(format!("sheraPo must have length 1 or {n1}"));
        }
        if cfg.rth.len() != n1 {
            return Err(format!("rth must have length {n1}"));
        }
        // ---- __init__ constants ----
        let cochlea_length = 0.035;
        let bm_mass = 0.5;
        let bm_impedance_factor = 1.0;
        let scala_width = 0.001;
        let scala_height = 0.001;
        let helicotrema_width = 0.001;
        let rho = 1e3;
        let greenwood_a = 20682.0;
        let greenwood_alpha = 61.765;
        let greenwood_b = 140.6;
        let stapes_area = 3e-6;
        let zweig_n = 1.5;
        let shera_mu_max = 3.0;
        let rme = 0.3045192500000000e12;
        let fs = cfg.samplerate;
        let dt = 1.0 / fs;

        // ---- initCochlea ----
        let bm_length = cochlea_length - helicotrema_width;
        let bm_width = scala_width;
        let _bm_mass = bm_mass * bm_impedance_factor;
        let zweig_mso = 2.0 * rho / (bm_width * scala_height);
        let zweig_l = 1.0 / (2.3030 * greenwood_alpha);
        let zweig_omega_co = 2.0 * PI * greenwood_a - greenwood_b;
        let zweig_mpo = (zweig_mso * libm_pow(zweig_l, 2.0)) / libm_pow(4.0 * zweig_n, 2.0);
        let ko = zweig_mpo * libm_pow(zweig_omega_co, 2.0);
        let x = linspace0(bm_length, n1);
        let dx = bm_length / (1.0 * n as f64);

        // ---- initMiddleEar ----
        let q0_factor = zweig_mpo * bm_width;
        let p0x = zweig_mso * dx / (1.0 * zweig_mpo * bm_width);
        let d_m_factor = -p0x * stapes_area * rme;
        let rk4_0 = -(bm_width * zweig_mpo) / stapes_area;
        let rk4g_0 = (zweig_mpo * bm_width) / (zweig_mso * stapes_area * dx);

        // ---- SetDampingAndStiffness ----
        let neg_alpha = -greenwood_alpha;
        let f_resonance: Vec<f64> =
            x.iter().map(|&xi| greenwood_a * libm_pow(10.0, neg_alpha * xi) - greenwood_b).collect();
        let mut ctr = argmin_abs_diff(&f_resonance, 100.0);
        if cfg.low_freq_irregularities {
            ctr = n + 1;
        }
        let onek = argmin_abs_diff(&f_resonance, 1000.0);
        let two_pi = 2.0 * PI;
        let mut omega: Vec<f64> = f_resonance.iter().map(|&f| two_pi * f).collect();
        omega[0] = zweig_omega_co;
        let omega2: Vec<f64> = omega.iter().map(|&w| w * w).collect();
        let sherad_factor = omega.clone();
        let shera_po: Vec<f64> =
            if cfg.shera_po.len() == 1 { vec![0.0 + cfg.shera_po[0]; n1] } else { cfg.shera_po.iter().map(|&p| 0.0 + p).collect() };
        let shera_p: Vec<f64> = shera_po.iter().map(|&p| p + 0.0).collect();
        let c = 120.8998691636393;

        // ---- initZweig ----
        let exact_delay: Vec<f64> = f_resonance.iter().map(|&f| shera_mu_max / (f * dt)).collect();
        let mut maxdelay = f64::NEG_INFINITY;
        for &e in &exact_delay {
            let d = e.floor() + 1.0;
            if d > maxdelay || d.is_nan() {
                maxdelay = d;
            }
        }
        if !(maxdelay.is_finite() && maxdelay >= 1.0) {
            return Err("invalid Zweig delay buffer length".into());
        }
        let ybuffer_lgt = maxdelay as usize;

        // ---- initGaussianElimination ----
        let zweig_ms: Vec<f64> = omega.iter().map(|&w| (zweig_mso * zweig_omega_co) / w).collect();
        let zweig_mp: Vec<f64> = omega.iter().map(|&w| ko / (zweig_omega_co * w)).collect();
        let mut zasq = vec![0.0; n1];
        let mut zasc = vec![0.0; n1];
        let mut zal = vec![0.0; n1];
        let mut zah = vec![0.0; n1];
        zasq[0] = 1.0;
        zasc[0] = 1.0 + zweig_mso * dx;
        zah[0] = -1.0 * zweig_omega_co / omega[1];
        for i in 1..n1 {
            zal[i] = -zweig_ms[i] * omega[i] / omega[i - 1];
        }
        for i in 1..n1 - 1 {
            zah[i] = -zweig_ms[i - 1] * omega[i] / omega[i + 1];
        }
        let dx2 = libm_pow(dx, 2.0);
        let oc_mpo = zweig_omega_co * zweig_mpo;
        for i in 1..n1 {
            zasq[i] = omega[i] * zweig_ms[i] * zweig_ms[i - 1] * dx2 / oc_mpo;
            zasc[i] = zasq[i] + zweig_ms[i] + zweig_ms[i - 1];
        }
        let tridiag = Tridiag::new(&zal, &zasc, &zah);

        // ---- compression_slope_param ----
        let slope = cfg.compression_slope;
        let yknee1 = 1.0 * 6.9183e-10;
        let yknee2 = 1.5488e-8;
        let pole_e = vec![0.0 + 0.31; n1];
        let v1 = 0.6807e-08 / 3.0 / (2.0f64).sqrt();
        let v2 = 26.490e-11 / 3.0 / (2.0f64).sqrt();
        let k1db = 20.0 + 20.0;
        let k1l = libm_pow(10.0, k1db / 20.0);
        let vknee1 = k1l * v1;
        let vst1db = 20.0 * v1.log10() + k1db;
        let vst2db = 20.0 * v2.log10() + k1db;
        let k2db = (vst1db - vst2db) / (1.0 - slope);
        let vknee2 = v2 * libm_pow(10.0, k2db / 20.0) * k1l;

        // ---- init_model: irregularities and nonlinearity constants ----
        let lf_limit = if cfg.zweig_irregularities { ctr } else { 0 };
        let factor = 100.0;
        let rth = cfg.rth.clone();
        let rth_norm: Vec<f64> = rth.iter().map(|&r| libm_pow(10.0, r / 20.0 / cfg.knee_var)).collect();
        let mut rthy1: Vec<f64> = rth_norm.iter().map(|&r| yknee1 * r).collect();
        let mut rthy2: Vec<f64> = rth_norm.iter().map(|&r| yknee2 * r).collect();
        let mut rthv1: Vec<f64> = rth_norm.iter().map(|&r| vknee1 * r).collect();
        let mut rthv2: Vec<f64> = rth_norm.iter().map(|&r| vknee2 * r).collect();
        let mut pole_s: Vec<f64> =
            (0..n1).map(|i| (1.0 + cfg.irr_pct * rth[i] / 2.0) * shera_po[i]).collect();
        for i in lf_limit.min(n1)..n1 {
            rthy1[i] = yknee1;
            rthy2[i] = yknee2;
            rthv1[i] = vknee1;
            rthv2[i] = vknee2;
            pole_s[i] = shera_po[i];
        }
        let mut sa = vec![0.0; n1];
        let mut sb = vec![0.0; n1];
        let mut const_nl1 = vec![0.0; n1];
        let mut cos_theta = vec![0.0; n1];
        let mut sin_theta = vec![0.0; n1];
        for i in 0..n1 {
            let ratio = rthv2[i] / rthv1[i];
            let theta0 = (((pole_e[i] - pole_s[i]) * factor) / (ratio - 1.0)).atan();
            let theta = theta0 / 2.0;
            let sfoc = (pole_s[i] * factor) / ratio;
            let se = ((PI - theta0) * 0.5).cos();
            sb[i] = sfoc * se;
            sa[i] = sfoc * (1.0 - se * se).sqrt();
            const_nl1[i] = theta.cos() / (2.0 * theta).cos();
            cos_theta[i] = theta.cos();
            sin_theta[i] = theta.sin();
        }

        // ---- delay-line ring layout ----
        // Upstream keeps one ring of YbufferLgt samples for every channel. The taps of channel i
        // never reach further back than its nominal delay floor(SheraMuMax/(f_i dt)) + 1
        // (SheraMu <= ~1.9 < SheraMuMax = 3 for poles in [-0.31, 0.31]), so `compact_delay`
        // gives each channel a ring of that length plus a margin (≈ 8 MB instead of 50 MB, and
        // far fewer TLB misses). The reachable ages are checked at every ZweigImpedance call.
        let ring_len: Vec<usize> = (0..n1)
            .map(|i| {
                if cfg.compact_delay {
                    let nominal = exact_delay[i].floor() + 1.0;
                    if nominal.is_finite() && nominal >= 0.0 {
                        ((nominal as usize) + 16).min(ybuffer_lgt)
                    } else {
                        ybuffer_lgt
                    }
                } else {
                    ybuffer_lgt
                }
            })
            .collect();
        let mut ring_off = vec![0usize; n1];
        let mut total = 0usize;
        for i in 0..n1 {
            ring_off[i] = total;
            total += ring_len[i];
        }
        let ybuffer = vec![0.0f32; total];

        let options = cfg.options;
        let mut m = Model {
            n,
            n1,
            fs,
            dt,
            bm_length,
            bm_width,
            zweig_mso,
            zweig_l,
            zweig_omega_co,
            zweig_mpo,
            ko,
            dx,
            q0_factor,
            p0x,
            d_m_factor,
            rk4_0,
            rk4g_0,
            c,
            ctr,
            onek,
            ybuffer_lgt,
            yknee1,
            yknee2,
            vknee1,
            vknee2,
            non_linearity: cfg.non_linearity,
            pole_update: cfg.pole_update,
            options,
            x,
            f_resonance,
            omega,
            omega2,
            sherad_factor,
            shera_po,
            shera_p,
            shera_d: vec![0.0; n1],
            shera_rho: vec![0.0; n1],
            shera_mu: vec![0.0; n1],
            pole_s,
            pole_e,
            rth,
            rthy1,
            rthy2,
            rthv1,
            rthv2,
            sa,
            sb,
            const_nl1,
            cos_theta,
            sin_theta,
            exact_delay,
            zweig_ms,
            zweig_mp,
            zasq,
            zasc,
            zal,
            zah,
            tridiag,
            ybuffer,
            ring_off,
            ring_len,
            wpos: vec![0; n1],
            taps: vec![[0.0; 5]; n1],
            ring_violation: None,
            zwp: 0,
            dev: vec![0.0; n1],
            yzweig: vec![0.0; n1],
            zrp1: vec![0; n1],
            g: vec![0.0; n1],
            right: vec![0.0; n1],
            qsol: vec![0.0; n1],
            pole_scratch: vec![0.0; n1],
            pole_scratch2: vec![0.0; n1],
            inv_rthv1: vec![0.0; n1],
            inv_sa: vec![0.0; n1],
            inv_abs_p: vec![0.0; n1],
            screen_ok: false,
            exact_pole_checks: 0,
            interp: [0.0; 4],
            last_t: 0.0,
            last_rhs_v0: 0.0,
            pole_updates: 0,
        };
        // `solve()` prologue: polecalculation with Vtmp = Ytmp = 0, then SheraParameters and
        // ZweigImpedance.
        m.polecalculation(&vec![0.0; n1], &vec![0.0; n1]);
        m.shera_parameters();
        m.zweig_impedance();
        for i in 0..n1 {
            m.inv_rthv1[i] = 1.0 / m.rthv1[i];
            m.inv_sa[i] = 1.0 / m.sa[i];
        }
        m.refresh_pole_screen();
        Ok(m)
    }

    /// `polecalculation()` (called once, before integration).
    pub fn polecalculation(&mut self, vtmp: &[f64], ytmp: &[f64]) {
        let factor = 100.0;
        let n1 = self.n1;
        match self.non_linearity {
            NonLinearity::Disp => {
                let w = self.omega[self.onek];
                for i in 0..n1 {
                    let yknee1cst = self.rthy1[i] * w;
                    let yknee2cst = self.rthy2[i] * w;
                    let yknee1f = yknee1cst / self.omega[i];
                    let yknee2f = yknee2cst / self.omega[i];
                    let yvect = (ytmp[i] / yknee1f).abs();
                    let theta0 = ((self.pole_e[i] - self.pole_s[i]) / ((yknee2f / yknee1f) - 1.0)).atan();
                    let theta = theta0 / 2.0;
                    let cos_t = theta.cos();
                    let sin_t = theta.sin();
                    let cos_t0 = 2.0 * (cos_t * cos_t) - 1.0;
                    let sfoc = self.pole_s[i] * factor * (yknee2f / yknee1f);
                    let se = sin_t;
                    let sb = sfoc * se;
                    let sa = sfoc * (1.0 - (1.0 * (se * se))).sqrt();
                    let sxp = (yvect - 1.0) * cos_t / cos_t0;
                    let r = sxp / sa;
                    let syp = sb * (1.0 + r * r).sqrt();
                    let sy = sxp * sin_t + syp * cos_t;
                    self.shera_p[i] = self.pole_s[i] + sy / factor;
                }
            }
            NonLinearity::Vel => {
                for i in 0..n1 {
                    self.shera_p[i] = self.vel_pole(i, vtmp[i]);
                }
            }
            NonLinearity::Linear => {
                self.shera_p.copy_from_slice(&self.pole_s);
            }
        }
        for i in 0..n1 {
            self.shera_p[i] = fmin(self.shera_p[i], self.pole_e[i]);
        }
    }

    /// VEL nonlinearity of `TLsolver`/`polecalculation`, before the `fmin` with `PoleE`.
    #[inline(always)]
    fn vel_pole(&self, i: usize, v: f64) -> f64 {
        let vvect = v.abs() / self.rthv1[i];
        let sxp = (vvect - 1.0) * self.const_nl1[i];
        let r = sxp / self.sa[i];
        let syp = self.sb[i] * (1.0 + r * r).sqrt();
        let sy = sxp * self.sin_theta[i] + syp * self.cos_theta[i];
        self.pole_s[i] + sy / 100.0
    }

    /// `SheraParameters()`.
    pub fn shera_parameters(&mut self) {
        let c = self.c;
        let two_pi = 2.0 * PI;
        for i in 0..self.n1 {
            let p = self.shera_p[i];
            let p2 = p * p;
            let a = (p + (p2 + c * (1.0 - p2)).sqrt()) / c;
            let d = 2.0 * (p - a);
            self.shera_d[i] = d;
            self.shera_mu[i] = 1.0 / (two_pi * a);
            let h = d / 2.0;
            self.shera_rho[i] = 2.0 * a * (1.0 - h * h).sqrt() * (-p / a).exp();
        }
    }

    /// `ZweigImpedance()`.
    pub fn zweig_impedance(&mut self) {
        let lgt = self.ybuffer_lgt;
        let lgt_i = lgt as i64;
        let zwp = self.zwp;
        let base = (zwp + lgt) as f64;
        let two_pi = 2.0 * PI;
        let dt = self.dt;
        let n1 = self.n1;
        {
            let (mu, om) = (&self.shera_mu[..n1], &self.omega[..n1]);
            let dev = &mut self.dev[..n1];
            let mud = &mut self.pole_scratch2[..n1];
            for i in 0..n1 {
                let exact = two_pi * mu[i] / (om[i] * dt);
                let mudelay = exact.floor() + 1.0;
                dev[i] = mudelay - exact;
                mud[i] = mudelay;
            }
        }
        for i in 0..n1 {
            // `((Zwp + YbufferLgt) - Mudelay) % YbufferLgt` in float64 (np.remainder), then cast
            // to int32. Both operands are integers below 2**53, so the float subtraction is exact
            // and the Python float remainder equals the Euclidean integer remainder.
            let diff = base - self.pole_scratch2[i];
            let di = diff as i64;
            let z1 = if (0..lgt_i).contains(&di) { di as usize } else { di.rem_euclid(lgt_i) as usize };
            self.zrp1[i] = z1;
            // Age (in writes) of column Zrp1; the taps span ages age1+1 .. age1-3.
            let age1 = if zwp >= z1 { zwp - z1 } else { zwp + lgt - z1 };
            let len = self.ring_len[i];
            let w = self.wpos[i];
            let start = if len == lgt {
                // full ring: same layout as upstream (w == zwp), column arithmetic mod L
                if z1 == 0 { lgt - 1 } else { z1 - 1 }
            } else {
                if age1 < 3 || age1 + 1 >= len {
                    if self.ring_violation.is_none() {
                        self.ring_violation = Some((i, age1));
                    }
                    0
                } else {
                    let a = age1 + 1;
                    if w >= a { w - a } else { w + len - a }
                }
            };
            let ring = &self.ybuffer[self.ring_off[i]..self.ring_off[i] + len];
            let mut t = [0.0f32; 5];
            if start + 5 <= len {
                t.copy_from_slice(&ring[start..start + 5]);
            } else {
                let mut j = start;
                for v in t.iter_mut() {
                    *v = ring[j];
                    j += 1;
                    if j == len {
                        j = 0;
                    }
                }
            }
            self.taps[i] = t;
        }
    }

    /// Upstream `Zrp, Zrp1, Zrp2, Zrp3` (derived from Zrp1; only used for inspection).
    pub fn zrp_all(&self) -> [Vec<usize>; 4] {
        let l = self.ybuffer_lgt;
        let z1 = self.zrp1.clone();
        let z0: Vec<usize> = z1.iter().map(|&z| (z + l - 1) % l).collect();
        let z2: Vec<usize> = z1.iter().map(|&z| (z + 1) % l).collect();
        let z3: Vec<usize> = z2.iter().map(|&z| (z + 1) % l).collect();
        [z0, z1, z2, z3]
    }

    /// Recompute the reciprocals used by the pole-update screen (after SheraP changed).
    pub fn refresh_pole_screen(&mut self) {
        let n1 = self.n1;
        let mut ok = true;
        for i in 0..n1 {
            let inv = 1.0 / self.shera_p[i].abs();
            self.inv_abs_p[i] = inv;
            if i >= 1 && !(self.shera_p[i].abs() >= 1e-6 && inv.is_finite()) {
                ok = false;
            }
        }
        for i in 0..n1 {
            ok &= self.inv_rthv1[i].is_finite() && self.inv_sa[i].is_finite();
        }
        self.screen_ok = ok;
    }

    /// Fast screen for the 1 % rule: the same candidate-pole formula evaluated with
    /// reciprocals instead of divisions (a few ulp from the exact value). Returns true only
    /// if every relative change is below 0.01·(1 − 1e-7); in that case the exact test of
    /// upstream is certainly false too (the screen's error is ~1e-13 relative to |SheraP|,
    /// six orders of magnitude below the margin), and nothing else depends on the
    /// candidates. Any NaN/inf or a value near the threshold falls through to the exact test.
    #[inline(always)]
    fn screen_says_no_update(&mut self, v: &[f64]) -> bool {
        if !self.screen_ok {
            return false;
        }
        let n1 = self.n1;
        let len = n1 - 1;
        {
            let (rv1, nl1, isa, sb, st, ct, ps, pe, old, iold) = (
                &self.inv_rthv1[1..n1],
                &self.const_nl1[1..n1],
                &self.inv_sa[1..n1],
                &self.sb[1..n1],
                &self.sin_theta[1..n1],
                &self.cos_theta[1..n1],
                &self.pole_s[1..n1],
                &self.pole_e[1..n1],
                &self.shera_p[1..n1],
                &self.inv_abs_p[1..n1],
            );
            let v = &v[1..n1];
            let out = &mut self.pole_scratch2[..len];
            for i in 0..len {
                let vv = v[i].abs() * rv1[i];
                let sxp = (vv - 1.0) * nl1[i];
                let r = sxp * isa[i];
                let syp = sb[i] * (1.0 + r * r).sqrt();
                let sy = sxp * st[i] + syp * ct[i];
                let p = (ps[i] + sy * 0.01).min(pe[i]);
                out[i] = (p - old[i]).abs() * iold[i];
            }
        }
        let r = &self.pole_scratch2[..len];
        let mut acc = [0.0f64; 4];
        let chunks = len / 4;
        for c in 0..chunks {
            for l in 0..4 {
                let x = r[4 * c + l];
                // `!(x <= acc)` keeps NaN, which then fails the final `< limit` test
                acc[l] = if x <= acc[l] { acc[l] } else { x };
            }
        }
        for &x in &r[4 * chunks..] {
            acc[0] = if x <= acc[0] { acc[0] } else { x };
        }
        let limit = 0.01 * (1.0 - 1e-7);
        acc.iter().all(|&a| a < limit)
    }

    /// Candidate poles from velocities and the upstream 1 % rule. Returns true if the poles were
    /// updated (SheraParameters and ZweigImpedance re-run).
    #[inline(always)]
    pub fn maybe_update_poles(&mut self, v: &[f64]) -> bool {
        if self.screen_says_no_update(v) {
            return false;
        }
        self.exact_pole_checks += 1;
        let n1 = self.n1;
        {
            let (rv1, nl1, sa, sb, st, ct, ps, pe) = (
                &self.rthv1[..n1],
                &self.const_nl1[..n1],
                &self.sa[..n1],
                &self.sb[..n1],
                &self.sin_theta[..n1],
                &self.cos_theta[..n1],
                &self.pole_s[..n1],
                &self.pole_e[..n1],
            );
            let v = &v[..n1];
            let out = &mut self.pole_scratch[..n1];
            for i in 0..n1 {
                let vvect = v[i].abs() / rv1[i];
                let sxp = (vvect - 1.0) * nl1[i];
                let r = sxp / sa[i];
                let syp = sb[i] * (1.0 + r * r).sqrt();
                let sy = sxp * st[i] + syp * ct[i];
                out[i] = fmin(ps[i] + sy / 100.0, pe[i]);
            }
        }
        // max(|new - old| / |old|) over channels 1..n with NaN propagation (np.max). Max is
        // order independent, so a 4-lane reduction gives the same answer.
        let (newp, oldp) = (&self.pole_scratch[1..n1], &self.shera_p[1..n1]);
        let mut acc = [f64::NEG_INFINITY; 4];
        let mut nan = [false; 4];
        let chunks = newp.len() / 4;
        for c in 0..chunks {
            for l in 0..4 {
                let i = 4 * c + l;
                let r = (newp[i] - oldp[i]).abs() / oldp[i].abs();
                nan[l] |= r.is_nan();
                acc[l] = if r > acc[l] { r } else { acc[l] };
            }
        }
        for i in 4 * chunks..newp.len() {
            let r = (newp[i] - oldp[i]).abs() / oldp[i].abs();
            nan[0] |= r.is_nan();
            acc[0] = if r > acc[0] { r } else { acc[0] };
        }
        let any_nan = nan.iter().any(|&b| b);
        let maxrel = acc.iter().fold(f64::NEG_INFINITY, |m, &x| if x > m { x } else { m });
        if !any_nan && maxrel > 0.01 {
            std::mem::swap(&mut self.shera_p, &mut self.pole_scratch);
            self.shera_parameters();
            self.zweig_impedance();
            self.refresh_pole_screen();
            self.pole_updates += 1;
            true
        } else {
            false
        }
    }

    /// Stimulus interpolation of `TLsolver`.
    #[inline(always)]
    fn stimulus(&self, frac: f64) -> f64 {
        let [a, b, c, d] = self.interp;
        let cminusb = c - b;
        b + frac * (cminusb - 0.1666667 * (1.0 - frac) * ((d - a - 3.0 * cminusb) * frac + (d + 2.0 * a - 3.0 * b)))
    }

    /// `TLsolver(t, y, model)` including its side effects on the model.
    ///
    /// Upstream order: nonlinear pole check (may run SheraParameters + ZweigImpedance),
    /// `Dev += frac`, `delay_line`, `Dev -= frac`, `calculate_g`, `calculate_right`,
    /// `solve_tridiagonal`, derivative assembly. Here the delay line, `calculate_g`,
    /// `calculate_right` and the forward Thomas sweep are fused in one loop (each element's
    /// arithmetic is unchanged), so that the independent per-channel work overlaps the
    /// latency-bound recurrence.
    pub fn tl_rhs(&mut self, t: f64, y: &[f64], dy: &mut [f64]) {
        let n1 = self.n1;
        assert!(y.len() == 2 * n1 && dy.len() == 2 * n1);
        let frac = (t - self.last_t) / self.dt;
        let f0 = self.stimulus(frac);
        let (v, yd) = y.split_at(n1);
        self.last_rhs_v0 = v[0];
        if self.non_linearity != NonLinearity::Linear && self.pole_update == PoleUpdate::Faithful {
            self.maybe_update_poles(v);
        }
        let faithful_dev = self.pole_update == PoleUpdate::Faithful;

        // channel 0: delay tap (only feeds YZweig[0], which upstream computes but never uses)
        {
            let d = self.dev[0] + frac;
            self.yzweig[0] = window_tap(&self.taps[0], d);
            if faithful_dev {
                self.dev[0] = d - frac;
            }
        }
        let g0 = self.d_m_factor * v[0];
        self.g[0] = g0;
        let drive = g0 + self.p0x * f0;
        self.right[0] = drive;
        let td = &self.tridiag;
        let mut prev = drive / td.b0;
        self.qsol[0] = prev;
        {
            let taps = &self.taps[..n1];
            let dev = &mut self.dev[..n1];
            let yz = &mut self.yzweig[..n1];
            let (shf, shd, om2, rho, zasq) = (
                &self.sherad_factor[..n1],
                &self.shera_d[..n1],
                &self.omega2[..n1],
                &self.shera_rho[..n1],
                &self.zasq[..n1],
            );
            let (ta, tm) = (&td.a[..n1], &td.m[..n1]);
            let g = &mut self.g[..n1];
            let right = &mut self.right[..n1];
            let q = &mut self.qsol[..n1];
            for i in 1..n1 {
                let d = dev[i] + frac;
                let yzi = window_tap(&taps[i], d);
                yz[i] = yzi;
                if faithful_dev {
                    dev[i] = d - frac;
                }
                let dtot = shf[i] * shd[i];
                let stot = om2[i] * (yd[i] + rho[i] * yzi);
                let gi = dtot * v[i] + stot;
                g[i] = gi;
                let ri = zasq[i] * gi;
                right[i] = ri;
                prev = kfma(-prev, ta[i], ri) * tm[i];
                q[i] = prev;
            }
            // backward sweep fused with Vderivative = Qsol - g
            let cp = &td.cprime[..n1];
            let (dv, dyd) = dy.split_at_mut(n1);
            let mut next = prev;
            dv[n1 - 1] = next - g[n1 - 1];
            for i in (0..n1 - 1).rev() {
                next = kfma(-next, cp[i], q[i]);
                q[i] = next;
                dv[i] = next - g[i];
            }
            dv[0] = self.rk4_0 * q[0] + self.rk4g_0 * drive;
            dyd.copy_from_slice(v);
        }
    }

    /// Bookkeeping after each output sample (`solve()` loop body after `r.integrate`).
    pub fn after_sample(&mut self, y: &mut [f64]) {
        let n1 = self.n1;
        y[0] = self.last_rhs_v0; // self.V1[0] = self.Vtmp[0] writes through the view of r.y
        let lgt = self.ybuffer_lgt;
        self.zwp = (self.zwp + 1) % lgt;
        for i in 0..n1 {
            let len = self.ring_len[i];
            let w = if self.wpos[i] + 1 == len { 0 } else { self.wpos[i] + 1 };
            self.wpos[i] = w;
            self.ybuffer[self.ring_off[i] + w] = y[n1 + i] as f32;
        }
        self.zweig_impedance();
    }
}

#[inline(always)]
fn fmin(a: f64, b: f64) -> f64 {
    // np.fmin: NaN-ignoring minimum (same as f64::min).
    a.min(b)
}

impl OdeSystem for Model {
    #[inline(always)]
    fn rhs(&mut self, t: f64, y: &[f64], dy: &mut [f64]) {
        self.tl_rhs(t, y, dy);
    }
    fn accepted(&mut self, _t: f64, y: &[f64]) -> bool {
        if self.pole_update == PoleUpdate::AcceptedStep && self.non_linearity != NonLinearity::Linear {
            let n1 = self.n1;
            let v = y[..n1].to_vec();
            self.maybe_update_poles(&v)
        } else {
            false
        }
    }
}

pub struct RunOutput {
    /// `Qsol[0]` after every output sample (unfiltered `oto_emission`).
    pub qsol0: Vec<f64>,
    pub v_probe: Vec<f64>, // samples × probes, row-major
    pub y_probe: Vec<f64>,
    pub stats: Stats,
    pub pole_updates: u64,
}

#[derive(Debug)]
pub struct RunError {
    pub error: DopriError,
    pub sample: usize,
    pub t: f64,
}

impl Model {
    /// `cochlea_model.solve()` on an already Puria-filtered stimulus. Integrates
    /// `stim.len() - 2` samples, like upstream.
    pub fn run(&mut self, stim: &[f64], probes: &[usize]) -> Result<RunOutput, RunError> {
        let n1 = self.n1;
        let length = stim.len().saturating_sub(2);
        let np = probes.len();
        let mut qsol0 = vec![0.0; length];
        let mut v_probe = vec![0.0; length * np];
        let mut y_probe = vec![0.0; length * np];
        let mut y = vec![0.0; 2 * n1];
        let mut t = 0.0f64;
        let mut ws = Workspace::new(2 * n1);
        let mut stats = Stats::default();
        let opt = self.options;
        for j in 0..length {
            if j > 0 {
                self.interp[0] = stim[j - 1];
            }
            self.interp[1] = stim[j];
            self.interp[2] = stim[j + 1];
            self.interp[3] = stim[j + 2];
            let xend = t + self.dt;
            dopri5::integrate(self, &mut t, &mut y, xend, &opt, &mut ws, &mut stats)
                .map_err(|e| RunError { error: e, sample: j, t })?;
            self.last_t = t;
            self.after_sample(&mut y);
            if let Some((ch, age)) = self.ring_violation {
                return Err(RunError { error: DopriError::RingTooShort { channel: ch, age }, sample: j, t });
            }
            for (k, &p) in probes.iter().enumerate() {
                v_probe[j * np + k] = y[p];
                y_probe[j * np + k] = y[n1 + p];
            }
            qsol0[j] = self.qsol[0];
        }
        Ok(RunOutput { qsol0, v_probe, y_probe, stats, pole_updates: self.pole_updates })
    }
}

#[cfg(test)]
mod tests {
    use super::py_mod_f64;

    #[test]
    fn euclid_remainder_matches_numpy_float_remainder() {
        for lgt in [1usize, 7, 12_701] {
            for zwp in [0usize, 3, 6, 12_700] {
                if zwp >= lgt {
                    continue;
                }
                for mud in [1.0f64, 2.0, 55.0, 12_700.0, 12_701.0, 30_000.0] {
                    let diff = (zwp + lgt) as f64 - mud;
                    let a = py_mod_f64(diff, lgt as f64) as i64;
                    let b = (diff as i64).rem_euclid(lgt as i64);
                    assert_eq!(a, b, "lgt={lgt} zwp={zwp} mud={mud}");
                }
            }
        }
    }
}
