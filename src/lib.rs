// © copyright 2017 Ghent University – Universiteit Gent, all rights reserved; this Derivative work is
// made available for non-commercial academic research purposes and subject to an UGent Academic
// License (see LICENSE and NOTICE).
// Changed files notice: Rust port (2026) of the cochlear core of Verhulstetal2018Model v1.2;
// this file contains the crate root and the Python (PyO3) binding.

pub mod dopri5;
pub mod kernels;
pub mod model;

#[cfg(feature = "python")]
mod python {
    use crate::dopri5::Options;
    use crate::kernels;
    use crate::model::{Config, Model, NonLinearity, PoleUpdate};
    use numpy::ndarray::Array2;
    use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
    use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
    use pyo3::prelude::*;
    use pyo3::types::PyDict;

    fn parse_nl(s: &str) -> NonLinearity {
        // upstream: 'disp' -> 1, 'vel' -> 2, anything else -> linear
        match s {
            "disp" => NonLinearity::Disp,
            "vel" => NonLinearity::Vel,
            _ => NonLinearity::Linear,
        }
    }

    /// Low-level handle on one initialised cochlear model (state after `init_model` and the
    /// `solve()` prologue). Most users want `verhulst2018_rs.simulate`.
    #[pyclass(name = "CochleaCore", module = "verhulst2018_rs._core", unsendable)]
    pub struct PyCore {
        m: Model,
    }

    #[pymethods]
    impl PyCore {
        #[new]
        #[pyo3(signature = (*, samplerate, sections, shera_po, rth, compression_slope=0.4,
            zweig_irregularities=true, non_linearity_type="vel", knee_var=1.0,
            low_freq_irregularities=true, irr_pct=0.05, rtol=1e-2, atol=1e-13, nsteps=500,
            pole_update="faithful", compact_delay=true))]
        #[allow(clippy::too_many_arguments)]
        fn new(
            samplerate: f64,
            sections: usize,
            shera_po: PyReadonlyArray1<f64>,
            rth: PyReadonlyArray1<f64>,
            compression_slope: f64,
            zweig_irregularities: bool,
            non_linearity_type: &str,
            knee_var: f64,
            low_freq_irregularities: bool,
            irr_pct: f64,
            rtol: f64,
            atol: f64,
            nsteps: i64,
            pole_update: &str,
            compact_delay: bool,
        ) -> PyResult<Self> {
            let pole_update = match pole_update {
                "faithful" => PoleUpdate::Faithful,
                "accepted_step" => PoleUpdate::AcceptedStep,
                other => return Err(PyValueError::new_err(format!("unknown pole_update {other:?}"))),
            };
            if !(rtol > 0.0 && atol >= 0.0) {
                return Err(PyValueError::new_err("rtol must be > 0 and atol >= 0"));
            }
            let cfg = Config {
                samplerate,
                sections,
                shera_po: shera_po.as_slice()?.to_vec(),
                compression_slope,
                zweig_irregularities,
                non_linearity: parse_nl(non_linearity_type),
                knee_var,
                low_freq_irregularities,
                irr_pct,
                rth: rth.as_slice()?.to_vec(),
                options: Options { rtol, atol, nmax: nsteps, ..Default::default() },
                pole_update,
                compact_delay,
            };
            Ok(PyCore { m: Model::new(&cfg).map_err(PyValueError::new_err)? })
        }

        /// Integrate `len(stim) - 2` samples of an already Puria-filtered stimulus.
        /// Returns a dict with `qsol0` (unfiltered oto_emission), `v_probe`, `y_probe`
        /// (samples × probes) and integrator statistics.
        #[pyo3(signature = (stim, probes=Vec::new()))]
        fn run<'py>(&mut self, py: Python<'py>, stim: PyReadonlyArray1<f64>, probes: Vec<usize>) -> PyResult<Bound<'py, PyDict>> {
            let n1 = self.m.n1;
            if let Some(&p) = probes.iter().find(|&&p| p >= n1) {
                return Err(PyValueError::new_err(format!("probe channel {p} out of range 0..{n1}")));
            }
            let stim = stim.as_slice()?.to_vec();
            let m = &mut self.m;
            let out = py.detach(move || m.run(&stim, &probes).map(|o| (o, probes.len())));
            let (out, np) = out.map_err(|e| {
                PyRuntimeError::new_err(format!(
                    "dopri5 failed ({}: {}) at output sample {} (t={} s)",
                    e.error.code(),
                    e.error.message(),
                    e.sample,
                    e.t
                ))
            })?;
            let len = out.qsol0.len();
            let d = PyDict::new(py);
            d.set_item("qsol0", out.qsol0.into_pyarray(py))?;
            let vp = Array2::from_shape_vec((len, np), out.v_probe).unwrap();
            let yp = Array2::from_shape_vec((len, np), out.y_probe).unwrap();
            d.set_item("v_probe", vp.into_pyarray(py))?;
            d.set_item("y_probe", yp.into_pyarray(py))?;
            d.set_item("rhs_calls", out.stats.rhs_calls)?;
            d.set_item("nfcn", out.stats.nfcn)?;
            d.set_item("nstep", out.stats.nstep)?;
            d.set_item("naccept", out.stats.naccept)?;
            d.set_item("nreject", out.stats.nreject)?;
            d.set_item("pole_updates", out.pole_updates)?;
            d.set_item("exact_pole_checks", self.m.exact_pole_checks)?;
            Ok(d)
        }

        /// One `TLsolver(t, y, model)` evaluation, with its side effects on the model.
        fn rhs<'py>(&mut self, py: Python<'py>, t: f64, y: PyReadonlyArray1<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
            let y = y.as_slice()?;
            if y.len() != 2 * self.m.n1 {
                return Err(PyValueError::new_err("y must have length 2*(sections+1)"));
            }
            let mut dy = vec![0.0; y.len()];
            self.m.tl_rhs(t, y, &mut dy);
            Ok(dy.into_pyarray(py))
        }

        fn set_interp(&mut self, a: f64, b: f64, c: f64, d: f64) {
            self.m.interp = [a, b, c, d];
        }

        fn set_last_t(&mut self, t: f64) {
            self.m.last_t = t;
        }

        /// Post-sample bookkeeping of `solve()`; returns the (possibly modified) state.
        fn after_sample<'py>(&mut self, py: Python<'py>, y: PyReadonlyArray1<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
            let mut y = y.as_slice()?.to_vec();
            self.m.after_sample(&mut y);
            Ok(y.into_pyarray(py))
        }

        /// Overwrite the delay-line buffer (shape (sections+1, YbufferLgt), float64 →
        /// stored as float32 taps) and the write pointer.
        fn set_ybuffer(&mut self, buf: PyReadonlyArray2<f64>, zwp: usize) -> PyResult<()> {
            let a = buf.as_array();
            let (r, c) = a.dim();
            if r != self.m.n1 || c != self.m.ybuffer_lgt {
                return Err(PyValueError::new_err("ybuffer shape mismatch"));
            }
            if zwp >= c {
                return Err(PyValueError::new_err("zwp out of range"));
            }
            // Ring slot s of channel i holds the sample written `age` writes ago, i.e. the
            // upstream column (zwp - age) mod YbufferLgt.
            let m = &mut self.m;
            m.zwp = zwp;
            for i in 0..r {
                let len = m.ring_len[i];
                let w = zwp % len;
                m.wpos[i] = w;
                for s in 0..len {
                    let age = (w + len - s) % len;
                    let col = (zwp + c - age) % c;
                    m.ybuffer[m.ring_off[i] + s] = a[[i, col]] as f32;
                }
            }
            m.ring_epoch_bump();
            m.zweig_impedance();
            Ok(())
        }

        fn set_vector(&mut self, name: &str, v: PyReadonlyArray1<f64>) -> PyResult<()> {
            let v = v.as_slice()?;
            let m = &mut self.m;
            let dst = match name {
                "SheraP" => &mut m.shera_p,
                "Dev" => &mut m.dev,
                _ => return Err(PyKeyError::new_err(name.to_string())),
            };
            if v.len() != dst.len() {
                return Err(PyValueError::new_err("length mismatch"));
            }
            dst.copy_from_slice(v);
            m.refresh_pole_screen();
            Ok(())
        }

        fn shera_parameters(&mut self) {
            self.m.shera_parameters();
        }

        fn zweig_impedance(&mut self) {
            self.m.zweig_impedance();
        }

        /// Read an internal vector by its upstream attribute name.
        fn vector<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyArray1<f64>>> {
            let m = &self.m;
            let as_f = |v: &Vec<usize>| v.iter().map(|&x| x as f64).collect::<Vec<f64>>();
            let v: Vec<f64> = match name {
                "x" => m.x.clone(),
                "f_resonance" => m.f_resonance.clone(),
                "omega" => m.omega.clone(),
                "omega2" => m.omega2.clone(),
                "Sherad_factor" => m.sherad_factor.clone(),
                "SheraPo" => m.shera_po.clone(),
                "SheraP" => m.shera_p.clone(),
                "SheraD" => m.shera_d.clone(),
                "SheraRho" => m.shera_rho.clone(),
                "SheraMu" => m.shera_mu.clone(),
                "PoleS" => m.pole_s.clone(),
                "PoleE" => m.pole_e.clone(),
                "Rth" => m.rth.clone(),
                "RthY1" => m.rthy1.clone(),
                "RthY2" => m.rthy2.clone(),
                "RthV1" => m.rthv1.clone(),
                "RthV2" => m.rthv2.clone(),
                "Sa" => m.sa.clone(),
                "Sb" => m.sb.clone(),
                "const_nl1" => m.const_nl1.clone(),
                "cosTheta" => m.cos_theta.clone(),
                "sinTheta" => m.sin_theta.clone(),
                "exact_delay" => m.exact_delay.clone(),
                "ZweigMs" => m.zweig_ms.clone(),
                "ZweigMp" => m.zweig_mp.clone(),
                "ZASQ" => m.zasq.clone(),
                "ZASC" => m.zasc.clone(),
                "ZAL" => m.zal.clone(),
                "ZAH" => m.zah.clone(),
                "Dev" => m.dev.clone(),
                "YZweig" => m.yzweig.clone(),
                "Zrp" => as_f(&m.zrp_all()[0]),
                "Zrp1" => as_f(&m.zrp1),
                "Zrp2" => as_f(&m.zrp_all()[2]),
                "Zrp3" => as_f(&m.zrp_all()[3]),
                "g" => m.g.clone(),
                "right" => m.right.clone(),
                "Qsol" => m.qsol.clone(),
                _ => return Err(PyKeyError::new_err(name.to_string())),
            };
            Ok(v.into_pyarray(py))
        }

        /// Read an internal scalar by its upstream attribute name.
        fn scalar(&self, name: &str) -> PyResult<f64> {
            let m = &self.m;
            Ok(match name {
                "dt" => m.dt,
                "bm_length" => m.bm_length,
                "ZweigMso" => m.zweig_mso,
                "ZweigL" => m.zweig_l,
                "ZweigOmega_co" => m.zweig_omega_co,
                "ZweigMpo" => m.zweig_mpo,
                "Ko" => m.ko,
                "dx" => m.dx,
                "q0_factor" => m.q0_factor,
                "p0x" => m.p0x,
                "d_m_factor" => m.d_m_factor,
                "RK4_0" => m.rk4_0,
                "RK4G_0" => m.rk4g_0,
                "ctr" => m.ctr as f64,
                "onek" => m.onek as f64,
                "YbufferLgt" => m.ybuffer_lgt as f64,
                "Zwp" => m.zwp as f64,
                "Yknee1" => m.yknee1,
                "Yknee2" => m.yknee2,
                "Vknee1" => m.vknee1,
                "Vknee2" => m.vknee2,
                "lastT" => m.last_t,
                "non_linearity" => m.non_linearity.code() as f64,
                "pole_updates" => m.pole_updates as f64,
                "ybuffer_floats" => m.ybuffer.len() as f64,
                _ => return Err(PyKeyError::new_err(name.to_string())),
            })
        }
    }

    /// Upstream `cubic_interpolate` (float32 arguments) as compiled with -O3 -ffast-math.
    #[pyfunction]
    fn cubic_interpolate(y0: f32, y1: f32, y2: f32, y3: f32, mu: f32) -> f64 {
        kernels::cubic_interpolate(y0, y1, y2, y3, mu)
    }

    /// Upstream `delay_line(Y, Zrp, Zrp1, Zrp2, Zrp3, Dev, out, M, N)`.
    #[pyfunction]
    fn delay_line<'py>(
        py: Python<'py>,
        ybuf: PyReadonlyArray2<f64>,
        zrp: Vec<usize>,
        zrp1: Vec<usize>,
        zrp2: Vec<usize>,
        zrp3: Vec<usize>,
        dev: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let a = ybuf.as_array();
        let (n, m) = a.dim();
        let dev = dev.as_slice()?;
        if [zrp.len(), zrp1.len(), zrp2.len(), zrp3.len(), dev.len()].iter().any(|&l| l != n) {
            return Err(PyValueError::new_err("length mismatch"));
        }
        let buf: Vec<f32> = a.iter().map(|&v| v as f32).collect();
        let mut out = vec![0.0; n];
        kernels::delay_line(&buf, &zrp, &zrp1, &zrp2, &zrp3, dev, &mut out, m);
        Ok(out.into_pyarray(py))
    }

    /// Upstream `solve_tridiagonal` (Thomas algorithm, fused as compiled upstream).
    #[pyfunction]
    fn solve_tridiagonal<'py>(
        py: Python<'py>,
        a: PyReadonlyArray1<f64>,
        b: PyReadonlyArray1<f64>,
        c: PyReadonlyArray1<f64>,
        r: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let (a, b, c, r) = (a.as_slice()?, b.as_slice()?, c.as_slice()?, r.as_slice()?);
        let n = r.len();
        if a.len() != n || b.len() != n || c.len() != n {
            return Err(PyValueError::new_err("length mismatch"));
        }
        let mut x = vec![0.0; n];
        kernels::Tridiag::new(a, b, c).solve(r, &mut x);
        Ok(x.into_pyarray(py))
    }

    #[pymodule]
    fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyCore>()?;
        m.add_function(wrap_pyfunction!(cubic_interpolate, m)?)?;
        m.add_function(wrap_pyfunction!(delay_line, m)?)?;
        m.add_function(wrap_pyfunction!(solve_tridiagonal, m)?)?;
        m.add("NOTICE", "© copyright 2017 Ghent University – Universiteit Gent, all rights reserved; this Derivative work is made available for non-commercial academic research purposes and subject to an UGent Academic License")?;
        Ok(())
    }

    #[allow(dead_code)]
    fn _unused(_: &Bound<'_, PyArray2<f64>>) {}
}
