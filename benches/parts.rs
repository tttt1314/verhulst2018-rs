use std::time::Instant;
use verhulst2018::dopri5::Options;
use verhulst2018::model::{Config, Model, NonLinearity, PoleUpdate};
use verhulst2018::kernels::window_tap;
fn main() {
    let n1 = 1001;
    let rth: Vec<f64> = (0..n1).map(|i| ((i as f64 * 0.618).fract() - 0.5) * 2.0).collect();
    let cfg = Config { samplerate: 100_000.0, sections: 1000, shera_po: vec![0.06; n1], compression_slope: 0.4,
        zweig_irregularities: true, non_linearity: NonLinearity::Vel, knee_var: 1.0, low_freq_irregularities: true,
        irr_pct: 0.05, rth, options: Options::default(), pole_update: PoleUpdate::Faithful, compact_delay: true };
    let mut m = Model::new(&cfg).unwrap();
    let v: Vec<f64> = (0..n1).map(|i| 1e-9 * ((i as f64) * 0.01).sin()).collect();
    let reps = 20000;
    let t0 = Instant::now();
    let mut c = 0;
    for _ in 0..reps { if m.maybe_update_poles(&v) { c += 1; } }
    println!("poles check: {:.2} us (updates {c}, exact {})", t0.elapsed().as_secs_f64() / reps as f64 * 1e6, m.exact_pole_checks);
    let t0 = Instant::now();
    for _ in 0..2000 { m.shera_parameters(); }
    println!("shera_parameters: {:.2} us", t0.elapsed().as_secs_f64() / 2000.0 * 1e6);
    let t0 = Instant::now();
    for _ in 0..2000 { m.zweig_impedance(); }
    println!("zweig_impedance: {:.2} us", t0.elapsed().as_secs_f64() / 2000.0 * 1e6);
    let t0 = Instant::now();
    let mut s = 0.0;
    for k in 0..reps { let fr = 0.01 * (k % 10) as f64; for i in 0..n1 { s += window_tap(&m.taps[i], m.dev[i] + fr); } }
    println!("delay line (rings): {:.2} us {s}", t0.elapsed().as_secs_f64() / reps as f64 * 1e6);
    let r: Vec<f64> = (0..n1).map(|i| (i as f64).cos()).collect();
    let mut x = vec![0.0; n1];
    let t0 = Instant::now();
    for _ in 0..reps { m.tridiag.solve(&r, &mut x); }
    println!("tridiag: {:.2} us {}", t0.elapsed().as_secs_f64() / reps as f64 * 1e6, x[3]);
}
#[allow(dead_code)]
fn unused() {}
