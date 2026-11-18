// © copyright 2017 Ghent University – Universiteit Gent, all rights reserved; this Derivative work is
// made available for non-commercial academic research purposes and subject to an UGent Academic
// License (see LICENSE and NOTICE).
// Micro-benchmark of the right-hand side and of a full click-length run (synthetic poles).
use std::time::Instant;
use verhulst2018::dopri5::Options;
use verhulst2018::model::{Config, Model, NonLinearity, PoleUpdate};

fn main() {
    let n1 = 1001;
    let rth: Vec<f64> = (0..n1).map(|i| ((i as f64 * 0.618).fract() - 0.5) * 2.0).collect();
    let cfg = Config {
        samplerate: 100_000.0,
        sections: 1000,
        shera_po: vec![0.06; n1],
        compression_slope: 0.4,
        zweig_irregularities: true,
        non_linearity: NonLinearity::Vel,
        knee_var: 1.0,
        low_freq_irregularities: true,
        irr_pct: 0.05,
        rth,
        options: Options::default(),
        pole_update: PoleUpdate::Faithful,
        compact_delay: true,
    };
    let mut m = Model::new(&cfg).unwrap();
    let y: Vec<f64> = (0..2 * n1).map(|i| 1e-9 * ((i as f64) * 0.01).sin()).collect();
    let mut dy = vec![0.0; 2 * n1];
    let reps = 20000;
    let t0 = Instant::now();
    for k in 0..reps {
        m.tl_rhs(1e-7 * (k % 10) as f64, &y, &mut dy);
    }
    let per = t0.elapsed().as_secs_f64() / reps as f64;
    println!("rhs: {:.2} us/call (checksum {:e})", per * 1e6, dy[17]);

    let mut stim = vec![0.0; 8202];
    for (i, s) in stim.iter_mut().enumerate().take(400).skip(100) {
        *s = 0.02 * ((i as f64) * 0.3).sin();
    }
    let mut m = Model::new(&cfg).unwrap();
    let t0 = Instant::now();
    let out = m.run(&stim, &[]).unwrap();
    let el = t0.elapsed().as_secs_f64();
    println!(
        "click-length run: {:.3} s, {} rhs calls ({:.2}/sample, {:.2} us/call incl. integrator)",
        el,
        out.stats.rhs_calls,
        out.stats.rhs_calls as f64 / 8200.0,
        el / out.stats.rhs_calls as f64 * 1e6
    );
}
