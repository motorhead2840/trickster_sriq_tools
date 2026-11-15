//! Host-side functional check (TileGym-style KERNEL_OK / HOST path without CUDA).

use sri_cutile_rs_stub::{mirror_abs2_tile, observable_upsilon_tile, oracle_samya_tile};

fn main() {
    let v = [1.0_f64, -2.0, 0.5, 3.0];
    let xi = [0.2_f64, -0.1, 0.4, -0.3];
    let thesis: Vec<f64> = v.iter().zip(xi.iter()).map(|(a, b)| a + b).collect();
    let anti: Vec<f64> = v.iter().zip(xi.iter()).map(|(a, b)| a - b).collect();
    let samya = oracle_samya_tile(&thesis, &anti).expect("oracle");
    let oracle_ok = samya
        .iter()
        .zip(v.iter())
        .all(|(s, t)| (s - t).abs() < 1e-12);

    let ya = [1.0_f64, 2.0, 3.0, 4.0];
    let yb = [0.5_f64, -0.5, 1.0, -1.0];
    let (w_a, w_b) = (0.55_f64, 0.45_f64);
    let (_antithesis, proj) = observable_upsilon_tile(&ya, &yb, w_a, w_b).expect("obs");
    let obs_ok = proj
        .iter()
        .zip(ya.iter().zip(yb.iter()))
        .all(|(p, (a, b))| (p - (w_a * a + w_b * b)).abs() < 1e-12);

    let mirror = mirror_abs2_tile(&ya, &yb).expect("mirror");
    let mirror_ok = mirror.iter().all(|&m| m >= 0.0);

    let pass = oracle_ok && obs_ok && mirror_ok;
    println!("oracle_ok={oracle_ok} obs_ok={obs_ok} mirror_ok={mirror_ok}");
    if pass {
        println!("<VALIDATOR_OUTPUT>");
        println!("VERDICT: KERNEL_OK");
        println!("</VALIDATOR_OUTPUT>");
        std::process::exit(0);
    }
    println!("<VALIDATOR_OUTPUT>");
    println!("VERDICT: KERNEL_FAIL");
    println!("</VALIDATOR_OUTPUT>");
    std::process::exit(1);
}
