//! CPU stub for Yoga–Buddhi tile ops (NVIDIA cuTile Python→Rust path).
//!
//! Compiles without CUDA Tile / `tileiras`. When a real `cutile-rs` backend
//! exists, keep these signatures and swap the body for tile kernels + IR dumps.

/// Oracle Samya: 0.5 * (thesis + antithesis). Latent / known-ξ chart.
pub fn oracle_samya_tile(thesis: &[f64], antithesis: &[f64]) -> Result<Vec<f64>, String> {
    if thesis.len() != antithesis.len() {
        return Err("thesis/antithesis length mismatch".into());
    }
    Ok(thesis
        .iter()
        .zip(antithesis.iter())
        .map(|(t, a)| 0.5 * (t + a))
        .collect())
}

/// Lemma Z1 observable path: antithesis functional + Υ_obs (explicit weights).
pub fn observable_upsilon_tile(
    y_a: &[f64],
    y_b: &[f64],
    w_a: f64,
    w_b: f64,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    if y_a.len() != y_b.len() {
        return Err("y_a/y_b length mismatch".into());
    }
    let mut antithesis = Vec::with_capacity(y_a.len());
    let mut projected = Vec::with_capacity(y_a.len());
    for (&ya, &yb) in y_a.iter().zip(y_b.iter()) {
        let a = (2.0 * w_a - 1.0) * ya + (2.0 * w_b) * yb;
        antithesis.push(a);
        projected.push(0.5 * (ya + a));
    }
    Ok((antithesis, projected))
}

/// Mirror M = |z|² on split real/imag channels.
pub fn mirror_abs2_tile(real: &[f64], imag: &[f64]) -> Result<Vec<f64>, String> {
    if real.len() != imag.len() {
        return Err("real/imag length mismatch".into());
    }
    Ok(real
        .iter()
        .zip(imag.iter())
        .map(|(r, i)| r * r + i * i)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oracle_recovers_signal() {
        let v = vec![1.0, -2.0, 0.5];
        let xi = vec![0.2, -0.1, 0.4];
        let thesis: Vec<f64> = v.iter().zip(xi.iter()).map(|(a, b)| a + b).collect();
        let anti: Vec<f64> = v.iter().zip(xi.iter()).map(|(a, b)| a - b).collect();
        let samya = oracle_samya_tile(&thesis, &anti).unwrap();
        for (s, target) in samya.iter().zip(v.iter()) {
            assert!((s - target).abs() < 1e-12);
        }
    }

    #[test]
    fn observable_matches_mmse_weights() {
        let ya = vec![1.0, 2.0, 3.0];
        let yb = vec![0.5, -0.5, 1.0];
        let w_a = 0.6;
        let w_b = 0.4;
        let (_a, proj) = observable_upsilon_tile(&ya, &yb, w_a, w_b).unwrap();
        for i in 0..ya.len() {
            let mmse = w_a * ya[i] + w_b * yb[i];
            assert!((proj[i] - mmse).abs() < 1e-12);
        }
    }
}
