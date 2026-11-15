# Track B report — error-law / Lindblad selection / BLP smoke

Live Braket notebook artifact:  
`s3://amazon-braket-sri-results-1/sri-braket/track-b/results/20261111T220744Z_track_b.json`  
Repo mirror: [`docs/benchmark-results/errorlaw-track-b/live_braket_track_b.json`](benchmark-results/errorlaw-track-b/live_braket_track_b.json)

## Honesty boundary

- Error-law fit is on a **small notebook grid** (\(N\in\{16,32,64,128\}\), \(K\in\{32,64\}\), SNR \(\in\{10,15,20\}\) dB). Coefficients are not interchangeable with the larger v2 reference fit.
- Notebook used \(s_{\mathrm{amp}}=0.3\); suite `DEFAULT_S_AMP` is `0.25`.
- Selection check verifies the **depolarizing** closed form only. Amplitude-damping is deferred.
- BLP cell is a **toy backflow indicator**, not the full density-matrix Breuer–Laine–Piilo measure in `sri.selection`.

## Results (live Braket notebook, 2026-11-11)

### Error-law held-out validation

Two-term law:

\[
\mathrm{MSE}(\hat\kappa)\approx \frac{\mathbb E|\tau|^{-2}}{K\cdot s}+\frac{\mathbb E|\tau|^{-4}}{K\cdot s^{2}}\bar R(G)
\]

| Quantity | Value |
|---|---:|
| \(\mathbb E\|\tau\|^{-2}\) | 1.197 |
| \(\mathbb E\|\tau\|^{-4}\) | 2.054 |
| \(N^{*}\) @ 20 dB (ring) | ≈ 699 |
| \(\hat c_f\) | 0.710 |
| \(\hat c_L\) | 0.055 |
| Held-out \(R^{2}\) | **0.798** |
| Within 2 residual σ | 0.833 |

`heldout_r2 > 0` and `pass_snr_ge_10 = true` → **PASS**

Reference (larger v2 set): \(c_f\approx1.128\), \(c_L\approx0.730\), \(R^{2}\approx0.765\). Notebook \(\hat c_L\) is lower because theory preds sit above MC on this coarse grid; \(R^{2}\) remains strong.

### Lindblad / selection steady state

\(F_{ss}=(r+1/d)/(r+1)\) at \(\Xi_{\mathrm{crit}}=0.9\), channel = depolarizing:

| \(d\) | \(r^{*}\) | \(F_{ss}\) theory | \(F_{ss}\) numeric | abs err |
|---:|---:|---:|---:|---:|
| 2 | 4.00 | 0.9 | 0.9 | \(2.2\times10^{-16}\) |
| 3 | 5.67 | 0.9 | 0.9 | \(1.1\times10^{-16}\) |
| 4 | 6.50 | 0.9 | 0.9 | \(1.1\times10^{-16}\) |
| 8 | 7.75 | 0.9 | 0.9 | \(1.1\times10^{-16}\) |

`max_abs_err ≈ 2.2e-16`, `pass = true` → **PASS**

### BLP non-Markovian smoke

| Indicator | Value |
|---|---:|
| \(N_{\mathrm{BLP}}\) Markov | 0.0 |
| \(N_{\mathrm{BLP}}\) memory kernel | 0.0318 |
| memory > Markov | true |

→ **PASS** (toy indicator)

## Campaign verdict

**Track B PASS** under the declared evidence bar:
1. Held-out error-law \(R^{2}\approx0.80\) on the notebook grid  
2. Depolarizing selection closed form matches step propagator to machine precision  
3. Toy BLP backflow separates Markov vs memory kernels without overclaiming full BLP  

## Next

- Amplitude-damping selection hardening  
- Full density-matrix BLP in `sri.selection` on a larger campaign grid  
- Optional re-fit at `s_amp=0.25` to align with suite default  
