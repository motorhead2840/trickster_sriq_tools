# Track A report — Ω-dit / complexity / Landauer (A1)

Live Braket notebook artifact:  
`s3://amazon-braket-sri-results-1/sri-braket/track-a/results/20261111T220302Z_track_a.json`  
Repo mirror: [`docs/benchmark-results/omega-landauer-track-a/live_braket_track_a.json`](benchmark-results/omega-landauer-track-a/live_braket_track_a.json)

## Honesty boundary

- Ω-dit in this suite is a **finite-grid** sample of \(\sigma e^{i\beta}(v+\xi)\), not a continuous physical field.
- N-scaling measures **this software implementation**. It does **not** validate yellow-paper hardware-native O(1)/O(0).
- Landauer numbers are \(k_B T\ln 2\) **theory checks only**. Circumvention is **not validated**.

## Results (live Braket notebook, 2026-11-11)

### Ω-dit harness
- dimension 64, 8 samples
- `integral_mean_abs ≈ 0.305`
- `finite = true` → **PASS**

### Complexity / O(1) software test
| N | Projector median (s) | Ω-integrate median (s) | Projection MSE |
|---:|---:|---:|---:|
| 256 | 7.50e-6 | 5.03e-5 | 0 |
| 1024 | 1.01e-5 | 7.78e-5 | 0 |
| 4096 | 1.71e-5 | 1.51e-4 | 0 |
| 16384 | 5.39e-5 | 5.38e-4 | 0 |

- Projector log-log slope ≈ **0.465** (\(R^2\approx0.917\))
- Ω-integrate log-log slope ≈ **0.560** (\(R^2\approx0.943\))
- `software_o1_refuted = true` → **PASS (honest)**

Projection MSE = 0 confirms the Yoga–Buddhi identity when antithesis is supplied.

### Landauer theory check
- Formula: \(Q_{\min}=k_B T\ln 2\) per erased bit
- At 300 K: \(\approx 2.871\times10^{-21}\) J/bit
- `circumvention_claim = not_validated` → **PASS (honest)**

## Campaign verdict

**Track A1 PASS** under the declared evidence bar:
1. Ω-dit executable and finite  
2. Software path not O(1)  
3. Landauer bound tabulated without overclaim  

## Next

Track B completed on branch `cursor/track-b-errorlaw-fd3c` (PR #16).  
Live artifact: `s3://amazon-braket-sri-results-1/sri-braket/track-b/results/20261111T220744Z_track_b.json`
