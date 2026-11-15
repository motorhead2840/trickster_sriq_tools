# SRI technical foundation and implementation boundary

For the integrated mathematical framework across the unveiling, tracking,
selection, graph, and Gaussian sensing experiments, see
[theoretical-framework.md](theoretical-framework.md).

## Source

The theoretical source for this proof of concept is **SINGULARITY-BASED REALIZATION INTERFACE (SRI) ARCHITECTURE: Continuous-Variable Hardware-Native O(1) Symplectic Error Annihilation & Autopoietic Phase-Transition Damping Engine**, supplied as `sri-provisional-patent-dissertation-5.pdf`. The companion **Ceraph SRI Architecture - VC Project Report** is explanatory material that reproduces the source's claims; it is not independent experimental evidence. Neither PDF is checked into this repository.

## Core equations and code mapping

| Source concept | Equation in the source | Reference implementation |
|---|---|---|
| Omega-dit | `|Ω⟩ = ∫ σ(t)e^(iβ(t))[v(t) + ξ(t)] dω` | `OmegaDit.integrand()` and finite-grid `integrate_omega()` |
| Bipartite thesis state | `H = H_V ⊕ H_H`; `Ψ = v + ξ` | `symplectic_project()` returns `thesis = value + noise` |
| Conjugate antithesis | `Ψ' = v - ξ` | `symplectic_project()` returns `antithesis = value - noise` |
| Symplectic / Yoga–Buddhi projector | Dissertation: \(\hat\Upsilon_{YB}=\tfrac12(I+\hat{\mathcal C}_S)\); Samya \(\tfrac12[(v+\xi)+(v-\xi)]=v\). Observable complement: Lemma Z1 / `observable_yoga_buddhi` | `symplectic_project()` / `oracle_paired_projection` (oracle); `observable_yoga_buddhi` (deployable). See [yoga-buddhi-projector.md](yoga-buddhi-projector.md) |
| Stability coefficient | `Ξ = 1 - Var(logits)/(E(logits) + ε)` | `stability_coefficient()` |
| V3.0 damping | `χ = λ₀e^(-κ(1-Ξ)) + Λmax sigmoid(α(Var(∇)/(σ²sat ε) - θcrit))` | `v3_damping()` and `sri_gpu_demo.py` |
| Logit rescaling | `logits_damped = logits_raw * e^(-χ)` | `damp_logits()` and `sri_gpu_demo.py` |
| Landauer lower bound | `ΔQ ≥ kBT ln 2` | `landauer_minimum_heat()` evaluates the one-bit lower bound |

`integrate_omega()` is a rectangular finite-grid approximation. The code uses Python floating-point values and complex numbers; it does not represent a continuous physical field or a quantum state.

## What the projector result assumes

Let the observed state be `x = v + ξ`. The identity returns `v` if a matching antithesis `v - ξ` is available. The implementation receives both `v` and `ξ`, then constructs both states. If only `x` is observed, infinitely many `(v, ξ)` pairs can produce it; the source equations do not specify an estimator that selects the correct pair. Therefore this test demonstrates an algebraic identity with known components—not blind denoising, truth discovery, or hallucination prevention.

For a length-`N` vector, materializing the vectors and computing their average performs work proportional to `N` in software. Each elementwise operation may be constant work per element, but the complete vector operation is not constant work with respect to vector length. The code does not establish a single-cycle hardware implementation or the dissertation's O(0)/O(1) hardware claims.

## V3.0 equation caveats

The reference and GPU implementations follow the printed Equation 12 and multiply `σ²sat` by `ε` in Circuit B's denominator. With the source's example-scale values and `ε = 10^-7`, the sigmoid saturates near `Λmax` even for small nonzero gradient variance. The same document's worked example compares the unscaled variance directly to `θcrit`; those two descriptions do not agree. The implementations preserve the printed formula rather than silently changing it to match the worked example. GPU runs include the equation text and actual output metrics in `sri_gpu_run.json`.

Equation 11's stated range `0 ≤ Ξ ≤ 1` is not guaranteed by its formula: negative or near-zero mean logits can produce values outside that interval or a singular denominator. The implementation reports the formula result without clipping and raises an error when the denominator is effectively zero. Equation 13's scalar rescaling can reduce variance by shrinking every logit, but that is not evidence that erroneous tokens were identified; it can also alter the resulting softmax distribution.

## Validation boundaries

- The unit tests validate deterministic arithmetic, parameter/domain checks, CSV output, and reproducibility on synthetic vectors.
- The simulation uses generated clean values and generated noise, so its projection MSE checks arithmetic against known ground truth.
- Runtime fields measure Python execution on the current CPU host. They are illustrative only and are not comparable to the source document's GPU claims.
- The source's entropy and free-energy statements are not derived in this code. A variance statistic alone does not determine a density matrix or its von Neumann entropy.
- The `sri_pqc.py` sample is not LWE, ML-KEM, or any cryptographic scheme. It provides the exact synthetic noise to the projector and must not be used to protect data.
- Existing benchmark/telemetry values in tracked CSV files are historical inputs, not outputs validated by this reference simulator.

Further work toward a product proof of concept requires, at minimum, a defined observable-only method for estimating the cancellation component, comparison against appropriate baselines on held-out data, and separately instrumented accelerator tests. Security and cryptographic functionality require independent designs and reviews.

## CV foundation sprint additions (axiomatic status)

The Month-1 CV foundation work adds a **deployable** observable-only path beside the legacy oracle projector. Convention in `sri.cv` is ħ = 2 (vacuum quadrature variance 1).

| Concept | Equation / law | Implementation | Axiom status |
|---|---|---|---|
| Two-mode sensing | \(y_A=\sqrt{\eta}(s+n)+q_A\), \(y_B=\sqrt{\eta}(k n)+q_B\) | `sri.cv.estimator.simulate_measurements` | Physical Gaussian model |
| Observable-only MMSE | \(\hat s = w^\top y\), \(w=\mathrm{Cov}(s,y)\mathrm{Cov}(y,y)^{-1}\) | `estimate_signal`, `wiener_weights` | Optimal for this Gaussian model |
| Oracle Yoga–Buddhi | \(\tfrac12[(v+\xi)+(v-\xi)]=v\) | `oracle_paired_projection` / legacy `symplectic_project` | Algebraic upper bound only |
| Legacy Ξ | \(1-\mathrm{Var}/(\mathbb{E}+\varepsilon)\) | `sri.metrics.stability.legacy_xi` | Unclipped; may leave [0,1] |
| Axiomatic Ξ_SNR | \(\mathbb{E}^2/(\mathbb{E}^2+\mathrm{Var}+\varepsilon)\) | `snr_stability` | Bounded in (0,1] |
| Gaussian entropy | \(S=\sum g(\nu_i)\) from symplectic eigenvalues | `gaussian_von_neumann_entropy` | Standard CV formula (ħ=2) |
| Circuit B branch P | \(\mathrm{Var}/(\sigma_{\mathrm{sat}}^2\varepsilon)\) | `DampingParameters(branch=P)` | Printed formula; saturates easily |
| Circuit B branch W | \(\mathrm{Var}/\sigma_{\mathrm{sat}}^2\) vs \(\theta_{\mathrm{crit}}\) | `DampingParameters(branch=W)` | Worked-example branch |
| CPTP attenuator | \(V\mapsto e^{-2\chi}V+(1-e^{-2\chi})V_{\mathrm{vac}}\) | `cptp_attenuate_covariance` | Gates illegal maps in tests |

Conjecture outcomes are recorded in `docs/conjecture-registry.md` and JSON artifacts under `docs/benchmark-results/`. This remains a classical simulation of CV quantum optics until Braket/photonics contact.

## NVIDIA embodiment scaffolds (method only)

| Stack | Product use | Status |
|---|---|---|
| cuTile Python→Rust (TileGym agent pattern) | Yoga–Buddhi tile ops for Sphota GPU substrate | `sri.omlang.tile_ref` + `embodiment/cutile_rs_stub`; see [cuda-tile-rust-embodiment.md](cuda-tile-rust-embodiment.md) |
| Transformer Engine + JAX dropless MoE | SREE Mentor backbone training without capacity-drop | `sri.sree.moe_te_config`; see [sree-mentor-jax-te-moe.md](sree-mentor-jax-te-moe.md) |

Neither scaffold inherits NVIDIA published TFLOPS or cuTile geomean until measured on SRI workloads.
