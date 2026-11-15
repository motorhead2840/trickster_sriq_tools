# CV foundation month report

**Scope:** classical simulation of continuous-variable (CV) quantum optics as the SRI research core; Colab A100 for promotion grids; AWS Braket dry-run for December.

## What changed

1. **Observable-only core** — `sri.cv.estimator` estimates the signal from \((y_A,y_B)\) via the Gaussian MMSE filter. Legacy oracle Yoga–Buddhi remains an upper-bound identity only.
2. **Invariants** — symplectic checks, vacuum-injected loss, CPTP attenuator (`sri.cv.invariants`, `sri.dynamics`).
3. **Axiomatic metrics** — `Ξ_SNR`, purity/Fisher proxies, Gaussian von Neumann entropy from symplectic eigenvalues (`sri.metrics`).
4. **Circuit B dual branch** — printed (P) vs worked-example (W) in `sri.dynamics.v3_damping`.
5. **Campaign + GPU CV demo** — `experiments/cv_photonics_campaign.py`, `sri_gpu_cv_demo.py`.
6. **Braket-ready** — dry-run adapter + December runbook; no live submit in this revision.

## Conjecture outcomes

Merged machine-readable status: [`docs/benchmark-results/conjecture_status.json`](benchmark-results/conjecture_status.json).

| ID | Status | Meaning |
|---|---|---|
| C1 | **confirmed** | Oracle identity holds; MMSE matches analytic Wiener without oracle noise |
| C2 | **confirmed** | Default squeezed-reference ordering matches closed forms |
| C3 | **confirmed** | `Ξ_SNR` in ≈[0.010, 0.931] on the foundation spike grid; legacy Ξ reached ≈−148 |
| C4 | **confirmed** | Entropy nonincreasing under CPTP; mean χ_P≈50.4 vs χ_W≈22.4 |

CPU/NumPy preflight also showed MMSE beating moving-average and median baselines on the two-mode task (reference mode available to MMSE only — see equal-resource notes in summaries).

## Remaining gaps

- Parameter-oracle MMSE coefficients (known \(V_s,V_n,k,\eta,r\)); calibration from data is future work.
- `sri_gpu_cv_demo.py` uses NumPy CV algebra; JAX GPU is detected for provenance but not required unless `--require-gpu`.
- No photonic hardware or live Braket QPU results yet.
- LLM checkpoint is a single pass / synthetic smoke unless Colab runs the legacy TruthfulQA harness once.

## Freeze candidates

Tag candidates after A100 confirmation artifacts are committed:

- `sri.cv` observable-only API
- Branch **W** as the preferred Circuit-B research default (Branch P retained for printed-formula fidelity)
- Conjecture registry statuses linked to benchmark JSON
