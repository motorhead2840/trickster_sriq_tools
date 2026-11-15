# Phase 3 experiment specification (preregistered)

**Status:** SPEC LOCKED before final benchmark execution  
**Date locked:** 2026-11-13  
**Vertical slice:** Deployable dual-mode CV sensing path (`observable_yoga_buddhi` / Lemma Z1 packaging)  
**Runner:** `experiments/phase3_vertical_slice.py` (CPU-only; no AWS/GPU required)

---

## 1. Why this use case

The repository already implements an **observable-only** Gaussian dual-mode estimator and an algebraic rewrite (`observable_yoga_buddhi`) that is Lemma Z1’s deployable projector. That path:

- runs end-to-end on CPU with NumPy;
- forbids clean signal / latent noise at inference;
- has conventional baselines (`single_mode_wiener`, moving average, median) and an oracle upper bound;
- matches a product-shaped sensing workflow (two readings → cleaned estimate).

### What this Phase 3 slice does **not** test

- Constant-time / \(\mathcal O(1)\) hardware claims  
- Quantum advantage or live Braket QPU execution  
- Landauer / thermodynamic circumvention  
- General LLM hallucination prevention or customer value  
- Full multi-layer composition / Theorem Zero / A5 (still stub)  
- Beating Wiener/MMSE: **by construction \(\hat\Upsilon_{\mathrm{obs}}\equiv\hat s_{\mathrm{MMSE}}\)** — equivalence is not a performance advantage  

---

## 2. Claim under test (P3)

Under the declared Gaussian dual-mode model \(\mathcal M_{\mathrm{CV}}\) with **known model parameters** (parameter oracle for Wiener weights; **not** latent \(s\) or \(n\)):

1. The deployable SRI path uses only \((y_A,y_B)\) at inference.  
2. \(\hat\Upsilon_{\mathrm{obs}}\) matches plain MMSE within relative tolerance \(10^{-10}\) (SRI-algebra ablation).  
3. Sample MSE matches analytic MMSE within the **5σ** sampling rule (same as C1).  
4. Across the preregistered seed×noise grid, mean **relative MSE reduction** versus the strongest conventional baseline **`single_mode_wiener`** is at least **8%**:

\[
R = 1 - \frac{\mathrm{MSE}_{\mathrm{SRI}}}{\mathrm{MSE}_{\mathrm{single}}}
\]

**Pass:** all of (1)–(3) hold on every cell, and mean \(R \ge 0.08\).  
**Fail:** any of (1)–(3) fail, or mean \(R < 0\).  
**Inconclusive:** (1)–(3) hold and \(0 \le\) mean \(R < 0.08\).

Moving-average and median filters are reported as weaker conventional references; they are **not** the primary baseline (they are easier to beat).

Oracle paired projection is reported as a **non-deployable** reference only.

---

## 3. Protocol

| Item | Value |
|---|---|
| Seeds | `201, 202, 203, 204, 205` (five; distinct from older 108–110 demos) |
| Shots per cell | `20000` |
| Signal variance \(V_s\) | `1.0` |
| Noise variance \(V_n\) grid | `0.5, 1.0, 2.0` |
| Efficiency \(\eta\) | `0.8` |
| Coupling \(k\) | `1.0` |
| Signal squeezing \(r_A\) | `0.0` |
| Reference squeezing \(r_B\) | `0.7` |
| Primary metric | Sample MSE vs latent \(s\) (evaluation only; \(s\) not fed to estimators) |
| Resource metrics | Wall seconds; samples/s; NumPy/Python versions |
| Primary baseline | `single_mode_wiener` |
| Ablation | `estimate_signal` (plain MMSE) vs `observable_yoga_buddhi` |
| Oracle | `oracle_paired_projection` (labeled non-deployable) |

---

## 4. C5 interaction

C5 (Lemma Z1 algebra + sample MSE + observable-only) has conflicting documentation (`confirmed` in some Markdown vs `open` in machine-readable registry awaiting runner). This Phase 3 runner **records** C5 against its written pass criterion using CPU artifacts. Promotion to `confirmed` occurs only if criteria are met and artifacts are preserved — not because a unit test alone passed.

---

## 5. Exclusions

- No AWS / Braket / CUDA required for the core verdict.  
- No editing of prior negative/exploratory artifacts.  
- No success-shaped fallback if \(R\) is weak: report fail/inconclusive plainly.
