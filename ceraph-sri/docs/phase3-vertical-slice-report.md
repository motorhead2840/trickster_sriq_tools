# Phase 3 report — Integrated falsifiable SRI vertical slice

**Verdict: PASS** against the preregistered criterion in [`phase3-vertical-slice-spec.md`](phase3-vertical-slice-spec.md).  
**Artifacts:** [`benchmark-results/phase3-vertical-slice/`](benchmark-results/phase3-vertical-slice/)  
**Runner:** `experiments/phase3_vertical_slice.py` (CPU, NumPy)

---

## 1. Vertical slice implemented

End-to-end **deployable dual-mode CV sensing** path:

`simulate_measurements` → `observable_yoga_buddhi(y_A, y_B)` → MSE vs latent \(s\) (scoring only)

Compared to:

| Method | Role |
|---|---|
| `single_mode_wiener` | **Primary** conventional baseline |
| moving average / median on \(y_A\) | Weaker conventional references |
| `estimate_signal` (plain MMSE) | **Ablation** of SRI packaging |
| `oracle_paired_projection` | Non-deployable upper bound |

**Honesty:** \(\hat\Upsilon_{\mathrm{obs}}\equiv\hat s_{\mathrm{MMSE}}\) by construction. Phase 3 does **not** claim an MSE advantage over Wiener. The practical claim is dual-mode observable sensing (SRI’s deployable path) vs single-mode conventional estimation under the declared model.

---

## 2. Reproduce

```bash
pip install -e ".[dev]"
python3 experiments/phase3_vertical_slice.py --output-dir docs/benchmark-results/phase3-vertical-slice
python3 -m pytest tests/test_phase3_vertical_slice.py -v
python3 -m pytest -m "not slow"
```

---

## 3. Measured results (15 cells: 5 seeds × 3 noise levels)

| Metric | Value |
|---|---:|
| Mean SRI / plain-MMSE sample MSE | **0.6105** |
| Mean single-mode MSE | **0.6972** |
| Mean relative reduction vs single-mode \(R\) | **0.1205 (12.05%)** |
| Pass threshold | \(R \ge 0.08\) |
| Mean MA / median MSE | 1.103 / 1.253 |
| Mean oracle MSE | ~0 (non-deployable) |
| Mean \|SRI − MMSE\| | ~4.4×10⁻¹⁶ |
| Wall time (full grid) | ~2.76 s CPU |
| Mean samples/s (estimator segment) | ~8.9×10⁷ |

### Per-noise regime (report uncertainty of “one number”)

| \(V_n\) | Mean SRI MSE | Mean single MSE | Mean \(R\) |
|---|---:|---:|---:|
| 0.5 | 0.599 | 0.636 | **0.058** (< 0.08 if judged alone) |
| 1.0 | 0.612 | 0.692 | 0.116 |
| 2.0 | 0.621 | 0.764 | 0.187 |

Aggregate mean \(R\) still clears 0.08. Low-noise regime alone would be **inconclusive** under the same threshold — preserved as a limitation, not hidden.

---

## 4. Verdict

**PASS** — observable-only, algebra≡MMSE, 5σ analytic match on all cells; mean \(R=0.1205\ge0.08\).

---

## 5. C5 status reconciliation

| Source (before Phase 3) | C5 |
|---|---|
| `docs/conjecture-registry.md` | confirmed (GPU elaborate narrative) |
| `docs/benchmark-results/conjecture_status.json` | **open** (“awaiting runner”) |
| `sri.registry.conjectures` defaults | open until `record_result` |

**Resolution:** Phase 3 CPU runner called `record_result("C5", "confirmed", …)` with preserved artifacts. Canonical JSON updated. Markdown registry notes both GPU provenance and Phase 3 CPU confirmation. Unit tests alone do not promote C5; the runner does.

---

## 6. Gaps remaining

| Gap | Status |
|---|---|
| Independent external replication | Not done |
| Blind calibration (unknown model parameters) | Not tested — weights use known \(V_s,V_n,\ldots\) |
| Equal-mode resource fairness | Dual-mode uses \(y_B\); single-mode does not |
| A5 / Lemma Z2 composition | Still open / stub |
| Customer demand / revenue | None established |
| O(1), Landauer, quantum advantage, LLM product reliability | Unsupported |

---

## 7. Single recommended next milestone

**Preregister and run a blind-calibration or mismatched-parameter stress test** of the same vertical slice: freeze weights from a training/calibration draw, evaluate on held-out draws with parameter error, and set a new pass/fail threshold for degradation vs oracle-parameter MMSE. That is the next *evidence* step; it is not “prove the paradigm.”

Optional parallel research question (not a Phase 3 claim): define a minimal executable A5 mark→antithesis map only if it has a falsifiable MSE/latency criterion on this same sensing path.
