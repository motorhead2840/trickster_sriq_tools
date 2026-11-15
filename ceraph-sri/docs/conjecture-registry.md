# Conjecture registry (CV foundation sprint)

Statuses are only `open`, `confirmed`, or `refuted`. Pass criteria are fixed before A100 promotion. Machine-readable copies live in `sri.registry.conjectures` and `docs/benchmark-results/conjecture_status.json`.

| ID | Status | Notes |
|---|---|---|
| C1 | **confirmed** | Oracle identity verified; MMSE is observable-only Wiener filter. |
| C2 | **confirmed** | Analytic MSE ordering checked against mse_optimal closed forms. |
| C3 | **confirmed** | xi_snr range=[0.0102667, 0.931353]; legacy may exit [0,1] |
| C4 | **confirmed** | entropy_ok=True; mean_chi_P=50.4281; mean_chi_W=22.4281 |
| C5 | **confirmed** | Lemma Z1: observable Yoga–Buddhi ≡ MMSE under \(\mathcal M_{\mathrm{CV}}\); involution latent-only. **Reconciled 2026-11-13:** machine-readable JSON had been left `open` (“awaiting runner”) while Markdown already said confirmed from GPU elaborate. Phase 3 CPU runner re-confirmed against the written criterion (algebra + 5σ + observable-only) and wrote `docs/benchmark-results/phase3-vertical-slice/phase3_summary.json`. Prior GPU: `docs/benchmark-results/braket-live/sphota-yoga-buddhi-elaborate/sagemaker-g4dn/pasteback.json`. |
| C6 | **open** (draft) | Lemma Z1b: raw-\(Y\) non-involution witness \(W_{\mathrm{raw}}=\mathbb E[\lVert\widehat C_S(Y)+y_A\rVert_2^2]>\varepsilon\). Empirical mean ≈ 1.28233 on elaborate grid; \(\varepsilon\) not yet preregistered with Hari. |

**Phase 3 note:** A practical dual-mode vs single-mode MSE advantage was preregistered and **passed** (`docs/phase3-vertical-slice-report.md`). That advantage is **not** “beating Wiener” — \(\hat\Upsilon_{\mathrm{obs}}\equiv\hat s_{\mathrm{MMSE}}\). A5 / Lemma Z2 composition remain open research questions (A5 still a stub).

Update status only via experiment runners that call `record_result` and commit JSON artifacts under `docs/benchmark-results/`.
