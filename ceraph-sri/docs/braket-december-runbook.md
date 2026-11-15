# AWS Braket live runbook (S0–S6 campaign)

Hard spend envelope: **$400** ($200 trial credits + $200 pocket).  
Alarms: **$50** and **$200**. Prefer SV1 / Hybrid before any QPU.

Setup checklist: [`docs/aws-braket-setup.md`](aws-braket-setup.md).  
Campaign report: [`docs/braket-campaign-report.md`](braket-campaign-report.md).

## Budget table

| Bucket | Cap | Stop condition |
|---:|---|---|
| Colab A100 | 2–4 h (Pro units) | After E1–E4 lock criteria pass |
| Braket SV1/DM1 | **$40** | After `sv1_smoke.json` status COMPLETED |
| Hybrid Jobs (E1–E4 + suites) | **$120** | After hybrid vs A100/CPU comparison passes |
| Optional QPU toy | **$80** | Only if SV1 + Hybrid green; ≤1000 shots, ≤4 qubits |
| Buffer | **$160** | Retries / S3 |

Per-job caps: SV1/DM1 smoke `cost_cap_usd=5`; Hybrid `cost_cap_usd=25`.

## Pass / fail table

| Phase | What it proves | Pass criteria |
|---|---|---|
| **S0** Dry-run gates | Cost gates + plan schema | `dry_run_ok`, estimate ≤ cap |
| **S1** Accuracy lock | C1–C4 classical CV | C1–C4 confirmed; Ξ_SNR∈(0,1]; Branch W; entropy↓ |
| **S2** Advantage suite | Beats classical filters | MMSE < MA/median/single-mode; sq-ref≥50% vs coherent; report sq-signal control |
| **S3** Efficiency suite | Scaling + $/lock estimate | MSE within 5σ across N; CSV + Hybrid cost table |
| **S4** SV1/DM1 smoke | Job plumbing | Task COMPLETED (or emulate); labeled non-CV |
| **S5** Hybrid E1–E4 | Provenance on managed compute | `compare_locks` pass; optional campaign bundle |
| **S6** Optional QPU | Hardware contact only | ≤4 qubits, ≤1000 shots, ≤$80; not CV MMSE |

## S0 — Dry-run + cost gates (no AWS spend)

```bash
PYTHONPATH=. python experiments/braket_smoke_plan.py \
  --output-dir docs/benchmark-results/braket-smoke
pytest tests/test_braket_live_path.py -q
```

## S1 — Accuracy lock (Colab A100 when available)

```bash
PYTHONPATH=. python experiments/e1_colab_lock.py \
  --output-dir docs/benchmark-results
# on A100:
PYTHONPATH=. python experiments/e1_colab_lock.py \
  --output-dir docs/benchmark-results --require-gpu
```

Or use [`tests/SRI_Foundation_A100.ipynb`](../tests/SRI_Foundation_A100.ipynb).

Exit criteria: C1–C4 confirmed; Ξ_SNR in (0,1]; Branch **W** frozen as research default; entropy nonincreasing.

## S2 — Advantage vs classical tech

```bash
PYTHONPATH=. python experiments/braket_advantage_suite.py \
  --output-dir docs/benchmark-results/braket-campaign/advantage
```

## S3 — Efficiency / N-scaling

```bash
PYTHONPATH=. python experiments/braket_efficiency_suite.py \
  --output-dir docs/benchmark-results/braket-campaign/efficiency
```

## S4 — SV1 smoke (≤$10); optional DM1

```bash
# Local plumbing (no AWS spend):
PYTHONPATH=. python experiments/braket_live_smoke.py \
  --output-dir docs/benchmark-results/braket-live --emulate

# Optional DM1 emulate:
PYTHONPATH=. python experiments/braket_live_smoke.py \
  --output-dir docs/benchmark-results/braket-live --emulate --device dm1 --n-qubits 4

# Live (needs credentials + BRAKET_S3_BUCKET + budget alarms):
BRAKET_SUBMIT=1 BRAKET_S3_BUCKET=your-bucket \
  PYTHONPATH=. python experiments/braket_live_smoke.py \
  --output-dir docs/benchmark-results/braket-live --live --device sv1
```

## S5 — Hybrid E1–E4 + campaign bundle (≤$120)

```bash
PYTHONPATH=. python experiments/braket_hybrid_e1_e4.py \
  --output-dir docs/benchmark-results/braket-live \
  --reference-lock docs/benchmark-results/e1_colab_lock.json \
  --local-hybrid --include-campaign-bundle

# Live Hybrid Job create (gated):
BRAKET_SUBMIT=1 BRAKET_S3_BUCKET=your-bucket \
  PYTHONPATH=. python experiments/braket_hybrid_e1_e4.py \
  --output-dir docs/benchmark-results/braket-live \
  --submit-hybrid
```

## S5b — Sphota Yoga–Buddhi elaborate (Lemma Z1 + Colab oracle) (≤$25)

Preferred next Hybrid-shaped math gate after A0–A5 / Z1 lock. Full notebook cells:
[`docs/sphota-yoga-buddhi-braket-runbook.md`](sphota-yoga-buddhi-braket-runbook.md).

```bash
PYTHONPATH=. python experiments/braket_sphota_yoga_buddhi_elaborate.py \
  --output-dir docs/benchmark-results/braket-live/sphota-yoga-buddhi-elaborate \
  --reference-lock docs/benchmark-results/e1_colab_lock.json

pytest tests/test_braket_sphota_elaborate.py -q
```

Pass: `verdict.pass_all`, `c5_status=confirmed`, upload `pasteback.json` + S3 URIs before theory edits.

## S5c — Provisional evidence campaign R1–R7 (patent enablement)

After C5 GPU stamp: widen enablement + disclosure pack for provisional filing.
Plan: [`docs/provisional-patent-evidence-plan.md`](provisional-patent-evidence-plan.md).  
Cells: [`docs/sagemaker-provisional-evidence-cells.md`](sagemaker-provisional-evidence-cells.md).

```bash
PYTHONPATH=. python experiments/braket_provisional_evidence_campaign.py \
  --output-dir docs/benchmark-results/braket-live/provisional-evidence
pytest tests/test_provisional_evidence_campaign.py -q
```


## S6 — Optional QPU (≤$80)

Only after S4–S5 green. One 2–4 qubit toy, ≤1000 shots. Document that this is **not** CV MMSE on a QPU.

```bash
# Example live QPU plan still goes through the same gates with a QPU ARN and cost_cap_usd≤80.
# Prefer packaging via build_smoke_plan(device_arn=<qpu_arn>, task=qubit_toy_ising, shots≤1000).
```

## Campaign report

```bash
PYTHONPATH=. python experiments/braket_campaign_report.py
```

## Safety gates in code

Live submit requires **all** of: `BRAKET_SUBMIT=1`, `plan.submit=True`, estimated cost ≤ `cost_cap_usd`, and `BRAKET_S3_BUCKET`.

## Success metrics

- SV1 (or emulate) artifact with classical CV shadow attached
- Advantage suite documents equal-resource wins vs MA/median/single-mode
- Efficiency suite publishes N-scaling + cost-normalized Hybrid estimate (no O(1) claim)
- Hybrid lock matches reference lock within MSE/Ξ tolerances
- Spend log updated; no surprise QPU spend
- Report separates classical CV from qubit hardware contact
