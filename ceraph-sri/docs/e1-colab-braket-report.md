# Experiment Set 1 — Colab sprint → Braket campaign (S0–S6)

Canonical campaign report: [`docs/braket-campaign-report.md`](braket-campaign-report.md).  
Runbook: [`docs/braket-december-runbook.md`](braket-december-runbook.md).  
Live AWS artifact: [`docs/benchmark-results/braket-live/live_aws_run.json`](benchmark-results/braket-live/live_aws_run.json).

## What ran

| Phase | Artifact | Mode |
|---|---|---|
| S0 | [`docs/aws-braket-setup.md`](aws-braket-setup.md), live gates in `sri.backends.braket_interface` | Code + checklist |
| S1 E1–E4 | [`docs/benchmark-results/e1_colab_lock.json`](benchmark-results/e1_colab_lock.json) | **CPU/NumPy lock** (A100 `--require-gpu` still optional) |
| S2 Advantage | [`docs/benchmark-results/braket-campaign/advantage/`](benchmark-results/braket-campaign/advantage/) | Classical equal-resource comparisons |
| S3 Efficiency | [`docs/benchmark-results/braket-campaign/efficiency/`](benchmark-results/braket-campaign/efficiency/) | N-scaling + Hybrid instance $ estimate |
| S4 SV1 | [`live_aws_run.json`](benchmark-results/braket-live/live_aws_run.json) | **Live AWS** SV1 COMPLETED |
| S5 Hybrid | [`live_aws_run.json`](benchmark-results/braket-live/live_aws_run.json) | **Live AWS Hybrid Job** on `ml.m5.large` SUCCESS |

## Live AWS ARNs (us-east-1)

| Kind | ARN |
|---|---|
| SV1 task | `arn:aws:braket:us-east-1:443387201727:quantum-task/8784b986-956a-451c-86ca-e4dd8502e5a3` |
| Hybrid job | `arn:aws:braket:us-east-1:443387201727:job/50cf12a3-82c0-446e-beb5-b49f30e5d131` |
| Bucket | `amazon-braket-sri-results-1` |
| SV1 cost | **$0.00375** simulator; QPU $0 |

## Colab exit criteria

All true on the CPU lock (A100 confirmation still recommended once):

- C1–C4 **confirmed**
- `Ξ_SNR` ∈ ≈[0.010, 0.931]
- Circuit-B research default frozen to **Branch W**
- Entropy nonincreasing under CPTP attenuator

## Classical CV vs qubit contact

- **E1–E4 numerics** (observable-only MMSE, axiomatic Ξ, Circuit B, entropy) are **classical Gaussian CV**.
- **SV1 / DM1 / QPU Ising smoke** is qubit hardware/simulator plumbing only — not CV MMSE on a QPU.
- **Hybrid Job** re-ran the classical CV lock on Braket-managed compute; sample MMSE **0.6137** matched the repo lock.

## Advantage / efficiency headlines

- MMSE sample MSE ≈ **0.614** vs MA ≈ **1.088**, median ≈ **1.235**, single-mode ≈ **0.694**
- Squeezed-signal analytic MSE remains the stronger matched quantum baseline vs squeezed-ref
- Efficiency suite: MSE within 5σ across N; cost table is Hybrid instance estimate only (not O(1))

## Hybrid vs reference

Live Hybrid Job: `pass=true` (analytic 0.6127, sample 0.6137; beats MA/median/single).
Local hybrid vs reference earlier: `comparison_to_reference.pass = true` (MSE Δ = 0, Ξ Δ = 0).

## Spend log

| Item | Spend |
|---|---:|
| Live SV1 | **$0.00375** |
| Live Hybrid Job | short `ml.m5.large` SUCCESS (instance $ from Billing) |
| Optional QPU | **not run** |
