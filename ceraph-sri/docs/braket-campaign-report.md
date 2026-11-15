# AWS Braket simulation campaign report

Generated: `2026-11-11T19:12:44.744702+00:00`  
Live AWS update: `2026-11-11T20:06:18Z`

## Honesty boundary

- **Classical CV** (E1–E4 / S1–S3 / Hybrid numerics): observable-only Gaussian MMSE,
  axiomatic Ξ_SNR, Circuit-B Branch W, CPTP entropy — proven on NumPy / Hybrid-shaped compute.
- **Qubit contact** (S4 SV1/DM1, optional S6 QPU): job plumbing / hardware contact only.
  Not CV MMSE on a QPU. Not yellow-paper O(1), Landauer, or photonic quantum advantage.

## Phase results

- **PASS** — S1 Accuracy lock (C1–C4)
- **PASS** — S2 Advantage vs classical filters (equal-resource)
- **PASS** — S3 Efficiency / N-scaling (Hybrid-shaped)
- **PASS** — S4 SV1 smoke (**live AWS**, status=COMPLETED)
- **PASS** — S5 Hybrid provenance (**live AWS Hybrid Job**, training SUCCESS)

## Accuracy headline numbers

- Mean MMSE sample MSE: **0.6137248633352207**
- Moving average MSE: 1.0881443514705091
- Median filter MSE: 1.2352300114315238
- Single-mode Wiener MSE: 0.6944325357306224
- Ξ_SNR range: [0.010266705723728333, 0.9313530236278]
- Entropy nonincreasing under CPTP: True

## Advantage (S2)

- Defensible win: **True**
- MMSE beats MA/median/single-mode: True/True/True
- Fraction squeezed-ref beats coherent: 0.8333333333333334
- Squeezed-signal stronger than squeezed-ref (honest control): True
- Equal-resource note: MMSE uses two optical modes (y_A, y_B). Moving-average/median baselines use y_A only. Prefer single_mode_mse for same-mode comparisons; two-mode MMSE vs coherent two-mode analytic is the physics-aligned contest.

## Efficiency (S3)

- Accuracy OK across N: **True**
- Estimated Hybrid instance cost for sweep: $1.4709394666624575e-07
- Scaling note: Wall time grows with Monte Carlo length and workspace size; this software path is O(N) in vector work — not a constant-time claim.

## Live AWS contact (user Braket notebook, us-east-1)

Artifact: [`docs/benchmark-results/braket-live/live_aws_run.json`](benchmark-results/braket-live/live_aws_run.json)

| Field | Value |
|---|---|
| Account | `443387201727` |
| Results bucket | `amazon-braket-sri-results-1` |
| SV1 task ARN | `arn:aws:braket:us-east-1:443387201727:quantum-task/8784b986-956a-451c-86ca-e4dd8502e5a3` |
| SV1 simulator cost | **$0.00375** (QPU $0) |
| Hybrid job ARN | `arn:aws:braket:us-east-1:443387201727:job/50cf12a3-82c0-446e-beb5-b49f30e5d131` |
| Hybrid instance | `ml.m5.large` |
| Hybrid lock | analytic MMSE **0.6127**, sample **0.6137**; beats MA/median/single |
| Weekly budget context | $100/week (user); SV1+Hybrid well under |

## Campaign verdict

- Campaign pass (S1–S5 with live AWS): **True**
- Live AWS tasks: **yes** (SV1 + Hybrid Job)

Spend discipline: see `docs/compute-unit-spend-log.md`.
