# Colab + AWS spend log (E1 → Braket campaign S0–S6)

Total hard envelope: **$400** AWS ($200 trial + $200 pocket) + existing Colab Pro units.  
User live weekly budget context: **$100/week** (2026-11-11).

## Colab A100

| Week / phase | Planned A100 hours | Actual | Notes |
|---|---:|---:|---|
| Foundation month (prior) | ≤70 | CPU preflight in CI | Prior sprint |
| **E1 lock / S1** | **≤3** | TBD on user Colab | `e1_colab_lock.py --require-gpu` |
| Agent VM re-lock | 0 | **CPU lock OK** | C1–C4 confirmed 2026-11-11 |

## AWS Braket

| Phase | Cap | Actual (agent VM) | Actual (user live) |
|---|---:|---:|---:|
| S0 dry-run / smoke plan | $0 | **$0** | — |
| S2 advantage suite | $0 (classical) | **$0** | — |
| S3 efficiency suite | $0 (classical; Hybrid $ estimate only) | **$0** | — |
| S4 SV1 smoke | $40 | **$0** (emulate) | **$0.00375** simulator; QPU $0 |
| S5 Hybrid E1–E4 | $120 | **$0** (local hybrid) | **COMPLETED** on `ml.m5.large` (instance $ TBD from Billing; short job) |
| S6 Optional QPU | $80 | **$0** | not run |
| Buffer | $160 | $0 | — |

### Live ARNs (us-east-1, account `443387201727`)

| Kind | ARN |
|---|---|
| SV1 task | `arn:aws:braket:us-east-1:443387201727:quantum-task/8784b986-956a-451c-86ca-e4dd8502e5a3` |
| Hybrid job | `arn:aws:braket:us-east-1:443387201727:job/50cf12a3-82c0-446e-beb5-b49f30e5d131` |
| Results bucket | `amazon-braket-sri-results-1` |

Full artifact: [`docs/benchmark-results/braket-live/live_aws_run.json`](benchmark-results/braket-live/live_aws_run.json).

Alarms: prefer **$50** / **$200**; user also tracking **$100/week**. Stop if either fires before review.

## Runtime discipline

- Develop/emulate without `BRAKET_SUBMIT`.
- Live submit only with `BRAKET_SUBMIT=1`, `plan.submit=True`, cost ≤ `cost_cap_usd`, and a Braket-compliant bucket (`amazon-braket-*`).
- Never commit AWS keys.
- Campaign report must label SV1/DM1/QPU as non-CV plumbing.
