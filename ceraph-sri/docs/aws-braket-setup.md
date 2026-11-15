# Phase 0 — AWS Braket unlock checklist ($400 envelope)

Complete this **before** the first live submit. Never commit access keys.

## 1. Enable Braket

1. Sign in to the AWS console.
2. Open **Amazon Braket** in `us-east-1` (or your chosen Braket region).
3. Accept any service terms / create the default results bucket if prompted.

## 2. Results bucket + IAM

Create (or reuse) an S3 bucket, e.g. `sri-braket-results-<accountid>`.

Minimal IAM actions for the experiment principal:

- `braket:CreateQuantumTask`
- `braket:GetQuantumTask`
- `braket:CancelQuantumTask`
- `braket:GetDevice`
- `s3:PutObject`, `s3:GetObject`, `s3:ListBucket` on that bucket
- For Hybrid Jobs: `braket:CreateJob`, `braket:GetJob`, plus the managed policy AWS documents for Hybrid Jobs

Export locally / in Colab secrets (names only — values stay in the secret store):

```bash
export AWS_DEFAULT_REGION=us-east-1
export BRAKET_S3_BUCKET=sri-braket-results-<accountid>
# AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY via env or instance role
```

## 3. Budgets and stop conditions

Create AWS Budgets (Billing console):

| Alarm | Action |
|---:|---|
| **$50** | Email alert — pause new QPU experiments |
| **$200** | Email alert — freeze all Braket until review |
| Hard envelope | **$400** total ($200 trial + $200 pocket) |

Per-job `cost_cap_usd` defaults: **$5** (SV1 smoke), **$25** (Hybrid Job).

## 4. Live submit gates in this repo

Live submission requires **all** of:

1. `BRAKET_SUBMIT=1`
2. `plan.submit=True`
3. `estimate_cost_usd(plan) <= plan.cost_cap_usd`
4. `BRAKET_S3_BUCKET` set

Otherwise the adapter dry-runs or emulates locally.

## 5. Verify

```bash
python experiments/braket_smoke_plan.py --output-dir docs/benchmark-results/braket-smoke
python experiments/braket_live_smoke.py --output-dir docs/benchmark-results/braket-live --emulate
# when credentials + budget alarms are ready:
BRAKET_SUBMIT=1 BRAKET_S3_BUCKET=... python experiments/braket_live_smoke.py \
  --output-dir docs/benchmark-results/braket-live --live
```
