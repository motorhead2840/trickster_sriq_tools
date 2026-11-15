# Sphota-hybrid Yoga–Buddhi — Braket validation runbook

Validates that dissertation \(\hat{\Upsilon}_{YB}\) + Lemma Z1 translate to executable
code across **Python / optional JAX / optional Julia**, packaged as a Braket
Hybrid–shaped job — including an **elaborate** suite that regresses Colab oracle
fixtures and emits theory-hooks for the next framework update.

Related: [yoga-buddhi-projector.md](yoga-buddhi-projector.md), [architecture-axioms.md](architecture-axioms.md), [cuda-tile-rust-embodiment.md](cuda-tile-rust-embodiment.md), [sree-mentor-jax-te-moe.md](sree-mentor-jax-te-moe.md), [braket-december-runbook.md](braket-december-runbook.md).

## Honesty boundary

- Numerics are **classical** (Gaussian dual-mode + latent oracle identity).
- Hybrid Job = managed compute provenance, **not** photonic CV on a QPU.
- SV1/DM1 optional plumbing only; do not read qubit toy results as Yoga–Buddhi.
- `tile_ref` / `cutile_rs_stub` prove algebra structure for a future cuTile Rust port — not Blackwell CUPTI geomean.
- Colab Untitled10 success = **oracle Samya** regression, not blind antithesis.

## What the elaborate suite checks

| Check | Proves | Fail means |
|---|---|---|
| `colab_oracle_samya_fixture` | Known-baseline cancellation (Untitled10) + latent involution across N | Oracle chart broken in package |
| `lemma_z1_config_seed_grid` | \(\hat\Upsilon_{\mathrm{obs}}\equiv\hat s\) on 6 configs × seeds; MSE≤5σ; raw-Y **not** involution | C5 / Lemma Z1 not deployable |
| `mirror_and_softwalls` | Mirror \(\ge0\); softwalls stay false | A5 channel or honesty flags wrong |
| `sphota_crystallize` | NumPy + tile_ref (+ JAX/Julia if present) | Sphota hybrid regression |
| `sree_te_moe_plan` | Mentor TE dropless MoE plan `CONFIG_OK` | SREE training scaffold broken |
| `cutile_rs_stub` | Rust `KERNEL_OK` or honest `PERF_SKIP` | Functional tile mirror broken |
| `compare_e1_colab_lock` | Loose continuity vs prior E1 MSE lock | Provenance drift (investigate) |

Artifacts: `elaborate_summary.json`, `pasteback.json`, `theory_hooks.json`, `hybrid_job_spec.json`.

---

## Local / CI gate (no AWS)

```bash
# Fast Sphota smoke
PYTHONPATH=. python experiments/sphota_yoga_buddhi_suite.py \
  --output-dir docs/benchmark-results/sphota-yoga-buddhi
pytest tests/test_sphota_yoga_buddhi.py tests/test_braket_sphota_elaborate.py -q

# Elaborate (recommended before Braket paste)
PYTHONPATH=. python experiments/braket_sphota_yoga_buddhi_elaborate.py \
  --output-dir docs/benchmark-results/braket-live/sphota-yoga-buddhi-elaborate \
  --reference-lock docs/benchmark-results/e1_colab_lock.json
```

Pass bar: `verdict.pass_all=true`, `c5_status=confirmed`, pasteback theory_hooks emitted.

**GPU stamp (2026-11-12):** SageMaker `ml.g4dn.xlarge` / Tesla T4 / JAX `cuda:0` pasteback archived at
`docs/benchmark-results/braket-live/sphota-yoga-buddhi-elaborate/sagemaker-g4dn/pasteback.json`.
C5 confirmed with that provenance; see [architecture-axioms.md](architecture-axioms.md).

---

## Braket notebook cells (conda_braket) — elaborate

Sync / clone this branch onto the instance first (`cursor/architecture-axioms-theorem-zero-fd3c` or main once merged).

### Cell E0 — env

```python
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path

# Adjust if the repo lives elsewhere on the instance:
REPO = Path(os.environ.get("SRI_REPO", ".")).resolve()
sys.path.insert(0, str(REPO))
os.chdir(REPO)

BUCKET = os.environ.get("BRAKET_S3_BUCKET", "amazon-braket-sri-results-1")
PREFIX = "sri-braket/sphota-yoga-buddhi-elaborate"
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
OUT = Path("/tmp/sphota_yb_elaborate")
print("repo", REPO)
print("bucket", BUCKET, "region", REGION)
print("python", sys.version)
```

### Cell E1 — elaborate run (full detail)

```python
from experiments.braket_sphota_yoga_buddhi_elaborate import run

summary = run(
    OUT,
    seeds=[108, 109, 110, 111],
    n_samples=8192,
    oracle_dimensions=[256, 1024, 4096, 16384],
    prefer_jax=True,
    prefer_julia=True,   # set False if julia missing on the image
    promote_c5=True,
    reference_lock=REPO / "docs/benchmark-results/e1_colab_lock.json",
    run_cutile_stub=True,  # set False if no rustc/cargo
    submit_hybrid=False,
    cost_cap_usd=25.0,
)

print(json.dumps(summary["verdict"], indent=2))
print(json.dumps(summary["theory_hooks"], indent=2))
print("backends", json.dumps(summary["checks"]["sphota_crystallize"].get("backends", {}), indent=2))
assert summary["verdict"]["pass_all"], summary["verdict"]["failed_checks"]
assert summary["verdict"]["c5_status"] == "confirmed"
```

### Cell E2 — per-check dump (do not skip on failure investigation)

```python
for name, payload in summary["checks"].items():
    print("====", name, "pass=", payload.get("pass"), "====")
    # Compact: drop huge row arrays when printing
    slim = {k: v for k, v in payload.items() if k != "rows"}
    if "rows" in payload:
        slim["n_rows"] = len(payload["rows"])
        slim["n_fail_rows"] = sum(1 for r in payload["rows"] if not r.get("pass", True))
    print(json.dumps(slim, indent=2)[:4000])
```

### Cell E3 — upload + paste-back pointers

```python
import boto3

ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
client = boto3.client("s3", region_name=REGION)
uris = {}
for fname in (
    "elaborate_summary.json",
    "pasteback.json",
    "theory_hooks.json",
    "hybrid_job_spec.json",
):
    key = f"{PREFIX}/results/{ts}_{fname}"
    body = (OUT / fname).read_bytes()
    client.put_object(Bucket=BUCKET, Key=key, Body=body)
    uris[fname] = f"s3://{BUCKET}/{key}"
print(json.dumps(uris, indent=2))
print("--- PASTE BACK ---")
print((OUT / "pasteback.json").read_text())
```

Paste back into chat: **`pasteback.json`** contents + the four S3 URIs.

### Cell E4 — optional Hybrid Job packaging only

```python
# Does not spend until BRAKET_SUBMIT=1 and Hybrid entrypoint is wired.
os.environ["BRAKET_S3_BUCKET"] = BUCKET
summary2 = run(
    OUT / "hybrid_pack",
    seeds=[108, 109, 110, 111],
    n_samples=8192,
    prefer_julia=False,
    run_cutile_stub=False,
    submit_hybrid=True,
    cost_cap_usd=25.0,
    reference_lock=REPO / "docs/benchmark-results/e1_colab_lock.json",
)
print(json.dumps(summary2["hybrid"]["dry_run"], indent=2))
print("estimate_usd", summary2["hybrid"]["estimate_usd"])
```

---

## Quick Sphota-only cells (lighter)

### Cell S0 — config

```python
import json, os
from datetime import datetime, timezone
from pathlib import Path

BUCKET = os.environ.get("BRAKET_S3_BUCKET", "amazon-braket-sri-results-1")
PREFIX = "sri-braket/sphota-yoga-buddhi"
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
print("bucket", BUCKET, "region", REGION)
```

### Cell S1 — crystallize

```python
from experiments.sphota_yoga_buddhi_suite import run

summary = run(
    Path("/tmp/sphota_yoga_buddhi"),
    seed=108,
    n_samples=4096,
    prefer_jax=True,
    prefer_julia=True,
    promote_c5=True,
)
print(json.dumps(summary["verdict"], indent=2))
assert summary["verdict"]["pass_lemma_z1"]
```

---

## After paste-back — theory update gate

Only when `pass_all` and S3 URIs are in hand, update:

1. `docs/architecture-axioms.md` — Z1 confirmed-on-Hybrid provenance; draft Z2 options with measured non-involution witness.
2. `docs/yoga-buddhi-projector.md` — bind empirical `mean_sq_obs_anti_plus_ya`.
3. `docs/conjecture-registry.md` / JSON — C5 confirmed with elaborate artifact paths.
4. `docs/theoretical-framework.md` — short Hybrid-classical subsection (still not photonic).

Do **not** rewrite softwalls from this run alone.
