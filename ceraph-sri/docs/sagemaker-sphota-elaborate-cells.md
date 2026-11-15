# SageMaker / Jupyter cells — Sphota Yoga–Buddhi elaborate (CUDA-aware)

Paste the fenced Python blocks below as successive notebook cells.

Prefer a **Python 3 / PyTorch / JAX** kernel — not `conda_braket`.

**Instances:**
- `ml.t3.large` — no GPU; cells fall back to CPU JAX.
- `ml.g4dn.xlarge` (T4) / `ml.g5.*` — Cell 1 asserts `jax.default_backend() == "gpu"` before the suite runs.

Related: [sphota-yoga-buddhi-braket-runbook.md](sphota-yoga-buddhi-braket-runbook.md).

---

## Cell 0 — clone / sync repo

```python
import os, sys, subprocess
from pathlib import Path

REPO_URL = os.environ.get(
    "SRI_REPO_URL",
    "https://github.com/Ceraph-Corporation/sri-architecture.git",
)
# Optional for private clone:
# os.environ["GITHUB_TOKEN"] = "ghp_..."
TOKEN = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
if TOKEN and "://" in REPO_URL:
    REPO_URL = REPO_URL.replace("https://", f"https://x-access-token:{TOKEN}@")

# Classic Notebook: /home/ec2-user/SageMaker/...
# Studio:          /home/sagemaker-user/...
# Override with SRI_ROOT if needed.
_home = Path.home()
_default = (
    _home / "SageMaker" / "sri-architecture"
    if (_home / "SageMaker").is_dir()
    else _home / "sri-architecture"
)
ROOT = Path(os.environ.get("SRI_ROOT", str(_default))).resolve()
BRANCH = os.environ.get("SRI_BRANCH", "main")

ROOT.parent.mkdir(parents=True, exist_ok=True)
if not (ROOT / ".git").exists():
    subprocess.check_call(["git", "clone", "--branch", BRANCH, REPO_URL, str(ROOT)])
else:
    subprocess.check_call(["git", "fetch", "origin"], cwd=ROOT)
    subprocess.check_call(["git", "checkout", BRANCH], cwd=ROOT)
    subprocess.check_call(["git", "pull", "--ff-only", "origin", BRANCH], cwd=ROOT)

os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
print("home", _home)
print("repo", ROOT)
print(subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip())
```

## Cell 1 — deps (CUDA-aware JAX, no conda_braket)

```python
import importlib, subprocess, sys

def pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

pip("numpy", "scipy", "networkx")

# CUDA present? (nvidia-smi). t3.large → CPU JAX.
has_cuda = False
try:
    out = subprocess.check_output(["nvidia-smi", "-L"], text=True, stderr=subprocess.STDOUT)
    print(out)
    has_cuda = "GPU" in out or "NVIDIA" in out
except Exception as e:
    print("no nvidia-smi / no GPU:", e)

if has_cuda:
    # g4dn T4 with driver CUDA 13.x → prefer cuda13 wheels (2026 JAX default GPU path).
    pip("--upgrade", "jax[cuda13]")
else:
    pip("jax", "jaxlib")

# Optional S3 pasteback
pip("boto3")

# Editable install of this repo (setuptools)
pip("-e", ".")

import numpy as np
print("numpy", np.__version__)
import jax
print("jax", jax.__version__, "backend", jax.default_backend(), "devices", jax.devices())
```

## Cell 2 — elaborate Sphota / Lemma Z1 suite

```python
import json
from pathlib import Path
from experiments.braket_sphota_yoga_buddhi_elaborate import run

OUT = Path("/tmp/sphota_yb_elaborate")
REF = Path("docs/benchmark-results/e1_colab_lock.json")

summary = run(
    OUT,
    seeds=[108, 109, 110, 111],
    n_samples=8192,                 # drop to 4096 if memory tight on t3.large
    oracle_dimensions=[256, 1024, 4096, 16384],
    prefer_jax=True,
    prefer_julia=False,             # Julia usually absent on SageMaker
    promote_c5=True,
    reference_lock=REF if REF.is_file() else None,
    run_cutile_stub=False,          # needs rustc; skip on SageMaker
    submit_hybrid=False,            # math gate only — not Braket Hybrid spend
    cost_cap_usd=25.0,
)

print(json.dumps(summary["verdict"], indent=2))
print(json.dumps(summary["theory_hooks"], indent=2))
print("jax backends block:")
print(json.dumps(summary["checks"]["sphota_crystallize"].get("backends", {}), indent=2))
assert summary["verdict"]["pass_all"], summary["verdict"]["failed_checks"]
print("PASS — pasteback at", OUT / "pasteback.json")
print((OUT / "pasteback.json").read_text())
```

## Cell 3 — per-check dump (if anything fails)

```python
for name, payload in summary["checks"].items():
    slim = {k: v for k, v in payload.items() if k not in {"rows", "plan"}}
    if "rows" in payload:
        slim["n_rows"] = len(payload["rows"])
        slim["n_fail_rows"] = sum(1 for r in payload["rows"] if not r.get("pass", True))
    print("====", name, "pass=", payload.get("pass"), "====")
    print(json.dumps(slim, indent=2)[:3000])
```

## Cell 4 — optional S3 upload (paste URIs back in chat)

```python
import os
from datetime import datetime, timezone
import boto3

BUCKET = os.environ.get("BRAKET_S3_BUCKET") or os.environ.get("SRI_S3_BUCKET")
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
PREFIX = "sri-braket/sphota-yoga-buddhi-elaborate"

if not BUCKET:
    print("Set BRAKET_S3_BUCKET or SRI_S3_BUCKET to upload; pasteback.json is still local:")
    print(OUT / "pasteback.json")
else:
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
        client.put_object(Bucket=BUCKET, Key=key, Body=(OUT / fname).read_bytes())
        uris[fname] = f"s3://{BUCKET}/{key}"
    print(json.dumps(uris, indent=2))
```

## Paste back

Send **`pasteback.json`** contents (Cell 2 printout) and any S3 URIs from Cell 4. That stamps Hybrid/SageMaker provenance before the next theory/framework edit.
