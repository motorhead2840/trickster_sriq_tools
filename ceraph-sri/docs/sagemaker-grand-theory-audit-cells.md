# SageMaker g4dn — Grand Theory Audit cells

Full campaign: provisional R1–R7 + Ω/Landauer scaling + softwall claim board + **3D histograms** + project report / patent annex.

**Honesty:** Landauer circumvention, TMEM O(0), and DeepSeek TFLOPS are *instrumented* and will show **NOT_PROVEN / REFUTED / REFERENCE_ONLY** unless future hardware embodiment changes that. Foundation claims can still `pass_all`.

---

## Cell G0 — pull branch / main

```python
import subprocess
from pathlib import Path
ROOT = Path("/home/ec2-user/SageMaker/sri-architecture")
subprocess.check_call(["git", "fetch", "origin"], cwd=ROOT)
subprocess.check_call(["git", "checkout", "cursor/grand-theory-audit-g4dn-fd3c"], cwd=ROOT)
subprocess.check_call(["git", "pull", "--ff-only", "origin", "cursor/grand-theory-audit-g4dn-fd3c"], cwd=ROOT)
```

## Cell G1 — deps (keep CUDA JAX from prior Cell 1)

```python
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "matplotlib"])
```

## Cell G2 — run grand audit

```python
import os, sys, json
from pathlib import Path

ROOT = Path("/home/ec2-user/SageMaker/sri-architecture")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from experiments.braket_grand_theory_audit import run

OUT = Path("/tmp/grand_theory_audit")
summary = run(
    OUT,
    seeds=[108, 109, 110],
    n_samples=4096,
    timing_n=500_000,
    scaling_dimensions=[256, 1024, 4096, 16384],
)
print(json.dumps(summary["verdict"], indent=2))
print("figures:", json.dumps(summary["figures"], indent=2))
print("--- PASTEBACK ---")
print((OUT / "pasteback.json").read_text())
assert summary["verdict"]["pass_all"]
```

## Cell G3 — show 3D histograms in notebook

```python
from IPython.display import Image, display, Markdown
for name, path in summary["figures"].items():
    display(Markdown(f"### {name}"))
    display(Image(filename=path))
display(Markdown((OUT / "PROJECT_REPORT_grand_theory_audit.md").read_text()[:2500]))
```

## Cell G4 — S3 upload

```python
from datetime import datetime, timezone
import boto3

BUCKET = "amazon-braket-sri-results-1"
REGION = "us-east-1"
PREFIX = "sri-braket/grand-theory-audit"
ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
client = boto3.client("s3", region_name=REGION)
uris = {}
files = list(OUT.glob("*.json")) + list(OUT.glob("*.md")) + list((OUT / "figures").glob("*.png"))
for path in files:
    key = f"{PREFIX}/results/{ts}/{path.name}"
    client.put_object(Bucket=BUCKET, Key=key, Body=path.read_bytes())
    uris[path.name] = f"s3://{BUCKET}/{key}"
print(json.dumps(uris, indent=2))
```
