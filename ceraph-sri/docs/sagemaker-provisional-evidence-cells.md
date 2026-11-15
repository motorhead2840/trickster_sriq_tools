# SageMaker cells — provisional evidence campaign (R1–R7)

Assumes repo already cloned at `/home/ec2-user/SageMaker/sri-architecture` and Cell 1 (CUDA JAX) already succeeded.

Pull latest `main` first if needed:

```python
import subprocess
from pathlib import Path
ROOT = Path("/home/ec2-user/SageMaker/sri-architecture")
subprocess.check_call(["git", "fetch", "origin"], cwd=ROOT)
subprocess.check_call(["git", "checkout", "main"], cwd=ROOT)
subprocess.check_call(["git", "pull", "--ff-only", "origin", "main"], cwd=ROOT)
```

---

## Cell P1 — run R1–R7 campaign

```python
import os, sys, json
from pathlib import Path

ROOT = Path("/home/ec2-user/SageMaker/sri-architecture")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from experiments.braket_provisional_evidence_campaign import run

OUT = Path("/tmp/provisional_evidence")
summary = run(
    OUT,
    seeds=[108, 109, 110],
    n_samples=4096,
    timing_n=1_000_000,
)
print(json.dumps(summary["verdict"], indent=2))
print("--- PASTEBACK ---")
print((OUT / "pasteback.json").read_text())
print("--- DISCLOSURE (titles) ---")
pack = json.loads((OUT / "disclosure_pack.json").read_text())
print([c["title"] for c in pack["supported_claim_sketches"]])
print("do_not_claim_yet:", pack["do_not_claim_yet"])
assert summary["verdict"]["pass_all"]
```

## Cell P2 — upload to S3

```python
from datetime import datetime, timezone
import boto3

BUCKET = "amazon-braket-sri-results-1"
REGION = "us-east-1"
PREFIX = "sri-braket/provisional-evidence"
OUT = Path("/tmp/provisional_evidence")

ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
client = boto3.client("s3", region_name=REGION)
uris = {}
for fname in (
    "provisional_evidence_summary.json",
    "pasteback.json",
    "disclosure_pack.json",
):
    key = f"{PREFIX}/results/{ts}_{fname}"
    client.put_object(Bucket=BUCKET, Key=key, Body=(OUT / fname).read_bytes())
    uris[fname] = f"s3://{BUCKET}/{key}"
print(json.dumps(uris, indent=2))
```

Paste **PASTEBACK** + S3 URIs back into chat for counsel packaging.
