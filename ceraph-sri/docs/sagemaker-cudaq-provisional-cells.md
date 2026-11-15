# Braket CUDA-Q notebook — Provisional dissertation breakthrough cells

Use the **CUDA-Q and Braket** Jupyter kernel on a GPU notebook (`ml.g4dn.xlarge` or better).
These cells prove dissertation Eqs 1–18 algebraically, Lemma Z1, Witness/V3, LWE-toy Claim 5,
IEEE-754 XOR+FMA proxies, and CUDA-Q fabric / conjugate-shadow kernels.

**Softwall honesty:** Landauer physical circumvention and TMEM O(0) stay `NOT_PROVEN` /
`SOFTWARE_PROXY_NOT_TMEM` until silicon embodiment — do not market those as proven from this run.

Branch: `cursor/cudaq-provisional-dissertation-fd3c`

---

## Cell C0 — clone / pull repo (classic SageMaker or Braket NBI)

```python
import os, getpass, subprocess, sys
from pathlib import Path

# === PASTE YOUR GITHUB PAT BETWEEN THE QUOTES (or leave "" to be prompted) ===
GITHUB_PAT = ""
# ============================================================================

HOME = Path.home()
ROOT = HOME / "SageMaker" / "sri-architecture"
if not ROOT.parent.exists():
    ROOT = HOME / "sri-architecture"

BRANCH = "cursor/cudaq-provisional-dissertation-fd3c"
REPO = "https://github.com/Ceraph-Corporation/sri-architecture.git"

if not GITHUB_PAT:
    GITHUB_PAT = getpass.getpass("GitHub PAT (repo read): ").strip()

auth = REPO.replace("https://", f"https://x-access-token:{GITHUB_PAT}@")
if ROOT.exists():
    subprocess.check_call(["git", "remote", "set-url", "origin", auth], cwd=ROOT)
    subprocess.check_call(["git", "fetch", "origin"], cwd=ROOT)
    subprocess.check_call(["git", "checkout", BRANCH], cwd=ROOT)
    subprocess.check_call(["git", "pull", "--ff-only", "origin", BRANCH], cwd=ROOT)
    subprocess.check_call(["git", "remote", "set-url", "origin", REPO], cwd=ROOT)
else:
    ROOT.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["git", "clone", "--branch", BRANCH, auth, str(ROOT)])
    subprocess.check_call(["git", "remote", "set-url", "origin", REPO], cwd=ROOT)

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", str(ROOT)])
print("ROOT", ROOT)
print("HEAD", subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip())
```

## Cell C1 — CUDA-Q + GPU sanity

```python
import cudaq, subprocess, shutil
print("cudaq", getattr(cudaq, "__version__", "?"))
print("gpus", cudaq.num_available_gpus())
if shutil.which("nvidia-smi"):
    print(subprocess.check_output(["nvidia-smi"], text=True))

# Prefer GPU simulator on g4dn / p-class
if cudaq.num_available_gpus() > 0:
    cudaq.set_target("nvidia")
    print("target → nvidia")
else:
    cudaq.set_target("qpp-cpu")
    print("target → qpp-cpu (no GPU visible)")
```

## Cell C2 — run full dissertation campaign

```python
import os, sys, json
from pathlib import Path

ROOT = Path.home() / "SageMaker" / "sri-architecture"
if not ROOT.exists():
    ROOT = Path.home() / "sri-architecture"
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from experiments.braket_cudaq_provisional_campaign import run

OUT = Path("/tmp/cudaq_provisional")
summary = run(
    OUT,
    seeds=[108, 109, 110],
    n_samples=4096,
    cudaq_qubits=6,
)
print(json.dumps(summary["verdict"], indent=2))
print("round_pass:", json.dumps({k: v.get("pass") for k, v in summary["rounds"].items()}, indent=2))
print("round_track:", json.dumps({k: v.get("track") for k, v in summary["rounds"].items()}, indent=2))
print("--- PASTEBACK ---")
print((OUT / "pasteback.json").read_text())
assert summary["verdict"]["pass_math"], "math enablement failed"
assert summary["verdict"]["pass_cudaq"], "CUDA-Q rounds failed — check Cell C1 target"
```

## Cell C3 — show reports

```python
from IPython.display import Markdown, display
display(Markdown((OUT / "PROJECT_REPORT_cudaq_provisional.md").read_text()))
display(Markdown((OUT / "PATENT_ANNEX_cudaq_dissertation_enablement.md").read_text()[:4000]))
```

## Cell C4 — S3 upload

```python
from datetime import datetime, timezone
import boto3, json
from pathlib import Path

BUCKET = "amazon-braket-sri-results-1"
REGION = "us-east-1"
PREFIX = "sri-braket/cudaq-provisional-dissertation"
ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
client = boto3.client("s3", region_name=REGION)
uris = {}
for path in list(OUT.glob("*.json")) + list(OUT.glob("*.md")):
    key = f"{PREFIX}/results/{ts}/{path.name}"
    client.put_object(Bucket=BUCKET, Key=key, Body=path.read_bytes())
    uris[path.name] = f"s3://{BUCKET}/{key}"
print(json.dumps(uris, indent=2))
```
