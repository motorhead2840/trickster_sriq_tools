# Braket / SageMaker — Foundation Atlas cells (patent attorney pack)

Branch: `cursor/foundation-atlas-braket-fd3c`

Tests F01–F23 (math spine + softwall honesty) and writes legend-rich 3D histograms + attorney report.

---

## Cell F0 — pull branch

```python
import getpass, subprocess, sys
from pathlib import Path

GITHUB_PAT = ""  # <<< PASTE TOKEN HERE or leave blank to prompt

HOME = Path.home()
ROOT = HOME / "SageMaker" / "sri-architecture"
if not ROOT.parent.exists():
    ROOT = HOME / "sri-architecture"

BRANCH = "cursor/foundation-atlas-braket-fd3c"
REPO = "https://github.com/Ceraph-Corporation/sri-architecture.git"
if not GITHUB_PAT:
    GITHUB_PAT = getpass.getpass("GitHub PAT: ").strip()
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
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "matplotlib"])
print("ROOT", ROOT)
print("HEAD", subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip())
```

## Cell F1 — run foundation atlas

```python
import os, sys, json
from pathlib import Path

ROOT = Path("/home/ec2-user/SageMaker/sri-architecture")
if not ROOT.exists():
    ROOT = Path.home() / "SageMaker" / "sri-architecture"
if not ROOT.exists():
    ROOT = Path.home() / "sri-architecture"
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from experiments.braket_foundation_atlas import run

OUT = Path("/tmp/foundation_atlas")
summary = run(OUT, seeds=[108, 109, 110], n_samples=4096)
print(json.dumps(summary["verdict"], indent=2))
print("foundation_pass:", json.dumps({k: v.get("pass") for k, v in summary["foundations"].items()}, indent=2))
print("--- PASTEBACK ---")
print((OUT / "pasteback.json").read_text())
assert summary["verdict"]["pass_all"]
```

## Cell F2 — show 3D histograms + legends

```python
from IPython.display import Image, display, Markdown
for name, path in summary["figures"].items():
    display(Markdown(f"### {name}"))
    display(Image(filename=path))
display(Markdown((OUT / "PATENT_ATTORNEY_FOUNDATION_ATLAS_REPORT.md").read_text()[:3500]))
```

## Cell F3 — S3 upload

```python
from datetime import datetime, timezone
import boto3, json
from pathlib import Path

BUCKET = "amazon-braket-sri-results-1"
REGION = "us-east-1"
PREFIX = "sri-braket/foundation-atlas"
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
