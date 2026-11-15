# SageMaker g4dn — Phase-2 Final Dissertation affirmation cells

Branch: `cursor/phase2-dissertation-braket-fd3c`

Produces: telemetry.json, pasteback.json, 4× 3D histograms, Phase-2 counsel dissertation.

---

## Cell P0 — pull + install

```python
import getpass, subprocess, sys
from pathlib import Path

GITHUB_PAT = ""  # <<< PASTE TOKEN HERE

HOME = Path.home()
ROOT = HOME / "SageMaker" / "sri-architecture"
if not ROOT.parent.exists():
    ROOT = HOME / "sri-architecture"

BRANCH = "cursor/phase2-dissertation-braket-fd3c"
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

## Cell P1 — run Phase-2 campaign (telemetry + 3D histograms)

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

from experiments.braket_phase2_dissertation_campaign import run

OUT = Path("/tmp/phase2_dissertation")
summary = run(
    OUT,
    seeds=[108, 109, 110],
    n_samples=4096,
    vocab=32000,
    batches=40,
)
print(json.dumps(summary["verdict"], indent=2))
print("figures:", json.dumps(summary["figures"], indent=2))
print("telemetry_keys:", list(json.loads((OUT / "telemetry.json").read_text()).keys()))
print("--- PASTEBACK ---")
print((OUT / "pasteback.json").read_text())
assert summary["verdict"]["pass_phase2_complete"]
assert (OUT / "telemetry.json").is_file()
assert len(summary["figures"]) >= 4
```

## Cell P2 — show 3D histograms + Phase-2 dissertation

```python
from IPython.display import Image, display, Markdown
for name, path in summary["figures"].items():
    display(Markdown(f"### {name}"))
    display(Image(filename=path))
display(Markdown((OUT / "PHASE2_DISSERTATION_SRI_BRAKET_AFFIRMED.md").read_text()[:4500]))
print("--- telemetry.json (head) ---")
print((OUT / "telemetry.json").read_text()[:2500])
```

## Cell P3 — zip histograms + S3 upload

```python
import base64, zipfile, json, boto3
from datetime import datetime, timezone
from pathlib import Path
from IPython.display import HTML, display

# Browser download zip (no FileLink 404)
zip_path = Path.cwd() / "phase2-histograms.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in sorted((OUT / "figures").glob("*.png")):
        zf.write(p, arcname=p.name)
        print("zipped", p.name)
b64 = base64.b64encode(zip_path.read_bytes()).decode("ascii")
display(HTML(
    f'<a download="phase2-histograms.zip" href="data:application/zip;base64,{b64}" '
    f'style="font-size:16px;font-weight:600">⬇ Download phase2-histograms.zip</a>'
))

# S3 archival
BUCKET = "amazon-braket-sri-results-1"
REGION = "us-east-1"
PREFIX = "sri-braket/phase2-dissertation"
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
