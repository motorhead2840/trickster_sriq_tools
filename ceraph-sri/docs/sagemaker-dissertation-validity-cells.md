# SageMaker g4dn — Final Dissertation validity cells (per chapter)

Branch: `cursor/dissertation-braket-validity-fd3c`

One **unique cell per Final Dissertation chapter (1–6)**. Run **V0** once, then **C1 → C6**, then optional **CALL** (full suite) and **S3**.

| Cell | Dissertation chapter | What it checks |
|---|---|---|
| **C1** | Ch. 1 Crisis / safety tax | §1.1 pathology context + §1.2 in-process vs 100–500 ms model |
| **C2** | Ch. 2 Ontology | CV-dit + bipartite signal/noise space |
| **C3** | Ch. 3 Quasicrystal / projector | Oracle cancellation, Fibonacci smoke, Lemma Z1 |
| **C4** | Ch. 4 Witness / Vyapti engine | V3 damping, mirror mark, user-query path |
| **C5** | Ch. 5 Free energy / hardware | Landauer+Helmholtz math + softwall board |
| **C6** | Ch. 6 Optical horizons | PPLN / optical future — must stay NOT_PROVEN |

Integration map: `docs/patent-provisional/06-final-dissertation-integration.md`

---

## Cell V0 — install from a local zip (NO GitHub PAT)

**Why:** Ceraph blocks classic PATs with lifetime > 7 days. Token `6224527288` is that blocked token — git **and** API zipball both 403 with it.

**Do this in your browser (logged into GitHub):**
1. Open https://github.com/Ceraph-Corporation/sri-architecture/releases/tag/dissertation-validity-bootstrap  
2. Download `sri-architecture-dissertation-validity.zip`  
3. In JupyterLab: upload that zip into `~/SageMaker/` (or `~/`)  
4. Run the cell below (no PAT).

```python
import shutil, subprocess, sys, zipfile
from pathlib import Path

HOME = Path.home()
CANDIDATES = [
    HOME / "SageMaker" / "sri-architecture-dissertation-validity.zip",
    HOME / "sri-architecture-dissertation-validity.zip",
    HOME / "SageMaker" / "sri-dissertation-validity.zip",
    HOME / "Downloads" / "sri-architecture-dissertation-validity.zip",
]
ZIP = next((p for p in CANDIDATES if p.is_file()), None)
if ZIP is None:
    # Also accept any uploaded zip matching the name pattern under SageMaker/
    sm = HOME / "SageMaker"
    matches = sorted(sm.glob("*dissertation*validity*.zip")) if sm.exists() else []
    matches += sorted(HOME.glob("*dissertation*validity*.zip"))
    ZIP = matches[0] if matches else None
assert ZIP is not None, (
    "Upload sri-architecture-dissertation-validity.zip to ~/SageMaker/ first.\n"
    "Get it from: https://github.com/Ceraph-Corporation/sri-architecture/releases/tag/dissertation-validity-bootstrap"
)

ROOT = HOME / "SageMaker" / "sri-architecture"
if not ROOT.parent.exists():
    ROOT = HOME / "sri-architecture"

staging = ROOT.parent / "_sri_unzip_bootstrap"
if staging.exists():
    shutil.rmtree(staging)
staging.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(ZIP) as zf:
    zf.extractall(staging)
extracted = next(p for p in staging.iterdir() if p.is_dir())
if ROOT.exists():
    shutil.rmtree(ROOT)
shutil.move(str(extracted), str(ROOT))
shutil.rmtree(staging, ignore_errors=True)

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", str(ROOT)])
print("SYNC_MODE local_zip")
print("ZIP", ZIP)
print("ROOT", ROOT)
print("has_campaign", (ROOT / "experiments/braket_dissertation_validity_campaign.py").is_file())
```

### Optional later: new GitHub token (only if you want git sync again)

- **Do not reuse** token `6224527288`.  
- Prefer a **fine-grained** PAT (Contents: Read on `Ceraph-Corporation/sri-architecture`).  
- Or classic PAT with expiration set explicitly to **7 days** (new token id).  

## Cell V0b — shared paths (run once after V0)

```python
import os, sys, json
from pathlib import Path
from IPython.display import Markdown, display

ROOT = Path("/home/ec2-user/SageMaker/sri-architecture")
if not ROOT.exists():
    ROOT = Path.home() / "SageMaker" / "sri-architecture"
if not ROOT.exists():
    ROOT = Path.home() / "sri-architecture"
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from experiments.braket_dissertation_validity_campaign import run_chapter, run

OUT = Path("/tmp/dissertation_validity")
OUT.mkdir(parents=True, exist_ok=True)
CHAPTER_RESULTS = {}
print("OUT", OUT)
```

---

## Cell C1 — Chapter 1: Crisis & safety tax

```python
r1 = run_chapter(1, OUT, vocab=32000, batches=40)
CHAPTER_RESULTS[1] = r1
print(json.dumps({"chapter": 1, "title": r1["title"], "pass_chapter": r1["pass_chapter"],
                  "check_passes": {k: v.get("pass") for k, v in r1["checks"].items()},
                  "safety_tax": {k: r1["checks"]["safety_tax_1_2"].get(k) for k in
                                 ("inprocess_median_ms", "post_filter_model_ms", "speedup_vs_post_filter")}}, indent=2))
print("--- PASTEBACK C1 ---")
print(Path(r1["pasteback_path"]).read_text())
display(Markdown(Path(r1["report"]).read_text()))
assert r1["pass_chapter"]
```

---

## Cell C2 — Chapter 2: CV-dit & bipartite ontology

```python
r2 = run_chapter(2, OUT)
CHAPTER_RESULTS[2] = r2
print(json.dumps({"chapter": 2, "title": r2["title"], "pass_chapter": r2["pass_chapter"],
                  "check_passes": {k: v.get("pass") for k, v in r2["checks"].items()}}, indent=2))
print("--- PASTEBACK C2 ---")
print(Path(r2["pasteback_path"]).read_text())
display(Markdown(Path(r2["report"]).read_text()))
assert r2["pass_chapter"]
```

---

## Cell C3 — Chapter 3: Quasicrystal manifold & cancellation projector

```python
r3 = run_chapter(3, OUT, seeds=[108, 109, 110], n_samples=4096)
CHAPTER_RESULTS[3] = r3
print(json.dumps({"chapter": 3, "title": r3["title"], "pass_chapter": r3["pass_chapter"],
                  "check_passes": {k: v.get("pass") for k, v in r3["checks"].items()},
                  "mbl_note": r3["checks"]["fibonacci_aah_smoke_3_1"].get("note")}, indent=2))
print("--- PASTEBACK C3 ---")
print(Path(r3["pasteback_path"]).read_text())
display(Markdown(Path(r3["report"]).read_text()))
assert r3["pass_chapter"]
# Softwall: Fibonacci smoke must not claim MBL proven
assert "not claimed" in (r3["checks"]["fibonacci_aah_smoke_3_1"].get("note") or "").lower() or \
       r3["checks"]["fibonacci_aah_smoke_3_1"].get("track") == "INSTRUMENT"
```

---

## Cell C4 — Chapter 4: Witness gate, V3, mirror

```python
r4 = run_chapter(4, OUT, vocab=32000)
CHAPTER_RESULTS[4] = r4
print(json.dumps({"chapter": 4, "title": r4["title"], "pass_chapter": r4["pass_chapter"],
                  "check_passes": {k: v.get("pass") for k, v in r4["checks"].items()},
                  "user_path": {k: r4["checks"]["user_query_witness_path"].get(k)
                                for k in ("decision", "confidence", "elapsed_ms")}}, indent=2))
print("--- PASTEBACK C4 ---")
print(Path(r4["pasteback_path"]).read_text())
display(Markdown(Path(r4["report"]).read_text()))
assert r4["pass_chapter"]
```

---

## Cell C5 — Chapter 5: Free-energy math & hardware softwalls

```python
r5 = run_chapter(5, OUT)
CHAPTER_RESULTS[5] = r5
board = r5["checks"]["softwall_board_5_2"]
print(json.dumps({"chapter": 5, "title": r5["title"], "pass_chapter": r5["pass_chapter"],
                  "check_passes": {k: v.get("pass") for k, v in r5["checks"].items()},
                  "math": {k: r5["checks"]["landauer_helmholtz_5_1"].get(k)
                           for k in ("delta_Q_350K", "F_final_Umin", "physical_circumvention")},
                  "softwalls": {k: board.get(k) for k in
                                ("landauer_circumvention", "tmem_msb_xor_o0",
                                 "blackwell_0p0062ms", "any_softwall_falsely_greened")}}, indent=2))
print("--- PASTEBACK C5 ---")
print(Path(r5["pasteback_path"]).read_text())
display(Markdown(Path(r5["report"]).read_text()))
assert r5["pass_chapter"]
assert board["landauer_circumvention"] == "NOT_PROVEN"
assert board["tmem_msb_xor_o0"] == "SOFTWARE_PROXY_NOT_TMEM"
assert board["blackwell_0p0062ms"] == "REFERENCE_ONLY_NOT_REMEASURED"
```

---

## Cell C6 — Chapter 6: Future optical horizons (softwall)

```python
r6 = run_chapter(6, OUT)
CHAPTER_RESULTS[6] = r6
opt = r6["checks"]["optical_future_6"]
print(json.dumps({"chapter": 6, "title": r6["title"], "pass_chapter": r6["pass_chapter"],
                  "check_passes": {k: v.get("pass") for k, v in r6["checks"].items()},
                  "optical": {k: opt.get(k) for k in
                              ("ppln_phase_conjugate", "analog_optical_core",
                               "falsely_greened", "track")}}, indent=2))
print("--- PASTEBACK C6 ---")
print(Path(r6["pasteback_path"]).read_text())
display(Markdown(Path(r6["report"]).read_text()))
assert r6["pass_chapter"]
assert opt["ppln_phase_conjugate"] == "FUTURE_EMBODIMENT_NOT_PROVEN"
assert opt["falsely_greened"] is False
```

---

## Cell CALL — optional full suite (all chapters + integration report)

```python
summary = run(OUT, seeds=[108, 109, 110], n_samples=4096, vocab=32000, batches=40)
print(json.dumps(summary["verdict"], indent=2))
print("--- chapter rollup ---")
print(json.dumps({c: CHAPTER_RESULTS[c]["pass_chapter"] for c in sorted(CHAPTER_RESULTS)}, indent=2))
print("--- PASTEBACK FULL ---")
print((OUT / "pasteback.json").read_text())
display(Markdown((OUT / "DISSERTATION_VALIDITY_INTEGRATION.md").read_text()))
assert summary["verdict"]["pass_dissertation_validity"]
assert all(CHAPTER_RESULTS[c]["pass_chapter"] for c in range(1, 7))
```

---

## Cell S3 — archival

```python
import boto3
from datetime import datetime, timezone

BUCKET = "amazon-braket-sri-results-1"
REGION = "us-east-1"
PREFIX = "sri-braket/dissertation-validity"
ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
client = boto3.client("s3", region_name=REGION)
uris = {}
files = list(OUT.rglob("*.json")) + list(OUT.rglob("*.md"))
for path in files:
    rel = path.relative_to(OUT).as_posix()
    key = f"{PREFIX}/results/{ts}/{rel}"
    client.put_object(Bucket=BUCKET, Key=key, Body=path.read_bytes())
    uris[rel] = f"s3://{BUCKET}/{key}"
print(json.dumps(uris, indent=2))
print("Paste PASTEBACK C1–C6 (and FULL if run) + these URIs back into chat.")
```
