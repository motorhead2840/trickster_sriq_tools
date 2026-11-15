# Enablement evidence & latency comparison (for counsel)

**Purpose:** Solid background for the provisional draft — what was measured, what “faster” means, and which git paths to cite.  
**Live campaign:** Final Dissertation validity on SageMaker g4dn, 2026-11-12 (`9ed38e0` pasteback era).  
**Raw pastebacks:** [`docs/benchmark-results/braket-live/dissertation-validity/sagemaker-g4dn/`](../benchmark-results/braket-live/dissertation-validity/sagemaker-g4dn/)

---

## 1. Is SRI faster than current tech?

**Yes — on the safety / stability intercept path, as an architectural latency comparison.**

| Path | What it is | Measured / modeled time |
|---|---|---|
| **SRI in-process** (Witness + Υ before Softmax / next-token choice) | Measured wall time on SageMaker g4dn | **~0.495 ms** median (full suite); Ch.1 cell **~0.490 ms** |
| **Current typical external safety filter** | Industry / Final Dissertation §1.2 band of post-inference filters | **100–500 ms** (band); **300 ms** = midpoint used as model |

### How much faster?

Using the full-suite median **0.495 ms**:

| Compare against | Ratio (external ÷ SRI) | Plain English |
|---|---|---|
| Band midpoint **300 ms** | **~606×** | About **six hundred times** faster than the modeled typical filter wait |
| Band low end **100 ms** | **~202×** | Still about **two hundred times** faster than a “fast” external filter in that band |
| Band high end **500 ms** | **~1010×** | About **a thousand times** faster than a slow external filter in that band |

**Phase‑2 earlier figure** (~0.473 ms → ~634× vs 300 ms) is the same class of result; validity re-affirmed ~0.495 ms → ~606×.

### What this does *not* claim (keep in the spec)

1. **300 ms is not a matched product bakeoff** on the same model, prompts, and quality target. It is the **modeled midpoint** of the 100–500 ms post-filter band cited in the Final Dissertation §1.2.  
2. This measures the **stability / cleaning intercept**, not full end-to-end chat latency (network, full forward pass, etc.).  
3. JAX on this run fell back to **CPU** (no CUDA jaxlib); CUDA-Q was present. GPU-native jaxlib may change absolute milliseconds; the architectural gap vs 100–500 ms remains the enablement story.  
4. Softwalls (Landauer circumvention, TMEM O(0), Blackwell 0.0062 ms, MBL, optical PPLN) are **not** part of the speed claim.

**Spec language counsel can use:**  
*“In embodiments, an in-process stability transform applied to model output scores prior to Softmax completes in less than one millisecond on commodity GPU-class instances, whereas post-inference safety filters commonly impose latencies on the order of 100 to 500 milliseconds.”*

---

## 2. What the live validity campaign proved (attorney checklist)

| Gate | Result | Use in provisional |
|---|---|---|
| `pass_dissertation_validity` | **true** | Reduction to practice / enablement narrative |
| Chapters 1–6 | **all pass** | Chapter-mapped support for Final Dissertation spine |
| Lemma Z1 | **true** | Independent claim: dual-measurement cleaned estimate |
| Witness / user-query path | **true** | Independent claim: pre-Softmax stability; FIG. 4, 5, 8 |
| Softwall honest | **true** | Keeps future embodiments out of enabled independents |
| Digest | `a8338bf9cbf7033d` | Pasteback integrity marker |

### Per-chapter enablement (C1–C6)

| Ch | Pass | What counsel should take |
|---|---|---|
| 1 | yes | Safety-tax architecture latency (**FIG. 1**) |
| 2 | yes | CV-dit + bipartition math (**FIG. 2**) |
| 3 | yes | Cancellation projector + Lemma Z1; MBL instrument-only (**FIG. 2, 3**) |
| 4 | yes | V3 Witness + mirror + user path present (**FIG. 4, 8**) |
| 5 | yes | Landauer/Helmholtz **math**; physical bypass softwall (**FIG. 6**) |
| 6 | yes | Optical/PPLN correctly **future only** (**FIG. 6**) |

---

## 3. Git paths for the attorney packet (copy set)

Assemble a zip or shared folder from these paths on branch `cursor/dissertation-braket-validity-fd3c` (or `main` after merge):

### A. Specification drafting

| Path | Role |
|---|---|
| `docs/patent-provisional/00-README.md` | Index, claim spine, softwalls |
| `docs/patent-provisional/01-whitepaper-sri-provisional.md` | Narrative + evidence tables |
| `docs/patent-provisional/02-technical-description-for-spec.md` | Spec paragraphs + Unicode equations + FIG refs |
| `docs/patent-provisional/03-mermaid-architecture-diagrams.md` | FIG. 1–8 (user-query flows) |
| `docs/patent-provisional/04-attorney-plain-language-guide.md` | Copy order + walkthrough |
| `docs/patent-provisional/06-final-dissertation-integration.md` | Dissertation ↔ enabled/softwall map |
| `docs/patent-provisional/07-enablement-evidence-and-latency.md` | **This file** — latency + evidence |

### B. Live enablement evidence

| Path | Role |
|---|---|
| `docs/benchmark-results/braket-live/dissertation-validity/sagemaker-g4dn/pasteback.json` | Full suite verdict |
| `.../pasteback-chapters.json` | C1–C6 detail |
| `.../CHAPTER_VALIDITY_ROLLUP.md` | Human rollup |
| `docs/benchmark-results/braket-live/phase2-dissertation/sagemaker-g4dn/` | Phase‑2 prior |
| `docs/benchmark-results/braket-live/sphota-yoga-buddhi-elaborate/sagemaker-g4dn/` | Lemma Z1 / C5 prior |
| `docs/sagemaker-dissertation-validity-cells.md` | Reproducible cells |
| `experiments/braket_dissertation_validity_campaign.py` | Executable enablement |

### C. Optional prior S3 prefixes

- Phase‑2: `s3://amazon-braket-sri-results-1/sri-braket/phase2-dissertation/results/20261112T072158Z/`  
- Foundation atlas / CUDA-Q prefixes listed in `00-README.md`

---

## 4. Suggested exhibit list for the provisional

1. **Exhibit A** — FIG. 1–8 (from `03`)  
2. **Exhibit B** — Technical description §§0–11 (from `02`)  
3. **Exhibit C** — Validity pasteback JSON + rollup (this campaign)  
4. **Exhibit D** — Softwall / future-embodiment list (`00` + Ch. 5–6)  
5. **Exhibit E** — Claim spine (observable Υ_obs; Witness; combined system)

---

## 5. One-paragraph latency statement (inventor → attorney)

> Current generative systems often add a post-inference safety or quality filter that commonly costs on the order of 100–500 milliseconds per turn. On SageMaker g4dn (2026-11-12), the SRI in-process Witness and cancellation path completed with a median wall time of approximately 0.495 milliseconds. Relative to the 300-millisecond midpoint of that external-filter band, that is about a 606× reduction in intercept latency; relative to the 100–500 ms band edges, about 202×–1010×. This is an architectural enablement measurement of the intercept—not a claim that full product end-to-end latency, calorimetry, or special hardware conjugation has been proven.
