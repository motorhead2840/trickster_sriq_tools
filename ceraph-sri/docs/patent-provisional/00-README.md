# Provisional patent drafting pack (Ceraph / SRI)

**Purpose:** Materials for inventing counsel to draft a **provisional patent specification**.  
**Foundation verdict:** Yes — Phase‑2 Braket affirmation + Lemma Z1/C5 + **Final Dissertation validity pass (SageMaker g4dn, chapters 1–6)** are a **solid theoretical/mathematical foundation** for a provisional *if claims track supported inventions* and softwalls stay out of enabled claims.

**Latency headline (read carefully):** Measured in-process intercept ~**0.495 ms** is **faster** than the modeled post-filter band midpoint **300 ms** by ~**606×** (architectural intercept comparison — see [07](07-enablement-evidence-and-latency.md)).

**Terminology note:** These drafts use standard computer-science / signal-processing language (bipartite Hilbert space, symplectic conjugate projector, MMSE estimator, pre-Softmax stability gate). Historical research codenames are not used in the specification prose.

## How to use this folder

| File | Audience | Use |
|---|---|---|
| [01-whitepaper-sri-provisional.md](01-whitepaper-sri-provisional.md) | Counsel + technical readers | Narrative of the invention, problem, solution, evidence |
| [02-technical-description-for-spec.md](02-technical-description-for-spec.md) | Patent attorney drafting the spec | Technical description with **Unicode equations**, **FIG. cross-refs**, layman explainers, claim sketches |
| [03-mermaid-architecture-diagrams.md](03-mermaid-architecture-diagrams.md) | Spec figures | Landscape Mermaid FIG. 1–8 + download links |
| [figures/](figures/) | Google Docs paste | **Downloadable** PNG/SVG + `sri-patent-figures-landscape.zip` |
| [04-attorney-plain-language-guide.md](04-attorney-plain-language-guide.md) | Patent attorney | Figure map + copy order + walkthrough + suggested spec outline |
| [05-investor-preseed-evaluation.md](05-investor-preseed-evaluation.md) | Founder / early investors | Stage evaluation, 2-day / 2-month plan, recommended pre-seed ask |
| [06-final-dissertation-integration.md](06-final-dissertation-integration.md) | Counsel + inventor | Final Dissertation chapters ↔ enabled / softwall / FIG. / Braket validity stages |
| [07-enablement-evidence-and-latency.md](07-enablement-evidence-and-latency.md) | Counsel + inventor | **“How much faster?”** + live validity evidence + git paths for attorney packet |
| [08-attorney-simple-ip-brief.md](08-attorney-simple-ip-brief.md) | **Patent attorney (first send)** | Short plain-language “what we protect” + FIG. numbers for Google Docs |
| [09-combined-whitepaper-braket-affirmed.md](09-combined-whitepaper-braket-affirmed.md) | Counsel + technical readers | **Architecture/IC-first** whitepaper; AI features secondary; equation images `E01`–`E30` |
| [10-combined-technical-specification.md](10-combined-technical-specification.md) | Patent attorney drafting the spec | Tech spec with LaTeX + PNGs; IC embodiments; full test catalog; VC claims |
| [11-attorney-architecture-ic-plain-language.md](11-attorney-architecture-ic-plain-language.md) | **Patent attorney** | Plain-English walkthrough of architecture → chip → AI; claim order |
| [attorney-docx/Ceraph_SRI_Attorney_Briefing.docx](attorney-docx/Ceraph_SRI_Attorney_Briefing.docx) | **Patent attorney (editable)** | Cover + TOC + portrait flowcharts + eqs E01–E30 as subsections |
| [equations/](equations/) | Spec / Word / PDF paste | Rendered equation PNGs `E01`–`E30`, `D1`–`D6` + VC histograms |

## Evidence anchors (do not detach)

- **Validity pastebacks (2026-11-12):** `docs/benchmark-results/braket-live/dissertation-validity/sagemaker-g4dn/` (`pass_dissertation_validity: true`, digest `a8338bf9cbf7033d`)
- Phase‑2 S3: `s3://amazon-braket-sri-results-1/sri-braket/phase2-dissertation/results/20261112T072158Z/`
- Foundation atlas S3: `s3://amazon-braket-sri-results-1/sri-braket/foundation-atlas/results/20261112T065601Z/`
- CUDA-Q provisional S3: `s3://amazon-braket-sri-results-1/sri-braket/cudaq-provisional-dissertation/results/20261112T064213Z/`
- C5 g4dn elaborate: `docs/benchmark-results/braket-live/sphota-yoga-buddhi-elaborate/sagemaker-g4dn/`
- Axioms: `docs/architecture-axioms.md`
- Repro cells: `docs/sagemaker-dissertation-validity-cells.md`  


## Softwalls (future embodiments only — not enabled claims)

- Physical Landauer circumvention / measured \(F=U_{\min}\)
- TMEM-resident MSB-XOR as hardware \(\mathcal O(0)\)
- Blackwell 0.0062 ms / 0.0082 ms tables not remeasured on this campaign
- Full Theorem Zero without Lemma Z2
- Production PQC break / MBL proof

## Suggested claim spine for provisional

1. **Architecture:** bipartite CV state + symplectic conjugation + cancellation projector  
2. **IC / apparatus:** programmable on-chip sign-bit conjugation + FMA cancellation (TMEM path as dependent/future)  
3. **Observable estimator:** dual-mode projector ≡ MMSE (Lemma Z1)  
4. **Secondary AI:** in-process Witness (\(\Xi\) + \(\chi_{v3}\)) before Softmax  
5. Optional dependent: multi-backend kernels; mirror; fabric Hamiltonian; optical / quasicrystal futures  
