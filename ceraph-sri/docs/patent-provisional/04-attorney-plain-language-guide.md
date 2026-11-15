# Attorney guide — figures + technical description

**Audience:** Patent counsel and co-inventors.  
**Figures:** [03-mermaid-architecture-diagrams.md](03-mermaid-architecture-diagrams.md)  
**Specification prose + Unicode equations:** [02-technical-description-for-spec.md](02-technical-description-for-spec.md)

---

## One-sentence invention

SRI is middleware that (1) forms a cleaned signal estimate from dual observable measurements when available, and (2) applies a stability transform to model output scores **before Softmax / next-token selection**, instead of relying only on a slow post-generation filter.

---

## Figure map (copy order for the formal draft)

| Copy first | FIG. | What it shows |
|---|---|---|
| 1 | **FIG. 1** | User query → prior post-filter path vs SRI path → result to user |
| 2 | **FIG. 4** | User query → pre-Softmax stability gate → reply |
| 3 | **FIG. 5** | Full chat turn with SRI intercept (user→API→model→SRI→user) |
| 4 | **FIG. 8** | User query → confidence gate → present or refuse |
| 5 | **FIG. 3** | User request → dual-measurement cleaned estimate |
| Support | **FIG. 2** | User query through signal–noise cancellation |
| Boundary | **FIG. 6** | User-facing product: enabled vs future |
| Optional | **FIG. 7** | Operator request → dual-sensor control → report to user |

Each major section in `02-technical-description-for-spec.md` names the matching FIG. so you can jump from prose → diagram.

---

## Walkthrough

### FIG. 1 — timing (user bookends)
User sends query → AI begins processing → **prior:** full answer then external filter → present to user; **SRI:** pre-Softmax scores → stability transform → tokens → present to user.

### FIG. 2 — cancellation on a live turn
User query → raw internal state → signal + noise → invert noise → cleaned estimate → continue generation → present to user.

### FIG. 3 — deployable dual measurement
User query/sensing request → acquire y_A and y_B → blend weights → correction → cleaned estimate → return to user. Latent noise is not an input. Tie to Lemma Z1 in §5 of file 02.

### FIG. 4 — Witness gate
User query → candidate scores → before Softmax → stability score → mild/strong scaling → Softmax / sample → loop → present to user.

### FIG. 5 — serving
User query → API → forward pass → SRI intercept → optional dual-measurement clean → stability rescale → sample → loop → stream to user.

### FIG. 6 — boundaries on a user turn
User query → if proven, run enabled core and answer; if not proven, keep as future only—user still served by enabled core.

### FIG. 7 — sensing / robotics
Operator/user request → two sensors → cleaned state → controller → optional scaling → actuators → report to user.

### FIG. 8 — product method
User query → stabilize before Softmax → draft → confidence check → present answer **or** no-answer indication to the user.

---

## Suggested specification outline

1. Field and background (safety tax; cite **FIG. 1**).  
2. Summary of invention.  
3. Brief description of drawings (**FIG. 1–8**).  
4. Detailed description following §§2–6 of file 02 with **FIG. 2–5**.  
5. Claims: Witness method; dual-measurement method; combined system; dependents.  
6. Future embodiments for softwalls (**FIG. 6**).

---

## Evidence to attach

1. **Latency / “how much faster”:** [07-enablement-evidence-and-latency.md](07-enablement-evidence-and-latency.md)  
2. **Live validity pastebacks:** `docs/benchmark-results/braket-live/dissertation-validity/sagemaker-g4dn/` (`pass_dissertation_validity: true`; chapters 1–6; digest `a8338bf9cbf7033d`)  
3. Phase‑2 / foundation atlas / C5 under `docs/benchmark-results/braket-live/`  
4. Softwall list in [00-README.md](00-README.md)  
5. Dissertation chapter map: [06-final-dissertation-integration.md](06-final-dissertation-integration.md)

### One-line answer for the inventor

**Yes, faster:** measured in-process ~**0.495 ms** vs modeled external-filter midpoint **300 ms** ≈ **606×** faster on the intercept (≈202×–1010× across the 100–500 ms band). Not a full-product E2E bakeoff.
