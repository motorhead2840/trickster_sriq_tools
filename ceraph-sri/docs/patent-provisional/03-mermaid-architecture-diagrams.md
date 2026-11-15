# Patent figures — landscape (downloadable)

All figures use **left → right** layout so text stays readable in Google Docs / Word (landscape page).  
**Download PNG/SVG:** [`figures/`](figures/) — open a file, or grab [`figures/sri-patent-figures-landscape.zip`](figures/sri-patent-figures-landscape.zip).

| FIG. | Download PNG | Download SVG | Title |
|---|---|---|---|
| **FIG. 1** | [fig-01.png](figures/fig-01-user-query-paths.png) | [fig-01.svg](figures/fig-01-user-query-paths.svg) | User query: post-filter vs SRI |
| **FIG. 2** | [fig-02.png](figures/fig-02-signal-noise-cancel.png) | [fig-02.svg](figures/fig-02-signal-noise-cancel.svg) | Signal–noise cancellation |
| **FIG. 3** | [fig-03.png](figures/fig-03-dual-measurement.png) | [fig-03.svg](figures/fig-03-dual-measurement.svg) | Dual-measurement cleaned estimate |
| **FIG. 4** | [fig-04.png](figures/fig-04-stability-gate.png) | [fig-04.svg](figures/fig-04-stability-gate.svg) | Pre-Softmax stability gate |
| **FIG. 5** | [fig-05.png](figures/fig-05-chat-intercept.png) | [fig-05.svg](figures/fig-05-chat-intercept.svg) | Full chat turn with SRI |
| **FIG. 6** | [fig-06.png](figures/fig-06-claim-boundary.png) | [fig-06.svg](figures/fig-06-claim-boundary.svg) | Enabled vs future claims |
| **FIG. 7** | [fig-07.png](figures/fig-07-dual-sensor-control.png) | [fig-07.svg](figures/fig-07-dual-sensor-control.svg) | Dual-sensor control |
| **FIG. 8** | [fig-08.png](figures/fig-08-confidence-gate.png) | [fig-08.svg](figures/fig-08-confidence-gate.svg) | Present or refuse |

Regenerate: `bash docs/patent-provisional/figures/render-landscape.sh`

---

## FIG. 1 — User query: post-filter path vs SRI path

*Reference numerals 100–136. Landscape.*

```mermaid
flowchart LR
  U100["100 User sends a query"] --> P102["102 AI begins processing"]
  P102 --> A112["112 Prior: finish full answer"]
  A112 --> A114["114 Prior: external filter"]
  A114 --> A116["116 Prior: emit / block / rewrite"]
  P102 --> B122["122 SRI: scores before Softmax"]
  B122 --> B124["124 SRI: in-process stability"]
  B124 --> B126["126 SRI: next token"]
  B126 --> B128{"128 More tokens?"}
  B128 -->|Yes| B122
  B128 -->|No| B130["130 Assemble answer"]
  A116 --> U136["136 Present to user"]
  B130 --> U136
```

**Drafting note:** Same user query; prior art filters after a full answer; SRI steadies scores before each next-token choice.

---

## FIG. 2 — User query through signal–noise cancellation

*Reference numerals 200–220. Landscape.*

```mermaid
flowchart LR
  U200["200 User query"] --> S202["202 Raw internal state"]
  S202 --> S204["204 Signal + noise"]
  S204 --> S206["206 Mark signal"]
  S204 --> S208["208 Mark noise"]
  S206 --> S210["210 Noise-inverted copy"]
  S208 --> S210
  S210 --> S212["212 Combine with original"]
  S212 --> S214["214 Cleaned estimate"]
  S214 --> S216["216 Continue generation"]
  S216 --> U220["220 Present to user"]
```

**Drafting note:** Conceptual cancellation on a live user turn.

---

## FIG. 3 — User request → dual-measurement cleaned estimate

*Reference numerals 300–320. Landscape.*

```mermaid
flowchart LR
  U300["300 User request"] --> M302["302 Start processing"]
  M302 --> M304["304 Measure y_A"]
  M302 --> M306["306 Measure y_B"]
  M304 --> M308["308 Blend weights"]
  M306 --> M308
  M308 --> M310["310 Noise-aware correction"]
  M310 --> M312["312 Cleaned estimate"]
  M312 --> M314["314 Use in answer / control"]
  M314 --> U320["320 Return to user"]
```

**Drafting note:** Dual measurements only — no hidden true noise input. Prefer for Lemma Z1 claims.

---

## FIG. 4 — User query → pre-Softmax stability gate → reply

*Reference numerals 400–424. Landscape.*

```mermaid
flowchart LR
  U400["400 User query"] --> W402["402 AI processing"]
  W402 --> W404["404 Candidate scores"]
  W404 --> W406["406 Before Softmax"]
  W406 --> W408["408 Stability score"]
  W408 --> W410{"410 High volatility?"}
  W410 -->|No| W412["412 Mild scaling"]
  W410 -->|Yes| W414["414 Strong scaling"]
  W412 --> W416["416 Rescale scores"]
  W414 --> W416
  W416 --> W418["418 Softmax / sample"]
  W418 --> W420{"420 More tokens?"}
  W420 -->|Yes| W404
  W420 -->|No| U424["424 Present to user"]
```

**Drafting note:** Softmax only after rescaling.

---

## FIG. 5 — Full chat turn with SRI intercept

*Reference numerals 500–520. Landscape.*

```mermaid
flowchart LR
  U500["500 User query"] --> C502["502 Serving API"]
  C502 --> C504["504 Model forward pass"]
  C504 --> C506["506 SRI intercept"]
  C506 --> C508["508 Optional dual-measure clean"]
  C508 --> C510["510 Stability rescale"]
  C510 --> C512["512 Softmax / sample"]
  C512 --> C514{"514 More tokens?"}
  C514 -->|Yes| C504
  C514 -->|No| C516["516 Assemble text"]
  C516 --> U520["520 Stream to user"]
```

**Drafting note:** Primary system embodiment for chat / API.

---

## FIG. 6 — Enabled claims vs future embodiments

*Reference numerals 600–622. Landscape.*

```mermaid
flowchart LR
  U600["600 User query"] --> D602{"602 Proven in evidence?"}
  D602 -->|Yes| D604["604 Enabled core"]
  D604 --> D606["606 Dual-measure clean"]
  D604 --> D608["608 Pre-Softmax gate"]
  D604 --> D610["610 Combined middleware"]
  D610 --> U620["620 Stabilized answer"]
  D602 -->|No| D612["612 Future only"]
  D612 --> D614["614 Hardware conjugation"]
  D612 --> D616["616 Landauer circumvention"]
  D612 --> D618["618 Pending later lemma"]
  D618 --> U622["622 Serve via enabled core"]
```

**Drafting note:** Softwalls stay future-only.

---

## FIG. 7 — Operator request → dual-sensor control

*Reference numerals 700–722. Landscape.*

```mermaid
flowchart LR
  U700["700 Operator request"] --> R702["702 Start processing"]
  R702 --> R704["704 Sensor A"]
  R702 --> R706["706 Sensor B"]
  R704 --> R708["708 Cleaned state"]
  R706 --> R708
  R708 --> R710["710 Controller"]
  R710 --> R712{"712 Unstable outputs?"}
  R712 -->|Yes| R714["714 Stability scaling"]
  R712 -->|No| R716["716 Pass through"]
  R714 --> R718["718 Drive actuators"]
  R716 --> R718
  R718 --> U722["722 Report to operator"]
```

**Drafting note:** Non-LLM embodiment with user/operator bookends.

---

## FIG. 8 — User query → confidence gate → present or refuse

*Reference numerals 800–820. Landscape.*

```mermaid
flowchart LR
  U800["800 User query"] --> P802["802 AI processing"]
  P802 --> P804["804 Model scores"]
  P804 --> P806["806 In-process stability"]
  P806 --> P808["808 Draft answer"]
  P808 --> P810["810 Confidence check"]
  P810 --> P812{"812 Above threshold?"}
  P812 -->|No| P814["814 No-answer / low confidence"]
  P812 -->|Yes| P816["816 Present answer"]
```

**Drafting note:** User query in; present or refuse out.

---

## Counsel usage

1. Prefer downloaded **PNG** in Google Docs (landscape page orientation).  
2. Formal drawing priority: **FIG. 1, 4, 5, 8**, then **FIG. 3**.  
3. Source Mermaid above stays citation-ready for edits.  
