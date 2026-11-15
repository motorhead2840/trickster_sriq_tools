# Investor evaluation and pre-seed ask (Ceraph / SRI)

**Audience:** Founder (Hari) planning the next ~2 days of foundation work and a ~2‑month path toward a credible pre-seed conversation.  
**Register:** Plain business language. Not a promise of funding or product success.  
**Technical honesty:** Aligns with Phase‑2 evidence and softwalls already documented in this folder.

---

## 1. Bottom-line evaluation

| Dimension | Rating | One-line read |
|---|---|---|
| Research / IP foundation | **Strong for stage** | Clear invention spine, provisional materials, measured in-process path |
| Evidence quality | **Promising, bounded** | Campaign gates passed; ~634× is architectural vs modeled filter midpoint |
| Product readiness | **Early** | Middleware idea clear; no design-partner production proof yet |
| Go-to-market readiness | **Early** | Buyer, price, and wedge workload still to be chosen |
| Investor narrative readiness (in ~2 months, if focused) | **Achievable** | Matched benchmark + thin demo + filed provisional + one design partner LOI |

**Verdict:** SRI is a **credible deep-tech pre-seed prospect** if the next two months convert research milestones into (a) a matched latency/quality benchmark, (b) a demable intercept on a real model path, and (c) a filed provisional with clean claim boundaries. It is **not yet** a Series A / “market ready production company” story. Pre-seed money should buy that conversion—not a claim that production scale is already done.

---

## 2. What is already valuable to investors

1. **Defined invention, not only a vibe:** in-process stability gate + dual-mode observable cleaning (Lemma Z1).  
2. **Artifact trail:** Phase‑2 campaign, S3 provenance, patent annex, softwall honesty (rare and trust-building).  
3. **Attorney pack:** whitepaper, technical description, plain figures, claim spine.  
4. **Clear non-claims:** Landauer circumvention and hardware O(0) are not oversold—good for sophisticated angels and deep-tech funds.

---

## 3. What still blocks a strong raise

1. **No matched baseline yet:** 0.473 ms in-path vs 300 ms modeled external band is a hypothesis, not a customer-grade bakeoff.  
2. **No design partner / paid pilot.**  
3. **No thin product surface** (SDK or serving plugin) a buyer can try in a day.  
4. **Hardware story incomplete** for GPU-native investors (some paths CPU; CUDA-Q paths on T4—fine if narrated precisely).  
5. **Team / GTM packaging** (who sells, to whom, against which incumbent filter stack).

---

## 4. Recommended next 2 days (foundation formalization)

Use the two days to lock **investor-grade clarity**, not new physics.

| Day focus | Deliverable |
|---|---|
| Day 1 | Freeze claim spine + softwalls; pick **one wedge workload** (e.g. chat safety intercept *or* dual-sensor robotics estimate—not both) |
| Day 1 | Write a 2-page investor one-pager from this memo + FIG. 1/5/8 |
| Day 2 | Spec the **matched benchmark protocol** (same model, same prompts, same hardware, quality metric, latency definition) |
| Day 2 | Sketch a **minimal demo**: pre-Softmax hook that logs Ξ, applies χ damping, shows before/after tokens |
| Day 2 | Counsel packet zip: provisional folder + Phase‑2 annex + figure list |

Exit criteria for the two days: you can explain SRI to an investor in under three minutes using FIG. 1 and FIG. 8, and you know exactly which experiment will replace the modeled 300 ms comparison.

---

## 5. Recommended next ~2 months (toward a promising raise)

These are **investor-credibility milestones**, not “full market ready.”

1. **Matched bakeoff (must-have):** in-process SRI vs a strong external filter baseline on one workload; report latency, quality, refusal rate; 3+ seeds; publish methods.  
2. **Thin demo / SDK:** drop-in pre-Softmax middleware on one open model stack.  
3. **Provisional filed** with plain figures and enabled-core claims only.  
4. **One design partner** (LOI or unpaid pilot with written problem statement).  
5. **Cost sheet:** cloud burn for demos + legal + one contractor if needed.

Optional stretch: robotics/sensing FIG. 7 demo only if the wedge is sensing; otherwise defer.

---

## 6. Funding ask estimate (pre-seed)

### Recommended ask: **USD $750k – $1.5M**

| Ask band | When it fits |
|---|---|
| **$500k – $750k** | Lean: founder-heavy, contractor legal/eng, 6–9 months runway, one wedge, no full-time sales yet |
| **$750k – $1.25M (sweet spot)** | File IP, matched benchmarks, thin product, 1–2 contractors, design-partner program, ~12 months runway |
| **$1.5M – $2.5M** | Only if you already have a design partner LOI and will hire a founding eng + GTM earlier; harder without traction |

**Suggested target to request:** **~$1.0M** (or local-currency equivalent), with a clear use-of-funds table. Round size can flex ±25% by geography and investor type (angel syndicate vs deep-tech pre-seed fund).

### Illustrative use of a $1.0M pre-seed (~12 months)

| Bucket | Share | Approx. | Purpose |
|---|---|---|---|
| Core engineering | 40% | $400k | Middleware, demo, matched benchmarks, reliability |
| Cloud / GPU / Braket | 15% | $150k | Reproducible campaigns, demos, partner pilots |
| IP / legal | 10% | $100k | Provisional → utility path, counsel, entity hygiene |
| Design partners / GTM | 15% | $150k | Pilots, travel, early solutions eng |
| Founder runway / ops | 15% | $150k | Living costs, accounting, compliance |
| Buffer | 5% | $50k | Slippage |

### What this raise should **not** try to buy

- Full production multi-tenant SaaS at scale  
- Proving Landauer circumvention or TMEM O(0) as the fundraising wedge  
- Broad multi-vertical expansion before one wedge works  

---

## 7. “Market ready” vs “investor ready” (important distinction)

| Phrase | Realistic horizon after a clean pre-seed |
|---|---|
| **Investor-ready narrative** | ~2 months of focused work (section 5) |
| **Design-partner production pilot** | ~3–6 months post-raise |
| **Market-ready product** (reliable paid offering) | ~9–18 months post-raise, contingent on pilot learning |

Do not tell investors the company is market ready now. Tell them Phase‑2 made the science **testable and filable**, and pre-seed buys the path to a **proven wedge**.

---

## 8. Risks to disclose (builds trust)

- Architectural latency ratio may shrink under matched baselines.  
- Quality must not collapse when damping is strong.  
- Incumbents can ship faster post-filters; differentiation is **in-process** placement + observable cleaning.  
- Deep-tech timelines slip; softwalls must stay softwalls.

---

## 9. Founder checklist before first serious investor meeting

- [ ] One wedge sentence  
- [ ] FIG. 1 + FIG. 8 in the deck  
- [ ] Softwall slide (what you do **not** claim)  
- [ ] Matched benchmark plan or early results  
- [ ] Provisional status (drafted / filed)  
- [ ] Ask amount + 12‑month use of funds  
- [ ] Design partner target list (10 names)  

---

## 10. How to proceed this week

1. Keep patent pack language plain (this PR’s figures + attorney guide).  
2. Spend two days on section 4 deliverables.  
3. Start the matched benchmark immediately after.  
4. Approach counsel with FIG. 1–5 + 8 and the enabled claim spine.  
5. Treat **~$1M pre-seed** as the working ask unless a design partner appears sooner—then consider the upper band.  
