# Technical report and whitepaper

Ceraph Corporation, Kerala, India. 15 November 2026. For investors and for technical diligence. chief@ceraphco.com. www.ceraphco.com.

This document combines three sources. The first is the 14 November investor memorandum of the labelled exam on one and four NVIDIA RTX PRO 6000 Blackwell Server Edition boards. The second is the counsel brief written from the same night. The third is the November 2026 full technical specification of the Singularity-based Realization Interface, which states further equations and further figures. Where a number comes only from that specification, and not from the labelled-exam JSON, the sentence says so. Planning figures are not revenue and not an offer of securities.

## 1. Abstract

The enabled invention is a way to damp a numerical series on a graphics processor while keeping a local center. The deviation from that center is multiplied by a shrinking exponential of a local instability. A host schedule chooses a tile width. Julia, JAX, and Triton compute the same map. A run is kept when they agree with a NumPy reference.

On 14 November 2026 the exam passed 17 of 18 labelled checks on one board and 16 of 17 on four boards. The miss, on both machines, was a pre-declared bar that flux retain at most half the variance of a real temperature series. The remaining fractions were about 0.510, 0.518, and 0.510. On that thermometer the flux error stayed near 5.819, 5.874, and 5.820 when the series was shifted by 5 °C. Raw damping, which multiplies the whole reading, rose from about 135 to about 266. Same-input drift was 0. A fused forty-pass kernel took 7.38 ms. The eager loop took 17.13 ms. Four boards finished 164,748 heavy jobs in 300 seconds, 549.2 per second, against 137.3 on one board. At 50 million nodes the traffic count was 1,459.0872 GB/s, 91.3643 percent of the stated 1,597 GB/s peak.

A fair repeat on 15 November, after compilation on all four boards, queued four jobs of the same heavy size in 29.2718 ms. The slowest board’s own clock was 8.2376 ms.

The November specification also states a peak variance reduction of 958,874 times at a noise standard deviation of 1,000, a median latency of about 0.0117 to 0.0125 ms, a cosine similarity of at least 0.979 in an extreme case, a residual estimator with 93.94 percent mean-squared-error reduction, and a locked 0.0120 ms over more than 1,600 samples. Those figures are reproduced here as the specification states them. They are not fields in the labelled-exam JSON.

Sentence-level refusal was tested later and is not part of the raise. Section 12 records those sheets.

## 2. The company

Ceraph Corporation is in Kerala, India. Correspondence is chief@ceraphco.com. The public site is www.ceraphco.com. As of the 14 November memorandum the company had no design-partner revenue. That memorandum did not create any. The present report does not create any.

The working ask in that memorandum is 1,250,000 US dollars for 18 months, inside a planning band of 1,000,000 to 2,000,000 US dollars. A lean band of 700,000 US dollars for 12 months is also written there. A design partner who already runs a graphics pipeline is assumed, for planning only, to pay on the order of 60,000 US dollars a year once a small library exists and a provisional is on file. The November specification, a different document, recommends a seed of 7.5 to 10 million US dollars and projects revenue that the 14 November memorandum does not adopt. This report follows the memorandum. The larger seed range is recorded only so a reader of the specification is not surprised. It is not the company’s ask.

## 3. What a reader is being shown

A noisy row arrives. A short neighborhood supplies a local center and a measure of how wild that neighborhood is. The map keeps the center and shrinks only the departure from it. Three programs must agree: Julia for the block, JAX for the value and the derivative, Triton for the kernel. The real series in the exam was the daily minimum temperature in Melbourne, 3,650 days from 1 January 1981, beginning at 20.7 °C, with noise the notebook added so the error had a known target.

No language model was present in that exam. The counsel brief of the same night says this in its first page. Later language experiments are a different chapter.

## 4. The flux equation

Let y be the series, μ a local center, ξ a local instability, and λ a positive constant. In every scored test λ was 1.

```
A(y) = μ + (y − μ) exp(−λ ξ)
```

The raw comparison is

```
R(y) = y exp(−λ ξ)
```

If the series and the center both shift by a constant c,

```
A(y + c) = A(y) + c
```

while

```
R(y + c) = R(y) + c exp(−λ ξ)
```

which is not R(y) + c unless the exponential is 1. On a thermometer, shrinking the center would report a cooler day than the day that happened. That is what raw damping did. Flux did not.

Forty iterations with a fixed center collapse to one exponential:

```
A_40(y) = μ + (y − μ) exp(−40 ξ)
```

On one board the maximum absolute difference between the kernel and this closed form was 9.5×10⁻⁷. On board 0 of the four-board machine it was 1.2×10⁻⁵, inside the exam bar of 1×10⁻⁴.

The derivative JAX checks, for the raw map, is

```
dA = exp(−λ ξ) dθ − λ θ exp(−λ ξ) dξ
```

The residual of that check was 0.

## 5. The ladder and the gate predicate

The density is the mean absolute instability. Below 0.5 the block is 128. Below 1 it is 256. Below 2 it is 512. Otherwise it is 1024. Julia’s macro `@sphota` and the Python twin returned the same four widths. The gate predicate is true when a stability index is at least 0.20. At 0.05 it was closed. At 0.20 and at 0.80 it was open. Julia matched Python on each labelled case.

On one million random points the block sweep gave 0.0157 ms at 128, 0.0120 ms at 256, 0.0131 ms at 512, and 0.0116 ms at 1024. The ladder chose 256 because the mean instability of that draw was below 1. The question asked only whether the choice was within 10 percent of the fastest block. It was, by 3.6 percent. It did not ask the ladder to win.

## 6. The labelled exam, side by side

The one-board stack was PyTorch 2.14.1 with CUDA 13.0, Triton 3.8.0, and JAX 0.12.0.1. The four-board stack in a companion suite was PyTorch 2.15.0 with CUDA 13.2 and Triton 3.9.0. Each board is an RTX PRO 6000 Blackwell Server Edition.

| Check | One board | Four boards |
|---|---|---|
| Exam score | 17 of 18 | 16 of 17 |
| Miss | V1, variance remaining 0.510 to 0.518 | the same three fractions |
| Block ladder | Julia and Python, four densities | the same |
| Gate at 0.05, 0.20, 0.80 | closed, open, open | the same |
| Sign bit and derivative | 0 and 0 | 0 and 0 on each board |
| Triton versus flux | 4.8×10⁻⁷ | 4.8×10⁻⁷ on each board |
| Synthetic flux, both offsets | 1.472 | 1.472 |
| Synthetic raw at offset 5 | 18.17 | 18.17 |
| Melbourne flux, seeds 11, 22, 33 | 5.819, 5.874, 5.820 | the same |
| Melbourne flux after +5 °C | unchanged | unchanged |
| Melbourne raw, then +5 °C | about 135, then about 266 | the same |
| Same-input drift | 0 | 0 |
| Closed form | 9.5×10⁻⁷ | 1.2×10⁻⁵ |
| Eager versus fused, 40 passes | 17.13 ms and 7.38 ms | queue 29.51 ms, together 8.28 ms |
| 50 million nodes | 0.548 ms, 1,459 GB/s, 91.4 percent | 0.549 ms, 1,457 GB/s, 91.2 percent |

More precision from the JSON, one board, synthetic seed 11: flux 1.4723521 and 1.4723523; raw 0.7735443 and 18.1708527. Melbourne seed 11: flux 5.8189759 at both offsets; raw 135.7728271 and 266.6606750; moving average 5.8973565; untouched 7.2355824. Variance remaining 0.5149549, 0.5176691, 0.5095068.

The synthetic series is the one on which the half-variance bar is met, at about 11.7 percent remaining. The thermometer is the series on which it is not. Both sentences belong in a specification. The deployable claim of that night is the preservation of the center. The stronger variance claim is met on the synthetic series and not yet met on this thermometer.

A large-glitch screen used 24 clean windows and 24 copies with a spike of 8 standard deviations on 15 percent of each window. The rule, fixed in advance, called a window a glitch when its z-scored variance exceeded 4. All 24 glitches fired and none of the clean windows did. Precision, recall, and F1 were 1.0. The spikes were gross by construction. This shows that a variance predicate can see a declared spike. It does not show anything about sentences.

## 7. Endurance, power, and the queue

One board, tiled thermometer, block 1024: 41,186 jobs in 300.0 seconds, 137.2839 per second. Mean draw 314.5305 W, peak 326.608 W. Temperature from 32.0 °C to 44.0 °C. Clock 2302 to 2317 MHz. Occupancy 98.55 percent. Energy on the order of 2,291 millijoules per job, as the memorandum computes it. An earlier random-series burn in the same notebook, block 256, finished 41,097 jobs at 137.0 per second, mean 288.6 W, 30 °C to 42 °C. The real series chose the wider block and drew more power. Both stayed finite.

Four boards, the random heavy job: 164,748 jobs, 549.1516 per second. Per board, means 279.2437 W, 283.3406 W, 279.0838 W, and 276.8883 W. Peaks 300.458 W, 303.782 W, 304.733 W, and 304.319 W. Temperatures 48 °C, 47 °C, 48 °C, and 47 °C. Clocks to 2422 MHz. Cluster mean about 1,119 W. Energy about 2,037 millijoules per job. The 600 W figure is the nameplate of one board, not a target.

The queue picture, with buffers already on the devices: 29.5101 ms against 8.2828 ms. That is about 3.56 times on one wave. The five-minute rate is 4.00 times, 549.2 against 137.3.

On 15 November the same heavy size, 16,777,216 values and 40 passes, was timed again after the kernel had been compiled on all four boards. Queued: 29.2718 ms. Slowest board: 8.2376 ms. The four clocks were 7.4879, 8.2376, 7.9172, and 7.4986 ms. The wall clock around the threads was 11.7879 ms and includes the Python join. Outputs were finite. The pass rule, slowest board under three quarters of the queued time, passed. A clock taken before compilation had finished on boards 1, 2, and 3 did not pass: 29.1735 ms queued and 38.1371 ms together. That attempt measured setup. The fair attempt measured the kernel.

## 8. The roofline and the sixteen-byte message

At 50,000,000 nodes, 16 bytes per node, one board: median 0.5482880 ms, 1,459.0872 GB/s, 91.3643 percent of 1,597. The twelve-byte count beside it, which omits the write, is 1,094.3154 GB/s. The sixteen-byte count is the one that includes the write. Board 0 of the four-board machine: 0.5491520 ms, 1,456.7915 GB/s, 91.2205 percent.

A different cell transferred 16 bytes and recorded 319,999,984 bytes not sent, relative to a full reduction of its buffers. Kernel time 17.5875 ms. Scalar sync 3.7208 ms. Bandwidth in that file 54.58 GB/s. A reader who adds 54.58 to 1,459, or who calls 54.58 the roofline, has mixed two experiments. This report keeps them apart. The architectural point of the 16 bytes is the message: ship a scalar state, not the tensor. The architectural point of 1,459.0872 is how fast one board reads and writes the series itself.

## 9. Equations stated in the November specification

The November 2026 full technical specification writes a longer list. Each item below is given as that document states it, followed by the status this repository can actually support.

Landauer. The specification writes ΔQ ≥ k_B T ln 2 per irreversible bit erasure, and then says the architecture avoids that erasure. The labelled exam did not meter joules of bit erasure. A reference calculation at room temperature is on the order of 10⁻²¹ joules per bit. It is a reference, not a measured saving. The specification’s claim that erasure is avoided entirely is a design sentence, not an exam result.

Omega-dit. The specification writes an integral of existence-density, a phase, a structural value, and a residual. The repository implements a discrete sum of the same factors in `omega_dit`. That sum is a code identity. It is not a continuous physical field.

Bipartite split. The specification separates a structural part from a residual. In the scored flux map the structural part is the local mean and the residual is the deviation. That is the split the chips ran.

Cancellation chain. Thesis is value plus disturbance. Conjugation is described as a sign-bit exclusive-or. Antithesis is value minus disturbance. The projector is one half of the identity plus the conjugation, and the average of thesis and antithesis is the value. When both value and disturbance are supplied, this is an algebra check, and the residual to the supplied value is on the order of 10⁻⁷ or smaller. When only the series is supplied, the center must be estimated. The flux map estimates it as a local mean. The average of a deviation and its exact sign flip is zero, so the result is that mean. On 5,000,000 states the telemetry file records variance 2.4656 to 0.0 in 10.1 ms and a sign-bit residual of 0.0. The labelled exam records sign error 0.0 on four boards. The specification’s figure of up to 958,874 times variance reduction, at a noise standard deviation of 1,000, is stated in that specification. It is not a field of the labelled-exam JSON, and this report does not treat it as the same measurement as the thermometer.

Entropy and free energy. The specification says residual variance driven toward zero collapses an entropy, and that a free-energy proxy F = U − T S falls toward a minimum. The exam measured variance remaining and mean squared error. It did not measure a thermodynamic free energy in joules.

Vocabulary stability. The specification writes Ξ = 1 − Var / (E of absolute value + ε). On real vocabulary logits of Qwen2.5-1.5B this coefficient sat between about −3.55 and −1.81, and a horizon of 0.98 abstained on every question, including correct ones. The signed-mean form used in an earlier SDK note produced large positive values on the same logits and did not separate true answers from false ones.

Dual circuit. The specification writes χ as a baseline exponential of Ξ plus a sigmoid of a variance threshold. With the constants used in the notebooks, λ0 = 10, κ = 0.5, Λmax = 50, the sigmoid saturates and χ sits near 51 on a full vocabulary. Multiplying every logit by one exp(−χ) does not change which token wins, unless the scale underflows and the text becomes noise. Both behaviors were observed, on different runs.

Residual estimator. The specification says a small convolutional estimator, trained with residual energy plus a structural term, reached 93.94 percent mean-squared-error reduction at noise standard deviation 4. That figure is the specification’s figure. The labelled exam’s quality numbers are the tables in section 6, not 93.94 percent.

The specification says all eighteen of its equations were numerically verified. This repository can verify the flux identity, the closed form, the sign-bit average, the ladder, and the derivative. It does not, in the labelled-exam JSON, contain a table of eighteen separate residuals. A counsel reading both documents should keep that distinction.

## 10. Where the code lives

`sri/omlang/abhaya.py` holds the damping map and the ladder. `sri/omlang/yoga_buddhi.py` holds the thesis, antithesis, and average when both parts are supplied, and the mirror product with a complex conjugate. `sri/omlang/hybrid_stack.py` accepts a run only when the tiers that are installed match NumPy. `sri/omlang/phase_cancel.py` estimates a center and cancels the deviation by the sign-bit average. `experiments/dgx_abhaya_round.py` holds the pre-registered numeric protocol, including the rule that a printed latency budget is not an input to a decision.

The 0.2379 ms figure that appears in March notebook cells is a budget that was typed into a cell. Later medians near 10,000 points sit around 0.012 ms, under that budget, and are launch-bound. Flatness at short length is not a proof that every larger program is flat. At 50 million nodes the median is 0.548 ms, which is the arithmetic showing itself.

## 11. A person asks a question

This section is the L4 notebook. It is not the Blackwell exam.

A fused kernel on that L4 prints 0.1158 ms, Shannon index 0.4872, collision 0.0302, and a signal ratio 4.61. An earlier sweep prints 0.0812 ms at batch 1 and 0.0884 ms at batch 32, 2.76 microseconds per sequence, 819.94 GB/s on that cell’s byte count.

The last suite crops the key-value cache when a step is rejected, and the host inserts a reference sentence the cell already contains. Injected logit noise has standard deviation 11.5.

Test 1: 17.64 tokens a second, 4 noise steps blocked, rollback at token 08 between “ lower” and “ higher,” collision 0.226, ratio 1.13. The printed sentence contains 2,200 W/(m·K) and about 400 W/(m·K).

Test 2: 19.08 tokens a second, 4 noise steps blocked, collision 0.405, ratio 1.45.

Test 3: 11.95 tokens a second, 8 noise steps blocked, collision 0.423, ratio 1.45. The text gives Mercury 88 days and Venus 225 days, then calls 137 days Mercury’s orbital period.

Without the crop, two of three answers in an earlier cell repeat a word. An ungated perturbed run prints 16.84 tokens a second and still says diamond has a lower thermal conductivity than copper.

## 12. Sentence sheets, and why they are not the raise

Hidden-state coefficient, horizon 0.20, base model, few-shot on every item: the gate never fired. Both columns were 12 correct, 0 wrong, 8 abstentions. The clean score is the prompt.

Phase velocity, horizon 0.05: the gate fired on every row. 12 correct and 6 wrong became 0 correct and 19 wrong.

Top-10 entropy, horizon 0.50: wrong fell from 6 to 0 and correct stayed 12. First-token values on few-shot prompts were at most 0.3866. On raw prompts they were at least 0.6313. The ocean item in the few-shot frame was the text “Pacific Ocean” at 0.2629.

Dialectical audit, twelve items, threshold 0.45: ungated 6 correct, 3 wrong, 3 abstentions; gated 5 correct, 0 wrong, 7 abstentions. Hydrogen was kept at 0.0402. “Pacific Ocean.” became “Unknown.” at 0.9976. TruthfulQA, 100 items: 23 and 23. ARC-Challenge, 100 items: 50 and 44.

These sheets are why a raise conversation should not be asked to fund a finished refusal of false sentences. They are also why a provisional should describe a logit governor only as an embodiment, with the log beside it.

## 13. Use of the working ask

The 14 November memorandum, in US dollars, for 18 months:

| Use | Basis | Amount |
|---|---|---|
| Founder | 18 months at 10,000 a month | 180,000 |
| Kernel engineer | 18 months, loaded near 16,000 a month | 288,000 |
| Research scientist | 12 months, loaded near 14,000 a month | 168,000 |
| Finance and operations, fractional | 2,000 a month | 36,000 |
| Counsel and the utility path | beyond the provisional | 120,000 |
| Entity, insurance, accounting | 18 months | 40,000 |
| Blackwell and other cloud | planned average 8,000 a month | 144,000 |
| Bench held for later embodiments | not spent as if those devices exist | 30,000 |
| Papers, site, two gatherings, travel | a small public face | 70,000 |
| Design-partner program | the company’s cost to show up | 40,000 |
| Recruiting | two hires | 20,000 |
| Reserve | about 9 percent | 114,000 |
| Working ask | 18 months | 1,250,000 |

People 672,000. Counsel and the entity 160,000. Cloud 144,000. The public face and the partner program 110,000. Bench and recruiting 50,000. Reserve 114,000.

The lean ask is 700,000 US dollars for 12 months: founder 120,000, engineer 192,000, scientist for 6 months 84,000, counsel 80,000, cloud 72,000, public face and partner program 50,000, operations 36,000, reserve 66,000.

Projected income in that memorandum, as a planning band:

| Year | Quiet | Working | One wedge |
|---|---|---|---|
| 1 | no paid partner, income about 0 | two pilots, about 120,000 | the same |
| 2 | one retainer, about 60,000 | four retainers, about 400,000 | six to eight, about 0.8 to 1.2 million |
| 3 | two retainers, about 150,000 | eight retainers, about 1.2 million | one vertical, 2.5 to 4 million |

Year 2 at about 400,000 does not cover a 1.25 million burn. The raise is the bridge. A consumer product, a multi-tenant cloud, and a claim of thermodynamic advantage sit outside all three columns. The specification’s Year-1 revenue of 1.2 to 1.8 million and Year-3 revenue of 9 to 12 million are not the memorandum’s table. This report does not adopt them.

## 14. Products named, and the boundary around them

The November specification names an education product and a security product as places the reliability layer might later sit. Those names are product intentions. They are not revenue, and they are not demonstrated by the thermometer exam. A library inside one existing graphics pipeline is the wedge the memorandum will fund. Anything that waits on a finished sentence-filter waits on a later experiment.

## 15. Risks

The thermometer result is a public daily series with noise the notebook added, not a customer’s sensor. The half-variance bar missed on that series. The glitch screen was easy by design. The language sheets do not support a product claim. An incumbent can ship a blur faster than a new company can ship a library. The advantage shown is the placement, in the same pass, and the preservation of the center when the reading is shifted. Deep-tech schedules slip. The next money is for a library and for one partner’s own series.

## 16. What counsel and an investor can hold at once

An investor can hold a night on which 17 of 18 checks passed, a real temperature that did not move when it was shifted, a fused kernel faster than an eager loop, four boards doing four times the jobs, temperatures under 50 °C, and one missed bar written in the same table. Counsel can hold the same night as a specification: a density, a ladder, a center-keeping map, an agreement test, a sign bit, and a short message. Both can hold, in a separate section, the sentence experiments and the L4 log, without being asked to treat them as the same result as the roofline.

The line the memorandum ends on is the line this report keeps. The absolute, the center, remains. What is taken away is the disturbance around it, and only where the measurement says so.

<!-- computed-appendix -->

## 17. A worked sweep of the shift identity

This section evaluates the flux map and the raw map on a single reading, on the CPU, with λ = 1. The reading is 25.7 and the local center is 20.7, so the deviation is 5. The same reading and the same center are then lifted by 5. Nothing here is a new graphics-processor run, and nothing here is the Melbourne file. The point of writing every row is that an investor can recompute any cell with a calculator and see the identity that the exam later recorded on real buffers.

The target for the error is the center itself, which is the right comparison when the question is how far the damped reading sits from the level. Flux error is |A − μ|. After the lift, the target lifts too, and the flux error is |A′ − (μ+5)|. Raw error is |R − μ| before the lift and |R′ − (μ+5)| after it. Raw looks better at large instability only because it has pulled the reading toward zero, which on this toy number happens to pass near the old center for one particular ξ and then misses the lifted center by a wide margin.

| ξ | Flux | Flux error | Flux after +5 | Raw | Raw error | Raw after +5 | Raw error after |
|---|---|---|---|---|---|---|---|
| 0.05 | 25.4561 | 4.7561 | 30.4561 | 24.4466 | 3.7466 | 29.2027 | 3.5027 |
| 0.10 | 25.2242 | 4.5242 | 30.2242 | 23.2543 | 2.5543 | 27.7785 | 2.0785 |
| 0.15 | 25.0035 | 4.3035 | 30.0035 | 22.1202 | 1.4202 | 26.4237 | 0.7237 |
| 0.20 | 24.7937 | 4.0937 | 29.7937 | 21.0414 | 0.3414 | 25.1350 | 0.5650 |
| 0.25 | 24.5940 | 3.8940 | 29.5940 | 20.0152 | 0.6848 | 23.9092 | 1.7908 |
| 0.30 | 24.4041 | 3.7041 | 29.4041 | 19.0390 | 1.6610 | 22.7431 | 2.9569 |
| 0.35 | 24.2234 | 3.5234 | 29.2234 | 18.1105 | 2.5895 | 21.6339 | 4.0661 |
| 0.40 | 24.0516 | 3.3516 | 29.0516 | 17.2272 | 3.4728 | 20.5788 | 5.1212 |
| 0.45 | 23.8881 | 3.1881 | 28.8881 | 16.3870 | 4.3130 | 19.5752 | 6.1248 |
| 0.50 | 23.7327 | 3.0327 | 28.7327 | 15.5878 | 5.1122 | 18.6205 | 7.0795 |
| 0.55 | 23.5847 | 2.8847 | 28.5847 | 14.8276 | 5.8724 | 17.7124 | 7.9876 |
| 0.60 | 23.4441 | 2.7441 | 28.4441 | 14.1045 | 6.5955 | 16.8485 | 8.8515 |
| 0.65 | 23.3102 | 2.6102 | 28.3102 | 13.4166 | 7.2834 | 16.0268 | 9.6732 |
| 0.70 | 23.1829 | 2.4829 | 28.1829 | 12.7622 | 7.9378 | 15.2452 | 10.4548 |
| 0.75 | 23.0618 | 2.3618 | 28.0618 | 12.1398 | 8.5602 | 14.5017 | 11.1983 |
| 0.80 | 22.9466 | 2.2466 | 27.9466 | 11.5478 | 9.1522 | 13.7944 | 11.9056 |
| 0.85 | 22.8371 | 2.1371 | 27.8371 | 10.9846 | 9.7154 | 13.1216 | 12.5784 |
| 0.90 | 22.7328 | 2.0328 | 27.7328 | 10.4488 | 10.2512 | 12.4817 | 13.2183 |
| 0.95 | 22.6337 | 1.9337 | 27.6337 | 9.9392 | 10.7608 | 11.8729 | 13.8271 |
| 1.00 | 22.5394 | 1.8394 | 27.5394 | 9.4545 | 11.2455 | 11.2939 | 14.4061 |
| 1.05 | 22.4497 | 1.7497 | 27.4497 | 8.9934 | 11.7066 | 10.7431 | 14.9569 |
| 1.10 | 22.3644 | 1.6644 | 27.3644 | 8.5548 | 12.1452 | 10.2191 | 15.4809 |
| 1.15 | 22.2832 | 1.5832 | 27.2832 | 8.1376 | 12.5624 | 9.7207 | 15.9793 |
| 1.20 | 22.2060 | 1.5060 | 27.2060 | 7.7407 | 12.9593 | 9.2467 | 16.4533 |
| 1.25 | 22.1325 | 1.4325 | 27.1325 | 7.3632 | 13.3368 | 8.7957 | 16.9043 |
| 1.30 | 22.0627 | 1.3627 | 27.0627 | 7.0041 | 13.6959 | 8.3667 | 17.3333 |
| 1.35 | 21.9962 | 1.2962 | 26.9962 | 6.6625 | 14.0375 | 7.9587 | 17.7413 |
| 1.40 | 21.9330 | 1.2330 | 26.9330 | 6.3375 | 14.3625 | 7.5705 | 18.1295 |
| 1.45 | 21.8729 | 1.1729 | 26.8729 | 6.0285 | 14.6715 | 7.2013 | 18.4987 |
| 1.50 | 21.8157 | 1.1157 | 26.8157 | 5.7344 | 14.9656 | 6.8501 | 18.8499 |
| 1.55 | 21.7612 | 1.0612 | 26.7612 | 5.4548 | 15.2452 | 6.5160 | 19.1840 |
| 1.60 | 21.7095 | 1.0095 | 26.7095 | 5.1887 | 15.5113 | 6.1982 | 19.5018 |

At instability 0.05 the shrinking factor exp(−ξ) is 0.951229. Flux returns 25.4561, which is the center 20.7 plus 5 times that factor. The flux error is 4.7561. After the lift, flux returns 30.4561, exactly 5 above the first flux value, and the error against the lifted center is the same 4.7561. Raw returns 24.4466 before the lift, an error of 3.7466 against 20.7, and 29.2027 after the lift, an error of 3.5027 against 25.7. The raw error grows because the lift itself was multiplied by 0.951229 instead of being kept. This is the same algebra as A(y+c) = A(y)+c and R(y+c) = R(y)+c·exp(−λξ), with λ = 1.
At instability 0.25 the shrinking factor exp(−ξ) is 0.778801. Flux returns 24.5940, which is the center 20.7 plus 5 times that factor. The flux error is 3.8940. After the lift, flux returns 29.5940, exactly 5 above the first flux value, and the error against the lifted center is the same 3.8940. Raw returns 20.0152 before the lift, an error of 0.6848 against 20.7, and 23.9092 after the lift, an error of 1.7908 against 25.7. The raw error grows because the lift itself was multiplied by 0.778801 instead of being kept. This is the same algebra as A(y+c) = A(y)+c and R(y+c) = R(y)+c·exp(−λξ), with λ = 1.
At instability 0.50 the shrinking factor exp(−ξ) is 0.606531. Flux returns 23.7327, which is the center 20.7 plus 5 times that factor. The flux error is 3.0327. After the lift, flux returns 28.7327, exactly 5 above the first flux value, and the error against the lifted center is the same 3.0327. Raw returns 15.5878 before the lift, an error of 5.1122 against 20.7, and 18.6205 after the lift, an error of 7.0795 against 25.7. The raw error grows because the lift itself was multiplied by 0.606531 instead of being kept. This is the same algebra as A(y+c) = A(y)+c and R(y+c) = R(y)+c·exp(−λξ), with λ = 1.
At instability 0.75 the shrinking factor exp(−ξ) is 0.472367. Flux returns 23.0618, which is the center 20.7 plus 5 times that factor. The flux error is 2.3618. After the lift, flux returns 28.0618, exactly 5 above the first flux value, and the error against the lifted center is the same 2.3618. Raw returns 12.1398 before the lift, an error of 8.5602 against 20.7, and 14.5017 after the lift, an error of 11.1983 against 25.7. The raw error grows because the lift itself was multiplied by 0.472367 instead of being kept. This is the same algebra as A(y+c) = A(y)+c and R(y+c) = R(y)+c·exp(−λξ), with λ = 1.
At instability 1.00 the shrinking factor exp(−ξ) is 0.367879. Flux returns 22.5394, which is the center 20.7 plus 5 times that factor. The flux error is 1.8394. After the lift, flux returns 27.5394, exactly 5 above the first flux value, and the error against the lifted center is the same 1.8394. Raw returns 9.4545 before the lift, an error of 11.2455 against 20.7, and 11.2939 after the lift, an error of 14.4061 against 25.7. The raw error grows because the lift itself was multiplied by 0.367879 instead of being kept. This is the same algebra as A(y+c) = A(y)+c and R(y+c) = R(y)+c·exp(−λξ), with λ = 1.
At instability 1.25 the shrinking factor exp(−ξ) is 0.286505. Flux returns 22.1325, which is the center 20.7 plus 5 times that factor. The flux error is 1.4325. After the lift, flux returns 27.1325, exactly 5 above the first flux value, and the error against the lifted center is the same 1.4325. Raw returns 7.3632 before the lift, an error of 13.3368 against 20.7, and 8.7957 after the lift, an error of 16.9043 against 25.7. The raw error grows because the lift itself was multiplied by 0.286505 instead of being kept. This is the same algebra as A(y+c) = A(y)+c and R(y+c) = R(y)+c·exp(−λξ), with λ = 1.
At instability 1.50 the shrinking factor exp(−ξ) is 0.223130. Flux returns 21.8157, which is the center 20.7 plus 5 times that factor. The flux error is 1.1157. After the lift, flux returns 26.8157, exactly 5 above the first flux value, and the error against the lifted center is the same 1.1157. Raw returns 5.7344 before the lift, an error of 14.9656 against 20.7, and 6.8501 after the lift, an error of 18.8499 against 25.7. The raw error grows because the lift itself was multiplied by 0.223130 instead of being kept. This is the same algebra as A(y+c) = A(y)+c and R(y+c) = R(y)+c·exp(−λξ), with λ = 1.

### Forty passes at ξ = 0.1

Start from the same deviation of 5, center 20.7, instability 0.1. Each pass replaces the deviation by deviation times exp(−0.1). The closed form after n passes is 20.7 + 5·exp(−0.1 n).

| Pass | Deviation remaining | Value | Closed form | Absolute gap |
|---|---|---|---|---|
| 1 | 4.52418709 | 25.22418709 | 25.22418709 | 0 |
| 2 | 4.09365377 | 24.79365377 | 24.79365377 | 0 |
| 5 | 3.03265330 | 23.73265330 | 23.73265330 | 0 |
| 10 | 1.83939721 | 22.53939721 | 22.53939721 | 0 |
| 20 | 0.67667642 | 21.37667642 | 21.37667642 | 0 |
| 40 | 0.09157819 | 20.79157819 | 20.79157819 | 0 |

After forty passes the deviation remaining is 0.09157819 and the value is 20.79157819. The kernel check on one board allowed a maximum absolute gap of 9.5×10⁻⁷ against this closed form on a long buffer, not on this one-number illustration. The illustration is here so the algebra is visible before the buffer result is trusted.

## 18. Eight sign-bit rows, computed here

Eight float32 samples were drawn from a normal generator with seed 11, which is the exam’s synthetic seed and not the exam’s buffer. Each sample’s bits were exclusive-ored with 0x80000000 as an unsigned constant, and the synthesis one-half of the sample plus the flip was formed. The synthesis is the center when the sample is a pure deviation. Here the center is zero, so the synthesis should be zero.

| Sample | Sign flip | One half of the sum |
|---|---|---|
| 0.03419277 | -0.03419277 | 0.00000000e+00 |
| 1.35974753 | -1.35974753 | 0.00000000e+00 |
| 1.22472107 | -1.22472107 | 0.00000000e+00 |
| -0.51030707 | 0.51030707 | 0.00000000e+00 |
| -0.29796952 | 0.29796952 | 0.00000000e+00 |
| -0.52738422 | 0.52738422 | 0.00000000e+00 |
| 0.56972635 | -0.56972635 | 0.00000000e+00 |
| -0.05606444 | 0.05606444 | 0.00000000e+00 |

The largest absolute synthesis on these eight rows is 0.000e+00. The labelled exam recorded a sign residual of 0 on the boards, and the 5,000,000-state block recorded variance 2.4656 before the same override and 0.0 after it, in 10.1 ms. Those are the hardware figures. The eight rows are a hand check of the bit operation.

## 19. The roofline arithmetic, written out

Fifty million nodes times 16 bytes is 800,000,000 bytes. The one-board median is 0.54828799 ms, which is 0.00054828799 seconds. Dividing bytes by seconds and by 10⁹ gives 1459.087222 GB/s. The file records 1,459.0872277590881 GB/s. The difference between the hand division and the file is rounding in the median’s printed digits; the file is the source. Dividing 1459.0872 by the stated peak of 1597 gives 91.3643 percent. The file’s percentage is 91.36425972192163.

The twelve-byte count, store omitted, is 600,000,000 bytes over the same median, 1094.3154 GB/s. The file records 1,094.3154208193162 GB/s. A slide that quotes the twelve-byte number as if it included the write is overstating the work. The sixteen-byte number is the one that matches three loads and one store only if the kernel’s actual traffic is four 4-byte values per node. The memorandum says the corrected traffic count is the sixteen-byte count. This report follows that correction.

The scalar-message cell is a different division. Sixteen bytes transferred, 319,999,984 bytes not sent relative to a full reduction, kernel 17.5875 ms, scalar sync 3.7208 ms, and 54.58 GB/s stored beside that message. 54.58 is not 1,459. An investor memo that adds them has mixed a long traversal with a short message.

Launch versus payload. Near ten thousand points the median sits around 0.012 ms because starting the kernel costs about that much. At fifty million points the median is 0.548 ms, about forty-five times longer, which is the payload. A claim of constant time is true only inside the launch-bound regime, and the specification’s 0.0117 to 0.0125 ms band should be read as that regime unless a file shows the same band at fifty million nodes. This repository’s fifty-million-node file does not show 0.012 ms.

## 20. Every cut of the ladder

Density is the mean absolute instability. The width changes at 0.5, at 1, and at 2. The labelled densities in the exam were 0.30, 0.79, 1.50, and 7.06. Julia and Python both returned 128, 256, 512, and 1024. The rows below are the rule, including the boundaries, so a later programmer does not guess whether 0.50 is the lower rung or the next one. In this statement a density below 0.5 is 128, below 1 is 256, below 2 is 512, and every remaining density, including 2, is 1024.

| Density | Width | Why |
|---|---|---|
| 0.00 | 128 | below 0.5 |
| 0.30 | 128 | below 0.5 |
| 0.49 | 128 | below 0.5 |
| 0.50 | 256 | at least 0.5 and below 1 |
| 0.79 | 256 | at least 0.5 and below 1 |
| 0.99 | 256 | at least 0.5 and below 1 |
| 1.00 | 512 | at least 1 and below 2 |
| 1.50 | 512 | at least 1 and below 2 |
| 1.99 | 512 | at least 1 and below 2 |
| 2.00 | 1024 | at least 2 |
| 7.06 | 1024 | at least 2 |

On the one-million-point sweep the times were 0.0157 ms at 128, 0.0120 ms at 256, 0.0131 ms at 512, and 0.0116 ms at 1024. The ladder’s choice of 256 was 3.6 percent behind the fastest of those four numbers, inside a 10 percent bar. At that length the ranking is not a law of the chip. The law the exam checked was agreement on the width.

The gate predicate is a different cut. A stability index of 0.05 is closed. 0.20 is open. 0.80 is open. The host is specified not to launch when the predicate is closed. That predicate was not the rule that failed on Hydrogen and Atlantic. Those failures used other functions of a logit vector, described in section 12.

## 21. The eighteen headings, one by one

The November 2026 specification says its eighteen equations were numerically verified, and it groups the last six as supporting mechanisms with everyday analogies. This section repeats each heading in language a diligence reader can check. Where the labelled exam has a residual, the residual is given. Where it does not, the section says so. Repeating a heading is not the same act as endorsing the specification’s strongest sentence about that heading.

### 3.1 Landauer’s bound

The November 2026 specification writes ΔQ ≥ k_B T ln 2 for each irreversible erasure of one bit.

k_B is 1.380649×10⁻²³ joules per kelvin. At 298.15 K, k_B T ln 2 is about 2.87×10⁻²¹ joules. At 300 K it is about 2.89×10⁻²¹ joules. That is the reference cost of erasing one bit in the Landauer accounting, not a quantity the labelled exam subtracted from the board’s electrical draw. The boards drew hundreds of watts. A watt is a joule per second. The Landauer reference for one bit is twenty orders of magnitude smaller than one second of board power. An investor who is told that the architecture “saves Landauer energy” should ask to see a joule meter on the erasure itself. This report does not have that meter. What it has is electrical draw, temperature, and a floating-point identity: a deviation averaged with its sign flip is the center. The specification’s sentence that erasure is avoided entirely, and that the system never writes a wrong answer, is a design sentence in that PDF. The sentence sheets in section 12 are the reason it is not repeated here as a result.

### 3.2 The omega-dit integral

The specification writes |Ω⟩ = ∫ σ(t) e^{iβ(t)} [υ(t) + ξ(t)] dω.

σ is called an existence density, β a phase, υ a structural value, and ξ a residual. The integral says: keep the steady part and the leftover in one object, with a phase, until a later step cancels the leftover. The code path that corresponds to this description is a discrete sum of those factors, not a measured continuous field. Nothing in the labelled-exam JSON is a sample of a physical wave named Ω. Diligence can treat the integral as the inventor’s notation for “do not threshold the series before the center and the deviation have been separated.” That separation is exactly what the flux map does, in a simpler notation: μ is kept, and (y − μ) is scaled. The exponential scale is a soft cancellation. The sign-bit average is a hard cancellation of a centered deviation. Both sit under this heading. They are different knobs, and the thermometer result belongs to the soft knob.

### 3.3 The split into a level and a leftover

The specification writes a direct sum of a structural subspace and a residual subspace.

On the chips that ran the exam, the structural part is the local center and the residual is the deviation from that center. No infinite-dimensional inner product was measured. The operational test of the split is the shift identity. If the leftover is really only the deviation, lifting every reading and the center by the same constant must lift the output by that constant and must leave the error unchanged. Flux did that on the synthetic series and on the Melbourne series. Raw damping, which multiplies the whole reading, did not. A reader who wants a picture can use the counsel brief’s: the song stays at its loudness, the clatter is turned down. A moving average turns both down. Raw damping turned the temperature down. Flux did not.

### 3.4 Thesis

The specification writes the thesis state as the structural value plus the disturbance.

If the program is handed both parts, the thesis is their sum. If it is handed only the series, the series already is that sum, and the program must estimate the split. The flux map estimates the structural value as a local mean. The quality of the result then depends on the quality of the mean, which is why a half-variance bar can pass on a synthetic series and miss on a thermometer. The algebra of “plus the disturbance” is not the same statement as “any estimate of the disturbance is accurate.”

### 3.5 Conjugation by the sign bit

The specification implements conjugation as an IEEE-754 sign-bit exclusive-or, and also writes it as a difference of projectors.

In float32 the sign is the high bit. Exclusive-or with 0x80000000 flips that bit and leaves the magnitude. The constant must be an unsigned 32-bit value. A signed 32-bit interpretation of the same bits overflows before the operation. On the labelled exam the residual of this negation against ordinary unary minus was 0 on the boards that were scored. On 5,000,000 states the telemetry summary records the same residual, 0.0. This is a format identity. It does not require a new instruction beyond a bitwise operation and a bitcast.

### 3.6 Antithesis

The specification writes the antithesis as the structural value minus the disturbance.

If conjugation has flipped only the disturbance, the antithesis is the thesis with that disturbance negated. Adding thesis and antithesis then cancels the disturbance. If conjugation is applied to the whole series rather than to the deviation, the average is zero, which is the wrong answer on a thermometer: the day would be reported as 0 °C. The enabling detail is what gets flipped. The deviation gets flipped, or, in the soft map, scaled. The center does not.

### 3.7 The projector

The specification writes Υ = ½ (I + Ĉ_S).

Applied to a deviation, this is one half of the deviation plus its sign flip. For a finite float32 deviation the two magnitudes match and the sum is zero, so the reconstructed sample is the center. Applied to a value that still contains the center, the same formula pulls the sample toward zero. That is the failure mode the shift test was written to catch. The projector is correct on the object it is defined on. The implementation has to put the center back, or apply the projector only to the deviation.

### 3.8 The final state

The specification writes the final state as one half of thesis plus antithesis, equal to the structural value, in one fused multiply-add.

When both parts are supplied and the negation is exact, the residual to the supplied structural value is on the order of a unit in the last place or smaller, and the unit tests of that algebra record a residual at that scale. The forty-pass flux kernel is a different check: it matches μ + (y − μ) exp(−40 ξ) within 9.5×10⁻⁷ on one board. The specification’s “one fused multiply-add” describes the hard cancellation. The scored forty-pass kernel is the soft map, repeated. Both are in the disclosure. A latency quoted for one of them is not a latency for the other. The fused soft kernel on the tiled thermometer took 7.38 ms. The eager loop took 17.13 ms. The specification’s median near 0.012 ms is a short-buffer figure in that document, consistent with launch-bound kernels, and it is not 7.38 ms and not 0.548 ms.

### 3.9 Entropy

The specification says that driving residual variance toward zero collapses an entropy.

Variance is what the exam measured. Entropy of a density matrix was not measured. On the thermometer, variance did not go to zero. About 51 to 52 percent remained, and the pre-declared bar of 50 percent was missed. On the synthetic series about 11.7 percent remained, and that bar was met. On the 5,000,000-state sign-bit block the recorded variance went from 2.4656 to 0.0, which is the hard cancellation, not the thermometer. An entropy claim that says “the state becomes pure” should not be hung on the thermometer row.

### 3.10 Free energy

The specification writes F = U − T S and says a proxy falls toward a minimum.

U and S were not instrumented as thermodynamic internal energy and entropy. The electrical proxy that was instrumented is the board draw: 314.5 W mean on the one-board thermometer burn, peak 326.6 W, and about 277 to 283 W mean on each of the four boards. Temperatures stayed at or below 48 °C. Those are healthy operating numbers against a 600 W nameplate. They are not a measurement that F collapsed from thousands to zero. If the specification’s proxy is a numerical free-energy stand-in computed from variance, it should be plotted from the same file as the variance, and labelled as a stand-in.

### 3.11 The stability coefficient

The specification writes Ξ = 1 − Var(logits) / (E[|logits|] + ε).

On the Qwen2.5-1.5B logits that were actually scored, this coefficient was negative, about −3.55 to −1.81, because the variance of a wide logit vector exceeds the mean absolute value. A horizon of 0.98 then abstains on every question, right or wrong. A different algebra that uses a signed mean instead of the mean absolute value produced large positive coefficients and also failed to separate true answers from false ones. The coefficient is a real function. On those sheets it was not a detector. The gate predicate in the numerical exam, which opens at a stability index of 0.20, was scored on labelled scalars, not on these logit rows. The two uses of the word “stability” should not be glued together in a claim.

### 3.12 The dual-circuit scale

The specification writes χ = λ0 exp(−κ(1−Ξ)) + Λmax · sigmoid(Var/σ² − θ), with λ0 = 10, κ = 0.5, θ = 0.98, Λmax = 50 in the notebooks.

On a full vocabulary the sigmoid term saturates and χ sits near 51 or 52. One scale applied to every logit does not change the argmax. The winning token stays the winning token, until float32 underflow, which produced garbage text rather than a corrected answer. A volume knob that turns the whole mix down does not remove one voice. The flux map is different because it does not scale the center. Any write-up that treats χ and the flux exponential as the same mechanism has collapsed that distinction.

### 3.13 Rescaling and the argmax

The specification groups logit rescaling with the supporting mechanisms.

If every logit is multiplied by the same positive factor, the softmax winner is unchanged and the probabilities change only by a monotone transform of the gaps. That is a theorem of the softmax, not a Blackwell result, and it is why a global χ cannot be the hallucination filter. The L4 governor does something else: it looks at a collision and a gap, and on rejection it crops the cache. That can change the continuation, especially when the host then inserts a reference string. The change comes from the crop and the reference, not from a global scale.

### 3.14 A logarithmic residual map

The specification names a Weber-Fechner logarithmic map of the residual.

Weber-Fechner, in psychology, is the observation that perceived intensity often tracks the logarithm of a stimulus. The specification offers it as an analogy for a compressive map of a residual. No labelled exam row reports a log-residual error beside the flux error. This report does not invent one. If counsel wants the log map enabled, it needs a formula, a tolerance, and a file. Until then it stays a named mechanism in the specification, not an example.

### 3.15 Two-mode squeezing

The specification names two-mode squeezing as variance redistribution without information loss.

In quantum optics, squeezing moves uncertainty from one variable to its partner. The specification uses that phrase for a numerical redistribution. The labelled exam does not contain a squeezing parameter or a pair of quadrature variances. The sign-bit average does the opposite of hiding variance in a partner: it zeroes a centered deviation. The flux map reduces the deviation by a known factor exp(−λξ) and leaves the center. Neither operation was recorded as an optical squeezing measurement. The phrase should not be read as a claim that the GPU performed a quantum-optics experiment.

### 3.16 An internal consistency check

The specification names an inherence check: consistency without an external table.

The check the exam actually ran is internal in a precise sense. The same input twice must match, and it did, with drift 0. The kernel must match the closed form. Julia and Python must return the same width. JAX must match the written derivative. Those are consistency checks against a formula the host already holds, not against a labelled answer key of English questions. They do not look up Melbourne’s temperature in a second source. They ask whether the machine did the arithmetic it was given.

### 3.17 The mirror product

The specification names mirror validation by complex conjugation.

The code forms the product of a complex value with its conjugate, which is the squared modulus, a non-negative real. That product is a useful checksum: it is unchanged by a pure phase, and it is zero only at zero. It is not, by itself, the flux map. A diligence reader can find it in the yoga-buddhi module. It should be claimed, if at all, as a conjugate-magnitude check, not as the thermometer result.

### 3.18 A fabric term

The specification names a fabric Hamiltonian that prefers neighboring terms to agree in direction.

No labelled-exam score is a lattice energy. The nearest operational cousin is the local center itself: a neighborhood agrees on a level, and the deviation is what disagrees. A smoothing prior that penalizes neighbor disagreement would be a different algorithm and would need its own residual. This report does not smuggle that prior into the 17-of-18 score.

## 22. Questions a first meeting will ask

The answers below are the answers this report will support. They are not a script for exaggerating the result, and they are not an offer of securities.

### What, in one paragraph, was measured?

On 14 November 2026 a fixed exam of a center-preserving damping map passed 17 of 18 questions on one RTX PRO 6000 Blackwell Server Edition and 16 of 17 on four of them. The miss was the same on both: flux left about 51 to 52 percent of the variance on a contaminated Melbourne daily-minimum series, against a bar of 50 percent. On that series the flux error stayed near 5.82 to 5.87 when the readings were shifted by 5 °C, and a map that multiplies the whole reading rose from about 135 to about 266. A fused forty-pass kernel took 7.38 ms against an eager 17.13 ms. Four boards completed 549.2 heavy jobs a second against 137.3 on one board. At 50 million nodes one board moved 1,459.0872 GB/s, 91.36 percent of the stated 1,597 GB/s peak.

### What should an investor not repeat?

Do not repeat that the method stops hallucinations. Do not repeat 958,874 times, 93.94 percent, or a cosine of 0.979 as if they were this exam. Do not add 54.58 GB/s to 1,459 GB/s. Do not say NVLink was measured. Do not say the L4 notebook measured the conductivity of diamond. Do not quote a 7.5 to 10 million dollar seed as Ceraph’s ask. The planning band in the 14 November memorandum is 1 to 2 million US dollars, with a working figure of 1.25 million and a design partner on the order of 60,000 dollars a year, none of it booked.

### Why is four boards not four times faster on a single wave?

Over five minutes the job rate was four times, 549.2 against 137.3, because each board did its own jobs for the whole interval. On one wave of four jobs, the queued time was 29.51 ms and the concurrent time was 8.28 ms, a ratio of about 3.56. The shortfall from 4.0 is the spread among boards and the host’s launch, not a failure of the arithmetic. The 15 November fair clock, taken after compilation, was 29.27 ms queued and 8.24 ms on the slowest board’s own event clock. The wall clock around the Python threads was 11.79 ms and includes the join. A clock taken before compilation had finished on the idle boards was 38.14 ms and is not the result.

### What is the local center, in practice?

It is a level estimated from a short neighborhood, or a level supplied with the series. The map adds that level back after scaling only the departure from it. If the neighborhood is a poor estimate, the error stays large even though the shift identity holds. That is the thermometer: the identity held, and half the variance remained. A customer series has to be scored the same way, with the bar written down before the run. The company’s next technical milestone is that scoring on a partner’s own readings, not a new slogan.

### What would a design partner actually receive?

A small library: the ladder, the flux map, the agreement check, and a way to time it without mixing compilation into the clock. The partner is assumed, for planning, to already have a graphics pipeline. The memorandum prices a year of that relationship on the order of 60,000 US dollars once the library exists and a provisional is on file. No such agreement is signed. The 40,000 dollars in the use-of-funds table is the company’s cost of showing up, not the partner’s payment.

### Where does the money go?

Of the 1,250,000 dollar working ask, people are 672,000, counsel and the entity 160,000, cloud 144,000, the public face and the partner program 110,000, bench and recruiting 50,000, and the reserve 114,000. The lean alternative is 700,000 dollars for 12 months. Year-2 planning income of about 400,000 dollars does not cover a 1.25 million burn. The raise is the bridge. A consumer product, a multi-tenant cloud, and a thermodynamic-advantage story are outside the memorandum’s three income columns.

### What is omega-dit, without the integral?

Separate the level from the leftover. Invert or scale only the leftover. Put the level back. Check that a second program did the same thing. If several chips are working, send a short state, not the whole array. The integral, the entropy, and the free-energy sentence are the inventor’s longer description of that sequence. The sequence a chip has already run is the flux map, the sign-bit average, the ladder, and the 16-byte message.

### What did the language tests actually decide?

They decided that several plausible refusal rules do not yet separate a wrong answer from a right one on the sheets that were run. A global scale preserves the winning token. A very sensitive phase rule destroyed right answers. An entropy rule that looked successful was comparing a taught prompt with an untaught prompt. A dialectical rule removed three invented answers, also removed a correct Pacific, left TruthfulQA unchanged at 23 of 100, and lowered ARC-Challenge from 50 to 44 of 100. That is why the raise is not priced as a finished refusal product.

### What is the L4 notebook, then?

It is a logged governor on a different chip. A kernel reduces a logit row. If a host rule rejects the step, the cache is cropped and a sentence the host already typed can be inserted. With that crop, one trial printed the host’s diamond-and-copper sentence after a split between “lower” and “higher.” Without the crop, earlier trials repeated a token. A third trial still misstated Mercury’s year. The kernel time 0.1158 ms and the batch-32 figure 2.76 microseconds are L4 prints. They are not the Blackwell roofline.

### What are the main technical risks?

The thermometer is a public series with noise the notebook added, not a customer’s sensor. The half-variance bar missed. The glitch screen was easy by design. An incumbent can ship a blur. The advantage to defend is the placement of the center-preserving map in the same pass, the agreement test, and the shift identity. Deep-tech schedules slip. The language claim is already retired from the core, which removes one risk and also removes one popular slide.

### What should the provisional contain?

The flux map, the ladder, the agreement rule, the sign bit, the scalar message, the measured examples including the miss, and the omega-dit algebra clearly marked where it is design rather than the exam. The logit crop can be a separate embodiment with the L4 log beside it. Semantic filtering should not be an independent claim. Counsel has a full draft in the patent packet that accompanies this report.

### Who is the company?

Ceraph Corporation, Kerala, India. chief@ceraphco.com. www.ceraphco.com. As of the 14 November memorandum there was no design-partner revenue. This report does not create any. Correspondence about the mathematics can use the file names in the repository. Correspondence about a check should not expect a security to be offered by this document.

## 23. Glossary

| Term | In this report |
|---|---|
| Abhaya, in the code | The name of the damping map that keeps a local center. In the note to Swami Sandeepananda Giri the same word is used in its ordinary sense, the absence of fear. The code does not claim the ordinary sense. |
| Sphota, in the code | The ladder that picks 128, 256, 512, or 1024. The traditional word is about meaning arising. The macro `@sphota` returns a width. |
| Omega-dit | The inventor’s name for the cancellation chain: split, conjugate the leftover, average. The continuous integral is the specification’s notation. The discrete operations are what the tests run. |
| Flux map | A(y) = μ + (y − μ) exp(−λξ), with λ = 1 in every scored test. |
| Raw map | R(y) = y exp(−λξ). The comparison that fails the shift. |
| Local center | μ. A level. On the thermometer, the day’s level rather than the shake. |
| Instability | ξ. How wild the neighborhood is. Non-negative. |
| Density | Mean absolute instability. The ladder’s input. |
| Closed form | μ + (y − μ) exp(−40ξ), the algebra of forty passes at fixed ξ. |
| Sign bit | The high bit of a float32. Flipped by exclusive-or with unsigned 0x80000000. |
| Synthesis | One half of a value plus its sign flip. On a deviation, the center. |
| Agreement | Julia with Python on the width, JAX with the derivative, Triton with the formula, the kernel with the closed form. |
| Roofline count | 1,459.0872 GB/s at 50 million nodes, 16 bytes per node, 91.36 percent of 1,597. |
| Scalar message | 16 bytes transferred in a different cell, 54.58 GB/s beside that cell, 319,999,984 bytes not sent. |
| V1 | The missed question. Variance remaining 0.5149549, 0.5176691, 0.5095068. |
| Fair clock | 15 November. Queued 29.2718 ms. Slowest event 8.2376 ms. Wall 11.7879 ms. Finite. |
| Unfair clock | Compilation still inside the parallel wall. Together 38.1371 ms. Not the result. |
| L4 governor | Logit reduction, cache crop, optional host reference. Not the four-board flux timing. |
| Collision C | Sum of squared probabilities on a logit row. Printed in the L4 log. |
| Design partner | A planning figure on the order of 60,000 US dollars a year. Unbooked. |
| Working ask | 1,250,000 US dollars for 18 months, inside a 1 to 2 million band. Not the specification’s 7.5 to 10 million. |
| Nameplate | 600 W per RTX PRO 6000 board. The burns sat near 280 to 315 W. |

## 24. Files a diligence room can open

The labelled one-board and four-board exam records are the JSON files the 14 November memorandum cites. The fair and unfair four-board clocks are `benchmarks/rtx_pro_6000_blackwell/four_vs_one_heavy_fair.json` and `four_vs_one_heavy.json`. The sign-bit block and the roofline summary live in `benchmarks/rtx_pro_6000_blackwell/hardware_tensor_telemetry.json`. The dialectical sheet is `yoga_buddhi_dialectical_audit.json` in the same directory. The flux map is `sri/omlang/abhaya.py`. The thesis and the conjugate product are `sri/omlang/yoga_buddhi.py`. The host protocol, including the rule that a printed latency budget is not an input, is `experiments/dgx_abhaya_round.py`.

The November 2026 full technical specification is a four-page PDF supplied with this compilation. Its 958,874-times row, its 93.94 percent estimator, its cosine bound, and its seed table are not duplicated into those JSON files by this report. The 14 November investor memorandum and the counsel brief of the same night are the narrative sources for sections 1 through 16, together with the 15 November fair clock and the later language sheets.

## 25. A reproduction outline

On a machine with one or four RTX PRO 6000 boards, install a PyTorch build that matches the CUDA driver. The 15 November machine used torch 2.14.1 with CUDA 13.0 because torch 2.15.0 was not on that wheel index; Triton 3.8.0 came with it. Load the series or the random job at length 16,777,216. Compile and warm the forty-pass kernel on every device that will be timed. Then time. For the queue, run four jobs in order on one device. For the concurrent figure, run one job on each device and read CUDA events. Do not put allocation inside that window.

For the identity, compute flux and raw at offset 0 and offset 5 with λ = 1 and confirm the flux error matches and the raw error does not. For the ladder, score densities 0.30, 0.79, 1.50, and 7.06. For the sign bit, use an unsigned constant. For the closed form, compare forty passes with exp(−40ξ). Record the variance remaining on the thermometer even if it misses 0.5. A reproduction that drops V1 is not a reproduction.

Language sheets, if repeated, should keep the prompt format identical between the columns being compared. A few-shot frame against a raw frame is a different experiment and has to be labelled as one.

## 26. Status table

| Statement | Where it lives | How this report uses it |
|---|---|---|
| Flux error unchanged by +5 °C on Melbourne | labelled exam | Example and claim support |
| Raw error about 135 then about 266 | labelled exam | The comparison |
| Variance remaining about 0.51 to 0.52 | labelled exam | The miss, kept in the table |
| 17 of 18 and 16 of 17 | labelled exam | The score |
| 7.38 ms fused, 17.13 ms eager | labelled exam | Kernel comparison |
| 137.3 and 549.2 jobs per second | labelled exam | Five-minute rate |
| 29.51 ms queued, 8.28 ms together | labelled exam | One wave on 14 November |
| 29.27 ms queued, 8.24 ms slowest event | 15 November fair file | The clock after compilation |
| 1,459.0872 GB/s, 91.36 percent | labelled exam | Roofline |
| 54.58 GB/s and 16 bytes | scaling cell | A different measurement |
| Sign residual 0, variance 2.4656 to 0.0 in 10.1 ms | telemetry summary | Hard cancellation |
| 958,874 times, 93.94 percent, cosine ≥ 0.979 | November specification only | Disclosed as that PDF, not as the exam |
| 0.0117 to 0.0125 ms | November specification | Short-buffer band, not 50 million nodes |
| L4 0.1158 ms, crop, host reference | L4 notebook | Separate embodiment |
| TruthfulQA 23 and 23, ARC 50 and 44 | language sheets | Why filtering is not claimed |
| Ask 1.25 million inside 1 to 2 million | 14 November memorandum | The planning band |
| Seed 7.5 to 10 million, Year-3 9 to 12 million | November specification | Not the company’s ask |

## 27. What remains to be done

Score the flux map on a series a partner actually records, with the half-variance bar, or a replacement bar, written before the run. Keep the shift test. Time the partner’s length with device events after compilation. If the omega-dit specification’s larger multipliers are to enter a filed example, produce them under that same protocol and deposit the file. Leave the sentence filter in the lab until a sheet separates false from true without a change of prompt format and without a host-typed answer.

The line the memorandum ends on is still the line of this report. The level remains. What is taken away is the disturbance around it, and only where the measurement says so.

## 28. The labelled exam, question by question

The counsel brief lists the questions in a table. This section writes each one as a paragraph so a reader who does not have the notebook can see what was asked, what came back, and which conclusion the question does not support. The one-board stack was PyTorch 2.14.1, CUDA 13.0, Triton 3.8.0, and JAX 0.12.0.1. The four-board companion suite in the memorandum was PyTorch 2.15.0, CUDA 13.2, and Triton 3.9.0. The questions below were the same exam.

### B1. Density 0.30

The question asks which block the ladder chooses when the mean absolute instability is 0.30. The rule says 128, because 0.30 is below 0.5. Julia’s `@sphota` and the Python twin both returned 128. The question is not whether 128 is the fastest width on that buffer. It is whether the two programs implement the same cut. They did. A later change to the cuts has to change both programs in the same commit, or the agreement test will fail on purpose.

### B2. Density 0.79

0.79 is at least 0.5 and still below 1, so the width is 256. Both languages returned 256. This is the width the one-million-point sweep also chose, and it landed 3.6 percent behind the fastest of the four timed widths. Passing B2 does not mean 256 is optimal. It means the ladder is deterministic and shared.

### B3. Density 1.50

1.50 is at least 1 and below 2, so the width is 512. Both languages returned 512. The cut at 2 is the next rung, not this one. A density of 1.50 that was accidentally sent to 1024 would be a disagreement and would fail the exam even if the kernel were fast.

### B4. Density 7.06

7.06 is above every cut, so the width is 1024. Both languages returned 1024. The five-minute thermometer burn also sat on this rung: the mean absolute instability of the tiled series was high enough that the host launched block 1024. That is why the real-series burn and the random burn are not interchangeable on power. The random burn in the same notebook used 256 and drew less.

### G1. Gate at 0.05

The predicate is closed below 0.20. At 0.05 the host is specified not to launch. Julia and Python both said closed. This is a labelled scalar, chosen before the run. It is not a measurement of a language model’s hidden state. When a later experiment set a different horizon, 0.05, on a phase velocity, that was a different predicate and it fired on every sentence. G1 does not license that later horizon.

### G2 and G3. Gate at 0.20 and at 0.80

Both values are at least 0.20, so the predicate is open, and both languages said open. The boundary itself, 0.20, is open. A stability index that is slightly under 0.20 is closed. The exam did not sweep the boundary at machine precision. It scored three labelled points. That is enough to lock the predicate as written, and not enough to claim a smooth detector of semantic error.

### S1. The sign bit

A float32 is bitcast, exclusive-ored with unsigned 0x80000000, and bitcast back. The residual against unary minus was 0 on the one-board exam and on each of the four boards. Section 18 repeats the bit operation on eight CPU samples so a reader can see a zero synthesis when the center is zero. The 5,000,000-state block in the telemetry summary is the long form of the same operation: variance 2.4656 to 0.0 in 10.1 ms, residual 0.0. S1 is a pass. It is not a pass of the half-variance bar.

### J1. The derivative

JAX was asked for the directional derivative of the raw map, exp(−λξ) dθ minus λ θ exp(−λξ) dξ, with λ = 1. The residual against that expression was 0. The derivative is a check that the runtime’s differentiation matches the algebra. It is taken on the raw map because that is the map whose derivative was written down for the check. It does not by itself prove the flux shift identity. F0, F1, and F2 do that.

### R0. Triton against the formula

The kernel’s output matched the flux formula within 4.8×10⁻⁷, on one board and on each of the four. That residual is the agreement tolerance the exam actually achieved, tighter than the closed-form bar of 1×10⁻⁴ used for forty passes. A specification can quote 4.8×10⁻⁷ as the observed residual. It should not quietly replace it with zero, and it should not quote it as a residual of a language-model loss.

### F0. Synthetic seed 11

8,192 points. Flux error 1.4723521 at the original offset and 1.4723523 after a shift of 5. Raw error 0.7735443, then 18.1708527. Before the shift, raw looks better. After the shift, raw is unusable and flux has not moved. This is the example that justifies keeping the center in the formula. Variance remaining on this synthetic series is about 11.7 percent, so the half-variance bar is met here. The exam does not let F0 stand in for the thermometer.

### F1 and F2. The thermometer, three seeds, both offsets

Melbourne daily minimum, 3,650 days from 1 January 1981, first value 20.7 °C, with 0.5 °C sensor noise and a 10 percent glitch added by the notebook. Seeds 11, 22, and 33 were fixed first. Seed 11 flux error is 5.8189759 at both offsets. The moving average on that seed is 5.8973565. The untouched error is 7.2355824. Raw is 135.7728271, then 266.6606750. Seeds 22 and 33 stay near 5.874 and 5.820 and likewise do not move when the series is shifted, while raw roughly doubles. F1 is the shift. F2 is the comparison with raw. Both passed. A kitchen reading near 11 °C that is multiplied toward zero is the lay picture of the raw column.

### V1. The miss

The question, written before the score, asked whether flux would keep at most half the variance of the contaminated thermometer. The fractions remaining were 0.5149549, 0.5176691, and 0.5095068. All three are over 0.5. The question is a miss on one board and the same miss on four boards. It stays in the score, 17 of 18 and 16 of 17, because dropping it would be a different exam. The synthetic series meets the bar. The hard sign-bit average zeroes a centered deviation. V1 says the soft map, on this real series, did not meet a bar the company had already set. An investor who is shown only F1 has been shown the identity and not the miss. This report shows both.

### D1. The same input twice

Drift was 0. The kernel is deterministic on the buffer it was given. D1 does not say the thermometer is noise-free. It says two evaluations of the same numbers produced the same numbers. That is the property a later regression test should lock, with a tolerance of a unit in the last place if a different GPU library is ever substituted.

### C1. Forty passes against the closed form

The algebra says forty passes at fixed ξ equal μ + (y − μ) exp(−40ξ). On one board the maximum absolute gap was 9.5×10⁻⁷. On board 0 of the four-board machine it was 1.2×10⁻⁵, inside the pre-declared bar of 1×10⁻⁴. Section 17 shows the same algebra on a single deviation of 5 at ξ = 0.1, where the gap of the pure formula against itself is zero. C1 is the kernel’s gap. A product does not have to run forty passes. If it runs them, C1 is the check that it did.

### T1. The ladder against the fastest block

Widths 128, 256, 512, and 1024 took 0.0157, 0.0120, 0.0131, and 0.0116 ms on one million random points. The ladder chose 256, 3.6 percent behind 1024, inside 10 percent. At this length the launch dominates, so the ranking can move on another day. The exam’s question was the 10 percent band, and it passed. It is not a claim that 256 is the roofline width. The roofline width is whatever the long buffer uses, and the long thermometer buffer used 1024 because its density was on that rung.

### E1. Fused against eager

On the tiled thermometer, 16,777,216 points and 40 passes, Triton took 7.3776321 ms and the eager loop took 17.1342716 ms. The fused kernel is the one that belongs in a library. The eager loop is the readable reference. E1 passed because fused was faster, not because either number is 0.012 ms. The specification’s short-buffer band is a different length.

### Q1. Four boards against the queue

Four heavy jobs in order took 29.510108 ms. The same four at once took 8.282783 ms. The ratio is about 3.56, and the question asked for at least twice, so Q1 passed. The five-minute rate, 549.1516 against 137.2839 per second, is the four-times figure and is a different sentence. On 15 November, after compilation, the queued time was 29.2718 ms and the slowest board’s event was 8.2376 ms. The thread wall was 11.7879 ms. An earlier wall of 38.1371 ms included setup and is not Q1.

Beside those questions, the glitch screen scored 24 spiked windows and 24 clean windows. A window was a glitch when its z-scored variance exceeded 4. The spikes were 8 standard deviations on 15 percent of the window. All 24 fired, and no clean window did. F1 of that screen is 1.0. The counsel brief says, and this report repeats, that the screen shows a variance predicate can see a declared spike. It was not a language-model exam. No model was loaded that night.

## 29. What each line of the working ask is for

The 14 November memorandum prices 18 months at 1,250,000 US dollars. The lines below say what the work is, in the same dollars. They are a plan. They are not salaries already paid, and they are not a forecast that a partner will appear on a date.

The founder line is 180,000 dollars, 10,000 a month for 18 months. It keeps one person on the specification, the partner conversations, and the decision of what not to claim. It is not a market salary for a finished company, and the memorandum does not pretend that it is.

The kernel engineer is 288,000 dollars, about 16,000 a month loaded, for 18 months. The work is the library: ladder, flux, agreement, and a timing harness that refuses to mix compilation into the clock. The 15 November fair clock is the kind of mistake this line exists to retire. The unfair clock was 38.14 ms because setup was inside the window. The fair clock was 8.24 ms on the slowest event. A partner should never be shown the first number as a speedup.

The research scientist is 168,000 dollars, about 14,000 a month for 12 months, not 18. The work is the next series: a partner’s own readings, a bar written first, and a written miss if the bar is missed. It is not a budget for a new hallucination claim. That claim is outside the raise.

Finance and operations, fractional, is 36,000 dollars, 2,000 a month. Entity, insurance, and accounting are 40,000. Together they are the cost of being a company that can sign a provisional engagement and a cloud account without mixing personal cards into the books.

Counsel and the path past a provisional are 120,000 dollars. The draft in the patent packet is the input to that line, not a substitute for it. Filing fees, a search, and a later utility are why the line is larger than a single provisional invoice. The inventor still has to confirm legal name, assignment, and which public notebooks count as a disclosure.

Cloud is 144,000 dollars, a planned average of 8,000 a month. That is continuation time on the class of board already measured, not a purchase of a new cluster. The bench line, 30,000 dollars, is held for later embodiments and is not to be spent as if those devices were already in the examples. Recruiting is 20,000 dollars for the two hires. The public face is 70,000 dollars: papers, the site, two gatherings, travel. The design-partner program is 40,000 dollars of the company’s own cost to show up. None of these lines is revenue.

The reserve is 114,000 dollars, about 9 percent. It is there because a deep-tech schedule slips, which the memorandum already lists as a risk. It is not a silent expansion of the seed into the specification’s 7.5 to 10 million dollar range.

People are 672,000. Counsel and the entity are 160,000. Cloud is 144,000. The public face and the partner program are 110,000. Bench and recruiting are 50,000. Reserve is 114,000. The sum is 1,250,000.

The lean alternative cuts the same plan to 12 months and 700,000 dollars: founder 120,000, engineer 192,000, scientist for 6 months 84,000, counsel 80,000, cloud 72,000, public face and partner program 50,000, operations 36,000, reserve 66,000. A reader who can fund only the lean plan is still funding the library and the filing. The scientist’s window is shorter, which means the partner series has to be chosen sooner.

Projected income in the memorandum, as a band and not a booking: year 1 about 0 in the quiet case and about 120,000 if two pilots pay; year 2 about 60,000, or about 400,000, or about 0.8 to 1.2 million if a wedge appears; year 3 about 150,000, or about 1.2 million, or 2.5 to 4 million in one vertical. Year 2 at 400,000 does not cover a 1.25 million burn. That gap is the reason for the raise. The specification’s Year-1 revenue of 1.2 to 1.8 million and Year-3 revenue of 9 to 12 million are a different document. This report does not use them to fill the gap.

## 30. How the three sources were combined

The investor memorandum supplied the exam, the company description, the use of funds, the year bands, and the risks. The counsel brief supplied the plain-language path: a neighborhood, a center, three languages, the thermometer, the song and the clatter, the photograph, the four musicians, and the sentence counsel may actually say. The November 2026 specification supplied the longer algebra: Landauer, the omega-dit integral, the split, the thesis and antithesis, the projector, entropy, free energy, the stability coefficient, the dual circuit, the supporting mechanisms, the residual estimator, and the product names. Sections 1 through 8 and 13 and 15 follow the memorandum and the brief. Sections 9, 21, and the status table follow the specification only where the sentence says so. Section 11 is the L4 log, which is in neither the memorandum nor the four-page specification’s Blackwell table. Section 12 is the later language sheets, included so a diligence reader does not meet them for the first time after a raise conversation.

Where the sources disagree, the labelled-exam JSON wins on a measured number, and the memorandum wins on the company’s ask. The specification does not get to overwrite either. That is the whole method of this compilation. A slide that takes the specification’s 958,874-times row and the memorandum’s 17-of-18 score and puts them in one breath has undone the compilation.

The products named in the specification, an education mentor and a security tool, are intentions. They are not in the use-of-funds table as separate builds. A library inside one graphics pipeline is the wedge the 1.25 million dollars can reach. A product that waits on a finished sentence filter waits on an experiment this report has already declined to call successful.

## 31. The operating picture, with the numbers attached

The counsel brief asks a reader to picture a song in a busy kitchen. The song is the level. The clatter is the deviation. Flux turns the clatter down and leaves the song. On the thermometer the song is the day’s minimum, and the flux error stayed between about 5.82 and 5.87, a little under the moving average, at both a shift of zero and a shift of 5 °C. Raw damping turned the song down too, and the error went from about 135 to about 266. The picture is accurate for the shift. It is not accurate if someone then says the clatter was silenced. About half the variance remained. The song is intact and the room is still partly noisy. That is V1, and it belongs in the picture.

The brief’s second picture is a photograph on a shaky day. The person should stay. The shake should go. The five-minute run is that attempt, repeated 41,186 times on one board, on the real series tiled to 16,777,216 points, every value finite, mean draw 314.5 W, peak 326.6 W, temperature 32 °C to 44 °C. Four boards, on the heavy random job rather than the thermometer, repeated their job 164,748 times, means near 277 to 283 W, peaks 47 °C and 48 °C. The photograph was not burnt. The nameplate of 600 W was not approached. Headroom is part of the result, because a library that only works at the thermal limit is not a library a partner can leave on.

The third picture is four musicians and a queue. One after another, 29.51 ms. Together, 8.28 ms. Over five minutes, 137.3 against 549.2 a second. The fair repeat is 29.27 ms against 8.24 ms on the device clock. The wall clock of 11.79 ms is the time a host thread spends waiting, and a partner’s dashboard should be taught the difference before a number is put on a slide. Short jobs do not get this help. Starting the job costs more than the arithmetic, which is why a second board does not hurry a tiny buffer, and why 0.012 ms must not be quoted as the cost of a fifty-million-node pass.

Taken together, the pictures are a library, a temperature, and a queue. They are not a mind. The sentence tests remain in section 12 for anyone who wants the mind. They do not change the pictures.

## 32. The boards themselves

Each example that this report calls a Blackwell result was run on an NVIDIA RTX PRO 6000 Blackwell Server Edition. The notebooks report 97,887 MiB of memory on the device, a 600 W limit, and a stated peak bandwidth of 1,597 GB/s. The one-board exam saw cuda:0. The four-board exam saw cuda:0 through cuda:3, the same part on each index. The 15 November fair clock was the same class of machine. The L4 results are not this part. A sentence that needs a bandwidth has to name which chip and which byte count.

The interconnect in these runs is PCIe. No NVLink figure was taken, and this report does not borrow one from a datasheet. The 16-byte message is interesting on PCIe precisely because the alternative is to move the series. The scaling cell recorded 16 bytes moved and 319,999,984 bytes not moved, against a full reduction of its buffers, with 20,000,000 nodes on the device. That saving is the architectural claim. The 54.58 GB/s beside it is the rate of that cell’s traffic, not the rate at which a 50-million-node flux traversal reads and writes.

Memory headroom and thermal headroom are part of the same description. A 97,887 MiB device running a 16,777,216-point float32 job is not full. The point of the five-minute burn is not that the memory was saturated. The point is that the job could be repeated 41,186 times on one board, and 164,748 times across four, without a non-finite value and without leaving the 30s to high 40s Celsius. Occupancy on the one-board thermometer burn averaged 98.55 percent, and the clock stayed between 2302 and 2317 MHz. On four boards the clocks reached 2422 MHz and occupancy ran from about 94.8 to 99.9 percent. Those are signs the devices were working, not idle, and still far from the 600 W cap. Mean draw on the thermometer was 314.53055 W with a peak of 326.608 W. An opening sample in that file is about 111 W, before the burn settles near 325 W. Quoting the opening sample as the load would understate the burn. Quoting 600 W would overstate it.

The four-board means were 279.24365 W, 283.34055 W, 279.0838 W, and 276.8883 W. The peaks were 300.458 W, 303.782 W, 304.733 W, and 304.319 W. The cluster mean is about 1,119 W, which is four boards, not one board mysteriously drawing a kilowatt. The memorandum’s energy figures, about 2,291 millijoules per job on the one-board thermometer accounting and about 2,037 millijoules per job on the cluster accounting, come from that sampled power divided across the jobs finished in 300 seconds. They are not a Landauer ledger. A reader who wants joules of bit erasure will not find them in these watts. A reader who wants to know whether the box can be left on will find a usable answer: yes, at roughly half the nameplate or below, on the jobs that were actually run.

## 33. The errors, in the order a calculator meets them

Synthetic seed 11, flux, offset 0: 1.472352147102356. Offset 5: 1.4723522663116455. The two numbers differ in the seventh digit, which is floating-point noise around an identity that says they are the same error. Raw, offset 0: 0.7735443115234375. Raw, offset 5: 18.170852661132812. The raw column improves on flux before the shift and then fails by more than a factor of ten. Any chart that shows only the first raw number is a chart of the unshifted toy series, where pulling toward zero happens to look like denoising.

Melbourne seed 11, flux, both offsets: 5.818975925445557. The identity here is exact in the printed file, not merely close. Raw: 135.7728271484375, then 266.6606750488281. Moving average: 5.8973565101623535. Untouched: 7.23558235168457. Flux beats the moving average by a small amount and beats the untouched series by a larger one, and it does so at both offsets. Raw does not compete once the readings sit far from zero. Seeds 22 and 33, in the memorandum’s rounding, are flux 5.874 and 5.820, likewise unmoved by the shift. The variance fractions, which are the miss, are 0.5149548649787903, 0.5176690816879272, and 0.5095068216323853. Writing them as “about half” is fair. Writing them as “under the bar” is not.

Same-input drift is 0. Closed form, one board: 9.5367431640625×10⁻⁷. The memorandum rounds this to 9.5×10⁻⁷. Four-board board 0: 1.2×10⁻⁵, under the bar of 1×10⁻⁴. Triton versus the formula: 4.76837158203125×10⁻⁷, which the memorandum calls 4.8×10⁻⁷. Sign and derivative: 0. None of these residuals are the 93.94 percent mean-squared-error reduction named in the November specification. That percentage remains the specification’s percentage.

Fused kernel 7.377632141113281 ms. Eager loop 17.1342716217041 ms. One-board median at 50,000,000 nodes: 0.54828799 ms, 1,459.0872277590881 GB/s, 91.36425972192163 percent of 1,597. Twelve-byte companion: 1,094.3154208193162 GB/s. Board 0 of the four-board machine: 0.54915202 ms, 1,456.791518121417 GB/s, 91.22050833571804 percent. Jobs: 41,186 in 300 seconds is 137.28391742259728 per second. Jobs: 164,748 in 300 seconds is 549.1516376799458 per second. Queued 29.510108000067703 ms against together 8.282783000026939 ms. Fair clock, 15 November: queued 29.27183961868286 ms, slowest event 8.237631797790527 ms, the four events 7.487936019897461, 8.237631797790527, 7.917215824127197, and 7.498591899871826 ms, thread wall 11.787933999812594 ms, outputs finite. The unfair clock, kept only as a warning, was queued 29.173471927642822 ms and together 38.13713599993207 ms.

These digits are here so a spreadsheet can be checked against the files without rounding drift being mistaken for a new result. The prose in the earlier sections uses the shorter forms. If a shorter form and a longer form ever disagree, the JSON wins.

## 34. A boundary around the document

This report is a technical record and a planning memorandum. It is not a prospectus, not a term sheet, and not an offer of securities. The figures of 1,000,000 to 2,000,000 US dollars, the working figure of 1,250,000, the lean figure of 700,000, and the design-partner figure on the order of 60,000 a year are unbooked plans from the 14 November memorandum. The November specification’s seed of 7.5 to 10 million, its 40/25/20/15 split of proceeds, and its revenue path from 1.2–1.8 million in year 1 to 9–12 million in year 3 are printed in that specification and are not adopted here.

Ceraph Corporation is in Kerala, India. The correspondence address for this record is chief@ceraphco.com, and the public site is www.ceraphco.com. No design-partner revenue is claimed. The provisional draft that travels with this report is a draft for counsel. It is not a filed application. The note to Swami Sandeepananda Giri that travels with it is a student’s report to his guru. It is not an endorsement, and the guru’s name is not part of the raise.

What the company can presently defend is a center-preserving map, a shared ladder, agreement across the runtimes that were installed, a sign-bit synthesis, a short message between processors, a fused kernel faster than an eager loop, four boards doing four times the heavy jobs over five minutes, temperatures under 50 °C, and a missed variance bar written in the same table as the passes. What it cannot presently defend is a semantic filter, a measured Landauer saving, a NVLink result, or the specification’s largest multipliers, until those multipliers are reproduced under the labelled protocol and deposited. The level of the reading remains. The disturbance is reduced only to the degree the chosen map, on the recorded series, actually reduces it.
