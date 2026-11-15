# Provisional technical whitepaper

Ceraph Corporation, Kerala, India. 15 November 2026. Draft for a provisional filing. Not a filed application. Not a legal opinion. chief@ceraphco.com. www.ceraphco.com.

This paper collects the mathematics, the implementations, and the measurements in one place. A sentence is marked as a measurement only when a file or a notebook log holds the number. A sentence marked as algebra is an identity. A sentence marked as a design note is a motive taken from Advaita and from earlier papers. Design notes are not chip results.

## 1. Parties and hardware

Applicant: Ceraph Corporation, Kerala, India. Correspondence: chief@ceraphco.com, www.ceraphco.com.

Measured boards: NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB, 600 W limit, stated memory peak 1,597 GB/s. A separate notebook used one NVIDIA L4. Software in the Blackwell exam included PyTorch, Triton, JAX on CUDA, and Julia.

## 2. Design motive, not a physical proof

The project uses three Advaita pictures as design motives.

The witness does not treat every fluctuation as a new fact. In the number map, a local center μ is estimated from the series and is added back after the deviation is scaled.

Fearlessness, abhaya, is the refusal to commit a step while a chosen instability is high. Several instability formulas were tried on word scores. Section 8 records which ones changed a score and which ones did not.

The pair of opposites, a value and its sign-reversed disturbance, can be averaged. If the disturbance is exactly negated, the average is the center. Section 5 writes that identity. It holds when the disturbance is an input, or when it is the deviation from a center computed from the series. It does not, by itself, know whether an English sentence is true.

No measurement in this repository shows a circumvention of Landauer’s limit, a photonic crystal, or a fabricated quasicrystal.

## 3. Objects

A series y of real numbers. A local center μ, either a window mean or a single mean of the whole series. A deviation δ = y − μ. An instability ξ, a scalar or a series of scalars, defined in each section where it is used. A positive scale λ. In the scored flux runs λ = 1.

A tile width chosen from the density, the mean of absolute instability:

- below 0.5, width 128
- below 1, width 256
- below 2, width 512
- otherwise 1024

Julia’s macro `@sphota` implements that ladder. The Python function `sphota_schedule` implements the same ladder.

## 4. The flux map and the raw map

Flux:

```
A(y) = μ + (y − μ) exp(−λ ξ)
```

Raw:

```
R(y) = y exp(−λ ξ)
```

Shift identity, algebra. Let every sample and the center move by a constant c. Then

```
A(y + c) = (μ + c) + ((y + c) − (μ + c)) exp(−λ ξ) = A(y) + c
```

so the error of A against a latent series that also shifts by c is unchanged. The raw map has no such cancellation:

```
R(y + c) = (y + c) exp(−λ ξ) = R(y) + c exp(−λ ξ)
```

which equals R(y) + c only when the exponential is 1. A CPU check that shifts the series and the center together produced a maximum absolute gap of 1.86×10⁻¹⁵ between A(y) and A(y+5) − 5.

Forty flux passes with μ and ξ held fixed:

```
A_40(y) = μ + (y − μ) exp(−40 ξ)
```

On one Blackwell board the maximum absolute difference between the kernel and this closed form was 9.5367432×10⁻⁷.

Recorded errors, synthetic seed 11:

| map | offset 0 | offset 5 |
|---|---|---|
| untouched | 6.9095788 | 6.9095788 |
| moving average | 1.6750112 | 1.6750112 |
| raw | 0.7735443 | 18.1708527 |
| flux | 1.4723521 | 1.4723523 |

Recorded errors, Melbourne daily minimums, seed 11:

| map | offset 0 | offset 5 °C |
|---|---|---|
| untouched | 7.2355824 | 7.2355824 |
| moving average | 5.8973565 | 5.8973565 |
| raw | 135.7728271 | 266.6606750 |
| flux | 5.8189759 | 5.8189759 |

Flux variance remaining on three thermometer seeds: 0.5149549, 0.5176691, 0.5095068. The pre-declared ceiling was 0.5. The check failed.

Directional derivative of the raw map, used by JAX:

```
dA = exp(−λ ξ) dθ − λ θ exp(−λ ξ) dξ
```

Certify on each of four boards: derivative error 0.0, Triton error 4.7683716×10⁻⁷, sign-bit error 0.0.

## 5. Sign-bit conjugation

For a finite float32 value x, flipping the IEEE-754 sign bit is the exclusive-or of the integer view with 0x80000000. Call that flip C(x). For finite x, C(x) = −x, up to the sign of zero.

Synthesis:

```
Υ(x) = (1/2) (x + C(x))
```

Then Υ(x) = 0 for finite x. On a deviation δ,

```
μ + (1/2) (δ + C(δ)) = μ
```

so the residual deviation has variance 0 when the flip is exact. The telemetry file records this on 5,000,000 states: variance 2.4656 to 0.0 in 10.1 ms, residual 0.0. A CPU execution of the same exclusive-or also produced variance 0. The 10.1 ms number is the file’s number.

If a second tensor called truth is supplied and the noise is truth’s difference from the data, the same average returns that supplied tensor. On four boards, with the supplied tensor equal to the constant 0.5 and a hidden sine in the data, the residual to 0.5 was 1.19×10⁻⁷ and the error to the sine was 1.5. The identity returns the tensor it was given.

## 6. Stability coefficients that were written down

Signed mean, used in the SDK note and in the Colab kernel:

```
ξ_signed = 1 − Var(x) / (E(x) + ε), ε = 10⁻⁷
```

On Qwen2.5-1.5B vocabulary logits this number was positive and large, about 4 to 16, because the mean logit is negative. A horizon of 0.20 stays open for true and false answers alike.

Absolute mean, Master White Paper equation:

```
ξ_abs = 1 − Var(x) / (E(|x|) + ε)
```

On the same logits ξ_abs sat between −3.5494 and −1.8051. The force χ sat between 51.0283 and 52.4596. Exact arithmetic left the argmax unchanged on all 20 questions. A refusal at 0.98 abstained on all 20.

Two-circuit force, as typeset:

```
χ = λ0 exp(−κ (1 − ξ)) + Λmax / (1 + exp(−α (Var / (σ_sat² + ε) − θ)))
```

With λ0 = 10, κ = 0.5, Λmax = 50, α = 10, θ = 0.98, the second term saturates near 50 on a full vocabulary, which is why χ is near 51.

Hidden-state form, layers 8 through 16, last token, absolute-mean coefficient: every value in the logged sheet was negative, about −0.65 to −2.11. Horizon 0.20 never fired. Counts stayed 12 correct, 0 wrong, 8 abstentions.

Phase velocity between adjacent layers:

```
ξ_phase = 1 − cosine(h_{L−1}, h_L)
```

At horizon 0.05 the hook fired on every logged row. Counts moved from 12 correct and 6 wrong to 0 correct and 19 wrong.

Top-10 entropy:

```
ξ_10 = H(p_(1) … p_(10)) / log(10)
```

with p renormalized on those ten. Horizon 0.50 on the max of the first three tokens. Few-shot prompts: first-token ξ from 0.0624 to 0.3866. Raw prompts: 0.6313 to 0.9029. Counts moved from 12 correct, 6 wrong, 2 abstentions to 12 correct, 0 wrong, 8 abstentions. The ocean item in the few-shot frame was the text “Pacific Ocean” at 0.2629.

Collision concentration and signal ratio, used in the L4 kernel:

```
C = Σ p_i²
SNR_z = (max x − mean x) / std(x)
```

The fused v4 print on the L4 is 0.1158 ms, Ξ 0.4872, C 0.0302, SNR_z 4.61. The earlier batch sweep on that machine is 0.0812 ms at batch 1 and 0.0884 ms at batch 32, which is 2.76 microseconds per sequence and 819.94 GB/s on that cell’s byte count.

Helmholtz-style score in `yoga_buddhi_dialectical_audit.json`, threshold 0.45. Twelve items. Ungated 6 correct, 3 wrong, 3 abstentions. Gated 5 correct, 0 wrong, 7 abstentions. Hydrogen F = 0.0402, kept. Ungated text “Pacific Ocean.” had F = 0.9976 and became “Unknown.” TruthfulQA, n = 100: thesis 23, Samya 23. ARC-Challenge, n = 100: thesis 50, Samya 44.

## 7. Bandwidth, rate, heat

Fifty million nodes, 16 bytes per node, one board: median 0.54828799 ms, 1,459.0872 GB/s, 91.3643 percent of 1,597. Four-board machine, board 0: 0.54915202 ms, 1,456.7915 GB/s, 91.2205 percent.

Heavy job, 16,777,216 float32 values, 40 passes.

| record | queued on one board | together |
|---|---|---|
| 14 November exam | 29.5101 ms | 8.2828 ms |
| 15 November, compiled first | 29.2718 ms | slowest board 8.2376 ms, wall 11.7879 ms |

Per-board clocks on 15 November: 7.4879, 8.2376, 7.9172, 7.4986 ms. Finite. The pass rule was the slowest board under three quarters of the queued time. It passed. An earlier clock the same morning included allocation and first compilation and did not pass. That clock was 29.1735 ms queued and 38.1371 ms together.

Five-minute rates: one board 41,186 jobs, 137.2839 per second, mean 314.5305 W, peak 326.608 W, 32.0 °C to 44.0 °C. Four boards 164,748 jobs, 549.1516 per second. Mean draws 279.2437, 283.3406, 279.0838, 276.8883 W. Peaks 48, 47, 48, 47 °C.

Sync file: 16 bytes transferred, 319,999,984 bytes not sent, kernel 17.5875 ms, sync 3.7208 ms, bandwidth 54.58 GB/s. Not the roofline.

Fused forty-pass kernel on one board 7.3776 ms. Eager loop 17.1343 ms.

## 8. The L4 governor, as logged

One NVIDIA L4. Model Qwen2.5-3B-Instruct. The host adds Gaussian noise of standard deviation 11.5 to logits on a window of steps. If a collision and a top-two ratio meet the cell’s rule, the host crops the key-value cache by the negative of the step just appended and continues with a reference string the cell contains.

Final cropped suite:

| test | tok/s | noise steps blocked | rollback | C | ratio |
|---|---|---|---|---|---|
| 1 | 17.64 | 4 | token 08, ` lower` vs ` higher` | 0.226 | 1.13 |
| 2 | 19.08 | 4 | token 04, ` liquid` vs ` a` | 0.405 | 1.45 |
| 3 | 11.95 | 8 | token 35, `Mer` vs `Or` | 0.423 | 1.45 |

Test 1’s printed sentence contains 2,200 W/(m·K) and about 400 W/(m·K), which are in the host string. Test 3 prints Mercury 88 days and Venus 225 days, then calls 137 days Mercury’s orbital period. The suite without the crop repeats `liquid` and `Sun`. An ungated perturbed run prints 16.84 tok/s and the text “Pure diamond has a lower thermal conductivity than copper.”

## 9. What a provisional can carry

The filing can carry the flux map, the ladder, the agreement test, the sign-bit average, the independent four-board job, and the 16-byte message, with the numbers in sections 4, 5, and 7, including the missed half-variance bar.

The filing can describe, as an embodiment, a logit-row reduction to Ξ, C, and SNR_z, and a cache crop followed by a host reference string, with the L4 log in section 8.

The filing should not claim that a language model has been shown to refuse false sentences in general. Section 6 is the record of why.

## 10. File index

- `docs/benchmark-results/rtx6000-labelled-exam/one_exam.json`
- `docs/benchmark-results/rtx6000-labelled-exam/four_exam.json`
- `benchmarks/rtx_pro_6000_blackwell/hardware_tensor_telemetry.json`
- `benchmarks/rtx_pro_6000_blackwell/rtx6000_4gpu_scaling.json`
- `benchmarks/rtx_pro_6000_blackwell/four_vs_one_heavy_fair.json`
- `benchmarks/rtx_pro_6000_blackwell/yoga_buddhi_dialectical_audit.json`
- `docs/investor/TECHNICAL_REPORT_BLACKWELL.md`
- `docs/patent/PATENT_SPECIFICATION_SIMPLIFIED.md`
- `docs/patent/PATENT_SPECIFICATION_FLOWCHARTS.md`
- `docs/swami-sandeepananda-giri-note.md`
- `docs/gemini-november-briefing.md`
