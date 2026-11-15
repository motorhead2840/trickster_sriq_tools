# Blackwell numeric record

Ceraph Corporation, Kerala, India. 15 November 2026. chief@ceraphco.com. www.ceraphco.com.

The enabled work in this memo is a flux map on a numerical series, a scheduled graphics kernel, and independent jobs on four boards. Hardware figures are the values in `benchmarks/rtx_pro_6000_blackwell/hardware_tensor_telemetry.json`, which match the 14 November labelled-exam logs at the recorded precision. A last check of the saved language sheets is recorded at the end. That work is not part of the raise.

## Planning band

The 14 November memorandum used a working ask of 1,250,000 US dollars, inside a planning band of 1,000,000 to 2,000,000 US dollars, and a design-partner figure on the order of 60,000 US dollars a year. None of this is booked revenue. The band is the cost of turning the measured map into a library and one design partner, not a claim that the money has been raised.

## Roofline

At 50,000,000 nodes, counted at 16 bytes per node, the recorded bandwidth is 1,459.0872 GB/s. That is 91.3643 percent of the stated peak in the same file. The median latency at that length is 0.5483 ms.

## Four boards

One board is recorded at 137.2839 jobs per second. Four boards are recorded at 549.1516 jobs per second. Four heavy jobs queued on one board took 29.5101 ms. The same four jobs launched together took 8.2828 ms. The pattern is one job per board. A short vector cut across four boards was slower in an earlier run, because the launch and the join cost more than the arithmetic.

A second timing on 15 November, after the kernel had been compiled on all four boards, used the same heavy size: 16,777,216 float32 values and 40 flux passes. Four jobs queued on board 0 took 29.2718 ms. The slowest board’s own clock, while all four ran together, was 8.2376 ms. The four per-board clocks were 7.4879 ms, 8.2376 ms, 7.9172 ms, and 7.4986 ms. The wall clock around the threads was 11.7879 ms, which includes the Python join. Outputs stayed finite. The pass rule was the slowest board clock under three quarters of the queued time. That rule passed. The first attempt the same morning did not, because allocation and first-time compilation on boards 1–3 were inside the parallel clock.

## Sixteen-byte message

The semantic-sync record transferred 16 bytes and left 319,999,984 bytes off the bus relative to a full reduction of its buffers. The bandwidth stored beside that message is 54.58 GB/s. That figure belongs to the sync record. It is not the 1,459.0872 GB/s roofline.

## Power and temperature

Peak temperatures on the four boards are 48.0 °C, 47.0 °C, 48.0 °C, and 47.0 °C. Mean draw is 279.2437 W, 283.3406 W, 279.0838 W, and 276.8883 W. The limit in the file is 600 W. Each mean sits near half of that limit, and every peak temperature is under 50 °C.

## Flux map

The recorded flux errors across a uniform shift of 5 are 1.4723521 and 1.4723523. Raw damping, which scales the whole series, is 0.7735443 before that shift and 18.1708527 after it. At the unshifted point the raw map has the lower error. After the shift it does not. The flux pair stays level.

A CPU recomputation of the same map, shifting the series and the center together, left a maximum absolute gap of 1.86×10⁻¹⁵ between the unshifted result and the shifted result minus 5.

## Sign-bit override

The telemetry file records an override on 5,000,000 states. Variance moves from 2.4656 to 0.0 in 10.1 ms, with a sign-bit residual of 0.0. A CPU check of the same exclusive-or, averaging a float32 vector with its sign-flipped copy, also produced variance 0.0. The 10.1 ms figure is the one stored in the telemetry file. The CPU clock time is not a substitute for it.

On the labelled exam, the same sign-bit step had residual 0.0 on each of the four boards. The half-variance bar on the thermometer was missed: remaining fractions 0.5149549, 0.5176691, and 0.5095068.

## Reserved for a later phase

Saved language sheets were scored once more against a fixed rule: the wrong count must fall, and the correct count may drop by at most one. They do not enter the claims.

The hidden-state gate on the base model never fired. Both columns were 12 correct, 0 wrong, and 8 abstentions.

Phase velocity at horizon 0.05 fired on every row. A sheet that was 12 correct and 6 wrong became 0 correct and 19 wrong.

Top-10 entropy at 0.50 reduced a wrong count from 6 to 0 and kept 12 correct answers. The first-token values split by prompt format. Few-shot lines ending in “Answer:” stayed at or below 0.3866. Raw questions sat at or above 0.6313. The known ocean item in that frame was the text “Pacific Ocean” at 0.2629.

`yoga_buddhi_dialectical_audit.json` records a 12-item bank. Ungated: 6 correct, 3 wrong, 3 abstentions. After a Helmholtz threshold of 0.45: 5 correct, 0 wrong, 7 abstentions. Hydrogen was kept at 0.0402. The ungated text “Pacific Ocean.” was replaced with “Unknown.” at 0.9976. On 100 TruthfulQA multiple-choice items the thesis score and the Samya score are both 23. On 100 ARC-Challenge items the thesis score is 50 and the Samya score is 44.

That body of runs is the reason a probability threshold on its own is not the product claim. The raise described above is for the numeric map, the scheduled kernel, and the four-board job pattern.

## L4 notebook, fused kernel v4

A separate notebook, run on one NVIDIA L4 and not on the four RTX PRO 6000 boards, times a fused Triton kernel and a Qwen2.5-3B-Instruct loop. Those timings are not the 1,459.0872 GB/s roofline and not the 3.55 times heavy-job speedup. The fair four-board clock remains 29.2718 ms queued and 8.2376 ms for the slowest board.

The batch sweep in that notebook, for an earlier kernel, prints:

| batch | latency ms | per sequence µs | GB/s |
|---|---|---|---|
| 1 | 0.0812 | 81.18 | 27.89 |
| 4 | 0.0814 | 20.36 | 111.20 |
| 16 | 0.0819 | 5.12 | 442.27 |
| 32 | 0.0884 | 2.76 | 819.94 |
| 64 | 0.2646 | 4.13 | 547.54 |

A later cell titled as the fused v4 kernel prints latency 0.1158 ms, Shannon Ξ 0.4872, collision C 0.0302, and logit SNR_z 4.61. Those four numbers are one print. They are not the batch-32 row.

The end-to-end cell that crops the cache with a negative offset logs three tests. Each test adds Gaussian noise of standard deviation 11.5 to the logits on a stated step window, and the cell supplies a reference sentence in the prompt when it rolls back.

Test 1 prints 17.64 tok/s, 4 noise steps blocked, and a rollback at token 08 between the pieces ` lower` and ` higher`, with C 0.226 and ratio 1.13. The printed answer is: “At room temperature, pure diamond has a thermal conductivity of approximately 2,200 W/(m·K), which is about 5 times higher than copper's thermal conductivity of about 400 W/(m·K).”

Test 2 prints 19.08 tok/s, 4 noise steps blocked, and a rollback at token 04 between ` liquid` and ` a`, with C 0.405 and ratio 1.45. The printed answer states boiling points of liquid nitrogen and liquid oxygen and does not repeat the word liquid.

Test 3 prints 11.95 tok/s, 8 noise steps blocked, and a rollback at token 35 between `Mer` and `Or`, with C 0.423 and ratio 1.45. The printed answer gives Mercury 88 days and Venus 225 days, then calls 137 days Mercury’s orbital period.

An earlier cell in the same notebook, without that crop, printed 17.65, 18.40, and 21.27 tok/s with 4 blocked steps each, and the second and third answers repeat `liquid` and `Sun`. The ungated loop that printed 16.84 tok/s is the perturbed baseline in the third cell, and its text still begins “Pure diamond has a lower thermal conductivity than copper.”

What this notebook records is a governor on one L4: a collision statistic and a top-two ratio, a rollback of the cache by the tokens just appended, and a reference string supplied by the cell. On the cropped run, the three logged noise windows are marked blocked, and the first answer contains the supplied conductivity comparison. It does not record a four-board language benchmark, and it does not replace the numeric flux results above.
