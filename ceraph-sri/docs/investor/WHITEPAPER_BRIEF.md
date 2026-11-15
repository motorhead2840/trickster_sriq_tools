# Whitepaper

Ceraph Corporation, Kerala, India. 15 November 2026. A brief of the technical report. chief@ceraphco.com. www.ceraphco.com. Planning figures are not revenue and not an offer of securities.

## Abstract

The Singularity-based Realization Interface cancels a residual that has been separated from a structural component. The thesis is the sum of the two. Conjugation flips only the residual, by subtraction or by an exclusive-or of the IEEE-754 sign bit with the unsigned mask 0x80000000. The projector, one half of the identity plus that conjugation, returns the structural component. When the residual that is flipped is the residual that was added, the identity is exact up to rounding.

On an NVIDIA RTX PRO 6000 Blackwell Server Edition, a November 2026 Colab series reports that identity in float16. The ratio of thesis variance to recovered variance runs from 26 times at noise standard deviation 5 to 10,026 times at noise standard deviation 100. Median latency on a 16 by 65,536 batch stays between 0.0216 and 0.0223 milliseconds. Cosine similarity to the true structural component stays at or above 0.99979. A free-energy proxy built as U − T ln(var) falls from about 140 to about 1.0. Those ratios are the algebra of an oracle cancellation: the score has the true structure, and the thesis still contains the injected noise. The same whitepaper states that this series does not reproduce an internal H100 figure of 35,037 times at 95.23 microseconds, nor an internal estimator at up to 94 percent mean-squared-error reduction from observed logits alone.

A second map keeps a local center and scales only the deviation. On a Melbourne temperature series its error was unchanged by a shift of 5 °C. A pre-declared half-variance bar on that series was missed. The planning case in the November venture report is a seed of 7.5 to 10 million dollars and a revenue path from 1.2–1.8 million in year 1 to 9–12 million in year 3. An earlier memorandum the same month used a working ask of 1.25 million dollars. Neither figure is booked.

## The cancellation

```
|Thesis⟩ = |υ⟩ + |ξ⟩
|Antithesis⟩ = |υ⟩ − |ξ⟩
|Final⟩ = ½ (|Thesis⟩ + |Antithesis⟩) = |υ⟩
```

The continuous-variable packaging is an integral of an amplitude, a phase, and the same sum υ + ξ. The code evaluates it as a discrete sum. Flipping the whole thesis, rather than the residual, averages to zero and destroys the level. That is the failure the center-preserving map was written to avoid:

```
A(y) = μ + (y − μ) exp(−λ ξ)
R(y) = y exp(−λ ξ)
```

A(y + c) = A(y) + c. R(y + c) is not R(y) + c unless the exponential is 1. With λ = 1, a synthetic series kept a flux error of 1.472 across a shift of 5, while the raw error moved from 0.774 to 18.17. The thermometer kept a flux error of 5.819 at both offsets, while the raw error moved from 136 to 267.

## The eighteen core equations

E01 is Landauer’s bound, k_B T ln 2 per erased bit, about 2.87×10⁻²¹ joules at room temperature. It is a reference. It was not metered on the die. E02 is the omega-dit integral. E03 splits the space into a structural subspace and a residual subspace. E04 through E08 are the thesis, the conjugation, the antithesis, the projector Υ = ½(I + Ĉ_S), and the recovered state. E09 says a residual variance driven to zero collapses an entropy. The printed tables show a variance ratio, not a density-matrix entropy. E10 is the free-energy proxy, and the printed collapse is about 140 to about 1.0. E11 is a stability coefficient. On the form that divides by a mean absolute value, Campaign A reports −6.57 on structured synthetic logits and −9.04 on pure noise. A different H100 note reports a non-negative coefficient with mean 0.3744. The formulas must not be mixed. E12 is a dual-circuit gain, an exponential of the stability coefficient plus a sigmoid of a variance threshold. E13 multiplies every logit by exp(−χ). One positive factor does not change which logit wins. E14 is a logarithmic witness of variance. E15 is a squeezing proxy: under loss, a squeezed quadrature has variance η e^(−2r) + (1 − η), in a classical simulation, not on an optical bench. E16 is an aspirational concomitance, “the relation is already in the substrate,” whose operational form is agreement of two programs with a formula. E17 is the mirror M(C) = |C|², mean 1.93 on 1,000 samples. E18 is a fabric Hamiltonian, on-site terms plus neighbor couplings, Hermitian so the norm is kept. No measured spectrum of that operator is in the tables.

## The March–July carry-over

When the residual is not supplied, cancellation becomes estimation. For a known Gaussian model the minimum-mean-square estimator is the Wiener conditional mean, with weights built from the covariances of the signal and the two observation channels. The observable projector matches that conditional mean when the weights are those Wiener weights. That is the lemma. It is not the sign bit. The sign bit needs the residual in hand.

Idempotency says projecting twice changes nothing. Involution says flipping the residual twice restores the thesis. The raw scaling map is not an involution under a shift; the center-preserving map is. The launch ladder for that map is 128, 256, 512, or 1024 as the mean absolute instability crosses 0.5, 1, and 2. A gate opens at stability index 0.20.

## What was measured, by campaign

Campaign A, the November Colab series on one RTX PRO 6000, float16, sixty iterations after warmup, batch 16 by 65,536:

| Noise std | Variance ratio | Median | Cosine |
|---|---|---|---|
| 5 | 26× | 0.0223 ms | 1.0000 |
| 10 | 101× | 0.0218 ms | 1.0000 |
| 25 | 627× | 0.0218 ms | 0.99999 |
| 50 | 2,500× | 0.0216 ms | 0.99995 |
| 100 | 10,026× | 0.0217 ms | 0.99979 |

Campaign B, prior internal, not reproduced by Campaign A: H100, 35,037 times at 95.23 microseconds, cosine 1.0000. A100, median 97.28 microseconds. L4, up to about 94 percent mean-squared-error reduction from observed logits at noise standard deviation 4, and about 91 percent when a high-quality noise estimate is already available. Mild damping at λ0 = 0.0183 is reported at a variance ratio of 1.01.

Campaign C, the labelled exam of 14 November: 17 of 18 on one board, 16 of 17 on four. The miss is 51 to 52 percent of thermometer variance remaining. Fused kernel 7.38 milliseconds against an eager 17.13. Five-minute rates 137.3 and 549.2 jobs per second. Roofline 1,459.0872 gigabytes per second at fifty million nodes, 91.36 percent of a stated 1,597. Temperatures at or below 48 °C. A fair four-board clock on 15 November, after compilation, was 29.27 milliseconds queued and 8.24 milliseconds on the slowest device.

A four-page specification states a further peak of 958,874 times at noise standard deviation 1,000 and a median near 0.012 milliseconds. That peak is not a row of the table above.

## Costs and revenue

The November venture report states a company bootstrapped at about 5,000 dollars and a seed of 7.5 to 10 million dollars over 18 to 24 months, for a software stack, two prototypes, support, and design partners. The stated staff range is 34 to 62 core technical people, plus 15 to 20 support people for each product. The stated split of proceeds is 40 percent engineering, 25 percent intellectual property, 20 percent prototypes, and 15 percent design partners and go-to-market. At 7.5 million dollars that is 3.00, 1.875, 1.50, and 1.125 million. At 10 million it is 4.00, 2.50, 2.00, and 1.50 million.

Stated planning revenue is 1.2 to 1.8 million in year 1, with an operating loss; 4.5 to 6.5 million in year 2, with a smaller loss; 9 to 12 million in year 3, with operating profit stated at 1.5 to 3.0 million. The year-1 band does not carry the raise. The raise is the bridge the table itself implies. No contract is reported as signed. The 14 November memorandum’s working ask of 1.25 million dollars remains the plan sized to a library and one partner, not to a staff of 34 to 62.

The two named products, an education platform and a device safety suite, are the revenue story those documents attach to the larger raise. A product that must refuse a false sentence is ahead of the sheets. A product that embeds the projector or the flux map in a pipeline a partner already runs is the nearer object.

## Conclusion

The identity is solid when the residual is known: thesis plus antithesis, halved, is the structure, and Campaign A’s cosine similarities and flat 0.022-millisecond median are a reading of that identity as the injected noise grows. The estimation problem, recovering the structure from observations alone, is the Wiener problem in a known Gaussian model and is otherwise still open at the standard the internal 94 percent note set. The center-preserving map is the form that keeps a level when the whole record is shifted, and it has a deposited miss on a variance bar. Eighteen equations name the core. The March–July line tells which of them are identities and which are estimators. The larger planning case is 7.5 to 10 million dollars and the three-year revenue path above. The smaller planning case is 1.25 million dollars. The residual is removed to the degree the chosen equation, on the recorded campaign, actually removes it.
