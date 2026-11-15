# Technical report

Ceraph Corporation, Kerala, India. 15 November 2026. For investors and for technical diligence. chief@ceraphco.com. www.ceraphco.com.

This report gathers the equations of the Singularity-based Realization Interface, also called the omega-dit architecture, from the November 2026 specification, the November 2026 whitepaper, the November 2026 venture report, the November 2026 patent specification, and the March–July mathematical carry-over that the repository still implements. Each equation is written in standard mathematical notation, then interpreted, then tied to the measurement campaign that actually speaks to it. Planning figures are not revenue and not an offer of securities.

## Abstract

The architecture separates a numerical state into a structural component and a residual, then cancels the residual by a conjugation. When both components are supplied, conjugation is subtraction of the residual, or, in IEEE-754 binary32, an exclusive-or of the sign bit with the unsigned mask 0x80000000. The projector is one half of the identity plus that conjugation. Applied to a thesis state that is the sum of structure and residual, the projector returns the structure. That identity is exact in exact arithmetic and is exact up to rounding when the residual that is conjugated is the residual that was added.

A November 2026 Colab series on an NVIDIA RTX PRO 6000 Blackwell Server Edition reports this oracle cancellation on float16 batches. Variance of the thesis divided by variance of the recovered tensor runs from 26 times at a noise standard deviation of 5, through 101, 627, and 2,500, to 10,026 times at a noise standard deviation of 100. Median latency on a 16 by 65,536 batch, sixty iterations after warmup, stays between 0.0216 and 0.0223 milliseconds. Cosine similarity of the recovered tensor to the true structural component stays between 0.99979 and 1.0000. On a 64 by 8,192 batch at noise standard deviation 8, the same series reports a variance ratio of 64.9 and cosine similarity 1.0, with a maximum absolute error printed as 1.9 times a negative power of ten whose exponent glyph does not survive text extraction of the PDF. A free-energy proxy F = U − T ln(var), on a 64 by 8,192 batch at noise standard deviation 12, is reported to fall from about 140 to about 1.0. A stability coefficient on structured synthetic logits is reported at −6.57 and on pure noise at −9.04. A mirror product on 1,000 complex samples has mean squared modulus 1.93.

Those ratios grow with the injected noise because the recovered tensor is being compared with a thesis that still contains that noise, while the score also has access to the true structural component. The November whitepaper states, in its own section on prior internal results, that this Blackwell series does not independently reproduce a proprietary H100 figure of 35,037 times at 95.23 microseconds, nor an internal L4 convolutional estimator reported at up to 94 percent mean-squared-error reduction from observed logits alone. A four-page specification of the same month states a further peak of 958,874 times at noise standard deviation 1,000, a median band of 0.0117 to 0.0125 milliseconds, a cosine similarity of at least 0.979, and a residual estimator at 93.94 percent. That peak is not a row of the scaling table above.

A separate campaign, the labelled exam of 14 November on one and four RTX PRO 6000 boards, tested a different map: a local center plus the deviation from that center, scaled by an exponential of a local instability. That map preserves the error under a uniform shift. On a Melbourne daily-minimum series the flux error was unchanged by a shift of 5 °C, while a map that scales the whole reading was not. A pre-declared bar that at most half the variance remain was missed. That campaign did not load a language model.

Eighteen equations form the November core, from Landauer’s bound through the fabric Hamiltonian. Twelve further relations are carried from the March–July and dissertation line: the Wiener estimator, the observable projector, involution, the sign bit, idempotency, and the flux shift identity. The investor section records the planning case written in the November venture report: a company bootstrapped at about 5,000 dollars, a stated seed of 7.5 to 10 million dollars over 18 to 24 months, and planning revenue of 1.2 to 1.8 million in year 1, 4.5 to 6.5 million in year 2, and 9 to 12 million in year 3, with a stated operating profit of 1.5 to 3.0 million in year 3. An earlier memorandum the same month used a working ask of 1.25 million dollars. Both are plans. Neither is booked.

## 1. How to read a number in this report

Four campaigns are easy to mix because they share a chip family and a vocabulary.

Campaign A is the November 2026 Colab series on one RTX PRO 6000 Blackwell Server Edition, as printed in the whitepaper, the venture report, and the patent specification of that month. The structural component used for the cosine score is the true component. The latency quoted with the scaling table is about 0.022 milliseconds.

Campaign B is prior internal measurement, which the same whitepaper says has not been independently replicated by Campaign A. It includes an H100 median of 95.23 microseconds and a variance ratio of 35,037, an A100 median of 97.28 microseconds, and an L4 estimator at up to about 94 percent mean-squared-error reduction when the noise estimate is learned from observed logits.

Campaign C is the labelled exam of 14 November, with a fixed answer key, on one board and on four. Its map is the center-preserving flux map. Its roofline at fifty million nodes is 1,459.0872 gigabytes per second. Its miss is the half-variance bar on the thermometer.

Campaign D is the four-page full specification’s peak row: 958,874 times at noise standard deviation 1,000, and a short-buffer median near 0.012 milliseconds. Campaign A’s table stops at noise standard deviation 100 and at about 0.022 milliseconds. Campaign D is cited only as that specification.

A variance ratio var(thesis) / var(recovered) is large whenever the thesis is dominated by noise and the recovered tensor has been returned to the structural component. It is the expected algebra of an oracle cancellation. It is not, by itself, a measurement of a filter that looked only at a finished English sentence. Where a result is an estimate from observed data alone, this report says so.

## 2. The object being cancelled

Let υ denote a structural component and ξ a residual of the same shape. The thesis, the antithesis, and the recovered state are

```
|Thesis⟩ = |υ⟩ + |ξ⟩
|Antithesis⟩ = |υ⟩ − |ξ⟩
|Final⟩ = ½ (|Thesis⟩ + |Antithesis⟩) = |υ⟩
```

The middle line is conjugation of the residual only. Conjugating the whole thesis instead of the residual returns zero, which is the wrong answer on any series whose level is not zero. That distinction is the entire engineering content of the sign bit: the mask 0x80000000 is applied to the residual, and the structural component is added back.

The continuous-variable notation packages the same pair with an amplitude and a phase,

```
|Ω⟩ = ∫ σ(t) e^{i β(t)} [υ(t) + ξ(t)] dω
```

The repository implements the integral as a discrete sum σ · β · (υ + ξ) · Δω. The sum is a code identity. It is not a measured electromagnetic field.

## 3. The eighteen core equations

### E01. Landauer bound

```
ΔQ ≥ k_B T ln 2
```

per irreversible erasure of one bit. Boltzmann’s constant is 1.380649×10⁻²³ joules per kelvin. At 298.15 kelvin the bound is about 2.87×10⁻²¹ joules per bit. The specification offers this bound as the cost that a pipeline pays when it writes a token and later erases it. The boards in Campaign C drew on the order of 280 to 315 watts. One second of that draw is on the order of 10² joules, about twenty-three orders of magnitude above the bound for one bit. No campaign in this report metered joules of bit erasure. The bound is the physical reference. A claim that the chip saved that quantity needs a different instrument. What the algebra does show is that an exact cancellation of a known residual does not need a second pass that overwrites the residual after the fact.

### E02. Omega-dit integral

The integral in section 2. σ is an existence density, β a geometric phase, υ the structural term, ξ the residual. Keeping υ and ξ inside one integrand is the specification’s way of saying that the split happens at the cancellation step rather than by an early threshold. Premature discretization, in that prose, means committing a hard 0 or 1 before the residual has been isolated. The discrete sum in `omega_dit` is the implemented object. Campaign A does not report a sample of the integral. It reports the projector on tensors.

### E03. Bipartite partition

```
H = H_V ⊕ H_H
```

H_V carries the structural component. H_H carries a residual treated as orthogonal to it. Orthogonality here is the modelling assumption that lets a projector kill one summand and keep the other. In the scored flux map the structural summand is a local mean and the residual summand is the deviation from that mean. In Campaign A the two summands are the tensors the notebook added. Neither campaign measured an inner product on an infinite-dimensional Hilbert space. The operational test of the split is whether changing only the residual changes only the residual after the projector, and whether a uniform shift of a level is preserved. Campaign A tests the first, because the true υ is known. Campaign C tests the second.

### E04. Thesis

```
|Thesis⟩ = |υ⟩ + |∇ξ⟩
```

The specification writes the residual as a gradient term. In the code the thesis is the sum of the supplied value and the supplied noise, without a separate spatial gradient. The gradient notation marks the residual as the part that is allowed to move. A reader should not look for a finite-difference stencil unless a notebook builds one. The thesis is the noisy observation when υ and ξ are both known, and it is the only input when they are not.

### E05. Symplectic conjugation

```
Ĉ_S = Π_V − Π_H
```

implemented, on a floating-point residual, as an exclusive-or of the sign bit. Π_V keeps the structural subspace and Π_H keeps the residual subspace, so their difference flips the residual and leaves the structure. On the latent chart where the state is the pair (υ, ξ), applying the flip twice restores the pair: Ĉ_S squared is the identity. That involution is a unit test in the yoga-buddhi module. It holds when the same ξ is flipped that was added. It does not hold for a raw scaling map that multiplies the whole observation by a factor other than one. That failure is E21 below.

### E06. Antithesis

```
|Antithesis⟩ = |υ⟩ − |ξ⟩
```

This is Ĉ_S applied to the thesis when Ĉ_S flips only ξ. Headphones are the specification’s analogy and they are accurate only at this line: the inverted wave has to be the noise, not the music. If the antithesis is −thesis, the average is zero and the music is gone.

### E07. Projector

```
Υ = ½ (I + Ĉ_S)
```

Υ is the operator that extracts what is invariant under conjugation. On a deviation, Υ returns zero and the reconstructed sample is the center. On the thesis, Υ returns υ. Idempotency, Υ² = Υ, is the statement that projecting twice does nothing further. The code checks the algebra. Campaign A’s cosine similarities near 1 are the empirical form of the same statement when υ is known and the arithmetic is float16.

### E08. Final state

```
|Final⟩ = ½ (|Thesis⟩ + |Antithesis⟩) = |υ⟩
```

in one fused multiply-add when the negation is exact. This is E07 applied to E04. Campaign A’s scaling table is a numerical reading of E08 under increasing ξ, with latency reported flat near 0.022 milliseconds. The four-page specification’s phrase “a single fused multiply-add” describes this hard cancellation. The forty-pass exponential kernel of Campaign C is a different operator, repeated. A latency for one is not a latency for the other.

### E09. Entropy collapse

Driving the residual variance toward zero collapses a von Neumann entropy S(ρ) of a state ρ, in the specification’s language. The quantity Campaign A actually prints is a variance ratio and a cosine, not a density-matrix entropy. If the recovered tensor equals υ and υ is a low-variance structural draw, the variance of the residual has been removed by construction and any entropy that was a function of that variance falls with it. That is a restatement of E08. It becomes a thermodynamic result only if S(ρ) is computed from a physical density operator and the heat is metered. Neither step is in the printed tables.

### E10. Free-energy proxy

```
F = U − T S
```

and, in Campaign A’s printed proxy,

```
F = U − T ln(var)
```

The proxy uses a logarithm of variance in place of an entropy. On a 64 by 8,192 batch at noise standard deviation 12, the whitepaper and the venture report state that this proxy fell from about 140 to about 1.0. That is a numerical collapse of the proxy they defined. It is consistent with E08, because cancelling a large residual collapses ln(var) of the residual-bearing tensor. It is not a measurement of Helmholtz free energy in joules, and it is not the electrical draw of the board. The specification’s further sentence, that the proxy falls from values in the thousands to near zero, is Campaign D’s wording and is a wider range than the 140 to 1.0 row.

### E11. Stability coefficient

Three formulas share the name. They are not interchangeable.

The November specification prints

```
Ξ = 1 − Var(logits) / (E[|logits|] + ε)
```

The legacy simulator prints the same shape with a signed mean in the denominator,

```
Ξ = 1 − Var(ℓ) / (E[ℓ] + ε)
```

which is not confined to the unit interval. The bounded signal-to-noise form used as a gate elsewhere lies in (0, 1].

Campaign A, on the specification’s shape, reports Ξ = −6.57 on structured synthetic logits and Ξ = −9.04 on pure noise. Both are negative because the variance exceeds the mean absolute value. The structured value is less negative, which is the direction the whitepaper says it intended. The same whitepaper says production logits with a larger magnitude relative to their variance would be expected to go positive. It does not report that they did.

A different H100 note inside the long technical specification reports a mean stability coefficient of 0.3744, with range 0.3619 to 0.3840, on “realistic logits,” and 0.0000 on pure high noise. That note is using a non-negative form. Putting −6.57 and 0.3744 in one sentence without naming the formula is a mistake. On the language sheets run later in November, the absolute-mean form sat near −3.55 to −1.81 and a horizon of 0.98 abstained on every item, correct ones included. The coefficient is a statistic of a vector. It has not been established as a detector of a false sentence.

### E12. Dual-circuit damping

```
χ = λ0 exp(−κ (1 − Ξ)) + Λmax · σ(α (√Var − θ_crit))
```

where σ is a sigmoid. The November notebooks also write the second term as Λmax times a sigmoid of a normalized variance minus a threshold, with λ0 = 10, κ = 0.5, θ = 0.98, and Λmax = 50. Those constants saturate the sigmoid on a full vocabulary, and χ then sits near 51. An H100 note in the long specification uses a mild λ0 of 0.0183 and reports a variance ratio of 1.01, which is a one-percent move. The two settings are different controllers. A global χ multiplies an entire logit vector by one positive factor and therefore does not change which entry wins, unless the factor underflows the floating-point format and the text becomes noise. Both behaviors were observed on language runs. χ is a gain. It is not a substitution of one token for another.

### E13. Logit rescaling

```
ℓ ← ℓ · exp(−χ)
```

This is the application of E12 to a logit row. Because exp(−χ) is one positive scalar, the argmax is invariant for any finite χ that does not underflow. The long specification’s sentence that rescaling “crushes unstable logits before token selection” is true of the magnitudes and false of the identity of the winner. A change of winner requires a non-uniform rule: a crop of a cache, a rewrite of one coordinate, or a residual that is not a global scale. Campaign A does not report a changed argmax. The L4 notebook reported elsewhere crops a cache and can insert a host string. That is a different mechanism from E13.

### E14. Weber–Fechner witness

```
H_W ∝ log(1 + Var / ε)
```

Weber–Fechner, in sensory physiology, is the observation that a just-noticeable difference often scales with the stimulus, so a logarithm linearizes perceived intensity. Here the specification uses the logarithm as a compressive reading of residual variance. No campaign prints a column of H_W beside a variance ratio. The formula is the named proxy. It is not an extra empirical result.

### E15. Two-mode squeezing proxy

The repository’s Gaussian model uses ħ = 2, so the canonical commutator is 2i and vacuum variance is one. An x-squeezed vacuum has variance e^(−2r). Under transmissivity η the measured quadrature variance is

```
η e^{−2r} + (1 − η)
```

The specification’s phrase “two-mode squeezing” points at this redistribution: uncertainty moved from one quadrature into its partner, with the product constrained by the commutator. The simulations that exercise it use classical random numbers and analytic conditioning. They are not a record from an optical table, and they are not a row of Campaign A. Squeezing the reference mode beats squeezing the signal mode only inside an interval of a gain parameter that can be empty. A slogan that any squeezing helps is not the formula.

### E16. Inherence

```
A ⇒ B on a substrate X
```

The specification’s prose: a structural relation that is already carried by the substrate can be checked without an external table. The repository’s operational form of that sentence is agreement of two implementations against a formula the host already holds: the same input twice, a kernel against a closed form, two languages against one ladder. Those checks do not consult a labelled answer key of English questions. The image in the equation catalog marks this relation as aspirational. This report keeps that mark. A high correlation in a histogram is not, by itself, a proof of concomitance.

### E17. Mirror validation

```
M(C) = C · conj(C) = |C|²
```

The product of a complex value with its conjugate is the squared modulus, real and non-negative. It is invariant under a pure phase and is zero only at zero. Campaign A reports a mean squared modulus of 1.93 on 1,000 complex samples, and says that value is consistent with the saturation behavior expected of that gate. The mirror is a checksum. It does not cancel a residual. It confirms that a coefficient is present at a magnitude.

### E18. Fabric Hamiltonian

```
H_fab = Σ_i h_i + Σ_{(ij)} J_ij
```

An on-site term plus a coupling over neighboring pairs. The specification says real eigenvalues of this operator imply no probability leaking out of the model. That is true of a Hermitian matrix: its evolution is unitary and the norm is kept. The repository also uses a graph Laplacian, whose trace penalty and spectral gap govern different design quantities and are not interchangeable with this Hamiltonian. No campaign prints a measured spectrum of H_fab on a GPU. The formula is the design expression for “neighbors should agree, and the generator should be Hermitian.”

The November specification states that all eighteen of these equations were numerically verified. Campaign A’s printed checks that a reader can point to are the projector error, the variance-ratio table, the free-energy proxy, the two stability values, and the mirror mean. E01, E14, E15, E16, and E18 do not have their own residual row in that table. Verification of an identity in a notebook is not the same statement as a residual deposited beside the labelled exam.

## 4. Relations carried from March through July

These twelve relations are the dissertation and simulator line. They explain when E08 is an identity and when it is an estimation problem.

### E19. Observable conjugate

Lemma Z1 replaces the latent residual, which the observer does not have, with a conjugate built from the observations. The latent antithesis υ − ξ is available only in an oracle experiment. An observable antithesis has to be a function of the channels that were actually measured.

### E20. Observable projector and the Wiener estimator

For a jointly Gaussian model

```
s ~ N(0, V_s),  n ~ N(0, V_n)
y_A = √η (s + n) + q_A
y_B = √η k n + q_B
```

the minimum-mean-square estimator is the conditional mean

```
ŝ = Cov(s, y) Cov(y, y)^{-1} y
MSE = V_s − Cov(s, y) Cov(y, y)^{-1} Cov(y, s)
```

with weights w = Cov(s, Y) Cov(Y)^{-1} and ŝ = w_A y_A + w_B y_B. The observable projector is constructed so that, when those weights are the Wiener weights, the projector and the conditional mean agree. That equivalence is the content of the lemma. It holds inside the Gaussian model with known parameters. Calibrating the covariances from data is a separate problem, and it is the problem the L4 convolutional estimator was attempting. Campaign A’s cosine of 1.0 is the oracle end of this story, where υ is known. The lemma is the non-oracle end, where only y_A and y_B are known and the weights have to be right.

### E21. Raw map, and why it is not an involution

```
R(y) = y exp(−λ ξ)
```

scales the whole observation. A uniform shift c then scales as well:

```
R(y + c) = R(y) + c exp(−λ ξ)
```

which equals R(y) + c only when the exponential is 1. The center-preserving map

```
A(y) = μ + (y − μ) exp(−λ ξ)
```

satisfies A(y + c) = A(y) + c when the center shifts with the data. Campaign C recorded the distinction on a synthetic series, seed 11: flux error 1.4723521 and 1.4723523 across a shift of 5, raw error 0.7735443 then 18.1708527. On the Melbourne daily minimum, seed 11, flux error 5.8189759 at both offsets, raw error 135.7728271 then 266.6606750. Forty passes of A at fixed ξ collapse to μ + (y − μ) exp(−40 ξ). The one-board kernel matched that closed form within 9.5×10⁻⁷.

### E22. Latency ratio

A validity ratio used in the earlier notebooks compares a kernel time with a budget or with a baseline time. The March figure 0.2379 milliseconds was a budget typed into a cell. Campaign A’s median near 0.022 milliseconds and the specification’s band near 0.012 milliseconds are short-batch figures. Campaign C at fifty million nodes is 0.548 milliseconds. Flatness inside a launch-bound regime is not flatness at the long length. Any ratio should name both times and both lengths.

### E23. Wiener weights

The weights in E20. They are optimal for the Gaussian model when the covariances are the true covariances. They are not learned by the sign-bit exclusive-or. The sign bit needs the residual in hand. The weights are what you use when you do not have it.

### E24. Analytic residual

The MSE expression in E20 is the analytic residual of the Gaussian model. It falls when the reference channel y_B carries information about n and the weight on that channel is the conditional weight. It does not fall merely because a projector symbol is written next to the observation. The formula is the ceiling of that model, not a GPU timing.

### E25. Idempotency

```
Υ² = Υ
```

Projecting a state that is already structural leaves it there. The unit tests cover this on the latent chart. It is the algebraic reason a second cancellation pass, given a perfect first pass, should be the identity. A measured change on a second pass means the first pass was not Υ.

### E26. Latent involution

```
Ĉ_S² = I
```

on the chart (υ, ξ). Flipping the residual twice restores the thesis. The test `latent_involution_holds` checks it. The hypothesis fails if the object being flipped is not the residual.

### E27. Sign-bit proxy

```
x → −x  by  XOR(bits(x), 0x80000000)
```

with the mask held as an unsigned 32-bit integer. A signed interpretation of the same bits overflows. Campaign C recorded a residual of 0 against unary minus on each scored board. A telemetry block of 5,000,000 states records variance 2.4656 before this override and 0.0 after it, in 10.1 milliseconds, residual 0.0. That is E08 realized by a bit operation on a centered deviation. It is the same algebra as Campaign A’s projector, in a different floating-point width and on a different buffer.

### E28. Intercept ratio

A general ratio of an intercepted path’s time to an unintercepted path’s time. The long specification states 0.234 milliseconds against 2.086 milliseconds for a five-pass baseline, a ratio of 8.93, and a later pair 0.120 milliseconds against 25.59 milliseconds, a ratio of 212.57, which that specification calls an energy-efficiency multiplier. Those pairs are statements in that specification. They are not rows of Campaign A’s 0.022-millisecond table, and they are not the Campaign C pair of 7.38 milliseconds fused against 17.13 milliseconds eager. Each pair needs its own baseline in the same sentence.

### E29. Advantage against baselines

The long specification compares a raw series, an exponential moving average, a Kalman filter, and an adaptive reference under Gaussian and impulsive noise with drift. It states that the adaptive reference held up in that audit, and it gives one qualitative example at a change time of 25 and a drift of 30 degrees. It does not print the variances of that cell in the extract used here. The adaptive reference is identified there with E11, E12, and E13. Campaign C’s moving-average comparison on Melbourne is a different baseline: flux error 5.819 against a moving-average error of 5.897 on seed 11. Flux was slightly better than the moving average on that seed, and raw damping was not in the contest once the readings sat far from zero.

### E30. Cancellation residual ratio

The phase-2 residual ratio is the normalized leftover after a cancellation pass. Campaign A’s maximum absolute error, printed as 1.9 times a negative power of ten on the 64 by 8,192 batch, is that kind of leftover, with the exponent not recoverable from the PDF text. Campaign C’s Triton-versus-formula residual is 4.8×10⁻⁷, and its closed-form gap is 9.5×10⁻⁷ on one board and 1.2×10⁻⁵ on board 0 of the four-board machine. Those are the deposited residuals. A ratio without a denominator is not a result.

### The ladder, carried with the flux map

Density is the mean absolute instability. The launch width is 128 below 0.5, 256 below 1, 512 below 2, and 1024 otherwise. Julia’s `@sphota` and the Python twin agreed on densities 0.30, 0.79, 1.50, and 7.06. A gate predicate opens at a stability index of 0.20. At 0.05 it was closed. This ladder schedules E21’s flux map. It was not the launch schedule of Campaign A’s float16 projector table.

## 5. Empirical ledger

Campaign A, RTX PRO 6000, float16, sixty iterations after warmup, batch 16 by 65,536. Variance ratio means var(thesis) / var(recovered). Cosine similarity is against the true structural component.

| Noise std | Variance ratio | Median latency | Cosine |
|---|---|---|---|
| 5 | 26× | 0.0223 ms | 1.0000 |
| 10 | 101× | 0.0218 ms | 1.0000 |
| 25 | 627× | 0.0218 ms | 0.99999 |
| 50 | 2,500× | 0.0216 ms | 0.99995 |
| 100 | 10,026× | 0.0217 ms | 0.99979 |

### 5.1 Why those five ratios have that shape

Assume the structural component υ is drawn with variance 1, the residual ξ is independent of υ and has standard deviation σ, and the projector returns υ exactly. Then

```
Var(thesis) = Var(υ + ξ) = Var(υ) + Var(ξ) = 1 + σ²
Var(recovered) = Var(υ) = 1
Var(thesis) / Var(recovered) = 1 + σ²
```

The five noise settings in the table are 5, 10, 25, 50, and 100. The five values of 1 + σ² are 26, 101, 626, 2,501, and 10,001. The printed ratios are 26, 101, 627, 2,500, and 10,026. The match is the sampling variation of a finite batch around that identity, not a second physical effect. A cosine of the recovered tensor against the true υ is 1 when the recovery is exact, and the printed cosines fall from 1.0000 only in the fifth digit at the largest noise, which is where float16 has the least room to hold a small signal next to a large residual.

This is the scientific content of Campaign A. The table confirms that the implementation performed the cancellation it was given, and that the time of that implementation did not grow across those five noise settings. It does not confirm a new scaling law beyond 1 + σ². An investor slide that presents 10,026 times as a property that will appear on a customer’s data, where σ is unknown and υ is not handed to the score, has left this derivation behind. On customer data the ratio cannot even be computed in this form, because Var(υ) is the unknown.

A CPU illustration of the same identity, with one million normal draws and an exact recovery, sits on 1 + σ² to sampling error and returns cosine 1. It is an illustration of the algebra. It is not a new graphics-processor run, and it is not a substitute for the printed medians of 0.0216 to 0.0223 milliseconds, which remain a property of the Colab machine.

The same identity explains the free-energy proxy. If the proxy contains a term −T ln(Var) and the variance that enters it is the residual variance, cancelling a residual of standard deviation 12 replaces a large variance with the variance of υ. The logarithm collapses, and a proxy that started near 140 can land near 1.0 without any change in a thermodynamic internal energy U. The collapse is real as a collapse of the proxy. It is the logarithm of the variance ratio in disguise.

Float16 is the right place to look for the small departure from a perfect cosine. A residual of standard deviation 100, added to a structure of standard deviation 1, asks a 10-bit mantissa to remember a part that is one percent of the noise, then to subtract the noise away. The printed cosine of 0.99979 at that setting, and a maximum absolute error on the order of a negative power of ten at noise standard deviation 8, are the rounding record of an otherwise exact identity. The whitepaper’s conclusion describes the reconstruction error as that small quantity. The exponent glyph does not survive text extraction of the PDF, so this report does not invent one.

Same campaign, other printed points: 64 by 8,192 at noise standard deviation 8, variance ratio 64.9, cosine 1.0, maximum absolute error 1.9 times a negative power of ten. For a unit-variance structure and an independent residual of standard deviation 8, 1 + 8² = 65. The printed 64.9 is that identity on a finite batch. Free-energy proxy on 64 by 8,192 at noise standard deviation 12, about 140 before and about 1.0 after. Stability −6.57 structured, −9.04 pure noise. Mirror mean squared modulus 1.93 on 1,000 samples.

Campaign B, as the whitepaper classifies it, prior internal and not reproduced by Campaign A: H100 variance ratio 35,037 at median 95.23 microseconds, P95 118.87 microseconds, cosine 1.0000, Triton sign-flip kernel launched. A100 pure-cancellation median 97.28 microseconds, described as flat when sequence length grew by 16, and about twice a three-pass baseline on that note’s accounting. L4 convolutional estimator up to 94 percent mean-squared-error reduction at noise standard deviation 4 when trained on residual energy, and about 91 percent when a high-quality noise estimate is already available. A hybrid residual in that note is described at about 7 to 14 percent. Mild damping at λ0 = 0.0183 on H100 is reported at a variance ratio of 1.01. A non-negative stability coefficient on that H100 note has mean 0.3744.

Campaign C, labelled exam, 14 November, center-preserving flux, λ = 1. One board 17 of 18, four boards 16 of 17. The miss is variance remaining 0.5149549, 0.5176691, and 0.5095068 on the contaminated Melbourne series. Synthetic flux errors 1.4723521 and 1.4723523. Fused forty-pass kernel 7.3776 milliseconds, eager loop 17.1343 milliseconds. Five minutes: 41,186 jobs and 137.28 per second on one board, mean 314.53 watts, peak 326.61 watts, 32 °C to 44 °C. Four boards: 164,748 jobs, 549.15 per second, means about 277 to 283 watts, peaks 47 °C and 48 °C. Queue 29.51 milliseconds against 8.28 milliseconds together. Fair clock of 15 November, after compilation: queued 29.27 milliseconds, slowest device event 8.24 milliseconds. Roofline at 50,000,000 nodes, 16 bytes per node: 1,459.0872 gigabytes per second, 91.36 percent of a stated peak of 1,597. A different cell moved 16 bytes and recorded 54.58 gigabytes per second. Those two bandwidths are not the same experiment.

The long technical specification adds further hardware sentences that this ledger keeps in their own column: Blackwell throughput stated at 1,359.88 gigabytes per second with a per-token intercept of 0.0299 milliseconds; an 8192 by 8192 grid stated at 0.38 milliseconds; bandwidth statements of about 1,755.5 and 1,771.76 gigabytes per second, the latter at a stated 134 million parameters; a 10,000-evaluation jitter note with mean 194.72 microseconds and P99.9 of 281.22 microseconds; a later jitter note with mean 194.87 microseconds, P99 219.15, P99.9 247.81. An A100 kernel median near 0.2 milliseconds is also stated there. These figures are not recomputed in this report. They sit beside Campaign C’s 1,459.0872 gigabytes per second rather than replacing it.

Campaign D remains the four-page specification’s 958,874 times at noise standard deviation 1,000 and the 0.0117 to 0.0125 millisecond band. It is not interpolated from Campaign A’s table. A reader who wants that peak in a filed example needs the file that produced it.

## 6. Projected costs and projected revenue

This section is the investor section. Every dollar figure in it is a planning figure taken from the November 2026 venture report and the accompanying technical specification, or, where noted, from the 14 November memorandum. None of it is contracted revenue. None of it is an offer of securities. The company is described in those documents as bootstrapped, with about 5,000 dollars on hand.

### 6.1 The raise those documents state

The venture report and the technical specification state a seed of 7.5 to 10 million US dollars over 18 to 24 months. The stated purpose is a software stack, including the language layer and an operating-system layer named in those documents, two commercial prototypes, customer support, and initial design-partner deployments. The technical specification also names a novel integrated-circuit design inside that same envelope. The stated team range is 34 to 62 people on the core technical staff, plus a customer-support group of 15 to 20 people for each of the two products. A headcount of that size, over 18 to 24 months, is the cost object the raise is being asked to carry. Those documents do not publish a salary line for each role. Finance still has to turn the headcount into a month-by-month cost before a close. What they do publish is the envelope and a split of proceeds.

The four-page specification states the split as 40 percent engineering, 25 percent intellectual property, 20 percent product prototypes, and 15 percent design partners and go-to-market. Applied to the two ends of the stated range, that split is arithmetic, not a new forecast:

| Use | Share | At 7.5 million | At 10 million |
|---|---|---|---|
| Engineering | 40 percent | 3.00 million | 4.00 million |
| Intellectual property | 25 percent | 1.875 million | 2.50 million |
| Product prototypes | 20 percent | 1.50 million | 2.00 million |
| Design partners and go-to-market | 15 percent | 1.125 million | 1.50 million |
| Envelope | 100 percent | 7.5 million | 10 million |

The intellectual-property line is the provisional, the search, and the later utility path, not a budget that by itself files a claim of semantic filtering. The prototype line is the two products named below. The engineering line is the stack and, in the longer specification’s wording, the chip design. A chip design and a software library do not have the same burn, and a reader should not assume the 40 percent is only salaries for kernel work.

An earlier memorandum, dated 14 November and written against Campaign C, used a smaller working ask of 1,250,000 dollars for 18 months, inside a band of 1 to 2 million, with a lean alternative of 700,000 dollars for 12 months, and a design-partner planning price on the order of 60,000 dollars a year. That memorandum’s people budget was two hires plus a founder, not a staff of 34 to 62. Both plans are in the November record. They describe different companies: a library and one partner, or a stack, two products, a support organization, and a chip effort. This report does not collapse them into one number. The dedicated planning case of the documents listed at the top of this section is the 7.5 to 10 million dollar case. The 1.25 million dollar case remains the plan that matches Campaign C’s narrower claim set.

### 6.2 Projected revenue

The venture report states internal planning revenue, and it says these are planning figures, not commitments.

| Year | Stated planning revenue | Stated operating note |
|---|---|---|
| 1 | 1.2 to 1.8 million | operating loss expected while development continues |
| 2 | 4.5 to 6.5 million | operating loss reduced |
| 3 | 9 to 12 million | operating profit stated at 1.5 to 3.0 million |

The stated sources of that revenue are institutional subscriptions for an education product and consumer and enterprise subscriptions for a security product. Year 1 at the top of its band, 1.8 million, does not cover a 7.5 million dollar raise spent inside the year. The raise is the bridge across the loss the table itself predicts. Year 3 profit of 1.5 to 3.0 million on 9 to 12 million of revenue is a margin of roughly one sixth to one quarter, which is a planning assumption about support costs and not a measured margin. No design-partner contract is reported as signed. The global artificial-intelligence infrastructure market, which the venture report says is projected to exceed 100 billion dollars by 2030, is an industry figure those documents cite. It is not Ceraph’s revenue.

### 6.3 What the products are, in this plan

The education product is described as a platform trained on curated academic material, including openly available university courses, and as an institutional subscription. The security product is described as a continuous safety layer for personal computers, tablets, and phones, using the cancellation and the mirror as anomaly suppression. Both are intentions attached to the raise. Neither is a measured revenue stream. A product that depends on refusing a false sentence depends on a result this report does not have. A product that embeds the oracle projector, or the center-preserving flux map, inside a pipeline the partner already runs, depends on the campaigns above and on a partner’s own series, which has not yet been scored.

### 6.4 Cost risks that sit inside the envelope

A staff of 34 to 62, plus two support groups, spends a seed of this size quickly if hiring runs ahead of a design partner. The specification puts a chip design in the same window as the software stack. Silicon tape-out is a different cost curve from a Triton kernel, and the documents do not publish a tape-out quote. Intellectual property at 25 percent of the raise is ample for counsel and for foreign filings, and it is also large enough to be spent on claims the measurements do not support. The disciplined use of that line is the claim set in the attorney brief that accompanies this report. The revenue table assumes both products find paying users inside three years. The venture report’s own open items are the honest constraint on that assumption: the gap between a public residual estimator and the internal estimator that reaches about 94 percent, end-to-end tests on open models against strong baselines, and one or two design partners in education or in a regulated setting. Until those items close, the year-1 band is a target, not a run rate.

## 7. What is not established

A global scale on a logit vector preserves the winning token. A stability coefficient that is negative on both structured logits and pure noise does not, on the sheets in hand, separate a true answer from a false one. TruthfulQA on 100 items scored 23 with a later dialectical rule and 23 without it. ARC-Challenge on 100 items scored 50 without that rule and 44 with it. Those sheets are why a reliability claim in this report is a claim about a numerical residual, not a claim about knowing when a sentence is false.

The half-variance bar on the Melbourne series was missed. About 51 to 52 percent of the variance remained under the soft flux map. The oracle projector of Campaign A is a harder cancellation, and it is harder because the residual it subtracts is the residual that was added. Transferring Campaign A’s 10,026-times ratio onto a series whose residual was never supplied would be a different experiment, and it has not been deposited.

Landauer’s bound was not metered on the die. NVLink was not measured. The 16-byte message at 54.58 gigabytes per second is not the 1,459.0872 gigabyte-per-second traversal. The lattice sentence in the patent specification, a map from a lattice ciphertext into a continuous-variable tensor followed by E08, is a proposed embodiment. The 0.38 millisecond figure in the long specification is a stated timing of an 8192 by 8192 grid. It is not a demonstrated break of a standardized post-quantum scheme.

## 8. One argument, in physical language

The papers call the flip a discrete symplectic conjugation. The word is justified inside the model they define, and it is easy to over-read. A symplectic map on a classical phase space preserves the symplectic form, and in particular sends a pair (q, p) to a pair whose signed area is unchanged. The operator Ĉ_S, on the chart (υ, ξ), sends (υ, ξ) to (υ, −ξ). Applied twice it returns the original pair, so it is an involution. It reverses the residual the way a momentum reversal reverses p and keeps q. That is the analogy. It does not, in any file in this record, come from a measured canonical form, a measured Poisson bracket, or an optical symplectic transformation on a laboratory bench. The squeezing discussion in E15 is the nearest the repository comes to that laboratory language, and the repository’s own note on it says the simulations are classical random numbers.

Read in that lighter sense, the eighteen equations are one argument with three layers.

The first layer is kinematics. E02 packages structure and residual in one object so they are not thresholded apart too early. E03 names the split. E04 through E08 say that if you have both pieces, the average of a state and its residual-reversed copy is the structure, in one fused multiply-add. E25 and E26 say the operation is a projection and an involution: doing it twice is the same as doing it once, and reversing twice is doing nothing. E27 is the same reversal written as a bit.

The second layer is bookkeeping for the cost of the other approach. E01 recalls that erasing a bit has a floor. E09 and E10 say that if the residual variance is the thing playing the role of disorder, then removing it collapses a disorder term and a free-energy proxy built from that disorder. E18 asks that the generator of any larger model be Hermitian, so probability is not silently lost at the boundary. None of this layer is a wattmeter reading. Campaign C’s watts are a wattmeter reading of a different fact: the boards drew roughly half their nameplate and stayed under 50 °C.

The third layer is what you do when you do not have both pieces. E11 and E12 and E13 are a gain scheduled by a statistic of the mixture. A gain cannot change an argmax. E14 compresses a variance so a large residual and a small one can be looked at on one scale. E16 and E17 are checks that do not consult an outside table: agreement of two programs, and a squared modulus. E19 through E24 are the honest estimator. Two noisy views of a Gaussian source have a best linear estimate, the conditional mean, and the observable projector is that estimate when its weights are the Wiener weights. E15’s squeezing formula belongs here too, as a way of moving uncertainty into the reference channel before that estimate, inside a model, not on a bench. E21 is the special estimator that does not need a second channel because the structure is defined as a local center: shrink the deviation, add the center back, and a uniform shift of the data shifts the center with it.

Campaign A is a confirmation of the first layer in float16, and section 5.1 is the reason the confirmation looks like a dramatic ratio. Campaign B is a partial confirmation of the third layer, on other chips, at the percentages those notes state, and the November whitepaper tells the reader not to treat Campaign A as a repetition of it. Campaign C is a confirmation of E21, including the place E21 did not meet a variance goal the company had set. Campaign D is a sentence in a four-page specification whose peak ratio is not a row of Campaign A. A technical report that keeps those four in one paragraph has not understood the argument. A technical report that drops any of them has not disclosed it.

## 9. What the raise is buying, in operations

Section 6 prices the envelope. This section says what the work is, so a cost is not only a percentage.

Engineering, at 40 percent, is the stack the specification names: the language layer, the operating-system layer, the kernels that already exist for the projector and for the flux map, and the chip design the longer specification places in the same window. The kernel work is the part Campaign A and Campaign C already constrain. The chip design is not constrained by a tape-out, and a reader should expect that part of the 40 percent to be a study and a specification unless a foundry quote is added. A staff of 34 to 62 is not required to keep a Triton kernel correct. It is the specification’s number for the wider stack. Hiring to that number before a design partner is using the seed as a payroll, which section 6.4 already flags.

Intellectual property, at 25 percent, is counsel, the search, the provisional, the utility, and foreign filings. The brief that accompanies this report is the input to the provisional. The line is large enough to be misspent on claims the tables do not support. The claims that fit the tables are the split, the flip, the average, the time on the tested batch, the scaling of 1 + σ² under a known residual, the gain control with its formulas named, and the level-preserving map with its miss included.

Prototypes, at 20 percent, are the education product and the safety product as software, not as revenue. Each is a place the projector or the flux map might later sit. Neither has a user study in these files. Support groups of 15 to 20 people per product are a cost that begins when a user exists. Spending them in year 1, while the venture report itself predicts an operating loss, is a choice to fund support ahead of use.

Design partners and go-to-market, at 15 percent, are the cost of sitting with one or two outside parties and of a public face. The venture report’s own next step is those one or two partners, in education or in a regulated setting, plus a public comparison of the blind estimator against strong baselines. That next step is also the only path from the oracle identity to a customer number. Until a partner’s series is scored with the bar written first, the year-1 revenue band has no measured input.

The smaller plan, 1.25 million dollars, buys a narrower slice of the same list: the flux library, the agreement tests, counsel for a provisional on that library, cloud time on the class of board already measured, and the cost of showing up for one partner. It does not buy a staff of 34 to 62, a chip, or two support organizations. A company that stands on Campaign C and E21 can raise the smaller plan without pretending to the larger one. A company that stands on the eighteen-equation stack, the two products, and the chip study is describing the larger plan, and should say that the dramatic ratios are 1 + σ² under a known residual.

## 10. Conclusion

The mathematical core is an exact cancellation when the structural component and the residual are both in hand, and it is an estimation problem when only their sum is in hand. The exact case is E04 through E08, realized either by subtraction or by the sign bit E27, and it is idempotent and involutive on the latent chart. Campaign A is a float16 reading of that exact case on an RTX PRO 6000: variance ratios from 26 times to 10,026 times as the injected noise grows, cosine similarity to the true structure at or above 0.99979, and a median latency that stays near 0.022 milliseconds across that sweep. The free-energy proxy falls from about 140 to about 1.0 because the variance it is built on falls. The stability coefficient in that series is negative on both a structured draw and a pure-noise draw, less negative on the structured draw.

The estimation case is E19 through E24. It is solved, inside a known Gaussian model, by the Wiener weights, and it is unsolved as a general observer of logits. The whitepaper says so: the internal estimator near 94 percent mean-squared-error reduction, and the H100 ratio of 35,037 times at 95.23 microseconds, are not what Campaign A reproduced. The center-preserving exponential, E21, is the map that survives a change of level. Campaign C showed that survival on a synthetic series and on a 3,650-day temperature record, showed a fused kernel faster than an eager loop, showed four boards doing four times the heavy jobs over five minutes, and missed a variance bar the company had written down in advance.

Eighteen equations name the November core. Twelve more carry the March–July theory that tells a reader which of those eighteen are identities, which are estimators, and which are still proxies without a residual row. The planning case attached to the broader reading of this architecture is a seed of 7.5 to 10 million dollars, a stated revenue path from 1.2–1.8 million in year 1 to 9–12 million in year 3, and a stated year-3 operating profit of 1.5 to 3.0 million, with the company presently bootstrapped at about 5,000 dollars. The narrower case, matched to the flux library and one partner, remains the 1.25 million dollar working plan. Both are unbooked.

What a diligence room can defend today is the identity, the conditions under which the identity becomes an estimate, the tables in section 5 with their campaign labels, and a raise sized to the plan the company chooses to stand on. What it cannot yet defend is a sentence filter, a metered Landauer saving, or a transfer of the oracle variance ratios onto data whose residual was never supplied. The level of a reading is kept when the map is built to keep it. The residual is removed to the degree the chosen equation, on the recorded campaign, actually removes it.
