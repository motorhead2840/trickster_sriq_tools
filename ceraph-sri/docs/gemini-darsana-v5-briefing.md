# Briefing for Gemini: SRI version 5

Ceraph Corporation, Kerala, India. 15 November 2026. chief@ceraphco.com. www.ceraphco.com.

This briefing is the engineering record of a conversation that read Narayana Guru’s Darśana Mālā through Nataraja Guru’s prologue, verse translation, commentary, and epilogue, for all ten visions, the introduction to Part Two, the Main Conclusions, and the Advaita Dīpikā. It is a summary of that mapping. It is not a reproduction of Nataraja Guru’s book. Do not reconstruct the commentary. Build from the rules, equations, and test numbers below.

SRI version 5 is the current design. It replaces earlier public wording that called the same work a constant-time quantum cancellation, a Landauer bypass, or a finished filter of false sentences. Those wordings are withdrawn. Version 5 is a classical calibration-and-control architecture on tiles, plus an oracle cancellation that is exact only when the residual is known, plus the measurements already deposited. Notebook v5, dated 11 November 2026, is the selection-rate model inside this design. The later November hardware numbers are an evidence annex. They do not restore a dropped claim.

## What you are being asked to build

Build the v5 stack as software on existing graphics processors, with a path toward a self-sensing digital tile. The unit is a tile that senses its own context, estimates a residual or a relative veil, corrects by a gain, and does not depend on a central controller. The value is in the estimator and in the damping behavior, not in the cancellation identity. The identity

```
y = (x + x') / 2 = v    when the residual used to build x' is the true residual
```

is classical subtraction. It holds up to floating-point rounding when the residual is known. A sign-bit exclusive-or with unsigned 0x80000000 flips a known float32 value. It does not discover an unknown residual. If the sign bit is applied to the whole observation, the result is −v − ε, not v − ε.

Every mapping below was tagged in the conversation as one of four grades. Keep the tags.

- Proven or measured: a theorem or a notebook number.
- Theory: a design rule with a stated test.
- Interpretation: a philosophical reading. Do not put it in a claim or a product sentence.
- Axiology: a value. Do not map it to mathematics. Mercy and non-harm, named at the close of the karma chapter, stay as the company’s values and are not equations.

## Axioms

The conversation opened from a paraphrase of Bṛhadāraṇyaka Upaniṣad 5.5.1: the outer syllables are truth, the middle is untruth, and truth holds the untruth on both sides. Hari’s reading, which is the design reading:

- Satyam is the whole, signal plus an irreducible error.
- Ṛtam is the signal v.
- Sa is magnitude and yam is phase.
- Ta is one complex veil τ over both.
- Anṛtam is the regime where error outweighs signal.

Axiom 1. The veil has a floor. Variance of the residual is at least σ²_min > 0. A balance B = ‖v‖² / ‖ε‖² must stay above a critical value. Driving the residual below the floor removes signal. The loop regulates B. It does not target zero error.

Axiom 2. The floor depends on context c. The tile estimates context as well as error.

Axiom 3. Each tile estimates its own context from a local window. There is no central controller. The open hardware question is how neighbors cross-check one another without a large delay.

## Vision 1. Projection

Source of method: Nataraja Guru’s prologue and epilogue to the first chapter, plus the ten verses. Adhyāropa is a deliberate supposition, later taken back. That is the working rule for every axiom: assume it, then remove what fails a test.

The ground, before a readout is chosen, does not yet have the predicates “exists” and “does not exist.” A state has no definite value for an observable until a basis is chosen. The readout is

```
y = P x + n
```

The tile’s own law, called dharma in the conversation, is a generator ℒ. A tile holds one steady state when ℒ has exactly one zero eigenvalue and every other eigenvalue has a negative real part. The spectral gap is how strongly it holds. Notebook v5 already computes the related fidelity and rate ratio, defined below.

Manifestation needs a drive against the veil. With the drive off, the veil wins. History selects a steady state only when a symmetry or a nonlinearity supplies more than one sector. A linear memory register forgets its seed. That conflict, between “the seed selects the state” and “the law has one fixed point,” was found inside the axioms and settled by a test: dharma is unique per symmetry sector, and the seed picks the sector.

A design principle, not a derived constant: expose a fraction of each tile’s state and keep the rest unread. The conversation recorded one quarter manifest and three quarters unread as Hari’s principle. A decoherence-free subspace does not produce that ratio. At two qubits the protected share in the test was 3/8, and it falls as the register grows. Do not print 3/4 as a measured constant.

The data-processing inequality was taken as a theorem: a projection cannot create information. In 8,000 trials a proper projection never increased mutual information. A side channel that was not a projection did so in 62 percent of trials.

Verse-level rules to keep: the law is time-invariant and the state evolves; light and dark are complementary projectors, P(I − P) = 0, and the split is per subspace, not one scalar; one latent register plays the roles of memory, will, and drive; without a reference, tiles look like unrelated absolutes, and with one reference they collapse to offsets; readout of a whole is one global solve, not a chain whose error grows with the graph’s resistance; a tile’s life is prepare, hold, reset.

Method rules from the prologue and epilogue, binding on v5:

- State which level a result lives on: invariant structure, or the time evolution.
- The measurement is inside the model.
- Units must match. The old stability coefficient mixes variance and a mean and is not portable across a rescaling.
- Do not report a tautology as a measurement. A free-energy proxy that is a logarithm of the variance it just reduced collapses by construction.
- Negative results stay in the record.
- Invert a projection only on its range, never on its null space.
- Unmanifest means not yet selected, not a second world and not zero.
- Tag every sentence as measured, proved, theory, or interpretation.
- The generator and the states it produces belong to one model class.
- Do not treat a fitted ordering as a measured process.

Two scientific asides in that prologue were refused as evidence: a steady-state cosmology, and Piccardi’s galactic-field chemistry.

## Vision 2. Taking the supposition back

The effect has no being apart from the cause. The reading y is a view of x, not a second substance. Estimation recovers the part of x that P shows, down to the floor the veil sets. After a correct cancellation there is no leftover “created world” to delete.

The unmanifest is the directions a given family of readouts does not select. It is not a rival region. Reading “what does not shine is not real” as “nothing real is unobservable” was marked as metaphysics.

Engineering shape: a constrained two-pass estimator. Descend from effects toward causes, stop with a rank limit so the latent count does not regress forever, then rebuild. Agreement and difference are ablation and intervention, not slogans. Estimate the state. Do not reify features of the reading.

The epilogue’s picture of one vertical flux that stays steady while expansion and contraction alternate is the conversation’s statement of an invariant. The test is: find quantities that stay fixed across many readout families.

Three zones, to be printed on every result: the working model and the data; the residuals, aliases, and prior–data mismatch; the invariant content after cancellation. The third is not a computable remainder.

Further method, binding: a baseline is required to fix a gauge; coordinates are conventions and invariants carry the content; ten copies of one newspaper are not ten confirmations, so tiles that share a reference are not independent samples; an unfalsifiable claim is interpretation; label each residual by kind before choosing a correction. Bergson’s claim that dilated clock time is unreal was rejected. Parity is not a symmetry of the weak interaction; only the method point was kept, that a reflection is a map and not an identity.

## Vision 3. Appearance

What is projected has no being apart from the ground. The important engineering sentence is vivarta rather than a transformation that uses the ground up: a classical readout leaves x unchanged. That fails for a quantum measurement, which disturbs the state. v5’s default tile is classical. Do not claim a non-disturbing quantum readout.

Other rules: a strong prior on low signal-to-noise data invents a confident false object, and more signal collapses it; one reading can support two hypotheses until a second, differently placed reference separates them; the same data under a richer model can resolve what a poor model cannot; an estimator cannot fully estimate itself. The blue of the sky was flagged as Rayleigh scattering in the light–air interaction, not as a retinal mistake. Appearance can belong to the interaction.

## Vision 4. Graded error

Māyā, read here as measurement and error, is everything that can be read through P. The chapter’s eight aspects became an error taxonomy: model error, data error, missing information, instrument error, and a non-identifiable remainder. Each has its own correction. A single noise term is not enough.

A non-identifiable remainder is a component in the null space of P. In the identifiability test, error in the null directions stayed flat (ratio 0.988 against the prior) while error in the range fell by a factor of about 1,000. A correlated prior reached its predicted floor and did not go below it.

Heisenberg uncertainty, entropy, and osmosis were kept as analogies, not as thermodynamics of the chip. The conversation refused to map the chapter’s remarks on caste.

## Vision 5. Normalization

A reading means something only against a reference. Two results carry the design.

The reference-tile rule, from the verse that an eye cannot see itself: a tile cannot calibrate itself on its own channel. The reference must be independent of the readout, it must carry two known levels, and without an anchor only relative quantities are identifiable. In the test, the rank of the calibration rose from 1 (self only) to 1 (self fed back) to 2 (one known level) to 3 (two known levels). Two tiles with no anchor had a gauge nullity of 2. Variance sat on the Cramér–Rao bound (ratio 1.001). One pre-registered check failed because the error-floor test was run where noise still hid a small bias; a post-hoc look sat on the analytic floor. The verdict was not edited.

Lock-in detection, an in-phase and quadrature readout, beat a power detector by 17.3 dB and landed at 1.005 times the Cramér–Rao standard deviation. Thirty-two percent of single-quadrature trials were off by more than half. Use both quadratures.

Quantum logic was checked as a formal tool, not as a computer: distributivity failed in 1,000 of 1,000 triples in ℂ² and in ℂ⁴, and commuting families stayed Boolean. That is textbook. It is not a speedup.

The four-layer grid, the two birds, and the strong reading of “being alone” were marked interpretation. Nataraja Guru’s reading of Michelson–Morley as a fixed absolute space was rejected; physics reads that experiment as no preferred rest frame. The travelling twin is not a myth. Those flags stay in v5 so they are not rebuilt.

## Part Two. From estimation to control

Nataraja Guru’s introduction to Part Two was read as the turn from estimation to control. The design rules that survived:

- Still some modes and pass others. The veil is a selector, not a mute.
- Reflection (a gain error) is cured by calibration against a reference. Refraction (a bent model) is cured only by fixing the model.
- A model must be smaller than what it models.
- A delayed effect needs a lag window. Immediate-only attribution failed whenever the true lag was above zero. Lags 0, 3, 7, and 12 were recovered in 200 of 200 runs each.
- Weber–Fechner is an approximate law. Stevens’ power law fits many senses better. Do not build the witness on the logarithm alone.
- Certitude needs two sides: a derivation and a test that was allowed to fail.

The catalogue of mystical states was recorded and not mapped.

## Vision 6. Instrument and action

Instrument and action stand on the same footing. The test is controllability, dual to observability. A random pair was controllable in 200 of 200 draws. A path of five tiles had full rank from an end drive and rank 3 from a drive at the centre. A symmetric network driven only at its centre is the failure mode.

The reference must also be isolated from the actuation path, not only from the readout. Leakage biases the calibration in proportion to the leak. At leakage 0.02, 0.05, 0.1, and 0.2 the bias was 0.013, 0.038, 0.081, and 0.170, matching the formula. A leak-aware correction cut the 0.2 case to 0.002.

Three competing modes need a competition coefficient above 1. With it, the largest seed won in 300 of 300 trials and the dominance ratio was at least 1,800. Without it, no mode dominated. A drive-held cycle (Hopf) had an amplitude spread of 2×10⁻¹³ across 50 starts and disappeared when the drive was removed. A linear pair has no cycle.

## Vision 7. Knowledge

Testimony is not a ground. A claim rests on perception, inference, and comparison, in the conversation’s engineering translation: a measurement, a derivation, and a comparison against a baseline. An outside report is weighed. It is never the ground.

Efficacy means an intervention changes the outcome. Correlation called a confounded component real in 100 of 100 cases; intervention classified the real driver in 100 of 100. Planted evidence needs an independent integrity channel. A confidence channel scored AUROC 0.499. An integrity channel scored 0.929, short of a pre-registered 0.95, and 0.983 against genuine fires only. A spoofed channel scored 0.462.

A four-state readout that may abstain cut accepted error from 0.1195 (forced answer) to 0.0018, but coverage was 0.097, outside a pre-registered band of 0.5 to 0.95. Abstention is real and the coverage target was too high for that confidence. A gated fallback beat a plain Kalman filter on a stationary series (mean squared error 0.036 against 0.859) and needed a re-arm after a jump (0.181 with re-arm, 2.27 without).

The remark that Newton’s Principia is a blind alley was marked as opinion and not used.

## Vision 8. Coupling

Coupling runs from weak, through exchange, to locked. The Kuramoto model reproduced the regimes: critical coupling 1.596, order parameter 0.047 at half that coupling and 0.938 at coupling 3.2 against a prediction of 0.937, locked fraction rising from 0.039 to 1.000 by coupling 10.

A hub and an all-to-all graph had almost equal accuracy at equal budget (mean-squared-error ratio 1.062) with 16 links against 120. The hub is a single point of failure. When the hub was removed, the hub design kept no relative calibration. The all-to-all graph kept a relative root-mean-square of 0.042, which was 14 percent of the prior standard deviation. The pre-registered expectation was 10 percent. The conclusion stands: a hub that is lost keeps nothing. Do not sell the hub as the more accurate graph. Sell it as the cheaper graph, and state the failure.

## Vision 9. The loop

Feedback stabilizes what an open loop cannot, and not for free. On a drifting plant, closed-loop root-mean-square was 0.20 against 46.5 open, and a driven oscillator’s energy was 2.7×10⁻⁹ against 1. A pre-registered check with white sensor noise failed because the sensor model was wrong; a post-hoc band-limited sensor gave 0.944 closed against 0.302 open. The verdict on the written test was a fail. The post-hoc is labelled.

Persistence after a correction is lag in a linear tile and true hysteresis only in a multistable tile. Loop area at the slowest sweep was 1.70 for the bistable tile and 0.020 for the linear tile. An exponential average persisted for 45 steps against 44.9 predicted.

Training loss is not parameter convergence. On an ill-conditioned problem the loss settled at step 60 and the parameters at step 28,200, about 470 times later. Report per-tile parameter error against the reference, not a global mean and not the loss. On a consensus problem the mean was conserved to 1.7×10⁻¹⁶. A ring took 82.5 steps against a torus at 7.5. A time-to-five-percent formula missed; the asymptotic rate equalled the spectral gap exactly (ratio 1.000, post hoc).

## Vision 10. Stopping

Three rules, all known mathematics, all useful:

Stop by annealing the gain once a windowed innovation sits inside ±2 standard deviations over square-root of the window, and re-arm with a cumulative-sum test. On a stationary problem the annealed mean squared error was 0.0011, a constant gain 0.0256, and a high gain 0.328. After a jump of the target, a frozen gain stayed at 2.08 and the re-armed rule recovered to 0.061.

Hand over by keeping the old state until the new one locks. Maximum excursion was 0.201 for make-before-break, 0.309 for an instant switch, and 2.157 if a gap was left. A pre-registered “too short overlap” of 0.1 of the lock time was already enough. The check failed. The rule still matters when the new state is not yet locked.

For a far target, grade the targets or damp the step. Plain Newton converged on 14.2 percent of starts. Graded targets and a damped Newton converged on all of them.

Uniqueness of a steady state is three checks: one zero eigenvalue of the generator, agreement of two long trajectories, and a leak or a drift that fails the check on purpose. A valid qubit model had nullity 1. A dephasing-only model had nullity 2. A drifting rate of 0.252 was detected.

Psychic powers, rebirth, and the most elect grade cut off from all input were recorded and are not design goals. The conversation’s own note was that those powers do not matter for the chapter’s end, and they do not matter for v5.

## Closing standard

The Main Conclusions ask for certitude from two sides, the way a theorem can be proved by two routes. For v5 that means every mapping ships with its mathematics and a test that was allowed to fail. Hysteresis after a correction, the identifiability limit, and the eye that cannot see itself are the three closing sentences. They are T22, T5, and T6.

Standing corrections, which override any earlier wording in this briefing:

- The “supreme controller” is not a being. It is the innate law that holds a thing’s steady state.
- Creation is not a second substance. The magician is a metaphor.
- One quarter manifest is a design wish. It is not a derived constant.
- Clock time, the horizon of a readout, and whatever is unread are three different things. Do not collapse them.
- The generator ℒ and the projector P are different operators.

## The omega-dit, rewritten

The older integral mixed a time-dependent integrand with an integral over frequency and produced a number, not a state. Noise inside the amplitude also survived the product of a coefficient with its conjugate. Version 5 keeps the intent and writes a state:

```
|Ω⟩ = ∫ ψ(ω) |ω⟩ dω,    ∫ |ψ|² dω = 1
ψ = σ e^{iφ}
ψ_veiled = τ ψ,    τ = e^{η + iξ}
ρ = E_κ[ |ψ_veiled⟩⟨ψ_veiled| ],    Tr ρ² < 1
Ξ = |E τ|² / E|τ|²    in [0, 1],    Ξ ≥ Ξ_crit(c)
ψ̂ = ψ_veiled / τ̂
```

Ξ = 1 would mean no veil, which Axiom 1 rules out. The mirror product sees the amplitude face of the veil and loses the phase face. The phase face appears when tiles are compared. For a d-level tile the projector onto one level is the average of the clock operators. For d = 2 it is one half of the identity plus a sign flip, which is the oracle projector.

What τ is in a physical tile is still open. Notebook v5 gives a simulation model only.

## Notebook v5, the selection model

Date 11 November 2026. NumPy complex128, seed 11. Ξ_crit = 0.90 is a declared floor on the fidelity F = ⟨t|ρ|t⟩, not the older consistency statistic. Suite wall 10.6 seconds on that CPU.

```
F_ss = (r + 1/d) / (r + 1)
r* = (Ξ_crit − 1/d) / (1 − Ξ_crit)
```

r is the ratio of the selecting rate to the veil rate. Measured r* matched the closed form: 4.000, 5.667, 6.500, and 7.750 at d = 2, 3, 4, and 8. F_ss at r = 2 was 0.833, 0.778, 0.750, and 0.708. Selecting 1 of 8 costs a slightly higher rate ratio than selecting 1 of 2, not eight times higher. Fidelity above 0.99 was not reached on the scanned grid. F_ss at r = 16 is 0.971 at d = 2 and 0.949 at d = 8. Reaching 0.99 needs r above 49 at d = 2 and above 86.5 at d = 8.

A pseudomode memory register helped selection against a T1/T2-type veil of the same effective rate: the fidelity gain was +0.105 at d = 2 and +0.059 at d = 8 when the memory was slow, and it shrank toward the Markov value as the memory rate rose. Information backflow was clear at d = 8 (non-Markovianity 0.075 at memory rate 0.5, down to 9.8×10⁻⁵ at memory rate 16). Extra pure dephasing washed the backflow out. A fixed selector holds the floor until the veil rate drifts, then collapses (late fidelity 0.658 at d = 2 after a tenfold jump). An oracle that sees the true veil rate holds about 0.911. A Mehra-style rule that sees only a noisy fidelity (noise 0.04) recovers after the jump to about 0.892 in about 3.8 time units, chatters on a stationary veil because its setpoint is exactly the floor, and did not hold the floor through a ramp (late fidelity about 0.876). The cost in that notebook is a rate proxy, not joules.

A smoke test of two and three qubits, ring against all-to-all, changed the fidelity of the first tile by less than 0.01. That is not a large-array wiring law.

Two working names for the veil, both simulation only:

- Ongoing decoherence, optionally with a damped bosonic memory register on the tile.
- A selector. Satyam, in this model, is the pointer state the dissipation holds. The declared floor is a lower bound on fidelity to that pointer.

## Unveiling, notebooks v1 to v4 and v6

N tiles share K unknown samples. Tile n sees y_n = τ_n ψ + w_n. Neighbors form a relative veil from inner products. A common complex scale of every τ leaves every relative veil unchanged. The absolute veil is unobservable. On a 64-tile grid the ungauged phase error was 0.82 and the gauge-fixed error was 0.017.

The mean squared error of the relative estimate has a floor shared by every layout, about 1/(K · SNR), plus a layout term proportional to the graph’s average effective resistance. A ring’s resistance grows with N. A grid’s grows as the logarithm. An expander’s stays bounded. The fitted floor and layout coefficients were near 1 at 20 dB (R² 0.98) and failed below 10 dB. Partial overlap makes the layout matter immediately: at N = 576 a ring’s error was 40 to 111 times a grid’s.

Tracking. The first gain rule lowered the gain when neighbor consistency fell, which is the wrong sign when the veil starts drifting faster. After a tenfold jump in process noise its late error was 0.59. A Mehra adaptive Kalman filter, which estimates process noise from a forgotten window of innovations and never sees the true veil, stayed with the oracle (0.0540 against 0.0544). A corrected gain that raises the gain when innovations exceed the nominal model tied that filter (0.0545). Use Mehra as the tracker. Keep the corrected gain only as the fallback that still uses a stability statistic, and only with that sign.

Gauge pinning. One known-good reference cut absolute error from a free range of about 0.09 to 3 down to about 10⁻³. Further references cut the remaining gauge error about as 1/M. A stale reference frozen at the start let absolute error grow to 0.49 against 0.037 when refreshed. One faulty reference ruined a single pin (error 0.25). Refresh the reference on the timescale of the drift. If only relative error matters, leave the gauge free.

Notebook v6, seed 13, vacuum variance 1 per quadrature, is a Gaussian simulation, not a quantum-advantage claim. Squeezing the reference beats squeezing the signal for a gain k between 0.0398 and 0.9515 at the stated slice (noise variance 16 times signal variance, transmissivity 0.85, squeeze parameter 1). At k = 0.5 the mean squared errors were 0.828 unsqueezed, 0.798 squeeze-signal, 0.700 squeeze-reference, and 0.648 for a split photon budget. The slogan “the reference always wins when k < 1” is false at finite noise. Quasiperiodic wiring did not beat an expander on effective resistance. Weak links raised the resistance. Do not assume that wiring helps.

## November hardware annex

These numbers are later than notebook v5. They belong in v5 as evidence. They do not replace the tile rules.

Oracle cancellation on an RTX PRO 6000, float16, batch 16 by 65,536, sixty iterations after warmup. If the structure has unit variance and the residual is independent with standard deviation σ, and the recovery returns the structure, the variance ratio is 1 + σ². The printed ratios 26, 101, 627, 2,500, and 10,026 sit on that identity at σ = 5, 10, 25, 50, and 100. Median time stayed between 0.0216 and 0.0223 milliseconds. Cosine similarity to the true structure stayed at or above 0.99979. That cosine uses the true structure. It is not a blind estimate.

A center-preserving map, local center plus the deviation scaled by e^(−λξ) with λ = 1, keeps its error under a uniform shift. Synthetic flux error 1.4723521 and 1.4723523 across a shift of 5. On the Melbourne daily minimum, seed 11, flux error 5.8189759 at both offsets, while a map that scales the whole reading moved from 135.77 to 266.66. A pre-declared half-variance bar was missed: 0.51495, 0.51767, and 0.50951 remained. Four boards finished 549.15 heavy jobs a second against 137.28 on one, at 47 to 48 °C. At 50 million nodes one board moved 1,459.0872 GB/s, 91.36 percent of a stated 1,597. A different cell moved 16 bytes at 54.58 GB/s. Do not add those bandwidths.

Language-model sheets did not establish a filter of false sentences. A global scale preserves the winning token. TruthfulQA on 100 items scored 23 and 23. ARC-Challenge scored 50 without a later dialectical rule and 44 with it. An internal note that a stability coefficient on language-model answers had AUROC 0.47 to 0.54 is chance. Leave that claim dropped.

A four-page specification’s peak of 958,874 times, and an H100 note of 35,037 times at 95.23 microseconds, are not rows of the 1 + σ² table. The November whitepaper says the Blackwell series did not reproduce the H100 figure or an internal estimator near 94 percent mean-squared-error reduction from observed logits alone. If you cite them, cite them as those documents, not as v5’s result.

## Dropped, and not to be rebuilt

- Latency independent of length. Work scales with the buffer. Short kernels are launch-bound. Long ones follow bytes.
- A bypass of Landauer’s bound, or a thermodynamic ground state of the chip. A board that draws hundreds of watts is many orders of magnitude above kT ln 2 per bit. The bound was not metered.
- A continuous-variable quantum computer, a symplectic capacity, or optical squeezing, as a property of the digital kernel. The sign flip has determinant −1. It is not a symplectic map. Squeezing in v6 is a Gaussian noise model.
- A guarantee of no false sentence.
- Variance ratios of ten thousand to a million as a property of blind data. Those ratios are the oracle identity. Blind estimation, where it has been stated, is the separate internal figure near 94 percent under favorable noise, and that figure was not reproduced by the November Blackwell table.

## Scoreboard you must not round up

Thirty-two tests, T1 through T32, were run in `sri_darshana_validation_v2` on CPU. Twenty-four passed every pre-registered check. Eight failed one check: T6, T13, T20, T21, T24, T26, T29, T31. Several failures were errors in the test, and the post-hoc numbers are labelled. No threshold was edited after a run. All thirty-two tests are known mathematics. None is a new theorem. None is a quantum speedup. The combination of these parts against a standard calibration stack (periodic recalibration, a PID loop, a Kalman filter) on real or emulated hardware has not been run. That comparison is the next test, not a result.

## Build order

1. Lock the dropped list and the three axioms in the repository README before any new kernel.
2. Implement the tile as y = P x + n, with a generator whose steady-state count is checked by the three uniqueness tests.
3. Implement relative unveiling and the gauge. Pin one reference. Isolate it from the readout and from the actuation path. Refresh it on the drift timescale.
4. Track with Mehra’s adaptive Kalman filter. Keep the corrected gain, with the sign that raises gain when innovations exceed the model, as the fallback.
5. Use lock-in, both quadratures, as the readout.
6. Add the stopping rule with a re-arm, and make-before-break handover.
7. Report per-tile parameter error. Do not report training loss as convergence.
8. Keep the oracle projector and the center-preserving flux map as separate modes, each labelled with the condition under which it is exact.
9. Put the v5 rate formula behind a declared fidelity floor. Do not describe the rate as joules.
10. Run the architecture against periodic recalibration, PID, and Kalman on an emulated tile array: calibration error, recovery time after a jump, links, and a stand-in for area. Pre-register the bars. Record misses.

Phase A hardware, when a chip is actually started: a digital tile with in-place timing-error detection, a local gain loop, and a runtime balance floor, on an FPGA and then a tiny tape-out. Phase B, later: memory tiles whose physical noise is the veil, with a calibration estimator. Neither phase has been taped out. Analog memory chips cited in the conversation (speech, phase-change memory, resistive memory, probabilistic bits) are other people’s silicon. They are not Ceraph’s measurements.

## How to talk about the source

Narayana Guru wrote the verses. Nataraja Guru wrote the translation, the prologues, the commentaries, and the epilogues. Hari supplied that text for the conversation and corrected the reading where a scientific aside in the commentary does not match experiment. The mapping to tiles, gauges, and tests is the conversation’s work. It is not a claim that the text is an algorithm, and it is not evidence that the philosophical reading is true. The tests check the mathematics a reading produced.

If you need a longer equation-by-equation account of the November kernel, use `docs/investor/TECHNICAL_REPORT_EQUATIONS.md` on main, and keep its campaign labels. Version 5 is that mathematics, under the axioms and the dropped list in this briefing.
