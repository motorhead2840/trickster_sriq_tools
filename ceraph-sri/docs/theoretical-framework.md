# SRI mathematical framework

## Scope and evidence standard

This document integrates the six supplied unveiling notebooks with the current
`sri` package. It describes the mathematical models implemented by the project;
it is not a proof of a new physical mechanism, a quantum-hardware result, or a
claim of product performance. The corresponding experiment drivers,
reference outputs, and regression snapshots are the reproducible project
record:

- `experiments/unveiling/v1` through `v6`
- `tests/regression/snapshots/v1_key_numbers.json` through `v6_key_numbers.json`

In this document, **standard result** means a known identity or consequence of
the explicitly stated model; **prediction** means a derived, model-dependent
claim that still needs proof or broader testing; and **simulation evidence**
means a result reported for a finite synthetic experiment. None of these
categories imply independent experimental replication.

## 1. Relative veil estimation on a graph

### Observation model

Tile \(n\) has a nonzero complex gain, or veil,
\[
\tau_n = \exp(\kappa_n), \qquad \kappa_n = \eta_n + i\xi_n ,
\]
and observes \(K\) complex samples
\[
y_{n,k} = \tau_n\psi_k + w_{n,k}.
\]
Here \(\psi_k\) is the common input and \(w_{n,k}\) is tile-local noise. For an
edge \((n,m)\) in a connected graph \(G\), define
\[
r_{nm} = \frac{\langle y_n,y_m\rangle}{\langle y_m,y_m\rangle}
       \approx \frac{\tau_n}{\tau_m}.
\]
The approximation improves in the high-SNR, sufficiently sampled regime. It is
not an observation of \(\tau_n\) itself.

Writing \(A\) for the weighted adjacency matrix and \(L=D-A\) for its
combinatorial Laplacian, log-amplitude differences are estimated by the
least-squares problem
\[
\hat\eta = \arg\min_{\mathbf 1^\top x=0}
    \sum_{(n,m)\in E} A_{nm}\bigl(x_n-x_m-z_{nm}\bigr)^2,\qquad
z_{nm}=\tfrac12\left(\log|r_{nm}|-\log|r_{mn}|\right).
\]
Its normal equations are \(L\hat\eta=b\), with a gauge constraint because
\(L\mathbf1=0\). Relative phase is estimated from the leading eigenvector of
the Hermitian connection matrix whose edge entries are
\(A_{nm}r_{nm}/|r_{nm}|\). Together these give a relative estimate
\(\hat\tau_n=\exp(\hat\eta_n)\hat u_n\).

### Identifiability and reference tiles

For any nonzero complex \(c\), replacing every \(\tau_n\) by \(c\tau_n\)
leaves all ratios unchanged. Thus the data identify relative veils only, up
to one global complex factor. A mean-zero convention or a post-hoc alignment
selects coordinates in this equivalence class; neither supplies an absolute
measurement. Known reference tiles can pin the factor.

For the linear log-amplitude problem with independent, equal-variance edge
noise and exact Dirichlet anchors \(S\), the covariance on unpinned vertices is
\(\sigma^2 L_{FF}^{-1}\), where \(F=V\setminus S\). Its diagonal is the
effective resistance to the shorted anchor set, scaled by \(\sigma^2\).
Noisy, stale, or mutually dependent references require a different covariance
model. In particular, a \(1/M\) reduction in gauge variance is only an
approximation when \(M\) reference errors are suitably independent and
comparably precise.

The v4 notebook's proposed additive split between absolute and relative MSE is
not a universal identity: in general, their errors can have covariance. It
requires an orthogonal gauge/error decomposition to hold without a cross-term.

## 2. Shared inputs, edge noise, and the proposed error law

The shared-input model is different from the classical independent-edge
measurement model used in graph least-squares scaling results. Expanding the
shared-input ratio at high SNR gives a first-order log-ratio perturbation of
the form
\[
\log r_{nm}=(\kappa_n-\kappa_m)+(a_n-a_m)+e_{nm}+\cdots ,
\]
where \(a_n\) is a tile-local first-order noise potential and \(e_{nm}\)
contains higher-order, edge-dependent terms. A graph difference
\(a_n-a_m\) is a **coboundary**: on a connected graph, least squares can absorb
it as a node potential instead of accumulating it along paths. At lower SNR,
finite-sample effects, products of noise, and denominator fluctuations can
break this structure.

For the full-overlap model, v2 proposes the leading-order prediction
\[
\operatorname{MSE}(\hat\kappa)\approx
c_{\rm floor}\frac{\mathbb E|\tau|^{-2}}{K\,s}
+c_{\rm layout}\frac{\mathbb E|\tau|^{-4}}{K\,s^2}\,\bar R(G),
\qquad
\bar R(G)=\frac{\operatorname{tr}(L^+)}{N},
\]
where \(s\) is **linear** SNR and \(L^+\) is the Moore-Penrose inverse. The
derived leading terms suggest coefficients near one; the coefficients are
model-dependent and must be estimated, not assumed. For a log-normal
amplitude veil with \(\eta_n\sim\mathcal N(0,s_a^2)\),
\[
\mathbb E|\tau|^{-p}=\exp\!\left(\tfrac12 p^2s_a^2\right).
\]
For a ring, \(\bar R=(N^2-1)/(12N)\); for a two-dimensional torus it grows
asymptotically like \(\log N\); for a family of well-connected expanders it
remains bounded. These graph facts concern the graph-Laplacian noise model,
not every observation model.

Equating the two terms gives the approximate ring crossover
\[
N^\star \approx 12s\,
\frac{\mathbb E|\tau|^{-2}}{\mathbb E|\tau|^{-4}},
\]
independent of \(K\) at this order. The exact finite-\(N\) crossing uses the
ring's exact \(\bar R\). This is a falsifiable prediction, not a proved
finite-sample law.

With partial overlap, neighboring tiles use different sample sets. The
first-order noise term can become edge-dependent and cease to be a coboundary.
Replacing \(K\) by \(K_{\rm eff}\approx\rho K\) for overlap fraction \(\rho\)
is a useful heuristic, not a general theorem. The expander experiment creates
random subsets with repaired edge overlap; that is not a geometric sliding
window.

The v2 notebook reports a global weighted fit
\(c_{\rm floor}=1.128\), \(c_{\rm layout}=0.730\), \(R^2=0.765\) over 360
synthetic points. Its per-SNR fits have negative \(R^2\) at 0 and 5 dB, so the
global fit should not be read as validation across the full parameter range.
The v1 notebook reports nearly flat graph-size scaling for shared inputs at
20 dB and stronger ring degradation at 0 dB; it separately reproduces the
expected graph dependence for injected independent edge noise. These are
finite simulation results, not asymptotic evidence.

## 3. Tracking a drifting veil

After relative estimation and gauge alignment to a held reference, v3 treats
log-amplitude and phase as separate scalar processes:
\[
x_{t+1}=a x_t+w_t,\qquad z_t=x_t+v_t,
\]
with process variance \(q\) and measurement variance \(r\). The Kalman
prediction and update are
\[
P^-_t=a^2P_{t-1}+q,\quad
\nu_t=z_t-a\hat x_{t-1},\quad S_t=P^-_t+r,\quad
g_t=P^-_t/S_t,
\]
\[
\hat x_t=a\hat x_{t-1}+g_t\nu_t,\qquad P_t=(1-g_t)P^-_t.
\]
At steady state, the predicted variance \(Q=P^-\) is the positive solution of
\[
Q^2+\bigl(r(1-a^2)-q\bigr)Q-qr=0,
\]
and the gain \(Q/(Q+r)\) increases with \(q/r\).

Innovation covariance matching uses an exponentially smoothed innovation
power \(C_t\) to estimate \(q\), for example
\[
\hat q_t=\max(q_{\min}, C_t-a^2P_{t-1}-r).
\]
This is unbiased only under the correctly specified steady-state model with
known \(r\) and settled covariance. Innovation power alone does not generally
identify both process and measurement noise. The notebook's
\(\Xi_\nu=\min(1,S_0/C)\) and its gain interpolation are adaptive heuristics;
their sign is chosen so unexpectedly large innovations lower \(\Xi_\nu\) and
raise the gain. The earlier consistency-to-gain rule is retained as a negative
control, not as the recommended tracker. Simulated jumps and ramps are useful
stress tests, but do not establish robustness to real drift.

For orientation, the supplied v3 notebook reports late-window RMS
\(\kappa=0.0311\pm0.0046\) for its initial-\(q\) Kalman filter versus
\(0.0312\pm0.0047\) for the oracle in one stationary 20 dB scenario; a fixed
gain of 0.05 gives \(0.3045\pm0.0254\). In one v4 full-overlap jump scenario,
Mehra gives 0.0501 RMS versus 0.0505 for the oracle, while the v1
consistency-gain baseline gives 0.7643. These are configuration-specific
Monte Carlo summaries, not guarantees or measured device performance.

## 4. Open-system dynamics and selection

The v5 model uses the Gorini-Kossakowski-Sudarshan-Lindblad equation
\[
\dot\rho=-i[H,\rho]+\sum_j\left(L_j\rho L_j^\dagger
-\tfrac12\{L_j^\dagger L_j,\rho\}\right).
\]
A damped auxiliary bosonic pseudomode coupled to a tile gives a Markovian
joint model whose reduced tile dynamics can display memory. A Lorentzian
spectral density and trace-distance backflow are properties of this
specified, truncated simulation; they do not imply that a physical device has
been built.

For the specific competition between the declared-target jumps
\(L_j=\sqrt{\gamma_{\rm sel}}|t\rangle\langle j|\) and a Markov depolarizing
channel of rate \(\Gamma\), the target population obeys
\[
\dot F=\gamma_{\rm sel}(1-F)+\Gamma(1/d-F).
\]
Consequently, with \(r=\gamma_{\rm sel}/\Gamma\),
\[
F_{\rm ss}=\frac{r+1/d}{r+1},\qquad
r^\star=\max\!\left(0,\frac{\Xi_{\rm crit}-1/d}{1-\Xi_{\rm crit}}\right).
\]
This threshold applies to that channel and objective only. For
\(\Xi_{\rm crit}\ge1\), no finite rate suffices (the limiting case
\(\Xi_{\rm crit}=1\) requires infinite selection); other noise channels need
their own steady-state analysis. The notebook's numerical steady-state checks
agree with the closed form for its small simulated systems.

## 5. Observable-only Gaussian sensing

The v6 reference-mode model is a separate sensing problem, not another
derivation of the tile-veil estimator. It assumes
\[
s\sim\mathcal N(0,V_s),\quad n\sim\mathcal N(0,V_n),\quad
y_A=\sqrt\eta(s+n)+q_A,\quad y_B=\sqrt\eta\,k n+q_B,
\]
with independent Gaussian \(q_A,q_B\). For the specified jointly Gaussian
model, the MMSE estimator is the Wiener conditional mean
\[
\hat s=\operatorname{Cov}(s,y)\operatorname{Cov}(y,y)^{-1}y,\qquad
\operatorname{MSE}=V_s-\operatorname{Cov}(s,y)\operatorname{Cov}(y,y)^{-1}
\operatorname{Cov}(y,s).
\]
This is optimal when the model parameters are known; learning/calibrating them
from data is a separate problem.

The photonics convention used by the code is \(\hbar=2\), so
\([x,p]=2i\), vacuum variance is one, and an x-squeezed vacuum has variance
\(e^{-2r}\). Under loss \(\eta\), the measured quadrature variance is
\(\eta e^{-2r}+(1-\eta)\). In this model, squeezing the reference (B) beats
squeezing the signal (A) exactly when \(u=k^2\) lies between the valid roots
of
\[
(\eta V_n)^2u^2-\eta V_n(\eta V_n-\mu-1)u+\mu=0,\qquad
\mu=\eta e^{-2r}+(1-\eta).
\]
The interval may be empty; “\(k<1\)” alone is not sufficient at finite noise.
All these are Gaussian-model results. The simulations use classical random
numbers and analytic conditioning, not optical hardware.

## 6. Graph design and wiring

The same graph Laplacian supports two distinct design quantities:
\(\operatorname{tr}(L^+)/N\) predicts average variance under independent
edge-noise least squares, while the spectral gap \(\lambda_2(L)\) governs
convergence of basic consensus iterations. They are related but not
interchangeable objectives. A low-resistance graph does not by itself prove
lower wall-clock time, energy, or physical error.

The v6 wiring study compares rings, grids, expanders, small-world, and
quasiperiodic/weighted candidates using these graph diagnostics and a small
synthetic unveiling task. In the recorded \(N=36\) slice, the ring's mean
resistance was higher than the torus and expander; the measured MSE intervals
overlapped. The tests do not establish a general advantage for any
quasiperiodic wiring. The Aubry–André inverse participation ratio is a
diagnostic of its separate on-site Hamiltonian, not an estimator-performance
score.

## 7. Do not overload \(\Xi\)

The notebooks and repository use several different quantities called
\(\Xi\). They are not interchangeable:

| Name | Definition / role |
|---|---|
| Ensemble veil coherence | \(\Xi_{\rm veil}=|\mathbb E\tau|^2/\mathbb E|\tau|^2=\exp[-(s_a^2+s_p^2)]\) for the stated independent Gaussian log-veil. |
| Neighbor consistency | `xi_from_consistency`: a gauge-invariant residual-consistency proxy for edge-ratio reconstruction. |
| Innovation score | \(\Xi_\nu=\min(1,S_0/C)\): compares observed innovation power with a nominal model. |
| Legacy logit score | `legacy_xi`: \(1-\operatorname{Var}(\ell)/(\mathbb E\ell+\epsilon)\), which is not guaranteed to lie in \([0,1]\). |
| Bounded logit score | `snr_stability`: a signal-power-to-total-power ratio in \((0,1]\) under its definition. |
| Selection target | \(\Xi_{\rm crit}\): a required target population/fidelity in the selection model. |

Use the qualified name and state space whenever reporting one of these values.
No one \(\Xi\) definition has been established as a universal stability or
truthfulness measure.

## 8. Implementation and evidence map

| Framework component | Current implementation | Evidence / limits |
|---|---|---|
| Relative estimation, gauge, overlap | `sri/unveiling.py`, `sri/graphs.py`, `sri/gauge.py` | Synthetic observation models; no ASIC measurements. |
| Error-law prediction | `sri/errorlaw.py` | Leading-order fit helpers; coefficients and low-SNR validity remain open. |
| Adaptive tracking | `sri/tracking.py` | Classical Kalman-family filters on simulated log-veils. |
| Lindblad / pseudomode / selection | `sri/selection.py`, `sri/dynamics/` | Truncated classical simulation; selection formula is channel-specific. |
| Gaussian reference sensing | `sri/cv/gaussian.py`, `sri/cv/estimator.py`, `sri/cv/invariants.py` | Parameter-oracle Gaussian model; no live photonic or Braket QPU result. |
| Wiring candidates | `sri/wiring_quasi.py`, `sri/graphs.py` | Small synthetic graph study; diagnostics do not imply hardware benefit. |

The notebook outputs are useful experimental records, but a saved cell output
is not an independent replication. For code-level reproduction use the
versioned drivers and `reference_results.json` files above; fast and slow
regressions live under `tests/regression`. Preserve the source revision,
configuration, random seeds, raw records, and outputs together when promoting
a result.

## 9. Open mathematical and engineering obligations

1. Prove a finite-sample error bound for the shared-input ratio estimator,
   including a remainder term and explicit assumptions on SNR, veil moments,
   overlap, and graph connectivity.
2. Test the two-term law and crossover on held-out parameter grids, with
   uncertainty intervals and an independent implementation; characterize
   where the low-SNR approximation fails.
3. Derive anchored covariance for noisy or drifting references and validate
   robust pinning under outliers.
4. Calibrate process and measurement noise from observations, then compare
   adaptive trackers against properly tuned baselines on preregistered drift
   scenarios.
5. Separate the CV sensing result from claims about SRI: build a physical
   protocol, estimate rather than assume model parameters, account for
   resources, and compare against relevant matched baselines.
6. Measure any claimed hardware latency, power, energy, or scaling on an
   implementation whose work and memory requirements are reported as functions
   of problem size.

Until these obligations are met, the defensible framework is a collection of
connected classical models, standard estimation/channel results, and
falsifiable SRI hypotheses—not a demonstrated quantum advantage,
hardware-native \(O(1)\) mechanism, or universal error-annihilation law.

## 10. Architecture axioms bridge (A0–A5 / Lemma Z1) — 2026-11-12

Complementary to §§1–9: the locked axiom set and observable Yoga–Buddhi rewrite live in
[architecture-axioms.md](architecture-axioms.md) and [yoga-buddhi-projector.md](yoga-buddhi-projector.md).

| Result | Status | Evidence |
|---|---|---|
| Lemma Z1 / C5 (\(\hat\Upsilon_{\mathrm{obs}}\equiv\hat s\)) | **confirmed** | SageMaker g4dn T4 / JAX GPU elaborate suite pasteback |
| Colab oracle Samya regression | **supported** | Same suite `colab_oracle_samya_fixture` |
| Raw-\(Y\) non-involution \(W_{\mathrm{raw}}\) (Z1b / draft C6) | **metric emitted** (~1.28233) | Pasteback `theory_hooks`; threshold \(\varepsilon\) open |
| Lemma Z2 (unveil vs select → Z1) | **open** | Needs Hari composition fork |
| Photonic QPU / Landauer / hardware \(\mathcal O(0)\) | **not claimed** | Softwalls |

This section does not replace §§1–9 unveiling/error-law obligations; it stamps the
Gaussian dual-mode Yoga–Buddhi deployable path under A0–A5.

## 11. Claim register for the next DGX round — 2026-11-14

The Colab archive and the bridge note state more than one result. This section
keeps every one of them that has a concrete measurement or a concrete next
cell. The round itself is [dgx-abhaya-round.md](dgx-abhaya-round.md). A claim
enters that round as something we expect the hardware to be able to show.
Promotion still follows the cell's decision rule.

Two readings of Abhaya-gate are both in the register, because the notebooks
use both shapes.

- **Raw map.** \(A(\Theta)=\Theta\exp(-\lambda\Xi)\) multiplies the whole tensor.
  On the locked zero-mean spike series (seed 11, length 8192, \(\lambda=1\),
  window 9) the CPU preview MSE is **0.774**, against **1.675** for a moving
  average and **6.910** for the raw observation. That reading is ready to
  confirm. On the same noise added to a carrier of mean 5, the raw map MSE
  rises to **18.17**. That boundary stays in the report so a later pitch does
  not silently switch series.
- **Flux map.** The dissertation and patent cells damp a deviation and keep a
  center. \(A\) is applied to \(\Theta=y-\mu\), with \(\Xi\) the window
  variance of \(y\), and the center is added back. The latent value is not an
  input. On both the zero-mean series and the offset carrier the CPU preview
  MSE is **1.472**, still under the pass bar \(0.95\) times the better of the
  observation and the moving average. Flux variance remaining is **0.117** of
  the observation variance, under the half-variance bar.

| Claim worth testing | Why it is in the round | Entering status |
|---|---|---|
| Flux Abhaya beats a moving average without seeing the latent value, including when the carrier is offset | Matches the Colab shape; CPU preview already clears \(0.95\times\) | Ready to confirm on the GPU |
| Raw Abhaya on the zero-mean spike series | Bridge equation as written; CPU preview MSE 0.774 | Ready to confirm |
| Flux variance falls by at least half | Sage-PC and patent cells drive a variance to the floor when the construction allows it; the blind preview already clears half | Ready to confirm |
| Paired cancellation residual under \(10^{-5}\) | Dissertation cell 10 reports residual 0; Patent cells report grounded variance 0.00 | Replicate. This cell is allowed to see the paired tensors |
| One Triton launch of \(A\) at least 10% sooner than eager PyTorch of the same map | Hybrid om-lang, Tier 3 of the bridge note | Scheduled |
| `@sphota` block within 10% of the best of 128, 256, 512, 1024 | The Julia macro is a real schedule | Scheduled |
| GPU time grows slower than linear in length (log-log slope \(< 1\)) | L40S map-reduce stayed near 0.04 ms across a wide range; the Blackwell kernel grew about 9.5 times while length grew 64 times | Scheduled. Prior sessions point this way |
| GPU time nearly flat (slope \(< 0.15\)) | The strong constant-time reading of those same cells | Scheduled |
| Median at length 8192 under the internal budget 0.2379 ms | Dissertation Triton median was 0.0082 ms | Replicate. The budget is an internal target written into that cell, not a timed competitor |
| Same-GPU method faster than a named current operator at matched quality | The product speed claim. On 14 Nov an average pool finished the heavier kernel sooner | Scheduled. Quality now has to travel with the timer |
| Fewer joules than that operator, temperature within 5 °C, and temperature under half the board limit | 30 °C at about 105 W on a 600 W RTX PRO 6000; 36 °C at about 85 W on a 350 W L40S | Scheduled |
| Short labeled answers: gated generation at least as accurate per second as the ungated model | Untitled4 suppressed 56 of 150 tokens and took 7.95 s against 5.95 s ungated, with no answer key | Scheduled when a local model is present |

Phase 3's dual-mode reduction of **0.1205** against single-mode Wiener stays a
supported quality result and is not re-litigated here. Lemma Z1 remains the
deployable sensing identity in §10.

The printed success lines in `Sree.ipynb` ("stability overcoming entropy"),
the 10,240-long tensor labeled as qubits, and the thermodynamic penalty
printed as zero are included as motivations for the variance and residual
cells. They are not separate device claims until a cell measures them.

## 12. Witness program — 2026-11-14

The path from this date is the dissertation's Witness architecture, not a
narrower benchmark portfolio. Every claim in the five-page appendix, and the
notebook mechanisms that tried to build it, are listed as future technology in
[future-tech-register.md](future-tech-register.md).

Tonight's DGX session is the digital asymptote: in-process damping, the
omega-dit tuple, sign-bit antithesis, a Fibonacci window, abstention, a dtype
ladder, and the timings already on the run card. The optical core, material
many-body localization, a witness cluster, and a calorimeter stay on that
register as later instruments. A cell that passes supports its own sentence.
It does not silently promote a later instrument.

## 13. Measured status after the four-GPU session — 2026-11-14

The session in
[rtx6000x4/SESSION.md](benchmark-results/rtx6000x4/SESSION.md) promotes the
following rows from “ready to confirm” or “scheduled” to measured. The machine
was four RTX PRO 6000 Blackwell Server Edition boards. PyTorch 2.14.1+cu130.
JAX backend `gpu` on `cuda:0`–`cuda:3` after the CUDA 13 plugin.

| Statement | Status | Evidence |
|---|---|---|
| Julia `@sphota`, JAX, and Triton implement \(A=\Theta e^{-\lambda\Xi}\) and agree with NumPy | **measured** | Block 256; errors \(\le 3.8\times 10^{-6}\) (Julia, JAX) and \(\sim 10^{-5}\) (Triton) |
| Flux form beats a width-9 moving average with the latent series withheld, at carrier offset 0 and 5 | **measured** | MSE 1.472 vs 1.675; bar 1.591; variance remaining 0.117 |
| Raw form on the zero-mean series | **measured** | MSE 0.774 |
| Raw form on a carrier of mean 5 | **not supported** | MSE 18.17 |
| Triton sign-bit XOR matches negation | **measured** | Error 0 on all four boards, \(\sim 0.010\) ms |
| Triton product from length \(2^{13}\) through \(2^{20}\) stays under 0.2379 ms and near 0.010 ms | **measured** | Four boards; launch-bound for this map |
| Block 256 is within 10% of the best of \(\{128,256,512,1024\}\) and was the minimum | **measured** | 0.0116 ms vs 0.0147, 0.0126, 0.0129; eager 0.0164 ms |
| Log-log slope of a one-line PyTorch map \(< 0.15\) | **measured for that map** | Slopes about 0; 0.017–0.019 ms |
| Paired average residual \(< 10^{-5}\) when both tensors are inputs | **measured** | \(1.94\times 10^{-7}\) |
| Silence gate on window variance, F1 \(\ge 0.5\) | **not supported** | Best F1 0.291 at threshold 8 |
| Four-way shard faster than one board on this kernel | **not supported** | 0.088 ms vs 0.047 ms at \(N=2^{22}\) |
| Flux map faster than an average pool | **not supported** | 0.064 ms vs 0.016 ms at \(N=2^{20}\) |
| Board power and temperature for the light loop | **measured** | \(\sim 86\) W, \(33\)–\(36^\circ\)C; idle \(\sim 30\) W, \(28\)–\(31^\circ\)C; limit 600 W |
| Physical Landauer calorimetry, PPLN, material MBL | **not this session** | No instrument |

The flux form is the reading that survives a change of carrier. The raw
exponential does not. Flat sub-millisecond times are a property of these
launch-bound maps. They update the timing record. They do not by themselves
close an asymptotic claim for every larger workload. The specification pair
for this section is
`docs/reports/phase4-technical-specification.pdf` and
`docs/reports/phase4-patent-specification.pdf`, with DOCX twins beside them.
