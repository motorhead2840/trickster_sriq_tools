# Correlated-noise continuous-variable sensing: simulation protocol

## Question and scope

Can measuring a correlated auxiliary optical mode reduce the mean squared error (MSE)
of estimating a classical signal displacement, and does squeezing the reference
mode improve on appropriately matched alternatives? This is a proposed Gaussian
optical sensing experiment, **not an implementation or validation of SRI's claimed
symplectic error annihilation**. It is not a quantum computer, quantum error
correction, or a demonstration of an advantage over all classical or quantum methods.

The simulation evaluates ideal quantum-optical quadrature distributions using
classical random-number generation and analytic Gaussian conditioning. The signal
is classical information encoded in light, not an arbitrary unknown quantum state.
The numerical evaluation uses the true signal to calculate MSE; no method receives it
at inference time.

## Physical model

Use two traveling optical modes A and B, homodyne detectors and a common phase
reference. Use the repository convention `ħ = 2`, so `[x,p] = 2i` and vacuum
has quadrature variances 1. Prepare A and B in vacuum or in an x-squeezed
vacuum with parameter `r`: the quadrature covariance is
`diag(exp(-2r), exp(2r))` and its mean photon number before displacement is
`sinh(r)^2`. In each independent trial:

1. Draw independent classical Gaussian displacements `s ~ N(0, Vs)` and
   `n ~ N(0, Vn)`. Modulate mode A by `s + n` and mode B by `k*n` in x.
   The parameter `k` describes correlated noise coupling. This assumes a physical
   auxiliary channel that shares the noise but not the signal; it is not obtained
   automatically from the noisy observation.
2. Apply independent pure-loss channels with the same total efficiency `eta` to
   each mode. They combine transmission and idealized detector efficiency and
   inject independent vacuum. Homodyne the x quadrature on each available mode.
3. Use both measurements, or A alone for single-mode baselines, to estimate `s`
   with the exact Gaussian posterior mean. The simulator knows `Vs`, `Vn`, `k`,
   `eta` and `r`; no clean signal or noise realization is supplied to the estimator.
   This is a parameter-oracle benchmark, not a calibrated implementation.

Equivalently, writing independent unit-variance Gaussians `zA`, `zB`, `lA`, `lB`
for squeezed-state and loss vacuum quadratures, respectively, the measurement
model is:

```
yA = sqrt(eta) * (s + n + exp(-rA)*zA) + sqrt(1-eta)*lA
yB = sqrt(eta) * (k*n  + exp(-rB)*zB) + sqrt(1-eta)*lB
```

The posterior is `E[s|y] = Cov(s,y) Cov(y,y)^-1 y`, and the corresponding
analytic MSE is `Vs - Cov(s,y) Cov(y,y)^-1 Cov(y,s)`. It is optimal for the
specified Gaussian model and observations, including among nonlinear estimators.
The code simulates the measured quadrature distribution, not a full state-vector
or density-matrix simulator. Homodyne consumes the measured optical states;
classical postprocessing does not recover an intact quantum state.

## Comparisons and resources

All conditions use the same signal and noise distributions and efficiency:

- Coherent signal alone (one mode, no reference).
- Coherent signal and coherent reference (two modes, classical optical baseline).
- Squeezed signal alone (one-mode quantum baseline).
- Squeezed signal and coherent reference (two-mode quantum baseline).
- Coherent signal and squeezed reference (proposed two-mode comparison).

For the two squeezed two-mode cases, the same squeezing `r` is allocated to
different modes; both use two modes and the same mean squeezing photons. The
coherent two-mode baseline uses no squeezing, i.e. less than the same upper
resource budget. This is not a proof of optimality over all classical probes or
all quantum protocols. The signal and reference displacement photons, squeezing
photons, mode count and trial count are recorded separately. Displacement energy
from the shared noise is an environmental/model input, not a free laboratory
resource; experiments must account for how it is coupled into the two modes.

The simulation does not measure source preparation efficiency, losses beyond
`eta`, local oscillator power, phase locking, detector electronics, runtime,
calibration, or control/computational energy. There is no error-correction code
and no modeled error-correction overhead; claiming a fault-tolerant computing
benefit would require a separate protocol and its full overhead. A real benchmark
must measure or bound these costs and include them for every method.

## Reproduce and interpret

From the repository root, for example:

```bash
python experiments/cv_photonics_benchmark.py --output-dir cv_results --seed 108 --shots 20000
```

For cross-backend checks on Colab, open `tests/CV_Photonics_A100_Validation.ipynb`
and upload the current four files from `experiments/` when prompted:
`cv_a100_validation.py`, `cv_a100_kernels.py`, `cv_photonics_benchmark.py` and
`cv_photonics_julia.jl`. The notebook checks for an A100, installs missing Python
packages and, if necessary, downloads an official Julia binary and verifies its
published checksum. It runs three seeds, validates each estimate against NumPy,
checks the sample MSE against the analytic prediction and downloads a ZIP with
reports and source. NumPy and Julia are CPU implementations; PyTorch, JAX and
Triton use the GPU. The Julia code solves its own posterior covariance system;
the Python GPU backends use coefficients from the shared Python analytic model.
Agreement across backends is an implementation check, not independent validation
of the physical assumptions. The A100 run reports no kernel timing.

The notebook uses installed packages where possible. If an installed JAX lacks
GPU support, install a CUDA-compatible JAX build, restart the runtime and rerun
the notebook; the check will not silently fall back to CPU. The output includes
software versions, driver, source hashes and random-input hashes. No model or
dataset download is required.

The JSON contains input parameters, theory and Monte Carlo MSEs, posterior
coefficients, resource accounting and a script hash. Comparisons report baseline
minus squeezed-reference MSE: positive favors the squeezed reference. The
approximate 95% normal intervals are for Monte Carlo sampling error, not
uncertainty about the physical model. Common random numbers are used across
conditions; they reduce simulation noise and do not imply paired measurements
of multiple incompatible optical states on the same hardware shot.

With the defaults (`Vs = Vn = k = 1`, `eta = 0.8`, `r = 0.7`), the analytic
MSEs are approximately 0.5024 for two coherent modes, 0.3876 for a squeezed
signal with coherent reference, and 0.4517 for a squeezed reference with coherent
signal. Thus the proposed reference squeezing improves on the coherent baseline
in this model but **loses to the matched two-mode squeezed-signal baseline**.
These are model predictions, not measured experimental results.

Before any confirmatory claim, preregister the task, physical noise correlations,
calibration and held-out procedures, resource budgets, baselines, practical effect
size, uncertainty method and decision rule. Replicate across independent seeds,
noise strengths, coupling and loss values. Check the effect against a trusted
quantum-optics implementation or analytic derivation, then conduct an experiment
with independently characterized sources and detectors if available. Use a
separate calibration dataset and report both ideal and estimated coefficients.
If the proposed method fails to beat a relevant baseline under the chosen
resource accounting, report that failure rather than changing the criterion after
seeing the results.

## Existing Colab artifacts

The supplied output shows a retrieval-resume run. Preserve both the original and
recovery ZIPs: the latter contains updated JSON and the factual plot, not all
original plots. In `sri_results.json`, inspect the numerical, scaling,
architecture and kernel records as well as `factual.paired_summary`,
`retrieval_status_counts` and per-question records. Compare retrieval and
internal accuracy on the same paired questions, not the differently sized
summary subsets. Repeat on independently selected seeds and additional datasets
and models before making general claims. The ZIPs and complete records are needed
to perform that analysis; the pasted summary alone is insufficient.
