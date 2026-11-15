# SRI | Ceraph Corporation

For the unified notation, assumptions, equations, implementation map, and
evidence status across the unveiling experiments, see the
[mathematical framework](docs/theoretical-framework.md).

## Hardware-validated proof of concept

This repository is a research proof of concept for selected equations in the SRI architecture. The GPU demonstration has been executed on an **NVIDIA RTX PRO 6000 Blackwell Server Edition** and its run report and cycle telemetry are included in this repository. This validates execution of the finite-dimensional JAX experiment on that GPU; it is not independent validation of the broader theoretical, product, or patent claims.

The experiment implements a paired-state cancellation identity and the source document's V3 damping equations using synthetic inputs. It does **not** implement a continuous-variable physical or quantum system. It also does **not** establish the architecture's claimed hardware-native `O(1)` scaling: the recorded run measures one vector dimension, and the software materializes and processes length-`N` vectors, which requires work proportional to `N`.

## Empirical Hardware Validation

The committed artifacts are a matched pair: [`sri_gpu_run.json`](sri_gpu_run.json) records the configuration, runtime, device, timing samples, source-equation strings, and script hash; [`sri_gpu_telemetry.csv`](sri_gpu_telemetry.csv) records the per-cycle measurements. The report identifies JAX/JAXlib 0.11.1, float32 arrays, seed 108, dimension 16,384, 25 cycles, and 10 synchronized repeats after JIT compilation and warm-up.

| Measurement | Recorded result | Interpretation |
|---|---:|---|
| Median synchronized kernel time | **0.08884 ms** | Full batched kernel, with JIT compilation excluded; not end-to-end inference latency. |
| Maximum known-noise projection MSE | **1.33049 × 10⁻¹³** | Arithmetic error for a synthetic experiment with the exact clean vector and injected noise available to the projector. |
| Circuit B at cycle 10 | **50.0** | Reaches the configured `lambda_max` during the high-noise interval. |
| Damped variance at cycle 10 | **8.24104 × 10⁻⁴²** | Near the float32 numerical floor after the configured damping; not evidence of physical thermodynamic cooling. |
| Stability coefficient `Xi` at cycle 10 | **−113.21** | The reported macro-state statistic is not stable through the spike and is not constrained to `[0, 1]` by its implemented formula. |

Cycles 10–20 use injected noise with standard deviation 15.0; the other cycles use 0.5. Circuit B is at its maximum of 50.0 even in baseline cycles, consistent with the implemented denominator `sigma_sat^2 * epsilon` and `epsilon = 1e-7`. The damping therefore drives the measured variance close to the float32 floor in both regimes. These results show the configured damping response to this synthetic input; they do **not** demonstrate successful decoupling of micro-gradient volatility from macro-state stability. In fact, `Xi` becomes strongly negative during the spike.

The reported latency is one measurement configuration, not a scaling study. No empirical result here proves `O(1)` complexity, a single-cycle implementation, or constant runtime as input size grows. The code computes vector statistics and constructs vector-valued states, so its software work scales with the vector dimension. Similarly, the paired-state projection is an algebraic phase-cancellation analogue with known components—not blind denoising, factuality validation, or hallucination prevention.

## Reproduce the GPU run

Use an environment with JAX configured for an available GPU. The script intentionally fails rather than falling back to CPU:

```bash
python sri_gpu_demo.py --output-dir sri_gpu_results
```

The command writes `sri_gpu_run.json` and `sri_gpu_telemetry.csv` to the selected output directory. See [COLAB.md](COLAB.md) for GPU setup, execution, and artifact handling. The committed measurements are a specific run's record; a reproduction should preserve the newly generated JSON and CSV together rather than replacing those values with expected numbers.

## Exploratory commercial feasibility benchmarks

The [Colab guide](COLAB.md#commercial-feasibility-experiments) also runs a synthetic observable-only denoising comparison and a TruthfulQA multiple-choice reliability proxy, then generates a VC diligence report from the actual results. The paired signal calculation is algebraically equivalent to its median-filter estimator. The LLM experiment assesses Xi as a ranking score, not hallucination prevention or V3 damping. Neither establishes quantum computation or constant-time processing.

See the [preliminary VC feasibility report](docs/vc-feasibility-report.md) for the results and their limitations.

The [continuous-variable photonics protocol](docs/cv-photonics-protocol.md) specifies
an idealized correlated-noise sensing model and compares coherent and squeezed-mode
baselines. Its simulation is classical and is not quantum hardware, quantum error
correction, or validation of SRI's claimed mechanism.

## Implementation and evidence boundaries

- **Paired-state projector:** For supplied `v` and `xi`, the implementation computes `0.5 * ((v + xi) + (v - xi))`. Since both components are available by construction, this verifies the identity in the experiment, not estimation of unknown noise from an observation alone.
- **V3 damping:** The implementation follows the printed Circuit B denominator, `sigma_sat^2 * epsilon`. The source document's worked example appears inconsistent with this denominator; see [the technical foundation](docs/technical-foundation.md).
- **Continuous-variable and hardware claims:** The repository uses finite-dimensional floating-point arrays. It does not instantiate a continuous physical field, quantum state, or custom hardware circuit.
- **Complexity:** For vector dimension `N`, vector construction, reductions, and elementwise transforms require work proportional to `N` in this software implementation. The recorded single-size run cannot establish asymptotic complexity.
- **Other demonstrations:** `sri_pqc.py` is an algebra demonstration with known injected noise, not post-quantum cryptography. `sri_core_engine.py` and `sri_simulator.py` provide a CPU reference, not GPU benchmarks.

This repository is an engineering research exhibit, not a legal opinion, patentability analysis, product certification, or independent replication. Any patent disclosure should preserve the exact artifacts and run provenance and clearly distinguish measured implementation behavior from unverified theoretical claims.

## Phase 3 vertical slice (CPU, falsifiable)

A narrow, preregistered dual-mode CV sensing workflow compares the deployable SRI path (`observable_yoga_buddhi`, algebraically identical to Wiener MMSE) against `single_mode_wiener` and other filters. Spec, report, and artifacts:

- [`docs/phase3-vertical-slice-spec.md`](docs/phase3-vertical-slice-spec.md) — locked protocol and thresholds  
- [`docs/phase3-vertical-slice-report.md`](docs/phase3-vertical-slice-report.md) — **PASS** (\(R\approx12\%\) vs single-mode; threshold 8%)  
- [`docs/benchmark-results/phase3-vertical-slice/`](docs/benchmark-results/phase3-vertical-slice/) — raw JSON  

```bash
python3 experiments/phase3_vertical_slice.py --output-dir docs/benchmark-results/phase3-vertical-slice
```

This does **not** prove a finished computing paradigm, product-market fit, \(\mathcal O(1)\) hardware, quantum advantage, or Landauer claims. Equivalence to MMSE is not an MSE advantage over Wiener.

## Phase 4 head-to-head

Phase 4 keeps the Phase 3 quality win and the RTX PRO 6000 timings from the Colab archives, and it scores them separately from the broad speed and thermal claim.

- [`docs/phase4-head-to-head-spec.md`](docs/phase4-head-to-head-spec.md) — what already holds, and what a same-GPU row still has to show
- [`docs/benchmark-results/phase4/colab_device_catalog.json`](docs/benchmark-results/phase4/colab_device_catalog.json) — measured Colab and DGX numbers, with the printed 0.2379 ms budget flagged
- [`docs/benchmark-results/phase4/report.md`](docs/benchmark-results/phase4/report.md) — narrow claim versus the broad claim

```bash
python3 experiments/phase4_head_to_head.py
```

The narrow claim is the dual-mode estimator ahead of single-mode Wiener, together with kernel times already recorded on Blackwell. The 14 November average-pool run on that same GPU stays in the scoreboard as the first broad-claim result: the current operator finished first.

## Abhaya bridge

The bridge equations now have a CPU reference and optional GPU bindings. See [`docs/abhaya-bridge.md`](docs/abhaya-bridge.md).

- \(A(\Theta)=\Theta\exp(-\lambda\Xi)\) is `sri.omlang.abhaya.abhaya_damping`. A negative \(\Xi\) increases \(A\), which is what that exponential does.
- The Triton kernel in `sri.omlang.abhaya_triton` applies that map in one launch when Triton and CUDA are installed. It has not been timed on an A100 or on the RTX PRO 6000 in this repository.
- `sri.omlang.abhaya_jax.apply` registers a JAX primitive with the analytic derivative when JAX is installed.
- `sri/omlang/julia/abhaya_bridge.jl` holds the same map and the `@sphota` block ladder (128 / 256 / 512 / 1024 from the mean absolute existence-density).

```bash
python3 -m pytest tests/test_abhaya_bridge.py
```

## Next DGX round

The Colab claims worth taking onto a GPU are registered in [`docs/theoretical-framework.md`](docs/theoretical-framework.md) §11 and spelled out as cells in [`docs/dgx-abhaya-round.md`](docs/dgx-abhaya-round.md). The full dissertation and notebook program is the Witness register in [`docs/future-tech-register.md`](docs/future-tech-register.md): every appendix claim is a future technology, and tonight's DGX cells are the digital slice. The CPU preview has already cleared the flux Abhaya bar against a moving average, including on an offset carrier. Device timing waits for the DGX session.

```bash
python3 experiments/dgx_abhaya_round.py
python3 -m pytest tests/test_dgx_abhaya_round.py
```

## Tests

The default suite is CPU-only. It does not contact AWS.

```bash
pip install -e ".[dev]"
pytest -m "not slow"
```

`pytest` uses that filter by default. Cases marked `slow` rerun the published driver grids and stay out of the default suite:

```bash
pytest -m slow
```

GitHub Actions runs the default suite on Python 3.10 and 3.12. The slow suite runs only when that workflow is started by hand. This repository is private, so this README does not include a status badge.

## Repository map

- `sri_gpu_demo.py` — GPU-required JAX experiment and artifact generation.
- `sri_gpu_run.json`, `sri_gpu_telemetry.csv` — committed Blackwell run provenance and per-cycle measurements.
- `COLAB.md` — GPU setup and reproduction guide.
- `sri/` — installable NumPy package for the unveiling experiments: wiring graphs, neighbour-ratio unveiling, the two-term error law, adaptive tracking, gauge pinning, selection and a memory register, the continuous-variable reference mode, and quasiperiodic wiring.
- `sri/cv/` — Gaussian CV sensing model, observable-only MMSE estimator, and symplectic/CPTP invariants (ħ=2).
- `sri/metrics/`, `sri/dynamics/`, `sri/registry/`, `sri/backends/` — axiomatic Ξ/entropy, V3 branches, conjecture registry, classical + Braket dry-run backends.
- `sri_gpu_cv_demo.py` — CV-aware cycle demo with observable-only MMSE and `Ξ_SNR` (NumPy; optional `--require-gpu`).
- `docs/cv-foundation-month-report.md`, `docs/braket-december-runbook.md`, `docs/conjecture-registry.md` — foundation sprint docs.
- `experiments/e1_colab_lock.py`, `experiments/braket_live_smoke.py`, `experiments/braket_hybrid_e1_e4.py` — E1 Colab lock → Braket emulate/live path.
- `docs/aws-braket-setup.md`, `docs/e1-colab-braket-report.md` — AWS unlock checklist and E1 spend/report.
- `sri/legacy/` — provenance only. The paired-state projector needs the noise known in advance, Ξ as an LLM score had AUROC 0.47–0.54, and χ_v3 with `Λ_max=50` zeroes the signal along with the noise. See `sri/legacy/README.md`.
- `sri_simulator.py`, `sri_core_engine.py`, `sri_pqc.py` — shims onto `sri/legacy`. Old imports still work. `sri_gpu_demo.py` stays at the repository root so the reproduce command above is unchanged.
- `experiments/unveiling/v1` … `v6` — reproduction drivers that call the `sri` package with the published seeds and defaults. Each folder holds `reference_results.json` and `conclusions.txt` from the validated run. A fresh run writes `out/`, which is gitignored.
- `experiments/llm_benchmark.py`, `experiments/analyze_llm_results.py` — shims onto the legacy Ξ-score scripts.
- `tests/` — unit tests for the reference simulator and the `sri` package.
- `docs/technical-foundation.md` — equation-to-code mapping, assumptions, and limitations.
- `docs/whitepapers/omega_theory_technical_addendum.md`, `docs/whitepapers/yellow_paper_sri_architecture.md`, `docs/whitepapers/Ceraph_HPC_QM_Whitepaper_Addendum.md`, `docs/whitepapers/Project_Ceraph_Instructions.md` — supplementary project documents.
- `docs/ARCHIVE.md` — sizes and last commits for binaries removed from the working tree.
- `pyproject.toml` — packaging metadata and pytest/ruff configuration. `sri` is the installable package. SciPy is required for sparse solves and the Lindblad matrix exponential. NetworkX is the fallback generator for a random regular wiring when the configuration model does not return a simple connected graph.
- `LICENSE` — proprietary licence. Copyright (c) 2026 Ceraph Corporation. All rights reserved.
