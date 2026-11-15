# Run the GPU proof of concept in Google Colab

For the distinct correlated-noise photonic sensing simulation and cross-backend
A100 validation, use [`tests/CV_Photonics_A100_Validation.ipynb`](tests/CV_Photonics_A100_Validation.ipynb)
and follow the [protocol](docs/cv-photonics-protocol.md). This is a classical
simulation of idealized optical measurements, not execution on quantum hardware.

This run uses the V3.0 equations as printed in the supplied dissertation. It does not change the Circuit B denominator or tune it to the document's worked example. Each run creates a CSV of per-cycle measurements and a JSON report containing the actual JAX backend/device, configuration, source-equation strings, script hash, and synchronized GPU timing.

## Colab steps

1. In Colab, choose **Runtime → Change runtime type → T4 GPU** or an **A100 GPU** runtime. For the requested A100 experiment, enable the A100 accelerator where available.
2. Clone or upload this repository into the runtime. In a notebook cell, run:

   ```python
   %cd /content/sri-quantum
   !git pull
   ```

   If you uploaded a ZIP instead, change to the extracted repository directory and skip `git pull`.
3. Check that the selected runtime exposes a GPU:

   ```python
   import jax
   print(jax.__version__, jax.default_backend(), jax.devices())
   ```

   The demo will stop if JAX has no active GPU backend; it will not silently produce a CPU result. Avoid reinstalling JAX unless this check shows no GPU device. If needed, use the CUDA-enabled JAX installation instructions appropriate to the CUDA runtime, then restart the Colab runtime and repeat the check.
4. Run the demonstration and tests:

   ```python
   !python sri_gpu_demo.py --require-a100 --output-dir /content/sri_gpu_results
   !python -m unittest discover -s tests -v
   ```

   Omit `--require-a100` if running on a different GPU. Defaults are 16,384 dimensions, 25 cycles, seed 108, and 10 synchronized repeats after JIT warm-up. The synthetic high-noise interval is cycles 10–20 inclusive.
5. Inspect and download the actual artifacts:

   ```python
   import json
   from pathlib import Path
   from google.colab import files

   result_dir = Path("/content/sri_gpu_results")
   report = json.loads((result_dir / "sri_gpu_run.json").read_text())
   print(json.dumps(report, indent=2))
   files.download(str(result_dir / "sri_gpu_run.json"))
   files.download(str(result_dir / "sri_gpu_telemetry.csv"))
   ```

   Share those generated files (or paste the complete JSON and upload the CSV) here for review. Do not copy example numbers into the report; use only the metrics produced by your runtime.

## How to interpret the experiment

- The inputs are synthetic; the exact clean vector and injected noise are both present in the computation. A low projection MSE confirms the paired-state arithmetic for those inputs, not recovery of unknown truth.
- `Xi`, Circuit A, Circuit B, `chi_v3`, and damped-logit variance are computed using the dissertation's printed Equations 11–13 and default parameters. In particular Circuit B uses `Var(gradient) / (sigma_sat^2 * epsilon)` inside the sigmoid, with `epsilon = 1e-7`.
- The measured time is for the synchronized, batched JAX kernel after compilation warm-up. It is not end-to-end inference latency, an FMA instruction count, or proof of O(1) scaling.
- Save the JSON and CSV together. The JSON is the run provenance and includes a SHA-256 of the Python script used.

## Commercial feasibility experiments

These are exploratory tests, not demonstrations of a working product. The signal benchmark compares
noisy inputs, a moving average, a median filter, and the paired identity using noise estimated from
the median-filtered observation. The paired result is mathematically the median filter, not an
independent SRI method. The known-clean oracle is an unattainable reference. Window sizes are
selected on separate synthetic training signals and evaluated on independently seeded signals.

The second experiment tests whether Xi ranks correct answers above incorrect ones better than
the model's answer-option confidence on TruthfulQA MC1. It selects a single answer-letter token
on the first 100 validation examples, with deterministically shuffled choices. This is a limited
multiple-choice proxy, not open-ended hallucination prevention. It does not evaluate V3 damping,
because this experiment does not specify or compute the required gradient. AUC can be undefined when
all answers are right or all are wrong. The answer scores use the single next-token forms
` A`, ` B`, etc.; they do not follow the standard full-choice likelihood protocol. No operating
threshold is fitted.

1. Select an A100 runtime. Obtain the **updated** repository files containing the `experiments/`
   directory. The old ZIP used for the earlier A100 run does not contain these files. Upload a ZIP
   of the updated checkout or the provided experiment bundle. Do not upload untrusted ZIPs.
2. Upload and extract it, then locate the experiment directory:

   ```python
   import tempfile, zipfile
   from pathlib import Path
   from google.colab import files

   uploaded = files.upload()
   if len(uploaded) != 1:
       raise ValueError("Upload exactly one updated repository ZIP")
   archive = Path(next(iter(uploaded)))
   workdir = Path(tempfile.mkdtemp(prefix="sri-study-"))
   with zipfile.ZipFile(archive) as z:
       z.extractall(workdir)  # Only extract a ZIP you trust.
   matches = list(workdir.rglob("experiments/signal_benchmark.py"))
   if len(matches) != 1:
       raise ValueError("Expected one updated checkout containing experiments/signal_benchmark.py")
   repo = matches[0].parent.parent
   for script in ("llm_benchmark.py", "make_vc_report.py"):
       if not (repo / "experiments" / script).is_file():
           raise FileNotFoundError(script)
   print("Repository:", repo)
   ```

3. Check dependencies. Colab normally supplies PyTorch; install missing packages only if the
   check fails, then restart the runtime and upload the updated ZIP again. The model and dataset
   also require network access. The model is downloaded from Hugging Face.

   ```python
   import importlib.util
   missing = [name for name in ("torch", "transformers", "datasets")
              if importlib.util.find_spec(name) is None]
   print("Missing dependencies:", missing)
   ```

   If any are missing, install **only** those shown, e.g. `%pip install transformers datasets`
   when those two are missing. Check CUDA availability after restarting; the LLM benchmark refuses
   CPU fallback. Versions are saved in the result provenance.

4. Run both studies in sequence, generate a report from their actual summaries, and download all
   artifacts together. Processing 100 examples and the model download can take several minutes.
   If anything fails, report the error; don't treat a partial run as a completed study.

   ```python
   import subprocess, tempfile, zipfile
   from pathlib import Path
   from google.colab import files

   results = Path(tempfile.mkdtemp(prefix="sri-commercial-results-"))
   subprocess.run(["nvidia-smi"], check=True)
   subprocess.run(["python", "experiments/signal_benchmark.py",
                   "--output-dir", str(results)], cwd=repo, check=True)
   subprocess.run(["python", "experiments/llm_benchmark.py",
                   "--require-a100", "--output-dir", str(results)], cwd=repo, check=True)
   subprocess.run(["python", "experiments/make_vc_report.py",
                   "--results-dir", str(results)], cwd=repo, check=True)

   bundle = results.parent / (results.name + ".zip")
   with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
       for name in ("signal_summary.json", "signal_records.csv", "llm_summary.json",
                    "llm_records.csv", "vc_feasibility_report.md"):
           z.write(results / name, arcname=name)
   files.download(str(bundle))
   print("Share the ZIP for independent review of the report and results.")
   ```

The small benchmark is not a statistical validation or a customer trial. The report must not
be presented as evidence of quantum computing, constant-time processing, cryptography, or a
deployed product. Preserve raw CSVs and JSONs; model and dataset revisions and script hashes are
recorded when available.

## CV foundation sprint (A100 → Braket-ready)

Use [`tests/SRI_Foundation_A100.ipynb`](tests/SRI_Foundation_A100.ipynb) on an A100 runtime. Prefer the compressed E1 lock (do **not** re-run the 324-point campaign on A100):

```bash
# Phase 1 — E1–E4 lock (stop Colab when exit criteria pass)
python experiments/e1_colab_lock.py --output-dir docs/benchmark-results --require-gpu

# Phase 2a — Braket SV1 (emulate until credentials+budgets ready)
python experiments/braket_live_smoke.py --output-dir docs/benchmark-results/braket-live --emulate
# BRAKET_SUBMIT=1 BRAKET_S3_BUCKET=... python experiments/braket_live_smoke.py \
#   --output-dir docs/benchmark-results/braket-live --live

# Phase 2b — Hybrid-equivalent E1–E4
python experiments/braket_hybrid_e1_e4.py \
  --output-dir docs/benchmark-results/braket-live \
  --reference-lock docs/benchmark-results/e1_colab_lock.json
```

Setup: [`docs/aws-braket-setup.md`](docs/aws-braket-setup.md). Spend: [`docs/compute-unit-spend-log.md`](docs/compute-unit-spend-log.md). Report: [`docs/e1-colab-braket-report.md`](docs/e1-colab-braket-report.md).
