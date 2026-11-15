# SRI feasibility: preliminary VC diligence report

## Status

SRI is an early-stage research project. Its GPU demo executes synthetic finite-dimensional
arithmetic; it does not demonstrate unknown-noise removal, hallucination prevention, quantum
computation, or constant-time vector processing. This report is preliminary, not independent
verification or a valuation. No customer, revenue, IP ownership, or investment-return claim can
be established from these experiments.

## Existing A100 run

A Colab run reported by the project used an NVIDIA A100-SXM4-80GB, 25 cycles, vector length
16,384, and 10 timed repeats. Its reported median synchronized batched-kernel time was
0.2097175 ms after compilation and warm-up, and maximum known-noise projection MSE was
1.33049e-13. Its script hash matched `sri_gpu_demo.py` in this checkout. The report was supplied
as pasted JSON; the A100 CSV was not provided, and this result was not independently replicated.
This is not end-to-end inference latency or a scaling study. Exact clean signal and noise were
supplied to the computation.

## Observable-only signal experiment

The accompanying `experiments/signal_benchmark.py` was run locally with its defaults:
100 synthetic training signals and 200 independently seeded test signals, each 128 samples,
with Gaussian noise standard deviation 0.5. The mean and median filters each selected a window
of 9 from widths 1, 3, 5 and 9 on the training signals. The local run yielded:

| Method | Mean test MSE |
|---|---:|
| Noisy observation | 0.25049 |
| Moving average | 0.03165 |
| Median filter | 0.04657 |
| Paired projection with median-estimated noise | 0.04657 |
| Known-clean oracle, not deployable | 0 |

The paired output differed from the median estimate by at most 2.22e-16 per element. That is
expected: the pairing is algebraically the estimator itself, not an independent denoising
algorithm. The moving average performed better on this synthetic test. Results are specific to
these settings, are not a real-world benchmark, and do not establish an SRI advantage. The
signal script SHA-256 for this run was
`863c7fc47d14900e3baa7e6af0c43d105a23194e1f2202565f43be2129b21cf0`.
The [summary](benchmark-results/signal/signal_summary.json) and
[per-signal records](benchmark-results/signal/signal_records.csv) are preserved with this report.

## LLM reliability proxy on A100

The project supplied the output of a standalone Colab cell using Qwen/Qwen2.5-0.5B-Instruct on
the first 100 TruthfulQA multiple-choice validation examples. Answer order was shuffled with a
fixed seed. The experiment ranks correct versus incorrect predictions using the model's
answer-option confidence and the repository's Xi statistic, with higher Xi prespecified as the
more reliable direction.

| Metric | Result | Exploratory 95% interval |
|---|---:|---:|
| Correct multiple-choice answers | 13 / 100 (13%) | 7.8% to 21.0% |
| Answer-option confidence AUROC | 0.6242 | 0.461 to 0.772 |
| Xi AUROC | 0.4651 | 0.273 to 0.658 |
| Xi minus confidence AUROC | -0.1592 | -0.372 to 0.046 |

In this sample Xi ranked correctness less well than the confidence baseline and below 0.5
(the no-discrimination reference); this is not evidence that Xi improves reliability. There
were only 13 correct predictions. The paired AUROC-difference interval includes zero, so this
exploratory sample does not establish a reliable population difference. The
reported 13% accuracy is specific to this answer-letter scoring procedure and sample, not the
model's standard TruthfulQA score.

The AUROC intervals use 5,000 paired, correctness-stratified bootstrap resamples (seed 108);
the accuracy interval is Wilson's interval. These are exploratory sampling intervals, not a
guarantee of generalization: the first 100 examples were not randomly selected, and the same
model and fixed choice ordering were used throughout. The data were not independently replicated.

The protocol scores next-token forms ` A`, ` B`, etc., rather than standard full-choice
likelihood. It does not test generated answers, hallucination prevention, V3 damping, inference
latency, or quantum processing. Public benchmark contamination and choice-order effects remain
possible. The supplied [summary](benchmark-results/llm/llm_summary.json) records the A100,
software versions and dataset fingerprint, but not a model commit. The supplied
[per-question records](benchmark-results/llm/llm_records.csv) reproduce the summary metrics and
Xi formula, but cannot independently establish that the model generated them. The
[analysis](benchmark-results/llm/llm_analysis.json) records the intervals and input hashes.
The standalone Colab cell was provided in the session, but an executed notebook was not supplied.
Reproduce the record analysis with:

```bash
python experiments/analyze_llm_results.py --results-dir docs/benchmark-results/llm
```

## Investment implications and next milestones

The investment hypothesis is that a demonstrably useful error estimator or AI-reliability
signal might become a product. The main risks are that the paired formula requires a separate
estimator, the damping behavior remains problematic, and no commercial advantage or ownership
of defensible IP has been established.

Before presenting a funding case, preserve an executed notebook and independently replicate the
studies. Test a distinct observable-only method on representative held-out customer
data against strong baselines. Prespecify success criteria and uncertainty intervals, replicate
across models or datasets, and measure end-to-end cost and latency. Independently validate the
results and obtain qualified IP and ownership review.

## Foundation-sprint update

The CV foundation month adds an observable-only Gaussian MMSE path (`sri.cv.estimator`),
axiomatic `Ξ_SNR`, dual Circuit-B branches, and a Braket dry-run adapter. These are classical
simulations of CV optics and control laws. They do not by themselves establish customer value,
quantum advantage, or investment return. See `docs/cv-foundation-month-report.md`.
