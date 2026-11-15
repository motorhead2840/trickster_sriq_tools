"""Generate a factual report from both completed benchmark summaries."""

import argparse
import json
from pathlib import Path


def format_auc(value):
    return "undefined (only one correctness class)" if value is None else f"{value:.4f}"


def render(signal, llm):
    s = signal["mean_mse"]
    r = llm["results"]
    sc = signal["configuration"]
    lc = llm["configuration"]
    device = llm["provenance"]["device"]
    return f"""# SRI: experimental feasibility and VC diligence

## Status and scope

These are exploratory research results, not independent validation or proof of a commercial product.
The signal experiment uses synthetic data. The LLM experiment is a multiple-choice proxy for
truthfulness; it does not evaluate open-ended hallucination prevention. The results are conditional
on the dataset, model, input distribution and experimental settings below.

## Observable-only signal denoising

Train/test signals: {sc['train_signals']}/{sc['test_signals']}; separate seeds
{sc['seed_train']}/{sc['seed_test']}; length {sc['length']}; Gaussian noise standard deviation
{sc['noise_std']}. Filter widths were selected on the training signals from
{sc['candidate_windows']}; selected widths: {sc['selected_windows']}.

| Method | Mean test MSE (lower is better) |
|---|---:|
| Noisy observation | {s['noisy_mse']:.8g} |
| Moving average | {s['moving_average_mse']:.8g} |
| Median filter | {s['median_mse']:.8g} |
| Paired projection using median-estimated noise | {s['paired_mse']:.8g} |
| Ground-truth oracle (not deployable) | {s['oracle_mse']:.8g} |

The paired calculation uses the observable input and a median-filter estimate. It is algebraically
equivalent to that estimate; maximum measured elementwise difference:
{signal['max_paired_estimator_difference']:.3g}. A good result therefore does not demonstrate
an improvement over the median estimator or an independent SRI denoiser.

## LLM reliability proxy

Model: `{lc['model']}`; dataset: `{lc['dataset']}`, `{lc['config']}` / `{lc['split']}`;
first {lc['first_n']} examples; deterministic choice shuffle seed {lc['seed']}; GPU: {device}.
The model chooses a single answer-letter token. This is not a full answer-generation benchmark.

| Metric | Result |
|---|---:|
| Multiple-choice accuracy | {r['accuracy']:.2%} ({r['correct']}/{lc['first_n']}) |
| Option-confidence AUROC for correctness | {format_auc(r['confidence_correctness_auc'])} |
| SRI Xi AUROC for correctness | {format_auc(r['xi_correctness_auc'])} |

AUROC tests whether a score ranks correct choices above incorrect choices, with higher Xi
prespecified as the stable direction. It does not show that the system prevents hallucinations.
No detection threshold, V3 damping, latency, or open-ended answers were evaluated. Small samples,
answer-order effects and possible model training-data contamination limit the conclusion.

## Feasibility and investment implications

GPU execution and reproducible arithmetic are feasible. These experiments do not establish an
incremental denoising advantage from the paired projector, a working hallucination-prevention
product, quantum computation, physical error correction, or constant-time processing. A
positive Xi AUROC or a result above the confidence baseline alone is exploratory, not proof of
commercial efficacy; replicate on additional held-out datasets and models with uncertainty
intervals and a prespecified operating threshold before claiming practical value.

A VC case would depend on a technically distinct observable-only method, evidence of incremental
performance on representative customer data, validated end-to-end cost and latency, customer
traction, team and verified IP ownership. No valuation, patentability, customer demand, or
investment return can be inferred from these benchmark results.

## Provenance

Signal script SHA-256: `{signal['provenance']['script_sha256']}`.
LLM script SHA-256: `{llm['provenance']['script_sha256']}`.
Model commit: `{llm['provenance']['model_commit']}`;
dataset fingerprint: `{llm['provenance']['dataset_fingerprint']}`.
Preserve the two summary JSON files and per-example CSVs with this report.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.results_dir
    signal = json.loads((directory / "signal_summary.json").read_text())
    llm = json.loads((directory / "llm_summary.json").read_text())
    report = render(signal, llm)
    (directory / "vc_feasibility_report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
