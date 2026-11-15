"""Verify uploaded LLM records and estimate paired, stratified bootstrap intervals."""

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

if __package__:
    from .llm_benchmark import auc
else:
    from llm_benchmark import auc


def percentile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def wilson(successes, count):
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    margin = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [center - margin, center + margin]


def analyze(records_path, summary_path, iterations=5000, seed=108):
    if iterations < 1:
        raise ValueError("iterations must be positive")
    summary = json.loads(summary_path.read_text())
    with records_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_count = summary["configuration"]["first_n"]
    if len(rows) != expected_count or not rows:
        raise ValueError("record count does not match summary")
    indices = set()
    parsed = []
    for row in rows:
        index = int(row["dataset_index"])
        if index in indices:
            raise ValueError(f"duplicate dataset index: {index}")
        indices.add(index)
        correct = int(row["correct"])
        if correct not in (0, 1) or correct != int(row["predicted_label"] == row["correct_label"]):
            raise ValueError(f"inconsistent correctness at index {index}")
        confidence, xi = float(row["option_confidence"]), float(row["xi"])
        mean, variance = float(row["logit_mean"]), float(row["logit_variance"])
        if not all(math.isfinite(x) for x in (confidence, xi, mean, variance)):
            raise ValueError(f"non-finite measurement at index {index}")
        if not 0 <= confidence <= 1 or variance < 0 or abs(mean + 1e-7) <= 1e-12:
            raise ValueError(f"invalid measurement at index {index}")
        if not math.isclose(xi, 1 - variance / (mean + 1e-7), rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError(f"inconsistent Xi at index {index}")
        parsed.append((correct, confidence, xi))
    if indices != set(range(expected_count)):
        raise ValueError("dataset indices do not match first_n")

    positive = [row for row in parsed if row[0]]
    negative = [row for row in parsed if not row[0]]
    if not positive or not negative:
        raise ValueError("both correctness classes are needed for AUROC")
    outcomes = [row[0] for row in parsed]
    conf_auc = auc([row[1] for row in parsed], outcomes)
    xi_auc = auc([row[2] for row in parsed], outcomes)
    expected = summary["results"]
    for name, actual in (("correct", len(positive)), ("incorrect", len(negative)),
                         ("accuracy", len(positive) / len(parsed)),
                         ("confidence_correctness_auc", conf_auc), ("xi_correctness_auc", xi_auc)):
        if not math.isclose(actual, expected[name], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{name} does not match summary")

    # Resample within outcome classes while keeping both scores paired for each question.
    rng = random.Random(seed)
    confidence_samples, xi_samples, differences = [], [], []
    for _ in range(iterations):
        sample = rng.choices(positive, k=len(positive)) + rng.choices(negative, k=len(negative))
        y = [row[0] for row in sample]
        c = auc([row[1] for row in sample], y)
        x = auc([row[2] for row in sample], y)
        confidence_samples.append(c)
        xi_samples.append(x)
        differences.append(x - c)

    def interval(values):
        return [percentile(values, 0.025), percentile(values, 0.975)]

    return {
        "validation": "Record counts, correctness, Xi formula and summary metrics agree.",
        "count": len(parsed),
        "correct": len(positive),
        "accuracy": len(positive) / len(parsed),
        "accuracy_wilson_95": wilson(len(positive), len(parsed)),
        "confidence_auc": conf_auc,
        "confidence_auc_bootstrap_95": interval(confidence_samples),
        "xi_auc": xi_auc,
        "xi_auc_bootstrap_95": interval(xi_samples),
        "xi_minus_confidence_auc": xi_auc - conf_auc,
        "xi_minus_confidence_auc_bootstrap_95": interval(differences),
        "method": {
            "bootstrap": "paired, stratified by correctness, percentile interval",
            "iterations": iterations, "seed": seed,
            "caveat": "Exploratory; intervals assume these examples represent a sampling population.",
        },
        "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in (records_path, summary_path)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=108)
    args = parser.parse_args()
    result = analyze(args.results_dir / "llm_records.csv", args.results_dir / "llm_summary.json",
                     args.iterations, args.seed)
    output = json.dumps(result, indent=2) + "\n"
    (args.results_dir / "llm_analysis.json").write_text(output)
    print(output)


if __name__ == "__main__":
    main()
