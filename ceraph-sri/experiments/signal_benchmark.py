"""Observable-only denoising benchmark; the paired projector adds no estimator."""

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import statistics
from pathlib import Path


WINDOWS = (1, 3, 5, 9)


def signals(seed, count, length, noise_std):
    rng = random.Random(seed)
    for _ in range(count):
        phase = rng.uniform(0, 2 * math.pi)
        frequency = rng.uniform(0.5, 3.0)
        step_at = rng.randrange(length // 4, 3 * length // 4)
        step = rng.uniform(-1, 1)
        clean = [
            math.sin(2 * math.pi * frequency * i / length + phase)
            + (step if i >= step_at else 0)
            for i in range(length)
        ]
        observed = [value + rng.gauss(0, noise_std) for value in clean]
        yield clean, observed


def filtered(values, width, method):
    radius = width // 2
    result = []
    for i in range(len(values)):
        window = values[max(0, i - radius):min(len(values), i + radius + 1)]
        result.append(statistics.median(window) if method == "median" else statistics.fmean(window))
    return result


def mse(actual, expected):
    return statistics.fmean((x - y) ** 2 for x, y in zip(actual, expected))


def paired_projection(observed, estimated_clean):
    # Construct the conjugate using only the observed input and an estimated noise.
    # Algebraically, this is exactly the clean estimator apart from rounding.
    estimated_noise = [x - value for x, value in zip(observed, estimated_clean)]
    return [(x + (x - 2 * noise)) / 2 for x, noise in zip(observed, estimated_noise)]


def select_width(data, method):
    return min(WINDOWS, key=lambda w: statistics.fmean(
        mse(filtered(observed, w, method), clean) for clean, observed in data
    ))


def run(output_dir, seed=108, train_count=100, test_count=200, length=128, noise_std=0.5):
    if train_count < 1 or test_count < 1 or length < 8 or noise_std <= 0:
        raise ValueError("Counts must be positive, length >= 8, and noise_std > 0")
    training = list(signals(seed, train_count, length, noise_std))
    test = list(signals(seed + 1, test_count, length, noise_std))
    widths = {method: select_width(training, method) for method in ("mean", "median")}
    rows = []
    for index, (clean, observed) in enumerate(test):
        mean = filtered(observed, widths["mean"], "mean")
        median = filtered(observed, widths["median"], "median")
        projected = paired_projection(observed, median)
        rows.append({
            "signal": index, "noisy_mse": mse(observed, clean),
            "moving_average_mse": mse(mean, clean), "median_mse": mse(median, clean),
            "paired_mse": mse(projected, clean), "oracle_mse": 0.0,
            "max_paired_estimator_difference": max(abs(a - b) for a, b in zip(projected, median)),
        })
    summary = {
        "experiment": "synthetic_signal_denoising",
        "configuration": {"seed_train": seed, "seed_test": seed + 1, "train_signals": train_count,
                          "test_signals": test_count, "length": length, "noise_std": noise_std,
                          "candidate_windows": list(WINDOWS), "selected_windows": widths},
        "mean_mse": {key: statistics.fmean(row[key] for row in rows) for key in
                     ("noisy_mse", "moving_average_mse", "median_mse", "paired_mse", "oracle_mse")},
        "max_paired_estimator_difference": max(row["max_paired_estimator_difference"] for row in rows),
        "notes": ["The median estimator uses only the noisy observation.",
                  "The paired calculation is algebraically the same estimator, not a new denoising method.",
                  "The zero-error oracle uses clean ground truth and is not deployable.",
                  "This is a synthetic in-distribution test, not a real-world efficacy study."],
        "provenance": {"python": platform.python_version(),
                       "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "signal_records.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "signal_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=108)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
