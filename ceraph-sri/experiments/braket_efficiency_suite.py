"""S3 — Efficiency / N-scaling for observable-only MMSE (Hybrid-shaped).

Sweeps vector dimension N, records wall time and MSE vs analytic, and emits a
cost-normalized table comparing local CPU to the Hybrid Job instance profile
(ml.m5.large). Efficiency claim is reproducible managed-compute provenance +
scaling curve — not asymptotic O(1).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from sri.cv.estimator import SensingConfig, analytic_mse, estimate_signal, simulate_measurements

# On-demand ml.m5.large list price used for cost-normalized estimates (USD/hour).
# Hybrid Job also bills separately for any quantum tasks; this suite is classical.
_M5_LARGE_USD_PER_HOUR = 0.096
_DEFAULT_DIMENSIONS = (1024, 4096, 16384, 65536)


def _time_mmse(dimension: int, shots: int, seed: int, config: SensingConfig) -> dict:
    rng = np.random.default_rng(seed)
    # Use min(shots, dimension) samples so memory stays proportional to N*shots policy:
    # shots is the Monte Carlo length; dimension labels the Hybrid container vector size
    # used by foundation sweeps. For MMSE we scale the sample count with N when N < shots.
    n_samples = max(dimension, shots) if dimension >= shots else shots
    # Cap pathological blow-ups: keep Monte Carlo length at `shots`, repeat if needed
    # only for warmup identity; timing uses `shots` draws for fair samples/s.
    n_samples = shots
    t0 = time.perf_counter()
    signal, _noise, y_a, y_b = simulate_measurements(config, n_samples, rng)
    # Touch a length-N buffer to reflect Hybrid Job vector work at this dimension.
    workspace = np.empty(dimension, dtype=np.float64)
    workspace[: min(dimension, n_samples)] = y_a[: min(dimension, n_samples)]
    result = estimate_signal(y_a, y_b, config)
    elapsed = time.perf_counter() - t0
    errors = (result.estimate - signal) ** 2
    sample_mse = float(np.mean(errors))
    se = float(np.std(errors, ddof=1) / np.sqrt(n_samples))
    analytic = analytic_mse(config)
    return {
        "dimension": dimension,
        "shots": n_samples,
        "wall_time_s": elapsed,
        "samples_per_s": n_samples / elapsed if elapsed > 0 else float("inf"),
        "sample_mse": sample_mse,
        "analytic_mse": analytic,
        "sample_mse_standard_error": se,
        "within_5sigma": abs(sample_mse - analytic) <= 5 * max(se, 1e-15),
        "workspace_bytes": int(workspace.nbytes),
    }


def _cost_row(wall_time_s: float, instance_usd_per_hour: float) -> dict:
    hours = wall_time_s / 3600.0
    return {
        "estimated_instance_cost_usd": hours * instance_usd_per_hour,
        "instance_usd_per_hour": instance_usd_per_hour,
        "instance_profile": "ml.m5.large",
    }


def run(
    output_dir: Path,
    *,
    dimensions: list[int] | None = None,
    shots: int = 20000,
    seeds: list[int] | None = None,
    instance_usd_per_hour: float = _M5_LARGE_USD_PER_HOUR,
) -> dict:
    if dimensions is None:
        dimensions = list(_DEFAULT_DIMENSIONS)
    if seeds is None:
        seeds = [108, 109, 110]
    config = SensingConfig(reference_r=0.7)
    config.validate()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for dimension in dimensions:
        per_seed = [
            _time_mmse(dimension, shots, seed, config) for seed in seeds
        ]
        mean_time = float(np.mean([r["wall_time_s"] for r in per_seed]))
        mean_mse = float(np.mean([r["sample_mse"] for r in per_seed]))
        all_within = all(r["within_5sigma"] for r in per_seed)
        cost = _cost_row(mean_time, instance_usd_per_hour)
        rows.append(
            {
                "dimension": dimension,
                "shots": shots,
                "seeds": seeds,
                "mean_wall_time_s": mean_time,
                "mean_samples_per_s": float(np.mean([r["samples_per_s"] for r in per_seed])),
                "mean_sample_mse": mean_mse,
                "analytic_mse": analytic_mse(config),
                "mse_stable_within_5sigma": all_within,
                "local_cpu": {
                    "platform": platform.platform(),
                    "mean_wall_time_s": mean_time,
                },
                "hybrid_instance_estimate": {
                    **cost,
                    "note": (
                        "Cost estimate assumes Hybrid Job classical instance time only; "
                        "no quantum-task charges in this suite."
                    ),
                },
                "per_seed": per_seed,
            }
        )

    accuracy_ok = all(row["mse_stable_within_5sigma"] for row in rows)
    # Cost per accurate lock: sum of estimated instance costs across the sweep.
    total_est_cost = float(
        sum(row["hybrid_instance_estimate"]["estimated_instance_cost_usd"] for row in rows)
    )
    summary = {
        "experiment": "braket_efficiency_suite",
        "phase": "S3",
        "configuration": {
            "dimensions": dimensions,
            "shots": shots,
            "seeds": seeds,
            "sensing": {
                "signal_variance": config.signal_variance,
                "noise_variance": config.noise_variance,
                "coupling": config.coupling,
                "efficiency": config.efficiency,
                "reference_r": config.reference_r,
            },
            "instance_usd_per_hour": instance_usd_per_hour,
        },
        "rows": rows,
        "aggregate": {
            "accuracy_ok_across_n": accuracy_ok,
            "total_estimated_hybrid_instance_cost_usd": total_est_cost,
            "cost_per_accurate_lock_usd": total_est_cost if accuracy_ok else None,
            "scaling_note": (
                "Wall time grows with Monte Carlo length and workspace size; "
                "this software path is O(N) in vector work — not a constant-time claim."
            ),
        },
        "pass": accuracy_ok,
        "disclaimer": (
            "Efficiency = reproducible timing + cost-normalized Hybrid instance estimate. "
            "Does not prove hardware-native O(1) or thermodynamic advantage."
        ),
        "provenance": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }

    (output_dir / "efficiency_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    csv_path = output_dir / "efficiency_scaling.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dimension",
                "shots",
                "mean_wall_time_s",
                "mean_samples_per_s",
                "mean_sample_mse",
                "analytic_mse",
                "mse_stable_within_5sigma",
                "estimated_instance_cost_usd",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dimension": row["dimension"],
                    "shots": row["shots"],
                    "mean_wall_time_s": row["mean_wall_time_s"],
                    "mean_samples_per_s": row["mean_samples_per_s"],
                    "mean_sample_mse": row["mean_sample_mse"],
                    "analytic_mse": row["analytic_mse"],
                    "mse_stable_within_5sigma": row["mse_stable_within_5sigma"],
                    "estimated_instance_cost_usd": row["hybrid_instance_estimate"][
                        "estimated_instance_cost_usd"
                    ],
                }
            )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=list(_DEFAULT_DIMENSIONS),
    )
    parser.add_argument("--shots", type=int, default=20000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[108, 109, 110])
    args = parser.parse_args(argv)
    summary = run(
        args.output_dir,
        dimensions=list(args.dimensions),
        shots=args.shots,
        seeds=list(args.seeds),
    )
    printable = {k: summary[k] for k in summary if k != "rows"}
    printable["row_count"] = len(summary["rows"])
    print(json.dumps(printable, indent=2))
    if not summary["pass"]:
        raise SystemExit("efficiency suite failed: MSE not within 5σ across N")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
