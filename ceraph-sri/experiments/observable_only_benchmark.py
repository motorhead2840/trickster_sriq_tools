"""Compare oracle paired projection vs observable-only CV MMSE and classical filters.

Preregisters C1/C2 evaluation. The estimator never receives the clean signal or
the latent noise at inference time for non-oracle methods.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
from pathlib import Path

import numpy as np

from sri.cv.estimator import (
    SensingConfig,
    analytic_mse,
    estimate_signal,
    oracle_paired_projection,
    simulate_measurements,
    single_mode_wiener,
)
from sri.cv.gaussian import mse_optimal
from sri.registry.conjectures import record_result, registry_as_dicts, reset_registry


def _moving_average(values: np.ndarray, width: int = 9) -> np.ndarray:
    radius = width // 2
    out = np.empty_like(values, dtype=np.float64)
    for i in range(len(values)):
        lo = max(0, i - radius)
        hi = min(len(values), i + radius + 1)
        out[i] = float(np.mean(values[lo:hi]))
    return out


def _median_filter(values: np.ndarray, width: int = 9) -> np.ndarray:
    radius = width // 2
    out = np.empty_like(values, dtype=np.float64)
    for i in range(len(values)):
        lo = max(0, i - radius)
        hi = min(len(values), i + radius + 1)
        out[i] = float(np.median(values[lo:hi]))
    return out


def run_seed(
    seed: int,
    shots: int,
    config: SensingConfig,
) -> dict:
    rng = np.random.default_rng(seed)
    signal, noise, y_a, y_b = simulate_measurements(config, shots, rng)
    mmse = estimate_signal(y_a, y_b, config)
    single = single_mode_wiener(y_a, config)
    # Classical filters on the signal-mode observation only (no reference).
    mean_est = _moving_average(y_a)
    median_est = _median_filter(y_a)
    oracle = oracle_paired_projection(signal, noise)

    def mse(est: np.ndarray) -> float:
        return float(np.mean((est - signal) ** 2))

    sample_errors = (mmse.estimate - signal) ** 2
    sample_mse = float(np.mean(sample_errors))
    se = float(np.std(sample_errors, ddof=1) / np.sqrt(shots))
    analytic = analytic_mse(config)
    coherent_ref = SensingConfig(
        signal_variance=config.signal_variance,
        noise_variance=config.noise_variance,
        coupling=config.coupling,
        efficiency=config.efficiency,
        signal_r=0.0,
        reference_r=0.0,
    )
    squeezed_signal = SensingConfig(
        signal_variance=config.signal_variance,
        noise_variance=config.noise_variance,
        coupling=config.coupling,
        efficiency=config.efficiency,
        signal_r=max(config.signal_r, config.reference_r),
        reference_r=0.0,
    )
    return {
        "seed": seed,
        "shots": shots,
        "oracle_paired_mse": mse(oracle),
        "mmse_two_mode_sample_mse": sample_mse,
        "mmse_two_mode_analytic_mse": analytic,
        "mmse_sample_standard_error": se,
        "mmse_within_5sigma": abs(sample_mse - analytic) <= 5 * max(se, 1e-15),
        "single_mode_mse": mse(single),
        "moving_average_mse": mse(mean_est),
        "median_filter_mse": mse(median_est),
        "analytic_coherent_two_mode_mse": analytic_mse(coherent_ref),
        "analytic_squeezed_signal_mse": analytic_mse(squeezed_signal),
        "analytic_squeezed_reference_mse": analytic,
        "weights": mmse.weights.tolist(),
    }


def run(
    output_dir: Path,
    seeds: list[int] | None = None,
    shots: int = 20000,
    signal_variance: float = 1.0,
    noise_variance: float = 1.0,
    efficiency: float = 0.8,
    coupling: float = 1.0,
    squeezing: float = 0.7,
) -> dict:
    if seeds is None:
        seeds = [108, 109, 110]
    reset_registry()
    config = SensingConfig(
        signal_variance=signal_variance,
        noise_variance=noise_variance,
        coupling=coupling,
        efficiency=efficiency,
        signal_r=0.0,
        reference_r=squeezing,
    )
    config.validate()
    per_seed = [run_seed(seed, shots, config) for seed in seeds]

    # C1: oracle ~0; MMSE matches analytic; never used s/n in estimate_signal.
    oracle_ok = all(row["oracle_paired_mse"] < 1e-20 for row in per_seed)
    mmse_ok = all(row["mmse_within_5sigma"] for row in per_seed)
    c1_status = "confirmed" if oracle_ok and mmse_ok else "refuted"

    # C2: ordering vs closed form across seeds.
    ordering_ok = True
    for row in per_seed:
        analytic_order = sorted(
            [
                ("coherent", row["analytic_coherent_two_mode_mse"]),
                ("sq_ref", row["analytic_squeezed_reference_mse"]),
                ("sq_sig", row["analytic_squeezed_signal_mse"]),
            ],
            key=lambda item: item[1],
        )
        # Closed-form ordering must be consistent with mse_optimal calls.
        a_coh = mse_optimal(signal_variance, noise_variance, coupling, efficiency, 0.0, 0.0)
        a_ref = mse_optimal(signal_variance, noise_variance, coupling, efficiency, 0.0, squeezing)
        a_sig = mse_optimal(signal_variance, noise_variance, coupling, efficiency, squeezing, 0.0)
        if not (abs(a_coh - row["analytic_coherent_two_mode_mse"]) < 1e-12):
            ordering_ok = False
        if not (a_ref <= a_coh + 1e-12 and a_sig <= a_ref + 1e-12):
            # Default protocol expects sq_sig best, then sq_ref, then coherent.
            # Allow equality only if squeezing is zero.
            if squeezing > 0 and not (a_sig < a_ref < a_coh):
                ordering_ok = False
        _ = analytic_order  # retained for artifact readability
    c2_status = "confirmed" if ordering_ok and mmse_ok else "refuted"

    artifact_csv = "observable_only_records.json"
    record_result(
        "C1",
        c1_status,
        artifact_paths=[str(output_dir / "observable_only_summary.json")],
        notes="Oracle identity verified; MMSE is observable-only Wiener filter.",
    )
    record_result(
        "C2",
        c2_status,
        artifact_paths=[str(output_dir / "observable_only_summary.json")],
        notes="Analytic MSE ordering checked against mse_optimal closed forms.",
    )

    summary = {
        "experiment": "observable_only_cv_mmse",
        "configuration": {
            "seeds": seeds,
            "shots": shots,
            "signal_variance": signal_variance,
            "noise_variance": noise_variance,
            "efficiency": efficiency,
            "coupling": coupling,
            "reference_squeezing": squeezing,
            "vacuum_convention": "hbar=2, VACUUM_VAR=1",
        },
        "per_seed": per_seed,
        "aggregate": {
            "mean_mmse_sample_mse": statistics.fmean(
                row["mmse_two_mode_sample_mse"] for row in per_seed
            ),
            "mean_moving_average_mse": statistics.fmean(
                row["moving_average_mse"] for row in per_seed
            ),
            "mean_median_filter_mse": statistics.fmean(
                row["median_filter_mse"] for row in per_seed
            ),
            "mean_single_mode_mse": statistics.fmean(
                row["single_mode_mse"] for row in per_seed
            ),
            "mmse_beats_moving_average": statistics.fmean(
                row["mmse_two_mode_sample_mse"] for row in per_seed
            )
            < statistics.fmean(row["moving_average_mse"] for row in per_seed),
            "mmse_beats_median": statistics.fmean(
                row["mmse_two_mode_sample_mse"] for row in per_seed
            )
            < statistics.fmean(row["median_filter_mse"] for row in per_seed),
        },
        "conjectures": registry_as_dicts(),
        "limitations": [
            "Classical Monte Carlo of an ideal Gaussian optical model.",
            "MMSE coefficients use known model parameters (parameter oracle), not unknown calibration.",
            "Moving-average/median baselines see only y_A; MMSE also uses y_B — equal-mode resource note applies.",
            "Oracle paired projection is not deployable.",
        ],
        "provenance": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "observable_only_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / artifact_csv).write_text(
        json.dumps(per_seed, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=20000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[108, 109, 110])
    args = parser.parse_args(argv)
    summary = run(args.output_dir, seeds=list(args.seeds), shots=args.shots)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
