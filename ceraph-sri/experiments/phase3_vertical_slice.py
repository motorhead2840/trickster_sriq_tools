"""Phase 3 falsifiable vertical slice: deployable dual-mode CV sensing path.

Preregistered protocol: docs/phase3-vertical-slice-spec.md
CPU-only. Observable path never receives clean s or latent n.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from sri.cv.estimator import (
    SensingConfig,
    analytic_mse,
    estimate_signal,
    observable_yoga_buddhi,
    oracle_paired_projection,
    simulate_measurements,
    single_mode_wiener,
)
from sri.registry.conjectures import record_result, registry_as_dicts, reset_registry

# --- Preregistered constants (must match docs/phase3-vertical-slice-spec.md) ---
SEEDS = [201, 202, 203, 204, 205]
SHOTS = 20_000
NOISE_GRID = [0.5, 1.0, 2.0]
SIGNAL_VARIANCE = 1.0
EFFICIENCY = 0.8
COUPLING = 1.0
SIGNAL_R = 0.0
REFERENCE_R = 0.7
PASS_RELATIVE_REDUCTION = 0.08
ALGEBRA_RTOL = 1e-10


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


def _mse(est: np.ndarray, signal: np.ndarray) -> float:
    return float(np.mean((est - signal) ** 2))


def run_cell(seed: int, noise_variance: float, shots: int) -> dict:
    config = SensingConfig(
        signal_variance=SIGNAL_VARIANCE,
        noise_variance=noise_variance,
        coupling=COUPLING,
        efficiency=EFFICIENCY,
        signal_r=SIGNAL_R,
        reference_r=REFERENCE_R,
    )
    config.validate()
    rng = np.random.default_rng(seed)
    signal, noise, y_a, y_b = simulate_measurements(config, shots, rng)

    t0 = time.perf_counter()
    sri = observable_yoga_buddhi(y_a, y_b, config)
    plain = estimate_signal(y_a, y_b, config)
    single = single_mode_wiener(y_a, config)
    wall_s = time.perf_counter() - t0

    # Ablation: SRI packaging vs plain MMSE (must match).
    algebra_max_abs = float(np.max(np.abs(sri.projected - plain.estimate)))
    algebra_ok = bool(
        np.allclose(sri.projected, plain.estimate, rtol=ALGEBRA_RTOL, atol=ALGEBRA_RTOL)
    )

    sample_errors = (sri.projected - signal) ** 2
    sample_mse = float(np.mean(sample_errors))
    se = float(np.std(sample_errors, ddof=1) / np.sqrt(shots))
    analytic = analytic_mse(config)
    within_5sigma = abs(sample_mse - analytic) <= 5.0 * max(se, 1e-15)

    single_mse = _mse(single, signal)
    ma_mse = _mse(_moving_average(y_a), signal)
    med_mse = _mse(_median_filter(y_a), signal)
    oracle_mse = _mse(oracle_paired_projection(signal, noise), signal)

    relative_vs_single = 1.0 - (sample_mse / single_mse) if single_mse > 0 else float("nan")

    return {
        "seed": seed,
        "noise_variance": noise_variance,
        "shots": shots,
        "sri_sample_mse": sample_mse,
        "plain_mmse_sample_mse": _mse(plain.estimate, signal),
        "analytic_mmse": analytic,
        "sample_standard_error": se,
        "mmse_within_5sigma": within_5sigma,
        "algebra_max_abs_diff": algebra_max_abs,
        "algebra_ok": algebra_ok,
        "single_mode_mse": single_mse,
        "moving_average_mse": ma_mse,
        "median_filter_mse": med_mse,
        "oracle_paired_mse": oracle_mse,
        "relative_reduction_vs_single": relative_vs_single,
        "wall_seconds_estimators": wall_s,
        "samples_per_second": shots / wall_s if wall_s > 0 else float("inf"),
        "weights": sri.weights.tolist(),
        "observable_only": True,
        "notes": "signal/noise used only for scoring and oracle; not passed to SRI/MMSE/single/MA/median",
    }


def decide(cells: list[dict]) -> dict:
    algebra_all = all(c["algebra_ok"] for c in cells)
    sigma_all = all(c["mmse_within_5sigma"] for c in cells)
    observable_all = all(c["observable_only"] for c in cells)
    mean_r = statistics.fmean(c["relative_reduction_vs_single"] for c in cells)

    gates = {
        "observable_only": observable_all,
        "algebra_matches_mmse": algebra_all,
        "sample_mse_within_5sigma": sigma_all,
        "mean_relative_reduction_vs_single": mean_r,
        "pass_threshold": PASS_RELATIVE_REDUCTION,
    }

    if not (observable_all and algebra_all and sigma_all):
        verdict = "fail"
        reason = "Failed observable-only, algebra, and/or 5σ analytic match gates."
    elif mean_r < 0.0:
        verdict = "fail"
        reason = "Mean relative MSE reduction vs single_mode_wiener is negative."
    elif mean_r < PASS_RELATIVE_REDUCTION:
        verdict = "inconclusive"
        reason = (
            f"Mean relative reduction {mean_r:.4f} is in [0, {PASS_RELATIVE_REDUCTION}) "
            "— not a preregistered practical advantage."
        )
    else:
        verdict = "pass"
        reason = (
            f"All gates held; mean relative reduction vs single_mode = {mean_r:.4f} "
            f">= {PASS_RELATIVE_REDUCTION}."
        )

    return {"verdict": verdict, "reason": reason, "gates": gates}


def run(output_dir: Path) -> dict:
    reset_registry()
    output_dir.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()
    cells = [
        run_cell(seed, vn, SHOTS)
        for vn in NOISE_GRID
        for seed in SEEDS
    ]
    wall_total = time.perf_counter() - t_all
    decision = decide(cells)

    # Per-regime aggregates
    by_noise: dict[str, dict] = {}
    for vn in NOISE_GRID:
        subset = [c for c in cells if c["noise_variance"] == vn]
        by_noise[str(vn)] = {
            "mean_sri_mse": statistics.fmean(c["sri_sample_mse"] for c in subset),
            "mean_single_mse": statistics.fmean(c["single_mode_mse"] for c in subset),
            "mean_relative_reduction_vs_single": statistics.fmean(
                c["relative_reduction_vs_single"] for c in subset
            ),
            "n_cells": len(subset),
        }

    # C5 record from this runner (CPU artifacts)
    c5_ok = (
        decision["gates"]["algebra_matches_mmse"]
        and decision["gates"]["sample_mse_within_5sigma"]
        and decision["gates"]["observable_only"]
    )
    c5_status = "confirmed" if c5_ok else "refuted"
    summary_path = str(output_dir / "phase3_summary.json")
    record_result(
        "C5",
        c5_status,
        artifact_paths=[summary_path],
        notes=(
            "Phase 3 CPU runner: Lemma Z1 algebra + 5σ sample MSE + observable-only. "
            f"P3 practical-advantage verdict={decision['verdict']} (separate from C5)."
        ),
    )

    script_path = Path(__file__).resolve()
    summary = {
        "experiment": "phase3_vertical_slice_cv_sensing",
        "spec": "docs/phase3-vertical-slice-spec.md",
        "preregistered": {
            "seeds": SEEDS,
            "shots": SHOTS,
            "noise_grid": NOISE_GRID,
            "signal_variance": SIGNAL_VARIANCE,
            "efficiency": EFFICIENCY,
            "coupling": COUPLING,
            "signal_r": SIGNAL_R,
            "reference_r": REFERENCE_R,
            "pass_relative_reduction_vs_single": PASS_RELATIVE_REDUCTION,
            "algebra_rtol": ALGEBRA_RTOL,
        },
        "decision": decision,
        "by_noise_variance": by_noise,
        "aggregate": {
            "n_cells": len(cells),
            "mean_sri_mse": statistics.fmean(c["sri_sample_mse"] for c in cells),
            "mean_plain_mmse_mse": statistics.fmean(c["plain_mmse_sample_mse"] for c in cells),
            "mean_single_mse": statistics.fmean(c["single_mode_mse"] for c in cells),
            "mean_moving_average_mse": statistics.fmean(c["moving_average_mse"] for c in cells),
            "mean_median_mse": statistics.fmean(c["median_filter_mse"] for c in cells),
            "mean_oracle_mse": statistics.fmean(c["oracle_paired_mse"] for c in cells),
            "mean_relative_reduction_vs_single": decision["gates"][
                "mean_relative_reduction_vs_single"
            ],
            "mean_algebra_max_abs_diff": statistics.fmean(
                c["algebra_max_abs_diff"] for c in cells
            ),
            "wall_seconds_total": wall_total,
            "mean_samples_per_second": statistics.fmean(c["samples_per_second"] for c in cells),
        },
        "cells": cells,
        "conjectures": registry_as_dicts(),
        "limitations": [
            "Classical Monte Carlo of an ideal Gaussian optical sensing model.",
            "Wiener weights use known model parameters (parameter oracle), not blind calibration.",
            "single_mode_wiener sees only y_A; SRI/MMSE also use y_B — dual-mode resource note applies.",
            "observable_yoga_buddhi ≡ estimate_signal by construction; packaging is not an MSE advantage over Wiener.",
            "Oracle paired projection is non-deployable.",
            "Does not establish O(1), quantum advantage, Landauer claims, LLM reliability, or customer demand.",
        ],
        "unsupported_claims": [
            "constant-time/O(1)",
            "quantum advantage or hardware QPU execution",
            "Landauer circumvention",
            "general AI reliability / hallucination prevention",
            "customer value / startup traction",
            "Theorem Zero / A5 composition",
            "beating Wiener/MMSE (algebraically identical)",
        ],
        "provenance": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
            "spec_sha256": hashlib.sha256(
                (Path(__file__).resolve().parents[1] / "docs" / "phase3-vertical-slice-spec.md")
                .read_bytes()
            ).hexdigest()
            if (Path(__file__).resolve().parents[1] / "docs" / "phase3-vertical-slice-spec.md").exists()
            else None,
        },
    }

    (output_dir / "phase3_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "phase3_cells.json").write_text(
        json.dumps(cells, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "conjecture_status.json").write_text(
        json.dumps({"conjectures": registry_as_dicts()}, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/benchmark-results/phase3-vertical-slice"),
    )
    args = parser.parse_args(argv)
    summary = run(args.output_dir)
    print(
        json.dumps(
            {
                "verdict": summary["decision"]["verdict"],
                "reason": summary["decision"]["reason"],
                "aggregate": summary["aggregate"],
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
