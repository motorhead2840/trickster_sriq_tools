"""Preregistered CV photonic parameter campaign (Week 3).

Grid: Vs,Vn ∈ {0.25,1,4}, k ∈ {0,0.5,1}, η ∈ {0.5,0.8,1}, r ∈ {0,0.3,0.7,1.0}.
Analytic MSE is always computed. Monte Carlo confirmation uses --shots (default 5000
for CPU preflight; use ≥20000 on A100 confirmation subsets).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
from pathlib import Path

import numpy as np

from sri.cv.estimator import SensingConfig, analytic_mse, estimate_signal, simulate_measurements
from sri.cv.gaussian import mse_optimal, squeezing_photons, total_photons


def _evaluate_point(
    vs: float,
    vn: float,
    k: float,
    eta: float,
    r: float,
    shots: int,
    rng: np.random.Generator,
) -> dict:
    configs = {
        "coherent_two_mode": SensingConfig(vs, vn, k, eta, 0.0, 0.0),
        "squeezed_signal": SensingConfig(vs, vn, k, eta, r, 0.0),
        "squeezed_reference": SensingConfig(vs, vn, k, eta, 0.0, r),
    }
    results = {}
    for name, config in configs.items():
        analytic = analytic_mse(config)
        sample_mse = None
        se = None
        if shots > 0:
            signal, _n, y_a, y_b = simulate_measurements(config, shots, rng)
            est = estimate_signal(y_a, y_b, config).estimate
            errors = (est - signal) ** 2
            sample_mse = float(np.mean(errors))
            se = float(np.std(errors, ddof=1) / np.sqrt(shots))
        resources = total_photons(vs, vn, k, config.signal_r, config.reference_r)
        results[name] = {
            "analytic_mse": analytic,
            "sample_mse": sample_mse,
            "sample_mse_standard_error": se,
            "resources": resources,
            "squeezing_photons": squeezing_photons(r) if r > 0 else 0.0,
        }
    return {
        "signal_variance": vs,
        "noise_variance": vn,
        "coupling": k,
        "efficiency": eta,
        "squeezing": r,
        "results": results,
        "squeezed_reference_beats_coherent": (
            results["squeezed_reference"]["analytic_mse"]
            < results["coherent_two_mode"]["analytic_mse"]
        ),
        "squeezed_signal_beats_reference": (
            results["squeezed_signal"]["analytic_mse"]
            <= results["squeezed_reference"]["analytic_mse"] + 1e-15
        ),
    }


def run(
    output_dir: Path,
    seed: int = 108,
    shots: int = 5000,
    confirmation_subset: bool = False,
) -> dict:
    vs_vals = (0.25, 1.0, 4.0)
    vn_vals = (0.25, 1.0, 4.0)
    k_vals = (0.0, 0.5, 1.0)
    eta_vals = (0.5, 0.8, 1.0)
    r_vals = (0.0, 0.3, 0.7, 1.0)
    grid = list(itertools.product(vs_vals, vn_vals, k_vals, eta_vals, r_vals))
    if confirmation_subset:
        # A100 confirmation: defaults neighborhood + a few corners.
        grid = [
            (1.0, 1.0, 1.0, 0.8, 0.7),
            (1.0, 1.0, 1.0, 0.8, 0.0),
            (1.0, 1.0, 0.5, 0.8, 0.7),
            (1.0, 4.0, 1.0, 0.5, 1.0),
            (0.25, 0.25, 1.0, 1.0, 0.3),
            (4.0, 4.0, 0.5, 0.5, 0.7),
        ]
        if shots < 20000:
            shots = 20000

    rng = np.random.default_rng(seed)
    points = [
        _evaluate_point(vs, vn, k, eta, r, shots, rng) for vs, vn, k, eta, r in grid
    ]

    summary = {
        "experiment": "cv_photonics_campaign",
        "configuration": {
            "seed": seed,
            "shots": shots,
            "confirmation_subset": confirmation_subset,
            "grid_size": len(points),
            "vs_vals": list(vs_vals),
            "vn_vals": list(vn_vals),
            "k_vals": list(k_vals),
            "eta_vals": list(eta_vals),
            "r_vals": list(r_vals),
            "vacuum_convention": "hbar=2",
        },
        "points": points,
        "aggregate": {
            "fraction_sq_ref_beats_coherent": float(
                np.mean([p["squeezed_reference_beats_coherent"] for p in points])
            ),
            "default_point_analytic": {
                "coherent": mse_optimal(1.0, 1.0, 1.0, 0.8, 0.0, 0.0),
                "squeezed_reference": mse_optimal(1.0, 1.0, 1.0, 0.8, 0.0, 0.7),
                "squeezed_signal": mse_optimal(1.0, 1.0, 1.0, 0.8, 0.7, 0.0),
            },
        },
        "resource_note": (
            "Comparisons report analytic MSE with per-point photon accounting; "
            "squeezed strategies use squeezing photons that coherent baselines do not."
        ),
        "provenance": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cv_campaign_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=108)
    parser.add_argument("--shots", type=int, default=0)
    parser.add_argument("--confirmation-subset", action="store_true")
    args = parser.parse_args(argv)
    # Default shots=0 means analytic-only full grid (fast CPU preflight).
    shots = args.shots
    if args.confirmation_subset and shots <= 0:
        shots = 20000
    summary = run(
        args.output_dir,
        seed=args.seed,
        shots=shots,
        confirmation_subset=args.confirmation_subset,
    )
    print(json.dumps({k: summary[k] for k in summary if k != "points"}, indent=2))
    print(f"wrote {args.output_dir / 'cv_campaign_summary.json'} points={len(summary['points'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
