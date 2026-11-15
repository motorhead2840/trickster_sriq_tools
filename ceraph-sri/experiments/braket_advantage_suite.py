"""S2 — Advantage vs current classical tech under equal-resource notes.

Wraps observable-only MMSE defaults and a CV campaign confirmation subset.
Reports squeezed-signal as the matched quantum baseline when it wins.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from experiments.cv_photonics_campaign import run as run_campaign
from experiments.observable_only_benchmark import run as run_observable


def _pass_criteria(observable: dict, campaign: dict) -> dict:
    agg = observable["aggregate"]
    mmse = float(agg["mean_mmse_sample_mse"])
    ma = float(agg["mean_moving_average_mse"])
    med = float(agg["mean_median_filter_mse"])
    single = float(agg["mean_single_mode_mse"])
    defaults = campaign["aggregate"]["default_point_analytic"]
    sq_sig = float(defaults["squeezed_signal"])
    sq_ref = float(defaults["squeezed_reference"])
    coherent = float(defaults["coherent"])
    fraction = float(campaign["aggregate"]["fraction_sq_ref_beats_coherent"])

    checks = {
        "mmse_beats_moving_average": mmse < ma,
        "mmse_beats_median": mmse < med,
        "mmse_beats_single_mode": mmse < single,
        "mse_reduction_vs_ma": ma - mmse,
        "mse_reduction_vs_median": med - mmse,
        "mse_reduction_vs_single_mode": single - mmse,
        "ordering_sq_sig_best": sq_sig <= sq_ref <= coherent + 1e-12,
        "squeezed_signal_beats_squeezed_ref": sq_sig < sq_ref,
        "fraction_sq_ref_beats_coherent": fraction,
        "fraction_sq_ref_beats_coherent_ge_lock": fraction >= 0.50,
        "lock_margins": {
            "mmse": mmse,
            "moving_average": ma,
            "median": med,
            "single_mode": single,
            "analytic_coherent": coherent,
            "analytic_squeezed_reference": sq_ref,
            "analytic_squeezed_signal": sq_sig,
        },
    }
    checks["defensible_win"] = (
        checks["mmse_beats_moving_average"]
        and checks["mmse_beats_median"]
        and checks["mmse_beats_single_mode"]
        and checks["fraction_sq_ref_beats_coherent_ge_lock"]
    )
    checks["honest_controls"] = {
        "squeezed_signal_stronger_when_equal_photons": checks[
            "squeezed_signal_beats_squeezed_ref"
        ],
        "note": (
            "Squeezed-signal is the stronger matched quantum baseline under equal "
            "squeezing photons; report it even when the proposed squeezed-ref wins "
            "vs coherent / classical filters."
        ),
    }
    return checks


def run(
    output_dir: Path,
    *,
    shots: int = 20000,
    seeds: list[int] | None = None,
    campaign_shots: int = 0,
    confirmation_subset: bool = True,
) -> dict:
    if seeds is None:
        seeds = [108, 109, 110]
    output_dir.mkdir(parents=True, exist_ok=True)
    obs_dir = output_dir / "observable"
    camp_dir = output_dir / "campaign"

    observable = run_observable(obs_dir, seeds=seeds, shots=shots)
    campaign = run_campaign(
        camp_dir,
        seed=seeds[0],
        shots=campaign_shots,
        confirmation_subset=confirmation_subset,
    )
    checks = _pass_criteria(observable, campaign)

    summary = {
        "experiment": "braket_advantage_suite",
        "phase": "S2",
        "equal_resource_note": (
            "MMSE uses two optical modes (y_A, y_B). Moving-average/median baselines "
            "use y_A only. Prefer single_mode_mse for same-mode comparisons; "
            "two-mode MMSE vs coherent two-mode analytic is the physics-aligned contest."
        ),
        "comparators": [
            "moving_average_y_a",
            "median_y_a",
            "single_mode_wiener",
            "coherent_two_mode",
            "squeezed_signal_matched_photons",
        ],
        "observable_aggregate": observable["aggregate"],
        "campaign_aggregate": campaign["aggregate"],
        "pass_criteria": checks,
        "pass": bool(checks["defensible_win"]),
        "paths": {
            "observable": str(obs_dir / "observable_only_summary.json"),
            "campaign": str(camp_dir / "cv_campaign_summary.json"),
        },
        "disclaimer": (
            "Classical Monte Carlo of an ideal Gaussian optical model. "
            "Not a photonic Braket QPU result and not yellow-paper O(1)/Landauer claims."
        ),
        "provenance": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    (output_dir / "advantage_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=20000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[108, 109, 110])
    parser.add_argument(
        "--campaign-shots",
        type=int,
        default=0,
        help="0 = analytic-only campaign grid (fast). >0 adds Monte Carlo.",
    )
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help="Use full CV campaign grid instead of confirmation subset.",
    )
    args = parser.parse_args(argv)
    summary = run(
        args.output_dir,
        shots=args.shots,
        seeds=list(args.seeds),
        campaign_shots=args.campaign_shots,
        confirmation_subset=not args.full_grid,
    )
    print(json.dumps({k: summary[k] for k in summary if k != "observable_aggregate"}, indent=2))
    if not summary["pass"]:
        raise SystemExit(f"advantage suite failed: {summary['pass_criteria']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
