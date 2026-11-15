"""Track A — honest Ω-dit / complexity / Landauer validation.

Goals (A1 evidence bar):
1. Exercise Ω-dit as an executable finite-grid object.
2. Measure wall-time vs dimension N for projector / V3 / Ω-integrate.
3. Fit a log-log slope; report whether software behavior is consistent with O(N).
4. Tabulate Landauer k_B T ln 2 as a *theory check only* (not measured silicon heat).

This suite deliberately refuses yellow-paper O(1)/O(0)/Landauer-circumvention claims.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np

from sri.legacy.simulator import (
    DampingParameters,
    OmegaDit,
    SimulationConfig,
    damp_logits,
    integrate_omega,
    landauer_minimum_heat,
    simulate,
    stability_coefficient,
    symplectic_project,
    v3_damping,
)

_DEFAULT_DIMENSIONS = (256, 1024, 4096, 16384, 65536)
_DEFAULT_TEMPERATURES_K = (1.0, 77.0, 300.0, 350.0)


def _time_call(fn, repeats: int = 5) -> dict:
    samples = []
    # Warm-up
    fn()
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    arr = np.asarray(samples, dtype=float)
    return {
        "repeats": repeats,
        "mean_s": float(arr.mean()),
        "median_s": float(np.median(arr)),
        "std_s": float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
        "min_s": float(arr.min()),
        "max_s": float(arr.max()),
    }


def _fit_loglog_slope(ns: list[int], times: list[float]) -> dict:
    """Ordinary least squares on log N vs log t. Slope ~1 ⇒ O(N)-like."""
    x = np.log(np.asarray(ns, dtype=float))
    y = np.log(np.asarray(times, dtype=float))
    if len(ns) < 2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return {"slope": None, "intercept": None, "r2": None}
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "interpretation": (
            "consistent_with_O(N)_software"
            if 0.7 <= slope <= 1.4
            else (
                "sublinear_or_noisy"
                if slope < 0.7
                else "superlinear_or_memory_bound"
            )
        ),
    }


def run_omega_dit_harness(dimension: int = 64, n_samples: int = 8, seed: int = 108) -> dict:
    """Executable Ω-dit finite-grid integral sanity check."""
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_samples):
        value = tuple(rng.normal(0.0, 1.0, size=dimension).tolist())
        xi = tuple(rng.normal(0.0, 0.25, size=dimension).tolist())
        samples.append(OmegaDit(sigma=1.0, beta=float(rng.uniform(0, 2 * math.pi)), value=value, xi=xi))
    d_omega = 1.0 / n_samples
    integrated = integrate_omega(samples, d_omega)
    magnitudes = [abs(z) for z in integrated]
    return {
        "dimension": dimension,
        "n_samples": n_samples,
        "d_omega": d_omega,
        "integral_mean_abs": float(np.mean(magnitudes)),
        "integral_max_abs": float(np.max(magnitudes)),
        "finite": all(math.isfinite(m) for m in magnitudes),
        "note": (
            "Finite equally-spaced Riemann sum of σ e^{iβ}(v+ξ). "
            "Not a continuous physical field or quantum state."
        ),
    }


def run_complexity_scaling(
    dimensions: list[int],
    *,
    cycles: int = 5,
    repeats: int = 5,
    seed: int = 108,
    omega_samples: int = 4,
) -> dict:
    rows = []
    for n in dimensions:
        rng = np.random.default_rng(seed + n)
        clean = tuple(rng.normal(10.0, 0.25, size=n).tolist())
        noise = tuple(rng.normal(0.0, 0.5, size=n).tolist())

        def project():
            return symplectic_project(clean, noise)

        project_stats = _time_call(project, repeats=repeats)
        projection = project()
        xi = stability_coefficient(projection.thesis)
        grad_var = float(np.var(np.asarray(noise)))

        def damp():
            result = v3_damping(xi, grad_var, DampingParameters())
            return damp_logits(projection.thesis, result.chi)

        damp_stats = _time_call(damp, repeats=repeats)

        omega_samples_list = [
            OmegaDit(
                1.0,
                0.1 * i,
                clean,
                tuple(rng.normal(0.0, 0.1, size=n).tolist()),
            )
            for i in range(omega_samples)
        ]

        def omega_work():
            return integrate_omega(omega_samples_list, 1.0 / omega_samples)

        omega_stats = _time_call(omega_work, repeats=max(2, repeats // 2))

        # Full synthetic cycle bundle (fewer repeats — heavier)
        sim_stats = _time_call(
            lambda: simulate(
                SimulationConfig(seed=seed, dimension=n, cycles=cycles, spike_start=1, spike_end=2)
            ),
            repeats=max(2, repeats // 2),
        )

        rows.append(
            {
                "dimension": n,
                "projector_median_s": project_stats["median_s"],
                "damping_median_s": damp_stats["median_s"],
                "omega_integrate_median_s": omega_stats["median_s"],
                "simulate_median_s": sim_stats["median_s"],
                "projector": project_stats,
                "damping": damp_stats,
                "omega_integrate": omega_stats,
                "simulate": sim_stats,
                "projection_mse": float(
                    np.mean((np.asarray(projection.projected) - np.asarray(clean)) ** 2)
                ),
            }
        )

    ns = [row["dimension"] for row in rows]
    fits = {
        "projector": _fit_loglog_slope(ns, [row["projector_median_s"] for row in rows]),
        "damping": _fit_loglog_slope(ns, [row["damping_median_s"] for row in rows]),
        "omega_integrate": _fit_loglog_slope(
            ns, [row["omega_integrate_median_s"] for row in rows]
        ),
        "simulate": _fit_loglog_slope(ns, [row["simulate_median_s"] for row in rows]),
    }
    # Software O(1) would require wall time roughly flat in N.
    software_o1_refuted = any(
        fit["slope"] is not None and fit["slope"] > 0.5 for fit in fits.values()
    )
    return {
        "dimensions": dimensions,
        "rows": rows,
        "loglog_fits": fits,
        "software_o1_claim_refuted": bool(software_o1_refuted),
        "verdict": (
            "Software path scales with N (O(N)-like). "
            "This refutes treating the present CPU/Python implementation as hardware-native O(1)."
            if software_o1_refuted
            else "Timing slopes inconclusive; do not claim O(1)."
        ),
    }


def run_landauer_theory_check(
    temperatures_k: list[float],
    *,
    hypothetical_bits_erased: list[int] | None = None,
) -> dict:
    if hypothetical_bits_erased is None:
        hypothetical_bits_erased = [1, 1024, 16384]
    rows = []
    for t in temperatures_k:
        q_min = landauer_minimum_heat(t)
        bit_rows = [
            {
                "bits": b,
                "ideal_lower_bound_joules": q_min * b,
                "note": "Ideal irreversible bit-erasure lower bound only; not measured device heat.",
            }
            for b in hypothetical_bits_erased
        ]
        rows.append(
            {
                "temperature_kelvin": t,
                "kT_ln2_joules": q_min,
                "hypothetical_erasures": bit_rows,
            }
        )
    return {
        "formula": "Q_min = k_B * T * ln(2) per erased bit",
        "k_B": 1.380649e-23,
        "rows": rows,
        "disclaimer": (
            "Theory check only. Does not measure silicon energy, does not prove "
            "Landauer circumvention, and does not equate float32 variance collapse with thermodynamics."
        ),
        "circumvention_claim": "not_validated",
    }


def run(
    output_dir: Path,
    *,
    dimensions: list[int] | None = None,
    temperatures_k: list[float] | None = None,
    repeats: int = 5,
    cycles: int = 5,
    seed: int = 108,
) -> dict:
    if dimensions is None:
        dimensions = list(_DEFAULT_DIMENSIONS)
    if temperatures_k is None:
        temperatures_k = list(_DEFAULT_TEMPERATURES_K)
    output_dir.mkdir(parents=True, exist_ok=True)

    omega = run_omega_dit_harness(dimension=64, n_samples=8, seed=seed)
    complexity = run_complexity_scaling(
        dimensions, cycles=cycles, repeats=repeats, seed=seed
    )
    landauer = run_landauer_theory_check(temperatures_k)

    summary = {
        "experiment": "omega_landauer_track_a",
        "track": "A1",
        "omega_dit_harness": omega,
        "complexity_scaling": {
            k: complexity[k]
            for k in complexity
            if k != "rows"
        },
        "complexity_rows": complexity["rows"],
        "landauer_theory_check": landauer,
        "pass_criteria": {
            "omega_dit_finite": omega["finite"],
            "software_o1_refuted_or_inconclusive_ok": True,
            "landauer_formula_ok": all(
                row["kT_ln2_joules"] > 0 for row in landauer["rows"] if row["temperature_kelvin"] > 0
            ),
            "no_circumvention_claim": landauer["circumvention_claim"] == "not_validated",
        },
        "honesty_boundary": [
            "Ω-dit here is a finite-dimensional complex sample, not a physical continuous manifold.",
            "N-scaling refutes software-native O(1) for this implementation; it does not implement TMEM XOR / single-cycle FMA hardware.",
            "Landauer table is k_B T ln 2 accounting only — not a joulemeter experiment and not circumvention.",
        ],
        "pass": bool(
            omega["finite"]
            and landauer["circumvention_claim"] == "not_validated"
        ),
        "provenance": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }

    (output_dir / "omega_landauer_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    csv_path = output_dir / "omega_complexity_scaling.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dimension",
                "projector_median_s",
                "damping_median_s",
                "omega_integrate_median_s",
                "simulate_median_s",
                "projection_mse",
            ],
        )
        writer.writeheader()
        for row in complexity["rows"]:
            writer.writerow(
                {
                    "dimension": row["dimension"],
                    "projector_median_s": row["projector_median_s"],
                    "damping_median_s": row["damping_median_s"],
                    "omega_integrate_median_s": row["omega_integrate_median_s"],
                    "simulate_median_s": row["simulate_median_s"],
                    "projection_mse": row["projection_mse"],
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
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--seed", type=int, default=108)
    args = parser.parse_args(argv)
    summary = run(
        args.output_dir,
        dimensions=list(args.dimensions),
        repeats=args.repeats,
        cycles=args.cycles,
        seed=args.seed,
    )
    printable = {k: summary[k] for k in summary if k != "complexity_rows"}
    print(json.dumps(printable, indent=2))
    if not summary["pass"]:
        raise SystemExit("Track A suite failed pass criteria")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
