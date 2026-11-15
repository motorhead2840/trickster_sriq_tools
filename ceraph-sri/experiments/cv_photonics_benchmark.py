"""Gaussian photonic sensing model with a correlated-noise reference mode."""

import argparse
import hashlib
import json
import math
import platform
import random
import statistics
from pathlib import Path


VACUUM_VARIANCE = 0.5


def posterior(signal_variance, noise_variance, efficiency, coupling, signal_r, reference_r=None):
    """Return ideal Gaussian posterior coefficients and MSE for homodyne outcomes."""
    signal_readout_variance = efficiency * (
        signal_variance + noise_variance + VACUUM_VARIANCE * math.exp(-2 * signal_r)
    ) + (1 - efficiency) * VACUUM_VARIANCE
    signal_covariance = math.sqrt(efficiency) * signal_variance
    if reference_r is None:
        coefficient = signal_covariance / signal_readout_variance
        return (coefficient,), signal_variance - coefficient * signal_covariance

    reference_variance = efficiency * (
        coupling ** 2 * noise_variance + VACUUM_VARIANCE * math.exp(-2 * reference_r)
    ) + (1 - efficiency) * VACUUM_VARIANCE
    cross_covariance = efficiency * coupling * noise_variance
    determinant = signal_readout_variance * reference_variance - cross_covariance ** 2
    signal_coefficient = signal_covariance * reference_variance / determinant
    reference_coefficient = -signal_covariance * cross_covariance / determinant
    return (signal_coefficient, reference_coefficient), (
        signal_variance - signal_coefficient * signal_covariance
    )


def run(output_dir, seed=108, shots=20000, signal_variance=1.0, noise_variance=1.0,
        efficiency=0.8, coupling=1.0, squeezing=0.7):
    parameters = (signal_variance, noise_variance, efficiency, coupling, squeezing)
    if (shots < 2 or not all(math.isfinite(x) for x in parameters)
            or signal_variance <= 0 or noise_variance < 0 or not 0 < efficiency <= 1
            or coupling < 0 or not 0 <= squeezing <= 3):
        raise ValueError("Require shots >= 2, finite parameters, positive signal variance, "
                         "nonnegative noise and coupling, 0 < efficiency <= 1, and 0 <= squeezing <= 3")

    conditions = {
        "coherent_signal_only": (0.0, None),
        "coherent_signal_and_reference": (0.0, 0.0),
        "squeezed_signal_only": (squeezing, None),
        "squeezed_signal_and_reference": (squeezing, 0.0),
        "squeezed_reference": (0.0, squeezing),
    }
    predictions = {
        name: posterior(signal_variance, noise_variance, efficiency, coupling, *squeezings)
        for name, squeezings in conditions.items()
    }
    rng = random.Random(seed)
    errors = {name: [] for name in conditions}
    for _ in range(shots):
        signal = rng.gauss(0, math.sqrt(signal_variance))
        noise = rng.gauss(0, math.sqrt(noise_variance))
        signal_vacuum, reference_vacuum = rng.gauss(0, 1), rng.gauss(0, 1)
        signal_loss, reference_loss = rng.gauss(0, 1), rng.gauss(0, 1)
        for name, (signal_r, reference_r) in conditions.items():
            y_signal = math.sqrt(efficiency) * (
                signal + noise + math.sqrt(VACUUM_VARIANCE) * math.exp(-signal_r) * signal_vacuum
            ) + math.sqrt((1 - efficiency) * VACUUM_VARIANCE) * signal_loss
            coefficients, _ = predictions[name]
            estimate = coefficients[0] * y_signal
            if reference_r is not None:
                y_reference = math.sqrt(efficiency) * (
                    coupling * noise + math.sqrt(VACUUM_VARIANCE) * math.exp(-reference_r) * reference_vacuum
                ) + math.sqrt((1 - efficiency) * VACUUM_VARIANCE) * reference_loss
                estimate += coefficients[1] * y_reference
            errors[name].append((estimate - signal) ** 2)

    results = {}
    for name, values in errors.items():
        results[name] = {
            "analytic_mse": predictions[name][1],
            "sample_mse": statistics.fmean(values),
            "sample_mse_standard_error": statistics.stdev(values) / math.sqrt(shots),
            "estimator_coefficients": list(predictions[name][0]),
        }
    comparisons = {}
    for baseline in ("coherent_signal_and_reference", "squeezed_signal_only",
                     "squeezed_signal_and_reference"):
        differences = [a - b for a, b in zip(errors[baseline], errors["squeezed_reference"])]
        mean = statistics.fmean(differences)
        se = statistics.stdev(differences) / math.sqrt(shots)
        comparisons[baseline] = {
            "analytic_mse_reduction": predictions[baseline][1] - predictions["squeezed_reference"][1],
            "paired_sample_mse_reduction": mean,
            "paired_normal_approx_95_interval": [mean - 1.96 * se, mean + 1.96 * se],
        }
    summary = {
        "experiment": "ideal_gaussian_cv_correlated_noise_sensing",
        "configuration": {"seed": seed, "shots": shots, "signal_variance": signal_variance,
                          "noise_variance": noise_variance, "efficiency": efficiency,
                          "reference_noise_coupling": coupling, "squeezing_r": squeezing,
                          "vacuum_quadrature_variance": VACUUM_VARIANCE},
        "results": results,
        "comparisons_vs_squeezed_reference": comparisons,
        "resources": {
            "squeezing_photons_per_squeezed_mode_before_displacement": math.sinh(squeezing) ** 2,
            "signal_displacement_photons_mean": signal_variance / 2,
            "noise_displacement_photons_signal_mean": noise_variance / 2,
            "noise_displacement_photons_reference_mean": coupling ** 2 * noise_variance / 2,
            "mode_counts": {"coherent_signal_only": 1, "coherent_signal_and_reference": 2,
                            "squeezed_signal_only": 1, "squeezed_signal_and_reference": 2,
                            "squeezed_reference": 2},
            "caveat": "Input-mode mean photons only, not wall-plug energy. Local oscillator, squeezing "
                      "generation, source, electronics, loss, calibration and computation costs are not measured.",
        },
        "limitations": [
            "Classical Monte Carlo of an ideal Gaussian quantum-optical measurement model, not quantum hardware.",
            "Known prior and noise parameters give all estimators oracle Gaussian posterior coefficients.",
            "Both quadrature channels have the same total efficiency; loss ports are independent vacuum.",
            "Simulated conditions share random draws for variance reduction; this is not a hardware paired trial.",
            "The signal is a classical random displacement, not an arbitrary unknown quantum state.",
            "This is homodyne measurement and classical estimation, not coherent quantum error correction.",
            "A squeezed-state gain in this restricted model is not a general quantum or computational advantage.",
            "No state preparation, detector, local oscillator, energy or error-correction overhead is timed or measured.",
        ],
        "provenance": {"python": platform.python_version(),
                       "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cv_photonics_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=108)
    parser.add_argument("--shots", type=int, default=20000)
    parser.add_argument("--signal-variance", type=float, default=1.0)
    parser.add_argument("--noise-variance", type=float, default=1.0)
    parser.add_argument("--efficiency", type=float, default=0.8)
    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--squeezing", type=float, default=0.7)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, seed=args.seed, shots=args.shots,
                         signal_variance=args.signal_variance, noise_variance=args.noise_variance,
                         efficiency=args.efficiency, coupling=args.coupling,
                         squeezing=args.squeezing), indent=2))


if __name__ == "__main__":
    main()
