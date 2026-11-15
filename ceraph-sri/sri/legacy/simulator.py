"""CPU reference model for the equations in the SRI provisional dissertation."""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pvariance
from typing import Iterable, Sequence


Vector = tuple[float, ...]
BOLTZMANN_CONSTANT = 1.380649e-23


def _finite_number(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must contain only finite numbers")
    return number


def _vector(values: Sequence[float], name: str) -> Vector:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of numbers")
    result = tuple(_finite_number(value, name) for value in values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _matching_vectors(*vectors: Vector) -> None:
    if any(len(vector) != len(vectors[0]) for vector in vectors[1:]):
        raise ValueError("all vectors must have the same length")


@dataclass(frozen=True)
class OmegaDit:
    """Finite-dimensional sample of sigma * exp(i beta) * (v + xi)."""

    sigma: float
    beta: float
    value: Vector
    xi: Vector

    def __post_init__(self) -> None:
        sigma = _finite_number(self.sigma, "sigma")
        beta = _finite_number(self.beta, "beta")
        if sigma < 0:
            raise ValueError("sigma must be non-negative")
        value = _vector(self.value, "value")
        xi = _vector(self.xi, "xi")
        _matching_vectors(value, xi)
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "xi", xi)

    def integrand(self) -> tuple[complex, ...]:
        phase = self.sigma * complex(math.cos(self.beta), math.sin(self.beta))
        return tuple(phase * (value + noise) for value, noise in zip(self.value, self.xi))


def integrate_omega(samples: Iterable[OmegaDit], d_omega: float) -> tuple[complex, ...]:
    """Approximate the dissertation's integral over a finite, equally spaced grid."""
    spacing = _finite_number(d_omega, "d_omega")
    samples = tuple(samples)
    if not samples:
        raise ValueError("at least one Omega-dit sample is required")
    width = len(samples[0].value)
    if any(len(sample.value) != width for sample in samples):
        raise ValueError("all Omega-dit samples must have the same vector width")
    return tuple(
        spacing * sum(sample.integrand()[index] for sample in samples)
        for index in range(width)
    )


@dataclass(frozen=True)
class Projection:
    thesis: Vector
    antithesis: Vector
    projected: Vector


def symplectic_project(value: Sequence[float], noise: Sequence[float]) -> Projection:
    """Apply the exact paired-state identity; `noise` must be the actual additive noise."""
    clean = _vector(value, "value")
    perturbation = _vector(noise, "noise")
    _matching_vectors(clean, perturbation)
    thesis = tuple(v + xi for v, xi in zip(clean, perturbation))
    antithesis = tuple(v - xi for v, xi in zip(clean, perturbation))
    projected = tuple((a + b) * 0.5 for a, b in zip(thesis, antithesis))
    return Projection(thesis, antithesis, projected)


def mean_variance(values: Sequence[float]) -> tuple[float, float]:
    vector = _vector(values, "values")
    return fmean(vector), pvariance(vector)


def stability_coefficient(logits: Sequence[float], epsilon: float = 1e-7) -> float:
    """Evaluate Equation 11 as printed, without clipping its result."""
    epsilon = _finite_number(epsilon, "epsilon")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    mean, variance = mean_variance(logits)
    denominator = mean + epsilon
    if abs(denominator) <= 1e-15:
        raise ValueError("mean(logits) + epsilon is too close to zero")
    return 1.0 - variance / denominator


@dataclass(frozen=True)
class DampingParameters:
    lambda_0: float = 1.0
    kappa: float = 4.0
    lambda_max: float = 50.0
    alpha: float = 10.0
    theta_crit: float = 5.0
    sigma_sat: float = 1.0
    epsilon: float = 1e-7

    def __post_init__(self) -> None:
        for name in ("lambda_0", "kappa", "lambda_max", "alpha", "theta_crit", "sigma_sat", "epsilon"):
            object.__setattr__(self, name, _finite_number(getattr(self, name), name))
        if min(self.lambda_0, self.kappa, self.lambda_max, self.alpha, self.theta_crit) < 0:
            raise ValueError("damping gains and threshold must be non-negative")
        if self.sigma_sat < 0 or self.epsilon <= 0:
            raise ValueError("sigma_sat must be non-negative and epsilon must be positive")


@dataclass(frozen=True)
class DampingResult:
    circuit_a: float
    circuit_b: float

    @property
    def chi(self) -> float:
        return self.circuit_a + self.circuit_b


def v3_damping(
    xi_stability: float,
    gradient_variance: float,
    parameters: DampingParameters = DampingParameters(),
) -> DampingResult:
    """Evaluate the two-circuit law, following the printed epsilon-scaled denominator."""
    stability = _finite_number(xi_stability, "xi_stability")
    variance = _finite_number(gradient_variance, "gradient_variance")
    if variance < 0:
        raise ValueError("gradient_variance must be non-negative")
    circuit_a = parameters.lambda_0 * math.exp(-parameters.kappa * (1.0 - stability))
    denominator = parameters.sigma_sat**2 * parameters.epsilon
    normalized_variance = variance / denominator if denominator else math.inf
    gate_input = parameters.alpha * (normalized_variance - parameters.theta_crit)
    if gate_input >= 0:
        sigmoid = 1.0 / (1.0 + math.exp(-min(gate_input, 745.0)))
    else:
        exp_value = math.exp(max(gate_input, -745.0))
        sigmoid = exp_value / (1.0 + exp_value)
    circuit_b = parameters.lambda_max * sigmoid
    return DampingResult(circuit_a, circuit_b)


def damp_logits(logits: Sequence[float], chi: float) -> Vector:
    force = _finite_number(chi, "chi")
    if force < 0:
        raise ValueError("chi must be non-negative")
    scale = math.exp(-min(force, 745.0))
    return tuple(value * scale for value in _vector(logits, "logits"))


def landauer_minimum_heat(temperature_kelvin: float) -> float:
    """Return k_B*T*ln(2), the ideal lower bound for erasing one bit."""
    temperature = _finite_number(temperature_kelvin, "temperature_kelvin")
    if temperature < 0:
        raise ValueError("temperature_kelvin must be non-negative")
    return BOLTZMANN_CONSTANT * temperature * math.log(2.0)


@dataclass(frozen=True)
class SimulationConfig:
    seed: int = 108
    dimension: int = 1024
    cycles: int = 25
    baseline_noise_std: float = 0.5
    spike_noise_std: float = 15.0
    spike_start: int = 10
    spike_end: int = 20
    damping: DampingParameters = DampingParameters()

    def __post_init__(self) -> None:
        if self.dimension <= 0 or self.cycles <= 0:
            raise ValueError("dimension and cycles must be positive")
        if self.spike_start < 0 or self.spike_end < self.spike_start:
            raise ValueError("noise spike interval must satisfy 0 <= start <= end")
        for name in ("baseline_noise_std", "spike_noise_std"):
            value = _finite_number(getattr(self, name), name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative")


def simulate(config: SimulationConfig = SimulationConfig()) -> list[dict[str, float | int]]:
    """Run reproducible synthetic signal/noise cycles and return host-side telemetry."""
    rng = random.Random(config.seed)
    clean = tuple(10.0 + rng.gauss(0.0, 0.25) for _ in range(config.dimension))
    records: list[dict[str, float | int]] = []

    for cycle in range(config.cycles):
        noise_std = (
            config.spike_noise_std
            if config.spike_start <= cycle <= config.spike_end
            else config.baseline_noise_std
        )
        noise = tuple(rng.gauss(0.0, noise_std) for _ in range(config.dimension))
        start = time.perf_counter()
        projection = symplectic_project(clean, noise)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        thesis_mean, thesis_variance = mean_variance(projection.thesis)
        gradient_variance = pvariance(noise)
        xi = stability_coefficient(projection.thesis)
        damping = v3_damping(xi, gradient_variance, config.damping)
        damped = damp_logits(projection.thesis, damping.chi)
        _, damped_variance = mean_variance(damped)
        mse = fmean((actual - expected) ** 2 for actual, expected in zip(projection.projected, clean))
        records.append(
            {
                "cycle": cycle,
                "noise_std": noise_std,
                "thesis_mean": thesis_mean,
                "thesis_variance": thesis_variance,
                "xi_stability": xi,
                "gradient_variance": gradient_variance,
                "circuit_a": damping.circuit_a,
                "circuit_b": damping.circuit_b,
                "chi_v3": damping.chi,
                "damped_variance": damped_variance,
                "projection_mse": mse,
                "projection_elapsed_ms": elapsed_ms,
            }
        )
    return records


def write_telemetry(records: Sequence[dict[str, float | int]], path: str | Path) -> None:
    if not records:
        raise ValueError("cannot write empty telemetry")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the CPU reference SRI simulation.")
    parser.add_argument("--seed", type=int, default=108)
    parser.add_argument("--dimension", type=int, default=1024)
    parser.add_argument("--cycles", type=int, default=25)
    parser.add_argument("--output", type=Path, default=Path("sri_demo_telemetry.csv"))
    args = parser.parse_args(argv)
    config = SimulationConfig(seed=args.seed, dimension=args.dimension, cycles=args.cycles)
    records = simulate(config)
    write_telemetry(records, args.output)

    spike_records = [row for row in records if row["noise_std"] == config.spike_noise_std]
    max_mse = max(float(row["projection_mse"]) for row in records)
    mean_chi = fmean(float(row["chi_v3"]) for row in records)
    print("SRI CPU reference simulation (synthetic known signal + supplied exact noise)")
    print(f"seed={config.seed} dimension={config.dimension} cycles={config.cycles}")
    print(f"noise-spike cycles={len(spike_records)}; max known-noise projection MSE={max_mse:.3e}")
    print(f"mean V3 damping force={mean_chi:.6g}; telemetry={args.output}")
    print("Projection MSE assumes the exact injected noise is available; it is not blind denoising.")
    print("Timing is host-side Python timing, not a GPU or hardware benchmark.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
