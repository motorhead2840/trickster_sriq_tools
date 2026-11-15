"""Local classical simulator backend for the CV sensing model."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from sri.cv.estimator import SensingConfig, analytic_mse, estimate_signal, simulate_measurements


@dataclass
class ClassicalGaussianBackend:
    name: str = "classical_gaussian"

    def run(
        self,
        config: SensingConfig,
        shots: int,
        rng: Generator,
    ) -> dict[str, float | list[float]]:
        signal, _noise, y_a, y_b = simulate_measurements(config, shots, rng)
        result = estimate_signal(y_a, y_b, config)
        mse = float(np.mean((result.estimate - signal) ** 2))
        return {
            "backend": self.name,
            "sample_mse": mse,
            "analytic_mse": analytic_mse(config),
            "weights": result.weights.tolist(),
            "shots": float(shots),
        }
