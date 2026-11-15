"""V3 dual-circuit damping with printed (P) and worked-example (W) branches.

Branch P: Circuit B uses Var(∇) / (σ_sat² · ε) — matches sri_gpu_demo / legacy.
Branch W: Circuit B compares Var(∇) / σ_sat² directly to θ_crit (no ε in denominator).

Research default after Experiment Set 1 lock: **Branch W** (`Branch.WORKED`).
Branch P is retained for printed-formula fidelity only.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sri.cv.invariants import attenuator_channel, is_physical_covariance

FloatArr = NDArray[np.float64]


class Branch(str, Enum):
    PRINTED = "P"
    WORKED = "W"


@dataclass(frozen=True)
class DampingParameters:
    lambda_0: float = 1.0
    kappa: float = 4.0
    lambda_max: float = 50.0
    alpha: float = 10.0
    theta_crit: float = 5.0
    sigma_sat: float = 1.0
    epsilon: float = 1e-7
    branch: Branch = Branch.WORKED

    def validate(self) -> None:
        for name in (
            "lambda_0",
            "kappa",
            "lambda_max",
            "alpha",
            "theta_crit",
            "sigma_sat",
            "epsilon",
        ):
            value = getattr(self, name)
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if min(self.lambda_0, self.kappa, self.lambda_max, self.alpha, self.theta_crit) < 0:
            raise ValueError("gains and threshold must be non-negative")
        if self.sigma_sat < 0 or self.epsilon <= 0:
            raise ValueError("sigma_sat must be non-negative and epsilon positive")


@dataclass(frozen=True)
class DampingResult:
    circuit_a: float
    circuit_b: float
    branch: str

    @property
    def chi(self) -> float:
        return self.circuit_a + self.circuit_b


def v3_damping(
    xi_stability: float,
    gradient_variance: float,
    parameters: DampingParameters | None = None,
) -> DampingResult:
    parameters = parameters or DampingParameters()
    parameters.validate()
    if gradient_variance < 0 or not np.isfinite(gradient_variance):
        raise ValueError("gradient_variance must be finite and non-negative")
    if not np.isfinite(xi_stability):
        raise ValueError("xi_stability must be finite")

    circuit_a = parameters.lambda_0 * float(
        np.exp(-parameters.kappa * (1.0 - xi_stability))
    )
    if parameters.branch is Branch.PRINTED:
        denominator = parameters.sigma_sat**2 * parameters.epsilon
        normalized = gradient_variance / denominator if denominator else np.inf
    else:
        denominator = parameters.sigma_sat**2 if parameters.sigma_sat > 0 else 0.0
        normalized = gradient_variance / denominator if denominator else np.inf
    gate = parameters.alpha * (normalized - parameters.theta_crit)
    circuit_b = parameters.lambda_max * float(1.0 / (1.0 + np.exp(-np.clip(gate, -60, 60))))
    return DampingResult(
        circuit_a=circuit_a,
        circuit_b=circuit_b,
        branch=parameters.branch.value,
    )


def apply_logit_damping(logits: ArrayLike, chi: float) -> FloatArr:
    if chi < 0 or not np.isfinite(chi):
        raise ValueError("chi must be finite and non-negative")
    scale = float(np.exp(-min(chi, 60.0)))
    return np.asarray(logits, dtype=np.float64) * scale


def cptp_attenuate_covariance(covariance: ArrayLike, chi: float) -> FloatArr:
    """Map χ onto the Gaussian CPTP attenuator; raise if the result is unphysical."""
    result = attenuator_channel(np.asarray(covariance, dtype=np.float64), chi)
    if not is_physical_covariance(result):
        raise ValueError("attenuator produced a non-physical covariance")
    return result
