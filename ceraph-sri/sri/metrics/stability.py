"""Legacy and axiomatic stability coefficients.

Legacy Equation 11 is reported without clipping. Axiomatic alternatives are
bounded in (0, 1] under the stated definitions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


def legacy_xi(logits: ArrayLike, epsilon: float = 1e-7) -> float:
    """Equation 11 as printed: 1 - Var / (E + ε). Not guaranteed in [0, 1]."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    values = np.asarray(logits, dtype=np.float64)
    if values.size == 0:
        raise ValueError("logits must be non-empty")
    mean = float(np.mean(values))
    variance = float(np.var(values))
    denominator = mean + epsilon
    if abs(denominator) <= 1e-15:
        raise ValueError("mean(logits) + epsilon is too close to zero")
    return 1.0 - variance / denominator


def snr_stability(logits: ArrayLike, epsilon: float = 1e-12) -> float:
    """Ξ_SNR = signal_power / (signal_power + variance) ∈ (0, 1]."""
    values = np.asarray(logits, dtype=np.float64)
    if values.size == 0:
        raise ValueError("logits must be non-empty")
    mean = float(np.mean(values))
    variance = float(np.var(values))
    signal_power = mean * mean
    return float(signal_power / (signal_power + variance + epsilon))


def purity_stability(variance: float, vacuum_var: float = 1.0) -> float:
    """Proxy purity for a thermal-like single quadrature: vacuum_var / (variance).

    Clipped into (0, 1] when variance ≥ vacuum_var. If variance < vacuum_var,
    returns 1.0 (cannot exceed vacuum purity in this proxy).
    """
    if variance < 0 or vacuum_var <= 0:
        raise ValueError("variance must be non-negative and vacuum_var positive")
    if variance <= vacuum_var:
        return 1.0
    return float(vacuum_var / variance)


def fisher_stability(fisher_information: float, epsilon: float = 1e-12) -> float:
    """Ξ_Fisher = 1 / (1 + 1/I) = I/(I+1) ∈ (0, 1) for I > 0."""
    if fisher_information < 0:
        raise ValueError("fisher_information must be non-negative")
    info = float(fisher_information)
    return float(info / (info + 1.0 + epsilon))


@dataclass(frozen=True)
class StabilityReport:
    legacy_xi: float
    xi_snr: float
    xi_purity: float
    xi_fisher: float


def stability_report(
    logits: ArrayLike,
    fisher_information: float | None = None,
    vacuum_var: float = 1.0,
    epsilon: float = 1e-7,
) -> StabilityReport:
    values = np.asarray(logits, dtype=np.float64)
    variance = float(np.var(values))
    # Classical Fisher for a Gaussian mean with known variance: N / variance.
    if fisher_information is None:
        fisher_information = values.size / max(variance, 1e-12)
    return StabilityReport(
        legacy_xi=legacy_xi(values, epsilon=epsilon),
        xi_snr=snr_stability(values),
        xi_purity=purity_stability(variance, vacuum_var=vacuum_var),
        xi_fisher=fisher_stability(fisher_information),
    )
