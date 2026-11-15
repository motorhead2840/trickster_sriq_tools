"""Observable-only estimators for the correlated-noise CV sensing model.

The legacy Yoga–Buddhi projector needs the exact noise vector. This module
estimates the classical signal displacement from homodyne outcomes alone,
using the Gaussian MMSE (Wiener) filter whose closed forms live in
``sri.cv.gaussian``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from sri.cv.gaussian import (
    VACUUM_VAR,
    mse_optimal,
    output_quadrature_var,
    wiener_weights,
)

FloatArr = NDArray[np.float64]


@dataclass(frozen=True)
class SensingConfig:
    """Classical signal + shared noise on two lossy optical modes."""

    signal_variance: float = 1.0
    noise_variance: float = 1.0
    coupling: float = 1.0
    efficiency: float = 0.8
    signal_r: float = 0.0
    reference_r: float = 0.0

    def validate(self) -> None:
        if self.signal_variance <= 0 or self.noise_variance < 0:
            raise ValueError("signal_variance must be positive; noise_variance >= 0")
        if not 0.0 < self.efficiency <= 1.0:
            raise ValueError("efficiency must lie in (0, 1]")
        if self.coupling < 0:
            raise ValueError("coupling must be non-negative")
        if min(self.signal_r, self.reference_r) < 0:
            raise ValueError("squeezing parameters must be non-negative")


@dataclass(frozen=True)
class EstimateResult:
    estimate: FloatArr
    weights: FloatArr
    analytic_mse: float


@dataclass(frozen=True)
class ObservableYogaBuddhiResult:
    r"""Lemma Z1: observable approximation of dissertation Yoga–Buddhi.

    Source (latent / oracle):
        \(\hat{\mathcal{C}}_S=\hat{\Pi}_V-\hat{\Pi}_H\),
        \(\hat{\Upsilon}_{YB}=\tfrac12(\hat{\mathbb{I}}+\hat{\mathcal{C}}_S)\),
        Samya \(|\Psi_{\mathrm{Samya}}\rangle=|v\rangle\).

    Deployable: \(\hat{\Upsilon}_{\mathrm{obs}}=\tfrac12(y_A+\widehat{\mathcal{C}}_S)=\hat s\).
    Involution \(\hat{\mathcal{C}}_S^{2}=I\) is required only on the latent
    \((v,\xi)\) chart — not on raw \(Y=(y_A,y_B)\).
    See ``docs/yoga-buddhi-projector.md``.
    """

    thesis: FloatArr
    antithesis: FloatArr
    projected: FloatArr
    weights: FloatArr
    analytic_mse: float


def mmse_coefficients(config: SensingConfig) -> FloatArr:
    """Return Wiener weights for (y_A, y_B); observable-only at inference."""
    config.validate()
    return wiener_weights(
        config.signal_variance,
        config.noise_variance,
        config.coupling,
        config.efficiency,
        config.signal_r,
        config.reference_r,
    )


def analytic_mse(config: SensingConfig) -> float:
    config.validate()
    return mse_optimal(
        config.signal_variance,
        config.noise_variance,
        config.coupling,
        config.efficiency,
        config.signal_r,
        config.reference_r,
    )


def estimate_signal(y_a: FloatArr, y_b: FloatArr, config: SensingConfig) -> EstimateResult:
    """Estimate s from measured quadratures only (no clean signal, no oracle noise)."""
    weights = mmse_coefficients(config)
    y_a = np.asarray(y_a, dtype=np.float64)
    y_b = np.asarray(y_b, dtype=np.float64)
    if y_a.shape != y_b.shape:
        raise ValueError("y_a and y_b must have the same shape")
    estimate = weights[0] * y_a + weights[1] * y_b
    return EstimateResult(estimate=estimate, weights=weights, analytic_mse=analytic_mse(config))


def observable_antithesis(y_a: FloatArr, y_b: FloatArr, config: SensingConfig) -> FloatArr:
    r"""Observable \(\widehat C_S\) for Lemma Z1: \((2w_A-1)y_A + 2w_B y_B\)."""
    weights = mmse_coefficients(config)
    y_a = np.asarray(y_a, dtype=np.float64)
    y_b = np.asarray(y_b, dtype=np.float64)
    if y_a.shape != y_b.shape:
        raise ValueError("y_a and y_b must have the same shape")
    return (2.0 * weights[0] - 1.0) * y_a + (2.0 * weights[1]) * y_b


def observable_yoga_buddhi(
    y_a: FloatArr, y_b: FloatArr, config: SensingConfig
) -> ObservableYogaBuddhiResult:
    r"""Deployable \(\hat{\Upsilon}_{\mathrm{obs}}\) approximating dissertation \(\hat{\Upsilon}_{YB}\).

    \(\hat{\Upsilon}_{\mathrm{obs}}=\tfrac12(y_A+\widehat{\mathcal{C}}_S)=\hat s_{\mathrm{MMSE}}\).
    Does not receive clean \(s\) or oracle noise \(n\). Equivalent to
    :func:`estimate_signal` by construction of \(\widehat{\mathcal{C}}_S\).
    """
    weights = mmse_coefficients(config)
    y_a = np.asarray(y_a, dtype=np.float64)
    y_b = np.asarray(y_b, dtype=np.float64)
    if y_a.shape != y_b.shape:
        raise ValueError("y_a and y_b must have the same shape")
    antithesis = (2.0 * weights[0] - 1.0) * y_a + (2.0 * weights[1]) * y_b
    projected = 0.5 * (y_a + antithesis)
    return ObservableYogaBuddhiResult(
        thesis=y_a,
        antithesis=antithesis,
        projected=projected,
        weights=weights,
        analytic_mse=analytic_mse(config),
    )


def simulate_measurements(
    config: SensingConfig,
    n_samples: int,
    rng: Generator,
) -> tuple[FloatArr, FloatArr, FloatArr, FloatArr]:
    """Draw (s, n, y_A, y_B). Callers must not feed s or n into the estimator."""
    config.validate()
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    s = rng.normal(0.0, np.sqrt(config.signal_variance), size=n_samples)
    n = rng.normal(0.0, np.sqrt(config.noise_variance), size=n_samples)
    sig_a = output_quadrature_var(config.signal_r, config.efficiency)
    sig_b = output_quadrature_var(config.reference_r, config.efficiency)
    q_a = rng.normal(0.0, np.sqrt(sig_a), size=n_samples)
    q_b = rng.normal(0.0, np.sqrt(sig_b), size=n_samples)
    y_a = np.sqrt(config.efficiency) * (s + n) + q_a
    y_b = np.sqrt(config.efficiency) * (config.coupling * n) + q_b
    return s, n, y_a, y_b


def single_mode_wiener(y_a: FloatArr, config: SensingConfig) -> FloatArr:
    """MMSE using mode A only (no reference). Observable-only."""
    config.validate()
    sig_a = output_quadrature_var(config.signal_r, config.efficiency)
    var_ya = (
        config.efficiency * (config.signal_variance + config.noise_variance) + sig_a
    )
    cov = np.sqrt(config.efficiency) * config.signal_variance
    weight = cov / max(var_ya, 1e-30)
    return weight * np.asarray(y_a, dtype=np.float64)


def oracle_paired_projection(clean: FloatArr, noise: FloatArr) -> FloatArr:
    """Legacy identity ½[(v+ξ)+(v-ξ)] = v. Requires oracle noise; upper bound only."""
    clean = np.asarray(clean, dtype=np.float64)
    noise = np.asarray(noise, dtype=np.float64)
    thesis = clean + noise
    antithesis = clean - noise
    return 0.5 * (thesis + antithesis)


def vacuum_variance() -> float:
    """Expose the module vacuum convention (ħ=2 ⇒ Var=1)."""
    return float(VACUUM_VAR)
