"""Continuous-variable reference mode and observable-only estimation.

Mode A carries a signal plus shared noise, s + n. Reference mode B carries
k · n. Both pass a lossy beam splitter of transmission η and are homodyned.
The estimator is the optimal linear (Wiener) map.

Convention: ħ = 2, so each vacuum quadrature has variance 1 (Weedbrook 2012).
"""
from __future__ import annotations

from sri.cv import estimator, gaussian, invariants
from sri.cv.estimator import (
    EstimateResult,
    SensingConfig,
    analytic_mse,
    estimate_signal,
    mmse_coefficients,
    oracle_paired_projection,
    simulate_measurements,
    single_mode_wiener,
)
from sri.cv.gaussian import (
    VACUUM_VAR,
    b_beats_a,
    eta_for_relative_advantage,
    k_interval_b_beats_a,
    monte_carlo_mse,
    mse_grid,
    mse_optimal,
    mse_optimal_array,
    mu_of,
    output_quadrature_var,
    r_from_photons,
    relative_advantage,
    split_r_photon_matched,
    squeeze_variance,
    squeezing_photons,
    strategies,
    total_photons,
    wiener_weights,
)
from sri.cv.invariants import (
    attenuator_channel,
    analytic_mc_agreement,
    beam_splitter_symplectic,
    is_physical_covariance,
    is_symplectic,
    loss_channel_covariance,
    omega_n,
    squeeze_symplectic,
)

__all__ = [
    "VACUUM_VAR",
    "EstimateResult",
    "SensingConfig",
    "analytic_mc_agreement",
    "analytic_mse",
    "attenuator_channel",
    "b_beats_a",
    "beam_splitter_symplectic",
    "estimator",
    "estimate_signal",
    "eta_for_relative_advantage",
    "gaussian",
    "invariants",
    "is_physical_covariance",
    "is_symplectic",
    "k_interval_b_beats_a",
    "loss_channel_covariance",
    "mmse_coefficients",
    "monte_carlo_mse",
    "mse_grid",
    "mse_optimal",
    "mse_optimal_array",
    "mu_of",
    "omega_n",
    "oracle_paired_projection",
    "output_quadrature_var",
    "r_from_photons",
    "relative_advantage",
    "simulate_measurements",
    "single_mode_wiener",
    "split_r_photon_matched",
    "squeeze_symplectic",
    "squeeze_variance",
    "squeezing_photons",
    "strategies",
    "total_photons",
    "wiener_weights",
]
