"""Continuous-variable reference mode.

Mode A carries a signal plus shared noise, s + n. Reference mode B carries
k · n. Both pass a lossy beam splitter of transmission η and are homodyned.
The estimator is the optimal linear (Wiener) map.

Convention: ħ = 2, so each vacuum quadrature has variance 1 (Weedbrook 2012).
A squeezed vacuum then has Var(x) = exp(−2r) and mean photon number sinh^2(r).

``xp`` is NumPy. The closed forms use ``xp`` so a later optional backend can
rebind it. This module does not import torch.
"""
from __future__ import annotations

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

xp = np

ComplexArr = NDArray[np.complex128]
FloatArr = NDArray[np.float64]
VACUUM_VAR = 1.0

def squeeze_variance(r: float) -> float:
    """Var(x) of an x-squeezed vacuum, ħ=2 convention (vacuum = 1)."""
    return float(xp.exp(-2.0 * r))

def output_quadrature_var(r: float, eta: float) -> float:
    """Var of the measured x after a lossy beam splitter of transmission η.

    σ² = η e^{-2r} + (1-η) * 1.  Unsqueeze (r=0) gives σ² = 1 for every η.
    """
    eta = float(xp.clip(eta, 0.0, 1.0))
    return eta * squeeze_variance(r) + (1.0 - eta) * VACUUM_VAR

def squeezing_photons(r: float) -> float:
    """Mean photon number sinh^2(r) of a squeezed vacuum."""
    return float(xp.sinh(r) ** 2)

def r_from_photons(nbar: float) -> float:
    """Squeezing parameter r whose squeezed vacuum has the given mean photon number."""
    nbar = max(float(nbar), 0.0)
    return float(xp.arcsinh(xp.sqrt(nbar)))

def split_r_photon_matched(r_budget: float) -> float:
    """Equal split of a sinh^2(r) photon budget across two modes."""
    return r_from_photons(0.5 * squeezing_photons(r_budget))

def total_photons(v_s: float, v_n: float, k: float, r_a: float, r_b: float) -> dict[str, float]:
    """Mean photon numbers, ħ=2: n̄ = (⟨x⟩² + ⟨p⟩²)/4 + (Var x + Var p)/4 - 1/2.

    Displacements live in x only, so the classical part is (V_s+V_n)/4 on A
    and (k² V_n)/4 on B; each squeezer adds sinh^2(r).
    """
    n_disp_a = 0.25 * (v_s + v_n)
    n_disp_b = 0.25 * (k ** 2 * v_n)
    n_sq_a = squeezing_photons(r_a)
    n_sq_b = squeezing_photons(r_b)
    return dict(
        n_A=n_disp_a + n_sq_a,
        n_B=n_disp_b + n_sq_b,
        n_disp=n_disp_a + n_disp_b,
        n_squeeze=n_sq_a + n_sq_b,
        n_tot=n_disp_a + n_disp_b + n_sq_a + n_sq_b,
    )

def mse_optimal_array(v_s, v_n, k, eta, r_a, r_b):
    """Closed-form MSE, broadcast over array k and/or eta."""
    k = xp.asarray(k, dtype=xp.float64)
    eta = xp.asarray(eta, dtype=xp.float64)
    sig_a = eta * xp.exp(-2.0 * r_a) + (1.0 - eta) * VACUUM_VAR
    sig_b = eta * xp.exp(-2.0 * r_b) + (1.0 - eta) * VACUUM_VAR
    a = eta * v_s
    b = eta * v_n
    c = eta * (k ** 2) * v_n
    num = sig_a * (c + sig_b) + b * sig_b
    den = (a + sig_a) * (c + sig_b) + b * sig_b
    out = v_s * num / xp.maximum(den, 1e-30)
    return out

def mse_optimal(v_s: float, v_n: float, k: float, eta: float, r_a: float, r_b: float) -> float:
    """Closed-form MSE of the optimal linear estimator of s given (X_A, X_B).

    X_A = √η (s+n) + q_A,  X_B = √η (k n) + q_B,
    Var(q) = η e^{-2r} + (1-η).  s~N(0,V_s), n~N(0,V_n) independent.
    MSE = V_s * [σ_A²(η k² V_n + σ_B²) + η V_n σ_B²]
             / [(η V_s + σ_A²)(η k² V_n + σ_B²) + η V_n σ_B²].
    """
    return float(xp.asarray(mse_optimal_array(v_s, v_n, k, eta, r_a, r_b)).reshape(-1)[0])

def strategies(r_budget: float) -> dict[str, tuple[float, float]]:
    """Named (r_A, r_B) allocations at a matched sinh^2(r) photon budget."""
    r_split = split_r_photon_matched(r_budget)
    return {
        "none": (0.0, 0.0),
        "squeeze_A": (float(r_budget), 0.0),
        "squeeze_B": (0.0, float(r_budget)),
        "split": (r_split, r_split),
    }

def mu_of(r: float, eta: float) -> float:
    """Measured-quadrature variance at squeezing r and transmission eta."""
    return output_quadrature_var(r, eta)

def b_beats_a(v_s: float, v_n: float, k: float, eta: float, r: float, tol: float = 0.0) -> bool:
    """Exact condition: squeeze-B has strictly smaller MSE than squeeze-A.

    With μ = η e^{-2r} + (1-η), the inequality reduces to
        η k² V_n ( η V_n (1-k²) - μ - 1 ) > μ
    and is possible only for k in (0,1) under sufficiently strong noise.
    Independent of V_s (the V_s prefactor cancels in the comparison).
    """
    if r <= 0 or eta <= 0 or v_n <= 0:
        return False
    mu = mu_of(r, eta)
    lhs = eta * (k ** 2) * v_n * (eta * v_n * (1.0 - k ** 2) - mu - 1.0)
    return bool(lhs > mu + tol)

def k_interval_b_beats_a(v_n: float, eta: float, r: float) -> tuple[float, float] | None:
    """Exact roots in k of the B-beats-A boundary, or None if the set is empty.

    Let α = η V_n, μ = η e^{-2r}+(1-η), u=k². The quadratic
        α² u² - α(α-μ-1) u + μ = 0
    has roots u±; B beats A for k² between them, when the discriminant is
    positive and the interval intersects (0,1).
    """
    if r <= 0 or eta <= 0 or v_n <= 0:
        return None
    mu = mu_of(r, eta)
    alpha = eta * v_n
    # D_inner = (α-μ-1)^2 - 4μ
    d_inner = (alpha - mu - 1.0) ** 2 - 4.0 * mu
    if d_inner < 0 or alpha <= 0:
        return None
    sqrt_d = xp.sqrt(d_inner)
    u_minus = ((alpha - mu - 1.0) - sqrt_d) / (2.0 * alpha)
    u_plus = ((alpha - mu - 1.0) + sqrt_d) / (2.0 * alpha)
    lo, hi = sorted((u_minus, u_plus))
    lo = max(lo, 0.0)
    hi = min(hi, 1.0)
    if hi <= lo + 1e-18:
        return None
    return float(xp.sqrt(lo)), float(xp.sqrt(hi))

def relative_advantage(mse_new: float, mse_base: float) -> float:
    """Fractional MSE reduction of one estimator against a baseline."""
    if mse_base <= 0 or not xp.isfinite(mse_base):
        return float("nan")
    return float((mse_base - mse_new) / mse_base)

def eta_for_relative_advantage(
    v_s: float, v_n: float, k: float, r: float, target: float = 0.01,
    strategy: str = "squeeze_B", n_scan: int = 400,
) -> float:
    """Smallest η in (0,1] at which the relative advantage over unsqueezed
    reaches `target` (e.g. 1%). Returns nan if it never does."""
    ras = strategies(r)
    r_a, r_b = ras[strategy]
    etas = xp.linspace(1e-3, 1.0, n_scan)
    hit = xp.nan
    for eta in etas:
        m0 = mse_optimal(v_s, v_n, k, eta, 0.0, 0.0)
        m = mse_optimal(v_s, v_n, k, eta, r_a, r_b)
        if relative_advantage(m, m0) >= target:
            hit = float(eta)
            break
    return hit

def wiener_weights(v_s: float, v_n: float, k: float, eta: float, r_a: float, r_b: float) -> FloatArr:
    """ŝ = w · (X_A, X_B).  w = Cov(s,X) Cov(X,X)^{-1}."""
    sig_a = output_quadrature_var(r_a, eta)
    sig_b = output_quadrature_var(r_b, eta)
    cxx = xp.array(
        [
            [eta * v_s + eta * v_n + sig_a, eta * k * v_n],
            [eta * k * v_n, eta * (k ** 2) * v_n + sig_b],
        ],
        dtype=xp.float64,
    )
    csx = xp.array([xp.sqrt(eta) * v_s, 0.0], dtype=xp.float64)
    try:
        w = xp.linalg.solve(cxx, csx)
    except xp.linalg.LinAlgError:
        w = xp.linalg.lstsq(cxx, csx, rcond=None)[0]
    return w

def monte_carlo_mse(
    v_s: float, v_n: float, k: float, eta: float, r_a: float, r_b: float,
    n_samples: int, rng: Generator,
) -> dict[str, float]:
    """Draw s, n, and quantum noises; apply the theoretical Wiener filter."""
    w = wiener_weights(v_s, v_n, k, eta, r_a, r_b)
    s = rng.normal(0.0, xp.sqrt(v_s), size=n_samples)
    n = rng.normal(0.0, xp.sqrt(v_n), size=n_samples)
    sig_a = output_quadrature_var(r_a, eta)
    sig_b = output_quadrature_var(r_b, eta)
    q_a = rng.normal(0.0, xp.sqrt(sig_a), size=n_samples)
    q_b = rng.normal(0.0, xp.sqrt(sig_b), size=n_samples)
    x_a = xp.sqrt(eta) * (s + n) + q_a
    x_b = xp.sqrt(eta) * (k * n) + q_b
    s_hat = w[0] * x_a + w[1] * x_b
    mse = float(xp.mean((s_hat - s) ** 2))
    theory = mse_optimal(v_s, v_n, k, eta, r_a, r_b)
    return dict(mse_mc=mse, mse_theory=theory, abs_err=abs(mse - theory), rel_err=abs(mse - theory) / max(theory, 1e-18))

def mse_grid(v_s, v_n, k_vals, eta, r_a, r_b) -> FloatArr:
    """Closed-form Wiener MSE on a grid of noise gains."""
    out = xp.empty(len(k_vals), dtype=xp.float64)
    for i, k in enumerate(k_vals):
        out[i] = mse_optimal(v_s, v_n, float(k), eta, r_a, r_b)
    return out
