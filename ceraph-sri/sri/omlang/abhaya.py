"""Abhaya damping and the discrete Ω-dit from the bridge note.

The damping rule is the scalar-tensor map

    A(Θ) = Θ · exp(-λ · Ξ)

implemented exactly, including growth when Ξ is negative. Mirror validation is
the existing elementwise product with the conjugate. Nothing here contacts
GCP, and a Triton launch is attempted only when Triton and CUDA are both
importable.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sri.omlang.yoga_buddhi import mirror_validation

FloatArr = NDArray[np.floating[Any]]

# Density thresholds for the launch schedule shared with the Julia @sphota macro.
_BLOCK_LADDER = (
    (0.5, 128),
    (1.0, 256),
    (2.0, 512),
)
_BLOCK_MAX = 1024


def omega_dit(
    sigma: FloatArr,
    beta: FloatArr,
    upsilon: FloatArr,
    xi: FloatArr,
    d_omega: FloatArr,
) -> float:
    """Discrete stand-in for ∫ σ · β · (υ + ξ) dω."""
    s = np.asarray(sigma, dtype=np.float64)
    b = np.asarray(beta, dtype=np.float64)
    v = np.asarray(upsilon, dtype=np.float64)
    x = np.asarray(xi, dtype=np.float64)
    dw = np.asarray(d_omega, dtype=np.float64)
    return float(np.sum(s * b * (v + x) * dw))


def abhaya_damping(theta: FloatArr, xi: FloatArr, lam: float) -> FloatArr:
    """A(Θ) = Θ · exp(-λ · Ξ). λ is a scalar. Θ and Ξ must broadcast."""
    state = np.asarray(theta, dtype=np.float64)
    instability = np.asarray(xi, dtype=np.float64)
    return state * np.exp(-float(lam) * instability)


def abhaya_jvp(
    theta: FloatArr,
    xi: FloatArr,
    lam: float,
    dtheta: FloatArr,
    dxi: FloatArr,
) -> FloatArr:
    """Directional derivative of A. dA = e^{-λΞ} dΘ - λ Θ e^{-λΞ} dΞ."""
    state = np.asarray(theta, dtype=np.float64)
    instability = np.asarray(xi, dtype=np.float64)
    scale = np.exp(-float(lam) * instability)
    return np.asarray(dtheta, dtype=np.float64) * scale + np.asarray(dxi, dtype=np.float64) * (
        state * scale * (-float(lam))
    )


def sigma_density(sigma: FloatArr) -> float:
    """Mean absolute existence-density. Empty input has density 0."""
    values = np.asarray(sigma, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 0.0
    return float(np.mean(np.abs(values)))


def sphota_schedule(n: int, sigma: FloatArr) -> dict[str, int]:
    """Choose a Triton block and grid from the length and the density of σ.

    The ladder is a scheduling rule shared with ``abhaya_bridge.jl``. It is
    not an occupancy measurement on any GPU.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    density = sigma_density(sigma)
    block = _BLOCK_MAX
    for edge, size in _BLOCK_LADDER:
        if density < edge:
            block = size
            break
    grid = 0 if n == 0 else math.ceil(n / block)
    return {"block": block, "grid": grid}


def abhaya_tiled(theta: FloatArr, xi: FloatArr, lam: float, block: int) -> FloatArr:
    """Apply Abhaya one tile at a time. The tiles match the full map."""
    if block < 1:
        raise ValueError("block must be positive")
    state = np.asarray(theta, dtype=np.float64).reshape(-1)
    instability = np.asarray(xi, dtype=np.float64).reshape(-1)
    if state.shape != instability.shape:
        raise ValueError("theta and xi length mismatch")
    out = np.empty_like(state)
    for start in range(0, state.size, block):
        stop = min(start + block, state.size)
        out[start:stop] = abhaya_damping(state[start:stop], instability[start:stop], lam)
    return out


def mirror_saturation(coefficients: NDArray[np.complexfloating[Any, Any]]) -> FloatArr:
    """M(C) = C · conj(C), returned as a real array."""
    return mirror_validation(coefficients)
