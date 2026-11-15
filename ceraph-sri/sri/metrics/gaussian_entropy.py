"""Gaussian von Neumann entropy from symplectic eigenvalues (ħ=2)."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sri.cv.invariants import omega_n

FloatArr = NDArray[np.float64]


def symplectic_eigenvalues(covariance: FloatArr) -> FloatArr:
    """Return symplectic eigenvalues ν_i ≥ 1 in the ħ=2 convention.

    Computed as absolute eigenvalues of i Ω^{-1} V, paired and sorted.
    The vacuum covariance is the identity and has ν_i = 1.
    """
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] % 2:
        raise ValueError("covariance must be even-dimensional square")
    n_modes = cov.shape[0] // 2
    omega = omega_n(n_modes)
    # Speeding via |eig(i Ω^{-1} V)|; Ω^{-1} = -Ω for this form.
    matrix = 1j * (-omega) @ cov
    eigs = np.linalg.eigvals(matrix)
    abs_eigs = np.sort(np.abs(eigs.real))  # numerical cleanup
    # Pair duplicates: take every other after sorting.
    nu = abs_eigs[1::2]
    if nu.size != n_modes:
        # Fallback: unique-style pairing by sorting descending abs and stride.
        abs_sorted = np.sort(np.abs(eigs))[::-1]
        nu = abs_sorted[::2][:n_modes]
        nu = np.sort(nu)
    return np.asarray(nu, dtype=np.float64)


def _g(nu: float) -> float:
    """Entropy contribution of one symplectic eigenvalue (ħ=2, nats)."""
    if nu <= 1.0 + 1e-12:
        return 0.0
    return float(
        (nu + 1.0) * np.log(nu + 1.0) - (nu - 1.0) * np.log(nu - 1.0)
    ) / 2.0


def gaussian_von_neumann_entropy(covariance: FloatArr) -> float:
    """S(ρ) = Σ g(ν_i) for Gaussian states (nats, ħ=2)."""
    return float(sum(_g(float(nu)) for nu in symplectic_eigenvalues(covariance)))


def purity_from_covariance(covariance: FloatArr) -> float:
    """P = Tr(ρ²) = 1 / Π ν_i for Gaussian states (ħ=2)."""
    nu = symplectic_eigenvalues(covariance)
    return float(1.0 / np.prod(nu))
