"""Physical and algebraic invariants for Gaussian CV maps.

Checks used to gate illegal transformations before they enter telemetry:
symplectic form preservation for linear optics, vacuum-injected loss
(complete positivity for Gaussian attenuators), and analytic/MC agreement.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sri.cv.gaussian import VACUUM_VAR

FloatArr = NDArray[np.float64]

# One-mode symplectic form in (x, p) ordering, ħ=2 convention.
OMEGA_1 = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)


def omega_n(n_modes: int) -> FloatArr:
    """Block-diagonal symplectic form for n modes."""
    if n_modes < 1:
        raise ValueError("n_modes must be positive")
    blocks = [OMEGA_1 for _ in range(n_modes)]
    return np.block(
        [
            [blocks[i] if i == j else np.zeros((2, 2)) for j in range(n_modes)]
            for i in range(n_modes)
        ]
    )


def is_symplectic(matrix: FloatArr, atol: float = 1e-8) -> bool:
    """Return True iff S Ω S^T = Ω (linear optics / Gaussian unitary)."""
    s = np.asarray(matrix, dtype=np.float64)
    if s.ndim != 2 or s.shape[0] != s.shape[1] or s.shape[0] % 2:
        return False
    n_modes = s.shape[0] // 2
    omega = omega_n(n_modes)
    return bool(np.allclose(s @ omega @ s.T, omega, atol=atol))


def loss_channel_covariance(
    covariance: FloatArr,
    efficiency: float,
    vacuum_var: float = VACUUM_VAR,
) -> FloatArr:
    """Apply identical pure-loss channels to each mode: V' = η V + (1-η) V_vac.

    For Gaussian states this is a standard CPTP attenuator with vacuum injection.
    """
    if not 0.0 <= efficiency <= 1.0:
        raise ValueError("efficiency must lie in [0, 1]")
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] % 2:
        raise ValueError("covariance must be even-dimensional square")
    vacuum = vacuum_var * np.eye(cov.shape[0], dtype=np.float64)
    return efficiency * cov + (1.0 - efficiency) * vacuum


def attenuator_channel(
    covariance: FloatArr,
    chi: float,
    vacuum_var: float = VACUUM_VAR,
) -> FloatArr:
    """CPTP damping map V ↦ e^{-2χ} V + (1 - e^{-2χ}) V_vac."""
    if chi < 0 or not np.isfinite(chi):
        raise ValueError("chi must be a finite non-negative number")
    scale = float(np.exp(-2.0 * chi))
    return loss_channel_covariance(covariance, scale, vacuum_var=vacuum_var)


def is_physical_covariance(covariance: FloatArr, atol: float = 1e-8) -> bool:
    """Uncertainty principle: V + iΩ ≽ 0 (ħ=2), plus symmetry and PSD."""
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] % 2:
        return False
    if not np.allclose(cov, cov.T, atol=atol):
        return False
    eig = np.linalg.eigvalsh(cov)
    if np.any(eig < -atol):
        return False
    n_modes = cov.shape[0] // 2
    omega = omega_n(n_modes)
    matrix = cov + 1j * omega
    herm_eigs = np.linalg.eigvalsh(matrix)
    return bool(np.all(herm_eigs >= -atol))


def beam_splitter_symplectic(theta: float = np.pi / 4) -> FloatArr:
    """Two-mode beam-splitter in (x1,p1,x2,p2) ordering."""
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array(
        [
            [c, 0.0, s, 0.0],
            [0.0, c, 0.0, s],
            [-s, 0.0, c, 0.0],
            [0.0, -s, 0.0, c],
        ],
        dtype=np.float64,
    )


def squeeze_symplectic(r: float) -> FloatArr:
    """Single-mode x-squeezer: x → e^{-r}x, p → e^{r}p."""
    return np.diag([np.exp(-r), np.exp(r)]).astype(np.float64)


def analytic_mc_agreement(
    analytic: float,
    sample: float,
    standard_error: float,
    n_sigma: float = 5.0,
) -> bool:
    """True if |analytic - sample| is within n_sigma sampling errors."""
    if standard_error < 0 or not np.isfinite(standard_error):
        return False
    tol = n_sigma * max(standard_error, 1e-15)
    return bool(abs(analytic - sample) <= tol)
