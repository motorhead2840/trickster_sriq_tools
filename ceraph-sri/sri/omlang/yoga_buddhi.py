"""Yoga–Buddhi algebra shared by NumPy / JAX / Julia sphota backends.

Source (dissertation Ch. 3):
  Ĉ_S = Π_V − Π_H
  Υ_YB = ½(I + Ĉ_S)
  |Ψ_Samya⟩ = Υ_YB |Ψ_Thesis⟩ = |v⟩

Deployable (Lemma Z1):
  Υ_obs = ½(y_A + Ĉ̂_S) = ŝ_MMSE
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArr = NDArray[np.floating[Any]]


def oracle_samya(value: FloatArr, noise: FloatArr) -> dict[str, FloatArr]:
    """Exact latent Yoga–Buddhi: thesis / antithesis / Samya."""
    v = np.asarray(value, dtype=np.float64)
    xi = np.asarray(noise, dtype=np.float64)
    if v.shape != xi.shape:
        raise ValueError("value and noise must share shape")
    thesis = v + xi
    antithesis = v - xi  # Ĉ_S on latent chart
    samya = 0.5 * (thesis + antithesis)
    return {
        "thesis": thesis,
        "antithesis": antithesis,
        "samya": samya,
        "value": v,
    }


def latent_involution_holds(value: FloatArr, noise: FloatArr, tol: float = 1e-12) -> bool:
    """Ĉ_S² = I on the latent (v, ξ) chart."""
    v = np.asarray(value, dtype=np.float64)
    xi = np.asarray(noise, dtype=np.float64)
    thesis = v + xi
    antithesis = v - xi
    restored = v - (antithesis - v)  # flip horizontal twice
    return bool(np.allclose(restored, thesis, rtol=tol, atol=tol))


def observable_upsilon(
    y_a: FloatArr,
    y_b: FloatArr,
    weights: FloatArr,
) -> dict[str, FloatArr]:
    """Lemma Z1 observable Υ̂_obs from Wiener weights (no s, no n)."""
    ya = np.asarray(y_a, dtype=np.float64)
    yb = np.asarray(y_b, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if ya.shape != yb.shape:
        raise ValueError("y_a and y_b must share shape")
    if w.shape != (2,):
        raise ValueError("weights must be length-2 Wiener coefficients")
    antithesis = (2.0 * w[0] - 1.0) * ya + (2.0 * w[1]) * yb
    projected = 0.5 * (ya + antithesis)
    mmse = w[0] * ya + w[1] * yb
    return {
        "thesis": ya,
        "antithesis": antithesis,
        "projected": projected,
        "mmse": mmse,
    }


def mirror_validation(c: NDArray[np.complexfloating[Any, Any]]) -> FloatArr:
    """Master-WP Mirror channel: M(C) = C · conj(C) → |C|² (existence-density mark)."""
    z = np.asarray(c)
    return np.real(z * np.conjugate(z)).astype(np.float64)
