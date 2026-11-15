"""CPU checks for the Abhaya map, its derivative, and the Sphota launch ladder."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sri.omlang.abhaya import (
    abhaya_damping,
    abhaya_jvp,
    abhaya_tiled,
    mirror_saturation,
    omega_dit,
    sphota_schedule,
)
from sri.omlang.abhaya_jax import available as jax_available
from sri.omlang.abhaya_triton import cuda_available, reference_matches_formula, triton_available

JULIA = Path("sri/omlang/julia/abhaya_bridge.jl")


def test_omega_dit_is_the_discrete_integral():
    sigma = np.array([1.0, 2.0])
    beta = np.array([0.5, 0.25])
    upsilon = np.array([3.0, 1.0])
    xi = np.array([1.0, -1.0])
    d_omega = np.array([0.1, 0.2])
    expected = 1.0 * 0.5 * (3.0 + 1.0) * 0.1 + 2.0 * 0.25 * (1.0 - 1.0) * 0.2
    assert omega_dit(sigma, beta, upsilon, xi, d_omega) == pytest.approx(expected)


def test_abhaya_matches_the_written_equation_including_growth():
    theta = np.array([2.0, -1.0, 0.5])
    xi = np.array([0.0, 1.0, -1.0])
    lam = 0.5
    got = abhaya_damping(theta, xi, lam)
    expect = theta * np.exp(-lam * xi)
    assert np.allclose(got, expect)
    assert got[2] > theta[2]


def test_jvp_matches_a_finite_difference():
    rng = np.random.default_rng(4)
    theta = rng.normal(size=16)
    xi = rng.normal(size=16)
    dtheta = rng.normal(size=16)
    dxi = rng.normal(size=16)
    lam = 1.25
    step = 1e-6
    base = abhaya_damping(theta, xi, lam)
    shifted = abhaya_damping(theta + step * dtheta, xi + step * dxi, lam)
    numeric = (shifted - base) / step
    analytic = abhaya_jvp(theta, xi, lam, dtheta, dxi)
    assert np.allclose(analytic, numeric, rtol=1e-5, atol=1e-5)


def test_tiles_match_the_full_map_and_the_schedule_follows_density():
    theta = np.linspace(-1.0, 1.0, 1000)
    xi = np.linspace(-0.5, 0.5, 1000)
    assert np.allclose(abhaya_tiled(theta, xi, 0.3, 128), abhaya_damping(theta, xi, 0.3))
    assert sphota_schedule(1000, np.array([0.1]))["block"] == 128
    assert sphota_schedule(1000, np.array([0.75]))["block"] == 256
    assert sphota_schedule(1000, np.array([1.5]))["block"] == 512
    assert sphota_schedule(1000, np.array([3.0]))["block"] == 1024
    assert sphota_schedule(1000, np.array([0.1]))["grid"] == 8
    assert sphota_schedule(0, np.array([0.1]))["grid"] == 0


def test_mirror_is_modulus_squared():
    coefficients = np.array([1.0 + 2.0j, -3.0 + 0.0j])
    got = mirror_saturation(coefficients)
    assert np.allclose(got, np.array([5.0, 9.0]))


def test_julia_source_uses_the_same_ladder_and_the_equation():
    text = JULIA.read_text(encoding="utf-8")
    for token in ("0.5", "1.0", "2.0", "128", "256", "512", "1024", "macro sphota", "exp."):
        assert token in text


def test_cpu_reference_is_the_fallback_and_gpu_paths_are_optional():
    theta = np.array([1.0, 2.0])
    xi = np.array([0.0, 1.0])
    assert np.allclose(reference_matches_formula(theta, xi, 1.0), abhaya_damping(theta, xi, 1.0))
    assert isinstance(triton_available(), bool)
    assert isinstance(cuda_available(), bool)
    assert isinstance(jax_available(), bool)
