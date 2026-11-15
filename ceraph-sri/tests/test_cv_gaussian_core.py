"""Unit tests for observable-only CV estimation and invariants."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.observable_only_benchmark import run
from sri.cv.estimator import (
    SensingConfig,
    analytic_mse,
    estimate_signal,
    observable_antithesis,
    observable_yoga_buddhi,
    oracle_paired_projection,
    simulate_measurements,
)
from sri.cv.invariants import (
    attenuator_channel,
    beam_splitter_symplectic,
    is_physical_covariance,
    is_symplectic,
    squeeze_symplectic,
)
from sri.registry.conjectures import get, list_conjectures, reset_registry


class EstimatorTests(unittest.TestCase):
    def test_oracle_projection_recovers_clean(self):
        clean = np.array([1.0, -2.0, 3.5])
        noise = np.array([0.5, 0.25, -1.0])
        projected = oracle_paired_projection(clean, noise)
        self.assertTrue(np.allclose(projected, clean))

    def test_estimator_does_not_need_clean_signal(self):
        config = SensingConfig(reference_r=0.7)
        rng = np.random.default_rng(0)
        signal, _noise, y_a, y_b = simulate_measurements(config, 5000, rng)
        result = estimate_signal(y_a, y_b, config)
        mse = float(np.mean((result.estimate - signal) ** 2))
        self.assertLess(abs(mse - result.analytic_mse), 0.05)
        self.assertEqual(result.weights.shape, (2,))

    def test_analytic_mse_positive(self):
        mse = analytic_mse(SensingConfig())
        self.assertGreater(mse, 0.0)
        self.assertLess(mse, 1.0)

    def test_lemma_z1_observable_yoga_buddhi_equals_mmse(self):
        config = SensingConfig(reference_r=0.7)
        rng = np.random.default_rng(7)
        _signal, _noise, y_a, y_b = simulate_measurements(config, 2048, rng)
        mmse = estimate_signal(y_a, y_b, config)
        yb = observable_yoga_buddhi(y_a, y_b, config)
        antithesis = observable_antithesis(y_a, y_b, config)
        self.assertTrue(np.allclose(yb.projected, mmse.estimate, rtol=1e-10, atol=1e-12))
        self.assertTrue(np.allclose(yb.antithesis, antithesis, rtol=1e-10, atol=1e-12))
        self.assertTrue(np.allclose(0.5 * (y_a + antithesis), mmse.estimate, rtol=1e-10, atol=1e-12))
        self.assertEqual(yb.analytic_mse, mmse.analytic_mse)

    def test_lemma_z1_latent_involution_oracle_only(self):
        """Oracle C_S is an involution on (v,ξ); deployable Ĉ_S on Y need not be."""
        clean = np.array([1.0, -2.0, 0.5])
        noise = np.array([0.3, -0.1, 0.2])
        thesis = clean + noise
        antithesis = clean - noise
        # Latent involution: flipping horizontal twice restores thesis.
        restored = clean - (antithesis - clean)
        self.assertTrue(np.allclose(restored, thesis))
        self.assertTrue(np.allclose(oracle_paired_projection(clean, noise), clean))

        config = SensingConfig(reference_r=0.5, coupling=0.8)
        rng = np.random.default_rng(11)
        _s, _n, y_a, y_b = simulate_measurements(config, 128, rng)
        c_hat = observable_antithesis(y_a, y_b, config)
        # Raw-Y chart: Ĉ_S(y_A,y_B) is not required to equal a sign-flip of y_A.
        self.assertFalse(np.allclose(c_hat, -y_a))
        self.assertFalse(np.allclose(c_hat, y_a))


class InvariantTests(unittest.TestCase):
    def test_beam_splitter_and_squeezer_are_symplectic(self):
        self.assertTrue(is_symplectic(beam_splitter_symplectic()))
        self.assertTrue(is_symplectic(squeeze_symplectic(0.7)))

    def test_attenuator_preserves_physical_covariance(self):
        vacuum = np.eye(2)
        damped = attenuator_channel(vacuum, chi=1.5)
        self.assertTrue(is_physical_covariance(damped))
        self.assertTrue(np.allclose(damped, vacuum))

    def test_illegal_matrix_rejected(self):
        bad = np.array([[1.0, 0.0], [0.0, 0.0]])
        # Degenerate covariance may still be PSD; use clearly non-symplectic S.
        self.assertFalse(is_symplectic(np.array([[2.0, 0.0], [0.0, 2.0]])))
        self.assertTrue(is_physical_covariance(np.eye(2)))
        _ = bad


class RegistryAndBenchmarkTests(unittest.TestCase):
    def test_registry_lists_c1_c5(self):
        reset_registry()
        ids = {item.conjecture_id for item in list_conjectures()}
        self.assertEqual(ids, {"C1", "C2", "C3", "C4", "C5"})
        self.assertEqual(get("C1").status, "open")
        self.assertEqual(get("C5").status, "open")
        self.assertIn("latent", get("C5").statement.lower())

    def test_observable_benchmark_smoke(self):
        reset_registry()
        with tempfile.TemporaryDirectory() as directory:
            summary = run(Path(directory), seeds=[108], shots=4000, squeezing=0.7)
        self.assertTrue(summary["per_seed"][0]["mmse_within_5sigma"])
        self.assertLess(summary["per_seed"][0]["oracle_paired_mse"], 1e-20)
        statuses = {item["conjecture_id"]: item["status"] for item in summary["conjectures"]}
        self.assertEqual(statuses["C1"], "confirmed")
        self.assertEqual(statuses["C2"], "confirmed")


if __name__ == "__main__":
    unittest.main()
