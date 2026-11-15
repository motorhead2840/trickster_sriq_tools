"""Fast checks for the sri package.

Published-number regression lives in tests/regression.
These tests cover shapes, the gauge ambiguity, and closed forms.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np

import sri
from sri.cv import k_interval_b_beats_a, mse_optimal, strategies
from sri.graphs import (
    is_connected,
    laplacian,
    make_graph,
    mean_resistance,
    ring_adjacency,
    ring_mean_resistance,
    torus_adjacency,
)
from sri.selection import f_ss_depolarizing, r_threshold
from sri.tracking import corrected_xi_gain, v1_xi_gain
from sri.unveiling import (
    estimate_decentralized,
    generate_observations,
    kappa_errors,
    relative_veil_matrix,
    sample_veils,
)
from sri.wiring_quasi import fibonacci_circulant, mod13_naive


class PackageBoundaryTests(unittest.TestCase):
    def test_public_import_does_not_load_legacy(self):
        self.assertIn("graphs", sri.__all__)
        self.assertNotIn("legacy", sri.__all__)
        source = Path(sri.__file__).read_text()
        tree = ast.parse(source)
        imported = [
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        ]
        self.assertFalse(any("legacy" in name for name in imported))


class GraphTests(unittest.TestCase):
    def test_ring_and_torus_shapes(self):
        ring = ring_adjacency(16)
        torus = torus_adjacency(16)
        self.assertEqual(ring.shape, (16, 16))
        self.assertEqual(torus.shape, (16, 16))
        self.assertTrue(np.allclose(ring, ring.T))
        self.assertTrue(np.allclose(np.diag(ring), 0))
        self.assertTrue(np.allclose(ring.sum(axis=1), 2))
        self.assertTrue(np.allclose(torus.sum(axis=1), 4))
        self.assertTrue(np.allclose(laplacian(ring) @ np.ones(16), 0, atol=1e-12))

    def test_ring_resistance_matches_the_closed_form(self):
        for n in (16, 64, 100):
            exact = (n ** 2 - 1) / (12.0 * n)
            self.assertAlmostEqual(ring_mean_resistance(n), exact)
            self.assertAlmostEqual(mean_resistance(ring_adjacency(n), kind="ring"), exact)

    def test_expander_is_connected_and_regular(self):
        rng = np.random.default_rng(7)
        adjacency = make_graph("expander", 36, rng, degree=4)
        self.assertEqual(adjacency.shape, (36, 36))
        self.assertTrue(is_connected(adjacency))
        self.assertTrue(np.allclose(adjacency.sum(axis=1), 4))


class GaugeTests(unittest.TestCase):
    def test_common_complex_factor_does_not_change_the_error(self):
        rng = np.random.default_rng(0)
        n = 16
        adjacency = make_graph("torus", n, rng)
        tau, _kappa = sample_veils(n, 0.25, 0.4, rng)
        observed, _psi = generate_observations(tau, 32, 20.0, rng)
        ratios = relative_veil_matrix(observed, adjacency)
        estimate = estimate_decentralized(ratios, adjacency)
        base = kappa_errors(estimate, tau)["mse_kappa"]
        scale = 0.6436835360828034 + 1.2646834437306493j
        self.assertAlmostEqual(kappa_errors(estimate, tau * scale)["mse_kappa"], base, places=10)
        self.assertAlmostEqual(kappa_errors(estimate * scale, tau)["mse_kappa"], base, places=10)
        ratios_scaled = (tau * scale)[:, None] / (tau * scale)[None, :]
        ratios_plain = tau[:, None] / tau[None, :]
        mask = adjacency > 0
        self.assertTrue(np.allclose(ratios_scaled[mask], ratios_plain[mask]))


class SelectionThresholdTests(unittest.TestCase):
    def test_r_star_at_xi_crit_0_9(self):
        # r* = (Ξ_crit - 1/d) / (1 - Ξ_crit). The d=3 value is 5.666..., shown as 5.667.
        expected = {
            2: 4.0,
            3: (0.9 - 1.0 / 3.0) / 0.1,
            4: 6.5,
            8: 7.75,
        }
        for dimension, value in expected.items():
            self.assertAlmostEqual(r_threshold(0.9, dimension), value, places=12)
        self.assertAlmostEqual(r_threshold(0.9, 3), 5.667, places=3)

    def test_steady_population_at_r_equals_2(self):
        for dimension in (2, 3, 4, 8):
            self.assertAlmostEqual(
                f_ss_depolarizing(2.0, dimension),
                (2.0 + 1.0 / dimension) / 3.0,
            )


class TrackerSignTests(unittest.TestCase):
    def test_v1_gain_rises_when_consistency_rises(self):
        self.assertGreater(v1_xi_gain(0.9), v1_xi_gain(0.2))

    def test_corrected_gain_rises_when_the_innovation_score_falls(self):
        low = corrected_xi_gain(0.2, g_nom=0.4, g_high=0.9)
        high = corrected_xi_gain(0.9, g_nom=0.4, g_high=0.9)
        self.assertGreater(low, high)


class ReferenceModeTests(unittest.TestCase):
    def test_published_slice_is_the_closed_form(self):
        # V_n / V_s = 16, eta = 0.85, r = 1, k = 0.5.
        interval = k_interval_b_beats_a(16.0, 0.85, 1.0)
        self.assertIsNotNone(interval)
        lo, hi = interval
        self.assertAlmostEqual(lo, 0.0398, places=3)
        self.assertAlmostEqual(hi, 0.9515, places=3)
        allocation = strategies(1.0)
        scores = {
            name: mse_optimal(1.0, 16.0, 0.5, 0.85, r_a, r_b)
            for name, (r_a, r_b) in allocation.items()
        }
        self.assertAlmostEqual(scores["none"], 0.828, places=3)
        self.assertAlmostEqual(scores["squeeze_A"], 0.798, places=3)
        self.assertAlmostEqual(scores["squeeze_B"], 0.700, places=3)
        self.assertAlmostEqual(scores["split"], 0.648, places=3)


class QuasiperiodicWiringTests(unittest.TestCase):
    def test_naive_step_13_disconnects_when_13_divides_n(self):
        adjacency = mod13_naive(52)
        self.assertEqual(adjacency.shape, (52, 52))
        self.assertFalse(is_connected(adjacency))

    def test_fibonacci_circulant_is_a_square_matrix(self):
        adjacency = fibonacci_circulant(36, degree=4)
        self.assertEqual(adjacency.shape, (36, 36))
        self.assertTrue(is_connected(adjacency))
        self.assertTrue(np.allclose(adjacency, adjacency.T))


if __name__ == "__main__":
    unittest.main()
