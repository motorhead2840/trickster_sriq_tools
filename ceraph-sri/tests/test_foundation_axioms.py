"""Tests for axiomatic stability, entropy, and V3 branches."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.foundation_axioms_sweep import run
from sri.dynamics.v3_damping import Branch, DampingParameters, v3_damping
from sri.metrics.gaussian_entropy import gaussian_von_neumann_entropy
from sri.metrics.stability import legacy_xi, snr_stability, stability_report
from sri.registry.conjectures import reset_registry


class StabilityTests(unittest.TestCase):
    def test_snr_bounded_when_legacy_is_not(self):
        logits = np.array([-2.0, 0.0, 2.0, 50.0])
        legacy = legacy_xi(logits)
        snr = snr_stability(logits)
        self.assertLess(snr, 1.0 + 1e-12)
        self.assertGreater(snr, 0.0)
        # Legacy can be wildly negative when variance >> mean.
        self.assertTrue(np.isfinite(legacy))

    def test_stability_report_fields(self):
        report = stability_report(np.ones(32))
        self.assertAlmostEqual(report.xi_snr, 1.0)
        self.assertAlmostEqual(report.xi_purity, 1.0)


class EntropyAndDampingTests(unittest.TestCase):
    def test_vacuum_entropy_zero(self):
        self.assertAlmostEqual(gaussian_von_neumann_entropy(np.eye(2)), 0.0, places=6)

    def test_branch_w_less_aggressive_on_tiny_variance(self):
        # Tiny gradient variance: Branch P saturates because of ε in the denominator.
        printed = v3_damping(0.9, 1e-3, DampingParameters(branch=Branch.PRINTED))
        worked = v3_damping(0.9, 1e-3, DampingParameters(branch=Branch.WORKED))
        self.assertGreater(printed.circuit_b, 40.0)
        self.assertLess(worked.circuit_b, printed.circuit_b)


class SweepSmokeTests(unittest.TestCase):
    def test_foundation_sweep_smoke(self):
        reset_registry()
        with tempfile.TemporaryDirectory() as directory:
            summary = run(
                Path(directory),
                seeds=[108],
                dimension=256,
                cycles=5,
                spike_start=2,
                spike_end=3,
            )
        self.assertIn(summary["conjectures"][2]["status"], {"confirmed", "refuted", "open"})
        self.assertGreater(summary["records_count"], 0)
        self.assertLessEqual(summary["aggregate"]["xi_snr_max"], 1.0 + 1e-9)
        self.assertGreater(summary["aggregate"]["xi_snr_min"], 0.0)


if __name__ == "__main__":
    unittest.main()
