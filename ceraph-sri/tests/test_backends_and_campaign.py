"""Smoke tests for Braket dry-run and CV campaign preflight."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.braket_smoke_plan import run as braket_run
from experiments.cv_photonics_campaign import run as campaign_run
from sri.backends.braket_interface import build_smoke_plan, dry_run_validate
from sri.backends.classical_gaussian import ClassicalGaussianBackend
from sri.cv.estimator import SensingConfig
import numpy as np


class BackendTests(unittest.TestCase):
    def test_classical_backend_mse_finite(self):
        out = ClassicalGaussianBackend().run(
            SensingConfig(reference_r=0.7),
            shots=2000,
            rng=np.random.default_rng(0),
        )
        self.assertTrue(np.isfinite(out["sample_mse"]))
        self.assertLess(abs(out["sample_mse"] - out["analytic_mse"]), 0.1)

    def test_braket_dry_run(self):
        plan = build_smoke_plan()
        result = dry_run_validate(plan)
        self.assertEqual(result["status"], "dry_run_ok")
        with tempfile.TemporaryDirectory() as directory:
            summary = braket_run(Path(directory))
        self.assertEqual(summary["validation"]["status"], "dry_run_ok")


class CampaignTests(unittest.TestCase):
    def test_analytic_grid_size(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = campaign_run(Path(directory), shots=0)
        self.assertEqual(summary["configuration"]["grid_size"], 3 * 3 * 3 * 3 * 4)


if __name__ == "__main__":
    unittest.main()
