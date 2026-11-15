"""Tests for Track A Ω-dit / complexity / Landauer suite."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.omega_landauer_suite import (
    run,
    run_landauer_theory_check,
    run_omega_dit_harness,
    _fit_loglog_slope,
)


class OmegaLandauerSuiteTests(unittest.TestCase):
    def test_omega_harness_finite(self):
        result = run_omega_dit_harness(dimension=16, n_samples=4, seed=1)
        self.assertTrue(result["finite"])
        self.assertGreater(result["integral_mean_abs"], 0.0)

    def test_landauer_positive_at_room_temp(self):
        result = run_landauer_theory_check([300.0], hypothetical_bits_erased=[1, 8])
        self.assertEqual(result["circumvention_claim"], "not_validated")
        self.assertGreater(result["rows"][0]["kT_ln2_joules"], 0.0)

    def test_fit_slope_near_one_for_linear(self):
        ns = [10, 20, 40, 80]
        times = [0.01 * n for n in ns]
        fit = _fit_loglog_slope(ns, times)
        self.assertAlmostEqual(fit["slope"], 1.0, places=6)
        self.assertEqual(fit["interpretation"], "consistent_with_O(N)_software")

    def test_suite_fast_path(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = run(
                Path(directory),
                dimensions=[64, 256, 1024],
                temperatures_k=[300.0],
                repeats=3,
                cycles=2,
                seed=108,
            )
            self.assertTrue(summary["pass"])
            self.assertTrue(summary["omega_dit_harness"]["finite"])
            self.assertTrue(
                (Path(directory) / "omega_complexity_scaling.csv").is_file()
            )
            self.assertTrue(
                (Path(directory) / "omega_landauer_summary.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
