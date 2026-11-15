"""Smoke test for grand theory audit campaign."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.braket_grand_theory_audit import run


class GrandTheoryAuditTests(unittest.TestCase):
    def test_audit_pass_and_figures(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = run(
                Path(directory),
                seeds=[108],
                n_samples=512,
                timing_n=20_000,
                scaling_dimensions=[256, 1024],
            )
            self.assertTrue(summary["verdict"]["pass_foundation"])
            self.assertTrue(summary["verdict"]["pass_integrity_softwalls"])
            self.assertEqual(
                summary["verdict"]["softwall_summary"]["deepseek_tflops_as_ours"],
                "REFERENCE_ONLY_NOT_OURS",
            )
            self.assertEqual(
                summary["verdict"]["softwall_summary"]["landauer_circumvention"],
                "NOT_PROVEN",
            )
            for name in ("pasteback.json", "telemetry.json", "grand_theory_audit_summary.json"):
                self.assertTrue((Path(directory) / name).is_file())
            figs = summary["figures"]
            if figs.get("status") == "skipped_no_matplotlib":
                self.assertTrue(Path(figs["figures_skipped"]).is_file())
            else:
                self.assertIn("hist3d_z1_mse_cube", figs)
                self.assertTrue(Path(figs["hist3d_z1_mse_cube"]).is_file())
            # Softwall honesty must never green Landauer / TMEM / DeepSeek-as-ours.
            self.assertNotEqual(
                summary["verdict"]["softwall_summary"]["tmem_msb_xor_O0"],
                "PROVEN",
            )
            self.assertFalse(summary["softwall_claim_board"]["any_softwall_falsely_greened"])


if __name__ == "__main__":
    unittest.main()
