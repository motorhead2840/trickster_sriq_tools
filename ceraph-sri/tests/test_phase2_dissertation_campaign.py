"""Smoke test for Phase-2 dissertation campaign."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.braket_phase2_dissertation_campaign import run


class Phase2DissertationTests(unittest.TestCase):
    def test_phase2_telemetry_and_figures(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = run(
                Path(directory),
                seeds=[108],
                n_samples=512,
                vocab=2048,
                batches=3,
            )
            self.assertTrue(summary["verdict"]["pass_phase2_complete"])
            self.assertEqual(summary["softwall"]["landauer_circumvention"], "NOT_PROVEN")
            self.assertGreaterEqual(len(summary["figures"]), 4)
            for path in summary["figures"].values():
                self.assertTrue(Path(path).is_file())
            self.assertTrue((Path(directory) / "telemetry.json").is_file())
            self.assertTrue((Path(directory) / "PHASE2_DISSERTATION_SRI_BRAKET_AFFIRMED.md").is_file())
            self.assertLess(
                summary["verdict"]["inprocess_median_ms"],
                summary["verdict"]["post_filter_model_ms"],
            )


if __name__ == "__main__":
    unittest.main()
