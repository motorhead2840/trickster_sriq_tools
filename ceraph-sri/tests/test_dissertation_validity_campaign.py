"""Smoke test for Final Dissertation validity campaign."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.braket_dissertation_validity_campaign import run, run_chapter


class DissertationValidityTests(unittest.TestCase):
    def test_validity_pass_and_softwalls(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = run(
                Path(directory),
                seeds=[108],
                n_samples=512,
                vocab=2048,
                batches=3,
            )
            self.assertTrue(summary["verdict"]["pass_dissertation_validity"])
            self.assertTrue(summary["verdict"]["softwall_honest"])
            self.assertTrue(summary["verdict"]["pass_lemma_z1"])
            self.assertTrue(summary["verdict"]["pass_user_query_path"])
            self.assertEqual(
                summary["stages"]["D9_softwalls"]["landauer_circumvention"],
                "NOT_PROVEN",
            )
            self.assertEqual(
                summary["stages"]["D12_chapter6_optical"]["ppln_phase_conjugate"],
                "FUTURE_EMBODIMENT_NOT_PROVEN",
            )
            self.assertTrue((Path(directory) / "telemetry.json").is_file())
            self.assertTrue((Path(directory) / "pasteback.json").is_file())
            self.assertTrue(
                (Path(directory) / "DISSERTATION_VALIDITY_INTEGRATION.md").is_file()
            )
            self.assertGreaterEqual(len(summary["integration_matrix"]), 10)

    def test_each_chapter_1_through_6(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for chapter in range(1, 7):
                result = run_chapter(
                    chapter,
                    root,
                    seeds=[108],
                    n_samples=512,
                    vocab=2048,
                    batches=3,
                )
                self.assertTrue(result["pass_chapter"], msg=f"chapter {chapter}")
                self.assertTrue(Path(result["pasteback_path"]).is_file())
                self.assertTrue(Path(result["report"]).is_file())


if __name__ == "__main__":
    unittest.main()
