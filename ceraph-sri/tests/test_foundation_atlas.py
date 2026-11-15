"""Smoke test for foundation atlas campaign."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.braket_foundation_atlas import run


class FoundationAtlasTests(unittest.TestCase):
    def test_atlas_pass_and_figures(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = run(Path(directory), seeds=[108], n_samples=512)
            self.assertTrue(summary["verdict"]["pass_foundation"])
            self.assertTrue(summary["verdict"]["pass_softwall_honesty"])
            self.assertEqual(summary["softwall"]["landauer_circumvention"], "NOT_PROVEN")
            self.assertIn("hist3d_sri_advantage_cube", summary["figures"])
            self.assertTrue(Path(summary["figures"]["hist3d_sri_advantage_cube"]).is_file())
            self.assertTrue((Path(directory) / "PATENT_ATTORNEY_FOUNDATION_ATLAS_REPORT.md").is_file())


if __name__ == "__main__":
    unittest.main()
