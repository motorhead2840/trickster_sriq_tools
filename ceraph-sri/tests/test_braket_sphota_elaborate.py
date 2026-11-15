"""Tests for the elaborate Sphota / Lemma Z1 Braket suite."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.braket_sphota_yoga_buddhi_elaborate import (
    check_colab_oracle_fixture,
    check_lemma_z1_grid,
    run as run_elaborate,
)
from sri.registry.conjectures import reset_registry


class ElaborateSuiteTests(unittest.TestCase):
    def test_colab_oracle_fixture_small(self):
        out = check_colab_oracle_fixture(seeds=[1], dimensions=[128, 512])
        self.assertTrue(out["pass"])

    def test_lemma_z1_grid_smoke(self):
        out = check_lemma_z1_grid(seeds=[108], n_samples=2048)
        self.assertTrue(out["pass"])
        self.assertGreaterEqual(len(out["rows"]), 6)

    def test_full_run_local_artifacts(self):
        reset_registry()
        with tempfile.TemporaryDirectory() as directory:
            summary = run_elaborate(
                Path(directory),
                seeds=[108, 109],
                n_samples=2048,
                oracle_dimensions=[256, 1024],
                prefer_julia=False,
                promote_c5=True,
                reference_lock=Path("docs/benchmark-results/e1_colab_lock.json"),
                run_cutile_stub=True,
            )
            self.assertTrue(summary["verdict"]["pass_core"])
            self.assertTrue(summary["verdict"]["pass_all"])
            self.assertEqual(summary["verdict"]["c5_status"], "confirmed")
            for name in (
                "elaborate_summary.json",
                "pasteback.json",
                "theory_hooks.json",
                "hybrid_job_spec.json",
            ):
                path = Path(directory) / name
                self.assertTrue(path.is_file(), name)
                json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
