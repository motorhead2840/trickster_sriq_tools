"""Tests for provisional-patent evidence campaign."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.braket_provisional_evidence_campaign import run


class ProvisionalEvidenceTests(unittest.TestCase):
    def test_campaign_passes_and_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = run(
                Path(directory),
                seeds=[108, 109],
                n_samples=1024,
                timing_n=50_000,
            )
            self.assertTrue(summary["verdict"]["pass_core"])
            self.assertTrue(summary["verdict"]["pass_all"])
            self.assertEqual(
                summary["verdict"]["provisional_foundation"],
                "solid_for_supported_claims",
            )
            for name in (
                "provisional_evidence_summary.json",
                "pasteback.json",
                "disclosure_pack.json",
            ):
                path = Path(directory) / name
                self.assertTrue(path.is_file(), name)
                json.loads(path.read_text(encoding="utf-8"))
            pack = json.loads((Path(directory) / "disclosure_pack.json").read_text())
            self.assertIn("Landauer circumvention", " ".join(pack["do_not_claim_yet"]))


if __name__ == "__main__":
    unittest.main()
