"""Smoke test for CUDA-Q provisional dissertation campaign (cudaq optional)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.braket_cudaq_provisional_campaign import run


class CudaqProvisionalCampaignTests(unittest.TestCase):
    def test_math_and_proxy_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = run(
                Path(directory),
                seeds=[108],
                n_samples=512,
                cudaq_qubits=4,
            )
            self.assertTrue(summary["verdict"]["pass_math"])
            self.assertTrue(summary["verdict"]["pass_proxy"])
            self.assertFalse(summary["softwall"]["any_softwall_falsely_greened"])
            self.assertEqual(summary["softwall"]["landauer_physical"], "NOT_PROVEN")
            self.assertEqual(
                summary["softwall"]["tmem_msb_xor_o0"],
                "SOFTWARE_PROXY_NOT_TMEM",
            )
            self.assertTrue((Path(directory) / "pasteback.json").is_file())
            # If cudaq is present in the environment, demand CUDA-Q greens.
            if summary["verdict"]["cudaq_present"]:
                self.assertTrue(summary["verdict"]["pass_cudaq"])


if __name__ == "__main__":
    unittest.main()
