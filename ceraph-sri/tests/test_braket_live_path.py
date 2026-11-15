"""Tests for Braket live gates, emulate path, hybrid comparison, and campaign suites."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.braket_advantage_suite import run as run_advantage
from experiments.braket_efficiency_suite import run as run_efficiency
from experiments.braket_hybrid_e1_e4 import (
    compare_locks,
    package_hybrid_job_spec,
    run_local_hybrid,
    try_submit_hybrid,
)
from experiments.braket_live_smoke import run as live_smoke
from sri.backends.braket_interface import (
    DM1_ARN,
    SV1_ARN,
    build_smoke_plan,
    dry_run_validate,
    emulate_quantum_task,
    estimate_cost_usd,
    resolve_device_arn,
    submit_quantum_task,
)
from sri.dynamics.v3_damping import Branch, DampingParameters


class LiveGateTests(unittest.TestCase):
    def test_dry_run_reports_cost_gate(self):
        plan = build_smoke_plan(task="qubit_toy_ising", shots=1000, cost_cap_usd=5.0)
        result = dry_run_validate(plan)
        self.assertEqual(result["status"], "dry_run_ok")
        self.assertTrue(result["within_cost_cap"])
        self.assertLessEqual(result["estimated_cost_usd"], plan.cost_cap_usd)

    def test_resolve_device_aliases(self):
        self.assertEqual(resolve_device_arn(device="sv1"), SV1_ARN)
        self.assertEqual(resolve_device_arn(device="dm1"), DM1_ARN)
        plan = build_smoke_plan(task="qubit_toy_ising", device="dm1", n_qubits=3)
        self.assertEqual(plan.device_arn, DM1_ARN)
        self.assertEqual(plan.parameters["n_qubits"], 3)

    def test_submit_blocked_without_env(self):
        plan = build_smoke_plan(task="qubit_toy_ising", submit=True, s3_bucket="bucket")
        with mock.patch.dict(os.environ, {"BRAKET_SUBMIT": "0"}, clear=False):
            with self.assertRaises(RuntimeError):
                submit_quantum_task(plan)

    def test_emulate_completed(self):
        plan = build_smoke_plan(task="qubit_toy_ising", shots=128)
        result = emulate_quantum_task(plan)
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(result["mode"], "local_emulate")
        self.assertEqual(sum(result["measurement_counts"].values()), 128)


class SmokeAndHybridTests(unittest.TestCase):
    def test_live_smoke_emulate(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = live_smoke(Path(directory), live=False, emulate=True, shots=256)
        self.assertEqual(summary["quantum_task"]["status"], "COMPLETED")
        self.assertIn("sample_mse", summary["classical_cv_shadow"])

    def test_live_smoke_dm1_emulate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = live_smoke(
                root, live=False, emulate=True, shots=64, device="dm1", n_qubits=3
            )
            self.assertEqual(summary["device"], "dm1")
            self.assertEqual(summary["quantum_task"]["status"], "COMPLETED")
            self.assertTrue((root / "dm1_smoke.json").is_file())

    def test_hybrid_local_and_compare(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            from experiments.e1_colab_lock import run as lock_run

            ref = lock_run(root / "ref", shots=1500, seeds=[108], dimension=256)
            (root / "ref" / "e1_colab_lock.json").write_text(
                __import__("json").dumps(ref), encoding="utf-8"
            )
            summary = run_local_hybrid(
                root / "hyb",
                shots=1500,
                seeds=[108],
                dimension=256,
                reference_lock=root / "ref" / "e1_colab_lock.json",
                include_campaign_bundle=False,
            )
            self.assertTrue(summary["comparison_to_reference"]["pass"])
            spec = package_hybrid_job_spec(root / "hyb")
            self.assertEqual(spec["cost_cap_usd"], 25.0)

    def test_try_submit_hybrid_gated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.dict(os.environ, {"BRAKET_SUBMIT": "0"}, clear=False):
                result = try_submit_hybrid(root)
            self.assertEqual(result["status"], "not_submitted")
            self.assertIn("spec", result)

    def test_branch_w_is_research_default(self):
        self.assertEqual(DampingParameters().branch, Branch.WORKED)


class CampaignSuiteTests(unittest.TestCase):
    def test_advantage_suite_fast(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = run_advantage(
                Path(directory),
                shots=1500,
                seeds=[108],
                campaign_shots=0,
                confirmation_subset=True,
            )
            self.assertTrue(summary["pass"])
            self.assertTrue(summary["pass_criteria"]["mmse_beats_moving_average"])
            self.assertIn("equal_resource_note", summary)

    def test_efficiency_suite_fast(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = run_efficiency(
                root,
                dimensions=[256, 1024],
                shots=1500,
                seeds=[108],
            )
            self.assertTrue(summary["pass"])
            self.assertEqual(len(summary["rows"]), 2)
            self.assertTrue((root / "efficiency_scaling.csv").is_file())


if __name__ == "__main__":
    unittest.main()
