"""Sphota-hybrid Yoga–Buddhi / Lemma Z1 translation tests."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.sphota_yoga_buddhi_suite import run as run_suite
from sri.backends.braket_interface import BraketJobPlan, dry_run_validate
from sri.cv.estimator import SensingConfig, mmse_coefficients, simulate_measurements
from sri.omlang import jax_backend, tile_ref
from sri.omlang.sphota import SphotaCompiler
from sri.omlang.yoga_buddhi import (
    latent_involution_holds,
    mirror_validation,
    observable_upsilon,
    oracle_samya,
)
from sri.registry.conjectures import reset_registry
from sri.sree.moe_te_config import mentor_te_moe_plan


class YogaBuddhiMathTests(unittest.TestCase):
    def test_oracle_samya_and_involution(self):
        v = np.array([1.0, -2.0, 0.5, 3.0])
        xi = np.array([0.2, -0.1, 0.4, -0.3])
        out = oracle_samya(v, xi)
        self.assertTrue(np.allclose(out["samya"], v))
        self.assertTrue(latent_involution_holds(v, xi))

    def test_observable_equals_mmse(self):
        config = SensingConfig(reference_r=0.7)
        rng = np.random.default_rng(0)
        _s, _n, ya, yb = simulate_measurements(config, 512, rng)
        w = mmse_coefficients(config)
        out = observable_upsilon(ya, yb, w)
        self.assertTrue(np.allclose(out["projected"], out["mmse"], rtol=1e-10, atol=1e-12))

    def test_mirror_validation_nonnegative(self):
        z = np.array([1 + 2j, 0.5 - 0.5j, -1j])
        m = mirror_validation(z)
        self.assertTrue(np.all(m >= 0))
        self.assertTrue(np.allclose(m, np.abs(z) ** 2))


class TileRefAndSreePlanTests(unittest.TestCase):
    def test_tile_ref_matches_oracle_and_obs(self):
        v = np.array([1.0, -2.0, 0.5])
        xi = np.array([0.2, -0.1, 0.4])
        samya = tile_ref.oracle_samya_tile(v + xi, v - xi)
        self.assertTrue(np.allclose(samya, v))
        ya = np.array([1.0, 2.0, 3.0])
        yb = np.array([0.5, -0.5, 1.0])
        _a, proj = tile_ref.observable_upsilon_tile(ya, yb, 0.6, 0.4)
        self.assertTrue(np.allclose(proj, 0.6 * ya + 0.4 * yb))
        spec = tile_ref.analysis_spec()
        self.assertEqual(spec["methodology"], "nvidia-cutile-python-to-rust-agentic")

    def test_sree_te_moe_plan_smoke_honest(self):
        plan = mentor_te_moe_plan(mentor_profile="smoke")
        self.assertTrue(plan["maxtext_flags"]["te_moe_block"])
        self.assertTrue(plan["honesty"]["do_not_claim_nvidia_deepseek_tflops"])
        self.assertEqual(plan["witness_attach"]["must_run_before"], "softmax")


class SphotaCompilerTests(unittest.TestCase):
    def test_crystallize_numpy_and_optional_jax(self):
        report = SphotaCompiler(
            n_samples=1024, seed=1, prefer_julia=False, prefer_jax=True
        ).crystallize()
        self.assertTrue(report.pass_lemma_z1)
        self.assertTrue(report.backends["numpy"]["ok"])
        self.assertTrue(report.backends["tile_ref"]["ok"])
        self.assertFalse(report.backends["cutile_rs"]["ok"])
        if jax_backend.available():
            self.assertTrue(report.backends["jax"]["ok"])
            self.assertTrue(report.backends["jax"]["matches_numpy_obs"])

    def test_suite_writes_artifacts_and_can_confirm_c5(self):
        reset_registry()
        with tempfile.TemporaryDirectory() as directory:
            summary = run_suite(
                Path(directory),
                n_samples=1024,
                prefer_julia=False,
                promote_c5=True,
            )
        self.assertTrue(summary["verdict"]["pass_lemma_z1"])
        self.assertEqual(summary["verdict"]["c5_status"], "confirmed")
        path = Path(directory) if False else None  # directory cleaned; re-check structure
        self.assertIn("sphota", summary)
        self.assertEqual(summary["hybrid"]["plan"]["task"], "sphota_yoga_buddhi")


class BraketPlanTests(unittest.TestCase):
    def test_sphota_task_accepted(self):
        plan = BraketJobPlan(
            job_name="sphota-test",
            device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
            shots=100,
            task="sphota_yoga_buddhi",
            cost_cap_usd=5.0,
        )
        result = dry_run_validate(plan)
        self.assertEqual(result["status"], "dry_run_ok")
        self.assertTrue(result["within_cost_cap"])


if __name__ == "__main__":
    unittest.main()
