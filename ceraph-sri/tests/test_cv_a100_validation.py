import importlib.util
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

if importlib.util.find_spec("numpy"):
    import numpy as np
    from experiments.cv_a100_validation import calculate, coefficients, configuration, run


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is required")
class CrossBackendModelTests(unittest.TestCase):
    def test_estimates_match_analytic_mse(self):
        vs, vn, eta, coupling, squeezing = 1.0, 1.0, 0.8, 1.0, 0.7
        z = np.random.default_rng(108).standard_normal((6, 50000), dtype=np.float32)
        entries = coefficients(vs, vn, eta, coupling, squeezing)
        estimates = calculate(np, z, entries, vs, vn, eta, coupling)
        errors = (estimates.astype(np.float64) - z[0]) ** 2
        self.assertEqual(estimates.shape, (5, 50000))
        for i, entry in enumerate(entries):
            se = errors[i].std(ddof=1) / math.sqrt(errors.shape[1])
            self.assertLess(abs(errors[i].mean() - entry[-1]), 6 * se)
        self.assertLess(entries[3][-1], entries[4][-1])
        self.assertLess(entries[4][-1], entries[1][-1])

    def test_zero_squeezing_conditions_match(self):
        z = np.random.default_rng(5).standard_normal((6, 100), dtype=np.float32)
        entries = coefficients(1, 1, 0.8, 1, 0)
        estimates = calculate(np, z, entries, 1, 1, 0.8, 1)
        np.testing.assert_array_equal(estimates[0], estimates[2])
        np.testing.assert_array_equal(estimates[1], estimates[3])
        np.testing.assert_array_equal(estimates[1], estimates[4])

    def test_analytic_no_noise_case(self):
        entries = coefficients(1, 0, 1, 1, 0)
        for entry in entries:
            self.assertAlmostEqual(entry[-1], 1 / 3)

    def test_invalid_configuration(self):
        for args in ((1, 1, 1, 0.8, 1, 0.7), (10, 1, 1, 0, 1, 0.7),
                     (10, math.nan, 1, 0.8, 1, 0.7)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                configuration(*args)

    @patch("experiments.cv_a100_validation.subprocess.run")
    @patch("experiments.cv_a100_validation.shutil.which", return_value="/mock/julia")
    @patch("experiments.cv_a100_validation.julia_estimates")
    def test_cpu_report_and_backend_mismatch(self, julia_estimates, which, subprocess_run):
        subprocess_run.return_value.stdout = "mock Julia"
        def fake_julia(z, executable, source, args):
            return calculate(np, z, coefficients(*args), *args[:4])
        julia_estimates.side_effect = fake_julia
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            report = run(path, shots=50000, gpu=False)
            self.assertEqual(report["status"], "completed")
            self.assertIn("julia_cpu", report["backend_checks"])
            self.assertTrue((path / "cv_a100_validation.json").is_file())
            julia_estimates.side_effect = lambda z, *args: np.zeros((5, z.shape[1]))
            with self.assertRaisesRegex(RuntimeError, "differs from NumPy"):
                run(path, shots=100, gpu=False)


if __name__ == "__main__":
    unittest.main()
