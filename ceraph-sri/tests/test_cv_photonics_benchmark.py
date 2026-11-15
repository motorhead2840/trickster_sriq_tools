import json
import math
import tempfile
import unittest
from pathlib import Path

from experiments.cv_photonics_benchmark import posterior, run


class CVPhotonicsTests(unittest.TestCase):
    def test_reference_has_no_effect_without_correlated_noise(self):
        single_coefficients, single_mse = posterior(1.0, 0.0, 0.8, 1.0, 0.0)
        coefficients, mse = posterior(1.0, 0.0, 0.8, 1.0, 0.0, 0.7)
        self.assertAlmostEqual(mse, single_mse)
        self.assertAlmostEqual(coefficients[0], single_coefficients[0])
        self.assertAlmostEqual(coefficients[1], 0.0)

    def test_reference_squeezing_helps_and_zero_squeezing_matches(self):
        coherent = posterior(1.0, 1.0, 0.8, 1.0, 0.0, 0.0)[1]
        squeezed = posterior(1.0, 1.0, 0.8, 1.0, 0.0, 0.7)[1]
        self.assertLess(squeezed, coherent)
        with tempfile.TemporaryDirectory() as directory:
            result = run(Path(directory), shots=100, squeezing=0)
        self.assertEqual(result["results"]["coherent_signal_and_reference"]["sample_mse"],
                         result["results"]["squeezed_reference"]["sample_mse"])

    def test_monte_carlo_matches_analytic_and_is_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            result = run(path, shots=12000)
            self.assertEqual(result, run(path, shots=12000))
            self.assertEqual(result, json.loads((path / "cv_photonics_summary.json").read_text()))
            for entry in result["results"].values():
                self.assertLess(abs(entry["sample_mse"] - entry["analytic_mse"]),
                                5 * entry["sample_mse_standard_error"])

    def test_invalid_parameters_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for kwargs in ({"shots": 1}, {"efficiency": 0}, {"efficiency": 1.1},
                           {"noise_variance": -1}, {"signal_variance": math.nan},
                           {"coupling": -1}, {"squeezing": 4}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    run(Path(directory), **kwargs)


if __name__ == "__main__":
    unittest.main()
