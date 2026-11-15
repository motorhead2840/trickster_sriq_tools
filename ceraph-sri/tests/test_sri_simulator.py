import csv
import math
import tempfile
import unittest
from pathlib import Path

from sri_simulator import (
    DampingParameters,
    OmegaDit,
    SimulationConfig,
    damp_logits,
    integrate_omega,
    landauer_minimum_heat,
    mean_variance,
    simulate,
    stability_coefficient,
    symplectic_project,
    v3_damping,
    write_telemetry,
)


class OmegaDitTests(unittest.TestCase):
    def test_integrand_and_finite_grid_integral(self):
        sample = OmegaDit(0.98, math.pi / 6, (4.0, -2.5), (0.35, -0.42))
        expected = tuple(
            0.98 * complex(math.cos(math.pi / 6), math.sin(math.pi / 6)) * value
            for value in (4.35, -2.92)
        )
        for actual, target in zip(sample.integrand(), expected):
            self.assertAlmostEqual(abs(actual - target), 0.0, places=12)
        integrated = integrate_omega((sample, sample), 0.5)
        for actual, target in zip(integrated, expected):
            self.assertAlmostEqual(abs(actual - target), 0.0, places=12)

    def test_rejects_mismatched_components(self):
        with self.assertRaises(ValueError):
            OmegaDit(1.0, 0.0, (1.0,), (1.0, 2.0))


class ProjectionTests(unittest.TestCase):
    def test_exact_paired_noise_cancellation(self):
        clean = (12.0, 0.0, 0.0)
        noise = (0.8, 4.3, -2.1)
        result = symplectic_project(clean, noise)
        self.assertEqual(result.thesis, (12.8, 4.3, -2.1))
        self.assertEqual(result.antithesis, (11.2, -4.3, 2.1))
        for actual, expected in zip(result.projected, clean):
            self.assertAlmostEqual(actual, expected)

    def test_rejects_bad_projection_inputs(self):
        with self.assertRaises(ValueError):
            symplectic_project((1.0,), (1.0, 2.0))
        with self.assertRaises(ValueError):
            symplectic_project((float("nan"),), (0.0,))


class DampingTests(unittest.TestCase):
    def test_stability_equation_and_domain_warning(self):
        self.assertAlmostEqual(stability_coefficient((9.0, 11.0)), 0.9)
        self.assertLess(stability_coefficient((1.0, 3.0, 30.0)), 0.0)
        with self.assertRaises(ValueError):
            stability_coefficient((-1e-7,))

    def test_two_circuit_law_follows_printed_epsilon_scaled_equation(self):
        params = DampingParameters(
            lambda_0=1.0, kappa=4.0, lambda_max=50.0, alpha=10.0,
            theta_crit=5.0, sigma_sat=1.0, epsilon=1e-7,
        )
        result = v3_damping(0.99, 2.0, params)
        self.assertAlmostEqual(result.circuit_a, math.exp(-0.04))
        self.assertAlmostEqual(result.circuit_b, 50.0)
        self.assertAlmostEqual(result.chi, result.circuit_a + result.circuit_b)
        self.assertLess(mean_variance(damp_logits((2.0, 4.0), result.chi))[1], 1e-40)

    def test_landauer_bound(self):
        self.assertAlmostEqual(
            landauer_minimum_heat(350.0),
            1.380649e-23 * 350.0 * math.log(2.0),
        )
        with self.assertRaises(ValueError):
            landauer_minimum_heat(-1.0)


class SimulationTests(unittest.TestCase):
    def test_seeded_simulation_is_reproducible_except_timing(self):
        config = SimulationConfig(seed=7, dimension=64, cycles=5, spike_start=2, spike_end=3)
        first = simulate(config)
        second = simulate(config)
        self.assertEqual(len(first), 5)
        for left, right in zip(first, second):
            left.pop("projection_elapsed_ms")
            right.pop("projection_elapsed_ms")
            self.assertEqual(left, right)
            self.assertLess(left["projection_mse"], 1e-28)
        self.assertEqual([row["noise_std"] for row in first], [0.5, 0.5, 15.0, 15.0, 0.5])

    def test_csv_telemetry_round_trip(self):
        records = simulate(SimulationConfig(seed=3, dimension=8, cycles=2))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "telemetry.csv"
            write_telemetry(records, output)
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["cycle"], "0")
        self.assertIn("projection_mse", rows[0])


if __name__ == "__main__":
    unittest.main()
