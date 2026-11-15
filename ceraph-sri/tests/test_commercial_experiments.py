import json
import tempfile
import unittest
from pathlib import Path

from experiments.analyze_llm_results import analyze, percentile, wilson
from experiments.llm_benchmark import auc, shuffled_choices
from experiments.make_vc_report import render
from experiments.signal_benchmark import paired_projection, run


class SignalBenchmarkTests(unittest.TestCase):
    def test_projection_is_estimator_not_oracle(self):
        observed = [1.0, -2.0, 3.0]
        estimated = [0.5, 0.0, 1.0]
        self.assertEqual(paired_projection(observed, estimated), estimated)

    def test_held_out_results_are_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            first = run(path, train_count=4, test_count=6, length=16)
            second = run(path, train_count=4, test_count=6, length=16)
            self.assertEqual(first, second)
            self.assertNotEqual(first["configuration"]["seed_train"], first["configuration"]["seed_test"])
            self.assertLess(first["max_paired_estimator_difference"], 1e-12)
            self.assertEqual(json.loads((path / "signal_summary.json").read_text()), first)
            self.assertEqual(len((path / "signal_records.csv").read_text().splitlines()), 7)


class LLMBenchmarkTests(unittest.TestCase):
    def test_auc_with_ties_and_missing_class(self):
        self.assertEqual(auc([0.5, 0.5, 0.9], [0, 1, 1]), 0.75)
        self.assertIsNone(auc([0.1, 0.2], [1, 1]))

    def test_choice_shuffle_preserves_answer(self):
        choices, labels = shuffled_choices(["wrong", "right", "other"], [0, 1, 0], 108)
        self.assertEqual(choices[labels.index(1)], "right")
        self.assertEqual((choices, labels), shuffled_choices(["wrong", "right", "other"], [0, 1, 0], 108))
        with self.assertRaises(ValueError):
            shuffled_choices(["a", "b"], [0, 0], 108)

    def test_analysis_recomputes_summary_and_detects_tampering(self):
        root = Path(__file__).resolve().parents[1] / "docs" / "benchmark-results" / "llm"
        result = analyze(root / "llm_records.csv", root / "llm_summary.json", iterations=20)
        self.assertEqual(result["correct"], 13)
        self.assertAlmostEqual(result["xi_auc"], 0.4650751547303271)
        self.assertEqual(percentile([0, 10], 0.25), 2.5)
        self.assertLess(wilson(13, 100)[0], 0.13)
        with tempfile.TemporaryDirectory() as directory:
            data = json.loads((root / "llm_summary.json").read_text())
            data["results"]["correct"] = 12
            altered = Path(directory) / "summary.json"
            altered.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "correct does not match"):
                analyze(root / "llm_records.csv", altered, iterations=1)

    def test_report_handles_undefined_auc(self):
        with tempfile.TemporaryDirectory() as directory:
            signal = run(Path(directory), train_count=4, test_count=4, length=16)
        llm = {
            "configuration": {"model": "example", "dataset": "example", "config": "mc",
                              "split": "validation", "first_n": 2, "seed": 108},
            "results": {"accuracy": 1.0, "correct": 2, "confidence_correctness_auc": None,
                        "xi_correctness_auc": None},
            "provenance": {"device": "A100", "script_sha256": "hash", "model_commit": "revision",
                           "dataset_fingerprint": "fingerprint"},
        }
        report = render(signal, llm)
        self.assertIn("undefined (only one correctness class)", report)
        self.assertIn("not establish", report)


if __name__ == "__main__":
    unittest.main()
