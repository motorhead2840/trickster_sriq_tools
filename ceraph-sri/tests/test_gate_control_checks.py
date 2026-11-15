"""CPU checks of the notebook gate kernel against the v8 selection test."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest

HERE = Path("docs/benchmark-results/gate-control-15nov")


def _load_checks():
    spec = importlib.util.spec_from_file_location("gate_kernel_checks", HERE / "kernel_checks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def report():
    return _load_checks().build_report(sizes=(17, 1000, 8192))


def test_kernel_never_changes_the_largest_logit(report):
    swept = report["largest_logit_changed"]
    assert swept["rows_tested"] == 216
    assert swept["rows_where_winner_changed"] == 0


def test_v8_drifts_are_reproduced_bit_for_bit(report):
    shift = report["v8_shift_test_seed_11"]
    assert shift["map_A_selected_by_v8"] == 2.0**-20
    assert shift["map_R_rejected_by_v8"] == 2.757575511932373
    assert shift["logged_in_v8_sheets"] == {"drift_A": 2.0**-20, "drift_R": 2.757575511932373}


def test_repo_damping_is_the_map_v8_rejected(report):
    shift = report["v8_shift_test_seed_11"]
    assert shift["repo_sri_omlang_abhaya_damping"] == shift["map_R_rejected_by_v8"]


def test_notebook_kernel_is_not_the_map_v8_selected(report):
    shift = report["v8_shift_test_seed_11"]
    assert shift["notebook_gate_kernel_cell0"] > 100.0
    assert shift["notebook_gate_logit_multiplier_on_this_series"] > 1.0
    assert shift["v8_map_A_departure_multiplier"] < 1.0


def test_kernel_turns_random_logits_into_near_certainty(report):
    for row in report["cell0_reproduction_on_random_logits"]["numpy_rows"]:
        assert row["top1_same"]
        assert row["top1_probability_before"] < 0.5
        assert row["top1_probability_after"] > 0.999


def test_stability_score_does_not_see_which_token_leads(report):
    confident = report["stability_score_examples"]["one_token_22_logits_ahead"]
    assert confident["xi"] > 0.999 and confident["collision"] > 0.999


def test_v4_kernel_passes_low_snr_rows_through():
    checks = _load_checks()
    x = (np.random.default_rng(0).standard_normal(4096) * 3.5).astype(np.float32)
    out, _, _, snr = checks.gate_v4(x)
    assert snr < 5.8
    assert np.array_equal(out, x)


def test_colab_cell_parses_and_keys_are_one_of_the_two_options():
    tree = ast.parse((HERE / "gate_control_colab.py").read_text())
    items = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "ITEMS"
    )
    assert len(items) == 40
    assert all(key in (a, b) and a != b for a, b, _, key in items)
    assert sum(key == a for a, _, _, key in items) == 20
