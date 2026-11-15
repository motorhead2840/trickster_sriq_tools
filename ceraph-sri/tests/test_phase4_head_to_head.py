"""Phase 4 keeps Colab measurements and refuses a printed budget as a competitor."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.phase4_head_to_head import (
    PHASE3_R,
    confirm_dual_mode,
    load_catalog,
    run,
)

CATALOG = Path("docs/benchmark-results/phase4/colab_device_catalog.json")
SPEC = Path("docs/phase4-head-to-head-spec.md")


def test_catalog_keeps_blackwell_medians_and_flags_the_printed_budget():
    catalog = load_catalog(CATALOG)
    by_id = {row["id"]: row for row in catalog["records"]}
    assert by_id["dissertation_triton_8192"]["median_ms"] == 0.0082
    assert by_id["dissertation_triton_8192"]["printed_budget_ms"] == 0.2379
    assert by_id["dissertation_triton_8192"]["printed_budget_is_measurement"] is False
    assert by_id["dissertation_triton_10000"]["median_ms"] == 0.0062
    assert by_id["dissertation_triton_10000"]["truth_vector_was_an_input"] is True
    assert by_id["jax_blackwell_2026_11_07"]["median_ms"] == 0.08884
    assert by_id["blackwell_vs_avg_pool_2026_11_14"]["baseline_faster"] is True
    assert by_id["phase3_dual_mode"]["mean_relative_reduction_vs_single_mode"] == PHASE3_R
    assert "0.2379" in SPEC.read_text(encoding="utf-8")
    assert "NOT_EVIDENCE" in SPEC.read_text(encoding="utf-8")


def test_printed_budget_cannot_be_relabeled_as_a_measurement(tmp_path: Path):
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    catalog["records"][1]["printed_budget_is_measurement"] = True
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    with pytest.raises(ValueError, match="printed budget"):
        load_catalog(path)


def test_confirmation_still_beats_single_mode_and_scoreboard_splits_claims(tmp_path: Path):
    confirmation = confirm_dual_mode(shots=4_000)
    assert confirmation["holds"] is True
    assert confirmation["mean_relative_reduction"] > 0.0
    board = run(CATALOG, tmp_path)
    by_id = {item["id"]: item["status"] for item in board["claims"]}
    assert board["narrow_claim"] == "SUPPORTED"
    assert board["broad_claim"] == "NOT_MET"
    assert by_id["printed_latency_budget"] == "NOT_EVIDENCE"
    assert by_id["same_gpu_faster_than_named_baseline"] == "NOT_MET"
    assert by_id["thermal_savings_against_a_named_accelerator"] == "OPEN"
    assert by_id["blackwell_execution"] == "SUPPORTED"
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Narrow claim:** SUPPORTED" in text
    assert "Broad claim" in text and "NOT_MET" in text
    assert "avg_pool1d" in text
