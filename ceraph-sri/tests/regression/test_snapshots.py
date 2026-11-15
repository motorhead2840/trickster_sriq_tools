"""The small snapshots are a faithful extract of the published JSON files."""
from __future__ import annotations

import json
from pathlib import Path

from tests.regression.extract import ROOT, SNAPSHOTS, _load
from tests.regression.support import SNAPSHOTS as SNAPSHOT_DIR
from tests.regression.support import load_snapshot


def test_snapshot_files_match_the_published_sources():
    for name, builder in SNAPSHOTS.items():
        saved = load_snapshot(name)
        fresh = json.loads(json.dumps(builder()))
        assert saved == fresh, name


def test_published_conjecture_copies_agree():
    primary = _load("docs/benchmark-results/conjecture_status.json")
    hybrid = _load("docs/benchmark-results/braket-live/e1_e4_hybrid/conjecture_status.json")
    assert [item["conjecture_id"] for item in primary["conjectures"]] == [
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
    ]
    # Canonical registry after Phase 3 C5 reconciliation.
    assert [item["status"] for item in primary["conjectures"]] == [
        "confirmed",
        "confirmed",
        "confirmed",
        "confirmed",
        "confirmed",
    ]
    # Hybrid campaign JSON is a frozen historical snapshot (C5 was still open there).
    # Preserve it; only require the same conjecture IDs, not identical statuses.
    assert [item["conjecture_id"] for item in hybrid["conjectures"]] == [
        item["conjecture_id"] for item in primary["conjectures"]
    ]
    hybrid_by_id = {item["conjecture_id"]: item["status"] for item in hybrid["conjectures"]}
    assert hybrid_by_id["C5"] == "open"
    assert all(hybrid_by_id[cid] == "confirmed" for cid in ("C1", "C2", "C3", "C4"))


def test_published_e1_aggregates_agree_across_copies():
    primary = _load("docs/benchmark-results/e1_colab_lock.json")
    hybrid = _load("docs/benchmark-results/braket-live/e1_e4_hybrid/e1_colab_lock.json")
    assert primary["aggregates"] == hybrid["aggregates"]
    assert primary["colab_exit_criteria"] == hybrid["colab_exit_criteria"]
    assert primary["circuit_b_research_default"] == "W"


def test_snapshot_directory_is_small():
    total = sum(path.stat().st_size for path in SNAPSHOT_DIR.glob("*.json"))
    originals = [
        ROOT / "experiments/unveiling/v1/reference_results.json",
        ROOT / "experiments/unveiling/v2/reference_results.json",
        ROOT / "experiments/unveiling/v3/reference_results.json",
    ]
    assert all(path.stat().st_size > 100_000 for path in originals)
    assert total < 100_000
    assert Path(ROOT / "experiments/unveiling/v2/reference_results.json").stat().st_size > total
