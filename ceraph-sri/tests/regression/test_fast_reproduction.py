"""Re-run the published E1 / observable / foundation settings.

Those grids are small enough for the default CPU suite and reproduce the
committed aggregates.
"""
from __future__ import annotations

from experiments.e1_colab_lock import run as run_e1
from experiments.foundation_axioms_sweep import run as run_foundation
from experiments.observable_only_benchmark import run as run_observable
from sri.registry.conjectures import reset_registry
from tests.regression.support import SIM_ATOL, SIM_RTOL, assert_close, load_snapshot, quiet


def test_observable_only_reproduces_published_aggregates(tmp_path):
    reset_registry()
    snap = load_snapshot("observable_only.json")
    with quiet():
        summary = run_observable(tmp_path / "observable", seeds=[108, 109, 110], shots=20000)
    assert summary["conjectures"][0]["status"] == "confirmed"
    assert summary["conjectures"][1]["status"] == "confirmed"
    assert snap["statuses"]["C1"] == "confirmed"
    assert snap["statuses"]["C2"] == "confirmed"
    assert_close(summary["aggregate"], snap["aggregate"], rtol=SIM_RTOL, atol=SIM_ATOL)
    assert summary["per_seed"][0]["mmse_two_mode_analytic_mse"] == snap["analytic_mse"]
    assert summary["per_seed"][0]["weights"] == snap["weights"]
    for row in summary["per_seed"]:
        assert_close(
            row["mmse_two_mode_sample_mse"],
            snap["sample_mse"][str(row["seed"])],
            rtol=SIM_RTOL,
            atol=SIM_ATOL,
            path=f"sample_mse[{row['seed']}]",
        )


def test_foundation_axioms_reproduce_published_aggregates(tmp_path):
    reset_registry()
    snap = load_snapshot("foundation_axioms.json")
    with quiet():
        summary = run_foundation(
            tmp_path / "foundation",
            seeds=[108, 109, 110, 111, 112],
            dimension=4096,
        )
    assert summary["conjectures"][2]["status"] == "confirmed"
    assert summary["conjectures"][3]["status"] == "confirmed"
    assert snap["statuses"]["C3"] == "confirmed"
    assert snap["statuses"]["C4"] == "confirmed"
    assert summary["records_count"] == snap["records_count"]
    assert_close(summary["aggregate"], snap["aggregate"], rtol=SIM_RTOL, atol=SIM_ATOL)


def test_e1_lock_reproduces_published_exit_criteria(tmp_path):
    reset_registry()
    snap = load_snapshot("e1_colab_lock.json")
    conjectures = load_snapshot("conjecture_status.json")
    assert conjectures["statuses"] == {
        "C1": "confirmed",
        "C2": "confirmed",
        "C3": "confirmed",
        "C4": "confirmed",
        "C5": "confirmed",
    }
    with quiet():
        lock = run_e1(
            tmp_path / "e1",
            shots=20000,
            seeds=[108, 109, 110],
            dimension=4096,
            require_gpu=False,
        )
    assert lock["circuit_b_research_default"] == snap["circuit_b_research_default"] == "W"
    assert lock["colab_exit_criteria"] == snap["exit"]
    assert all(lock["colab_exit_criteria"].values())
    lock_statuses = {
        item["conjecture_id"]: item["status"] for item in lock["conjectures"]
    }
    # E1 historical snapshot covers C1–C4; C5 (Lemma Z1) joins the registry as open.
    assert {key: lock_statuses[key] for key in ("C1", "C2", "C3", "C4")} == snap["statuses"]
    assert lock_statuses.get("C5") == "open"
    assert_close(
        lock["aggregates"]["observable"],
        snap["observable_aggregate"],
        rtol=SIM_RTOL,
        atol=SIM_ATOL,
    )
    assert_close(
        lock["aggregates"]["foundation"],
        snap["foundation_aggregate"],
        rtol=SIM_RTOL,
        atol=SIM_ATOL,
    )
