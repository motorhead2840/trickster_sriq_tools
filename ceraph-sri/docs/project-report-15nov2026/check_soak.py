"""Recompute the 15 November 40-minute soak logs.

The logs are the two JSON files in data/. Nothing here contacts a board.
Run from the repository root:

    python docs/project-report-15nov2026/check_soak.py
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FLOOR = 2**-19  # the drift the logs record, and the floor the index uses


def load(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def index_of(board: dict) -> float:
    """The efficiency index, from the formula in the afternoon documents."""
    jobs_per_joule = board["jobs_per_sec"] / board["mean_watts"]
    thermal = max(0.0, (board["max_temp_c"] - 60.0) / 25.0) ** 2
    power = max(0.0, (board["peak_watts"] - 540.0) / 600.0) ** 2
    return jobs_per_joule / (1.0 + thermal + power)


def sample_rows(document: dict, gpu: int) -> list[dict]:
    return [sample["boards"][gpu] for sample in document["time_series"]]


def check(document: dict) -> dict:
    """Raise AssertionError if a headline number disagrees with the samples."""
    summary = document["soak_summary"]
    boards = document["per_board"]
    elapsed = [sample["elapsed_s"] for sample in document["time_series"]]
    gaps = [b - a for a, b in zip(elapsed, elapsed[1:])]

    rate = summary["total_jobs_completed"] / summary["actual_duration_s"]
    assert abs(rate - summary["cluster_throughput_q_B_jobs_per_sec"]) < 0.02
    assert sum(b["jobs_completed"] for b in boards) == summary["total_jobs_completed"]
    assert abs(sum(b["jobs_per_sec"] for b in boards) - summary["cluster_throughput_q_B_jobs_per_sec"]) < 0.01
    logged_v = summary["cluster_value_appreciation_index_V"]
    assert abs(sum(b["value_appreciation_V02"]["value_appreciation_index_V"] for b in boards) - logged_v) < 1e-6
    assert sum(b["scalar_16B_messages_sent"] for b in boards) == summary["total_16B_scalar_messages"]
    assert sum(b["v13_gate_verifications"] for b in boards) == summary["total_v13_gate_verifications"]
    energy = summary["cluster_mean_watts"] / summary["cluster_throughput_q_B_jobs_per_sec"] * 1000.0
    assert abs(energy - summary["cluster_energy_mj_per_job"]) < 5e-4
    assert min(gaps) > 14.9 and max(gaps) < 15.2

    for board in boards:
        assert board["jobs_completed"] % 500 == 0
        assert board["jobs_completed"] // 500 == board["scalar_16B_messages_sent"]
        assert board["post_burn_shift_drift_f32"] == FLOOR
        assert board["post_burn_xor_cosine"] == 1.0
        assert board["numerical_faults"] == 0 and board["all_finite"] is True
        assert board["peak_watts"] < 600.0 and board["max_temp_c"] <= 78.0
        assert abs(index_of(board) - board["value_appreciation_V02"]["value_appreciation_index_V"]) < 0.002
        rows = sample_rows(document, board["gpu"])
        power = [row["power_w"] for row in rows]
        temp = [row["temp_c"] for row in rows]
        clock = [row["sm_mhz"] for row in rows]
        assert abs(statistics.fmean(power) - board["mean_watts"]) < 1e-4
        assert max(power) == board["peak_watts"]
        assert min(temp) == board["min_temp_c"] and max(temp) == board["max_temp_c"]
        assert min(clock) == board["min_sm_mhz"] and max(clock) == board["peak_sm_mhz"]
        assert set(row["limit_w"] for row in rows) == {600.0}
    return summary


def fleet_ratio() -> float:
    one = load("v8_enterprise_40min_soak_B1.json")["soak_summary"]
    eight = load("v8_enterprise_40min_soak_B8.json")["soak_summary"]
    return eight["cluster_throughput_q_B_jobs_per_sec"] / one["cluster_throughput_q_B_jobs_per_sec"]


def main() -> None:
    one = load("v8_enterprise_40min_soak_B1.json")
    eight = load("v8_enterprise_40min_soak_B8.json")
    check(one)
    check(eight)
    ratio = fleet_ratio()
    print(f"one board  {one['soak_summary']['cluster_throughput_q_B_jobs_per_sec']} jobs/s")
    print(f"eight      {eight['soak_summary']['cluster_throughput_q_B_jobs_per_sec']} jobs/s")
    print(f"ratio      {ratio:.6f}   ({ratio / 8:.6%} of eight times)")
    print("checks passed")


if __name__ == "__main__":
    main()
