"""Phase 4 scoreboard.

Reads the Colab and DGX catalog, re-checks the dual-mode quality path on CPU,
and writes a scoreboard. A printed latency budget cannot count as a competitor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sri.cv.estimator import (
    SensingConfig,
    observable_yoga_buddhi,
    simulate_measurements,
    single_mode_wiener,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "docs" / "benchmark-results" / "phase4" / "colab_device_catalog.json"
PHASE3_R = 0.1205
PHASE3_THRESHOLD = 0.08
CONFIRM_SHOTS = 8_000
CONFIRM_SEED = 401
CONFIRM_NOISE = (1.0, 2.0)


def load_catalog(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("phase") != 4:
        raise ValueError("catalog phase must be 4")
    ids = [row["id"] for row in data["records"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate catalog ids")
    for row in data["records"]:
        if row.get("printed_budget_is_measurement") is True:
            raise ValueError(f"{row['id']} marks a printed budget as a measurement")
        if "printed_budget_ms" in row and row.get("printed_budget_is_measurement") is not False:
            raise ValueError(f"{row['id']} must flag printed_budget_is_measurement false")
    return data


def confirm_dual_mode(shots: int = CONFIRM_SHOTS) -> dict:
    """Short CPU check that the deployable path still beats single-mode Wiener."""
    config_base = dict(
        signal_variance=1.0,
        coupling=1.0,
        efficiency=0.8,
        signal_r=0.0,
        reference_r=0.7,
    )
    cells = []
    for noise in CONFIRM_NOISE:
        config = SensingConfig(noise_variance=noise, **config_base)
        rng = np.random.default_rng(CONFIRM_SEED + int(noise * 10))
        signal, _noise, y_a, y_b = simulate_measurements(config, shots, rng)
        sri = observable_yoga_buddhi(y_a, y_b, config).projected
        single = single_mode_wiener(y_a, config)
        sri_mse = float(np.mean((sri - signal) ** 2))
        single_mse = float(np.mean((single - signal) ** 2))
        reduction = 1.0 - sri_mse / single_mse
        cells.append(
            {
                "noise_variance": noise,
                "shots": shots,
                "sri_mse": sri_mse,
                "single_mode_mse": single_mse,
                "relative_reduction": reduction,
            }
        )
    mean_r = float(np.mean([cell["relative_reduction"] for cell in cells]))
    return {
        "seed_base": CONFIRM_SEED,
        "shots": shots,
        "cells": cells,
        "mean_relative_reduction": mean_r,
        "holds": mean_r > 0.0 and all(cell["sri_mse"] < cell["single_mode_mse"] for cell in cells),
    }


def _measured_medians(catalog: dict) -> list[float]:
    medians = []
    for row in catalog["records"]:
        if row["kind"] != "measured_kernel_time":
            continue
        if "median_ms" in row:
            medians.append(float(row["median_ms"]))
        if "latency_ms" in row:
            medians.extend(float(v) for v in row["latency_ms"])
        if "average_latency_ms" in row:
            medians.extend(float(v) for v in row["average_latency_ms"])
    return medians


def score(catalog: dict, confirmation: dict) -> dict:
    medians = _measured_medians(catalog)
    blackwell = [
        row for row in catalog["records"]
        if "RTX PRO 6000" in row.get("device", "") and row["kind"] == "measured_kernel_time"
    ]
    head = next(row for row in catalog["records"] if row["id"] == "blackwell_vs_avg_pool_2026_11_14")
    cpu_rows = next(row for row in catalog["records"] if row["id"] == "blackwell_kernel_vs_cpu_2026_11_14")
    peak = max(item["cpu_over_gpu"] for item in cpu_rows["rows"])
    sub_ms = [value for value in medians if value < 1.0]
    claims = [
        {
            "id": "dual_mode_beats_single_mode",
            "status": "SUPPORTED" if confirmation["holds"] else "NOT_MET",
            "detail": (
                f"Phase 3 locked mean R={PHASE3_R:.4f} against threshold {PHASE3_THRESHOLD}. "
                f"This run's confirmation mean R={confirmation['mean_relative_reduction']:.4f}."
            ),
        },
        {
            "id": "blackwell_execution",
            "status": "SUPPORTED" if blackwell else "NOT_MET",
            "detail": f"{len(blackwell)} measured RTX PRO 6000 kernel records are in the catalog.",
        },
        {
            "id": "submillisecond_on_recorded_sizes",
            "status": "SUPPORTED" if sub_ms else "NOT_MET",
            "detail": "Smallest recorded kernel time is "
            + (f"{min(medians):.4f} ms." if medians else "missing."),
        },
        {
            "id": "faster_than_same_host_cpu_at_large_length",
            "status": "SUPPORTED" if peak > 1.0 else "NOT_MET",
            "detail": f"Peak CPU/GPU ratio on the 14 November Blackwell kernel is {peak:.1f}.",
        },
        {
            "id": "same_gpu_faster_than_named_baseline",
            "status": "NOT_MET" if head["baseline_faster"] else "SUPPORTED",
            "detail": (
                f"{head['baseline_name']} median {head['baseline_median_ms']:.3f} ms, "
                f"Ceraph kernel median {head['ceraph_median_ms']:.3f} ms, "
                f"temperatures {head['baseline_max_temperature_c']} °C and "
                f"{head['ceraph_max_temperature_c']} °C."
            ),
        },
        {
            "id": "thermal_savings_against_a_named_accelerator",
            "status": "OPEN",
            "detail": (
                "Board readings exist for one L40S and one RTX PRO 6000. "
                "No second accelerator ran the same job."
            ),
        },
        {
            "id": "printed_latency_budget",
            "status": "NOT_EVIDENCE",
            "detail": "0.2379 ms is stored and excluded. It is not a timed competitor.",
        },
    ]
    narrow = all(
        item["status"] == "SUPPORTED"
        for item in claims
        if item["id"] in {
            "dual_mode_beats_single_mode",
            "blackwell_execution",
            "submillisecond_on_recorded_sizes",
        }
    )
    broad = next(item for item in claims if item["id"] == "same_gpu_faster_than_named_baseline")
    return {
        "phase": 4,
        "narrow_claim": "SUPPORTED" if narrow else "NOT_MET",
        "broad_claim": broad["status"],
        "confirmation": confirmation,
        "claims": claims,
        "sentence": (
            "The deployable dual-mode path remains ahead of single-mode Wiener, "
            "and the archived kernels have already been timed on an RTX PRO 6000 Blackwell, "
            "including sub-millisecond medians at the small sizes. "
            "The same card's average-pool comparison is the open race: "
            "the current operator finished first, so the broad speed and thermal claim stays unmet "
            "until a later same-GPU row reverses it."
        ),
    }


def render_report(board: dict) -> str:
    lines = [
        "# Phase 4 report — head-to-head scoreboard",
        "",
        f"**Narrow claim:** {board['narrow_claim']}  ",
        f"**Broad claim (faster and cooler than a named current method on the same GPU):** {board['broad_claim']}",
        "",
        board["sentence"],
        "",
        "## Confirmation",
        "",
        f"Mean relative reduction versus single-mode Wiener on the short CPU check: "
        f"**{board['confirmation']['mean_relative_reduction']:.4f}**.",
        "",
        "| Noise variance | SRI MSE | Single-mode MSE | R |",
        "|---:|---:|---:|---:|",
    ]
    for cell in board["confirmation"]["cells"]:
        lines.append(
            f"| {cell['noise_variance']:.1f} | {cell['sri_mse']:.4f} | "
            f"{cell['single_mode_mse']:.4f} | {cell['relative_reduction']:.4f} |"
        )
    lines.extend(["", "## Claims", ""])
    for claim in board["claims"]:
        lines.append(f"- `{claim['id']}` — **{claim['status']}**. {claim['detail']}")
    lines.extend(
        [
            "",
            "## Next row that would move the broad claim",
            "",
            "Same GPU, same tensor, a named current operator, and this kernel or the dual-mode estimator. "
            "Record median time, board power, and temperature for both. "
            "The broad claim becomes SUPPORTED when the Ceraph side matches quality and finishes sooner or uses fewer joules.",
            "",
        ]
    )
    return "\n".join(lines)


def run(catalog_path: Path, output_dir: Path) -> dict:
    catalog = load_catalog(catalog_path)
    confirmation = confirm_dual_mode()
    board = score(catalog, confirmation)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scoreboard.json").write_text(json.dumps(board, indent=2) + "\n", encoding="utf-8")
    report = render_report(board)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return board


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the Phase 4 head-to-head claim.")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs" / "benchmark-results" / "phase4",
    )
    args = parser.parse_args()
    board = run(args.catalog, args.output_dir)
    print(json.dumps({"narrow_claim": board["narrow_claim"], "broad_claim": board["broad_claim"]}, indent=2))


if __name__ == "__main__":
    main()
