"""Merge S1–S5 campaign artifacts into a VC-safe Markdown + JSON report."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def _load(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _status_line(ok: bool | None, label: str) -> str:
    if ok is True:
        return f"- **PASS** — {label}"
    if ok is False:
        return f"- **FAIL** — {label}"
    return f"- **PENDING** — {label}"


def build_report(
    *,
    results_root: Path,
    lock_path: Path,
    advantage_path: Path,
    efficiency_path: Path,
    smoke_path: Path,
    hybrid_path: Path,
    spend_log_path: Path | None = None,
) -> dict:
    lock = _load(lock_path)
    advantage = _load(advantage_path)
    efficiency = _load(efficiency_path)
    smoke = _load(smoke_path)
    hybrid = _load(hybrid_path)

    exit_ok = bool(lock and lock.get("colab_exit_criteria", {}).get("c1_c4_confirmed"))
    advantage_ok = bool(advantage and advantage.get("pass"))
    efficiency_ok = bool(efficiency and efficiency.get("pass"))
    smoke_status = (smoke or {}).get("quantum_task", {}).get("status")
    smoke_mode = (smoke or {}).get("quantum_task", {}).get("mode")
    smoke_ok = smoke_status in {"COMPLETED", "skipped"}
    hybrid_compare = (hybrid or {}).get("comparison_to_reference") or {}
    hybrid_ok = bool(hybrid_compare.get("pass")) if hybrid_compare else None
    hybrid_submit = (hybrid or {}).get("hybrid_submit", {})
    live_aws = smoke_mode == "live" or hybrid_submit.get("status") == "submitted"

    phases = {
        "S1_accuracy": {"pass": exit_ok, "artifact": str(lock_path)},
        "S2_advantage": {"pass": advantage_ok, "artifact": str(advantage_path)},
        "S3_efficiency": {"pass": efficiency_ok, "artifact": str(efficiency_path)},
        "S4_sv1_smoke": {
            "pass": smoke_ok,
            "mode": smoke_mode,
            "status": smoke_status,
            "artifact": str(smoke_path),
            "live_aws": smoke_mode == "live",
        },
        "S5_hybrid": {
            "pass": hybrid_ok,
            "submit_status": hybrid_submit.get("status"),
            "artifact": str(hybrid_path),
            "live_aws": hybrid_submit.get("status") == "submitted",
        },
    }
    campaign_pass = all(
        phases[key]["pass"] is True
        for key in ("S1_accuracy", "S2_advantage", "S3_efficiency", "S4_sv1_smoke")
    ) and (hybrid_ok is True or hybrid_ok is None)

    obs = (lock or {}).get("aggregates", {}).get("observable", {})
    found = (lock or {}).get("aggregates", {}).get("foundation", {})
    eff_agg = (efficiency or {}).get("aggregate", {})

    md_lines = [
        "# AWS Braket simulation campaign report",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Honesty boundary",
        "",
        "- **Classical CV** (E1–E4 / S1–S3 / Hybrid numerics): observable-only Gaussian MMSE,",
        "  axiomatic Ξ_SNR, Circuit-B Branch W, CPTP entropy — proven on NumPy / Hybrid-shaped compute.",
        "- **Qubit contact** (S4 SV1/DM1, optional S6 QPU): job plumbing / hardware contact only.",
        "  Not CV MMSE on a QPU. Not yellow-paper O(1), Landauer, or photonic quantum advantage.",
        "",
        "## Phase results",
        "",
        _status_line(exit_ok, "S1 Accuracy lock (C1–C4)"),
        _status_line(advantage_ok, "S2 Advantage vs classical filters (equal-resource)"),
        _status_line(efficiency_ok, "S3 Efficiency / N-scaling (Hybrid-shaped)"),
        _status_line(smoke_ok, f"S4 SV1/DM1 smoke (mode={smoke_mode}, status={smoke_status})"),
        _status_line(hybrid_ok, f"S5 Hybrid provenance (submit={hybrid_submit.get('status')})"),
        "",
        "## Accuracy headline numbers",
        "",
        f"- Mean MMSE sample MSE: **{obs.get('mean_mmse_sample_mse', 'n/a')}**",
        f"- Moving average MSE: {obs.get('mean_moving_average_mse', 'n/a')}",
        f"- Median filter MSE: {obs.get('mean_median_filter_mse', 'n/a')}",
        f"- Single-mode Wiener MSE: {obs.get('mean_single_mode_mse', 'n/a')}",
        f"- Ξ_SNR range: [{found.get('xi_snr_min', 'n/a')}, {found.get('xi_snr_max', 'n/a')}]",
        f"- Entropy nonincreasing under CPTP: {found.get('entropy_nonincreasing_under_cptp', 'n/a')}",
        "",
        "## Advantage (S2)",
        "",
    ]
    if advantage:
        pc = advantage.get("pass_criteria", {})
        md_lines.extend(
            [
                f"- Defensible win: **{advantage.get('pass')}**",
                f"- MMSE beats MA/median/single-mode: "
                f"{pc.get('mmse_beats_moving_average')}/"
                f"{pc.get('mmse_beats_median')}/"
                f"{pc.get('mmse_beats_single_mode')}",
                f"- Fraction squeezed-ref beats coherent: "
                f"{pc.get('fraction_sq_ref_beats_coherent')}",
                f"- Squeezed-signal stronger than squeezed-ref (honest control): "
                f"{pc.get('squeezed_signal_beats_squeezed_ref')}",
                f"- Equal-resource note: {advantage.get('equal_resource_note')}",
                "",
            ]
        )
    else:
        md_lines.append("- Advantage artifact missing.")
        md_lines.append("")

    md_lines.extend(
        [
            "## Efficiency (S3)",
            "",
            f"- Accuracy OK across N: **{eff_agg.get('accuracy_ok_across_n', 'n/a')}**",
            f"- Estimated Hybrid instance cost for sweep: "
            f"${eff_agg.get('total_estimated_hybrid_instance_cost_usd', 'n/a')}",
            f"- Scaling note: {eff_agg.get('scaling_note', 'n/a')}",
            "",
            "## AWS contact",
            "",
            f"- Live AWS tasks this run: **{'yes' if live_aws else 'no (emulate / local hybrid)'}**",
            f"- Smoke mode: `{smoke_mode}`",
            f"- Hybrid submit: `{hybrid_submit.get('status')}`",
            "",
            "## Campaign verdict",
            "",
            f"- Campaign pass (S1–S4 + hybrid compare if present): **{campaign_pass}**",
            "",
            "Spend discipline: see `docs/compute-unit-spend-log.md`. "
            "Live submit still requires `BRAKET_SUBMIT=1`, cost caps, and `BRAKET_S3_BUCKET`.",
            "",
        ]
    )
    if spend_log_path and spend_log_path.is_file():
        md_lines.append(f"- Spend log path: `{spend_log_path}`")
        md_lines.append("")

    report = {
        "experiment": "braket_campaign_report",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phases": phases,
        "campaign_pass": campaign_pass,
        "live_aws": live_aws,
        "headlines": {
            "mean_mmse_sample_mse": obs.get("mean_mmse_sample_mse"),
            "mean_moving_average_mse": obs.get("mean_moving_average_mse"),
            "mean_median_filter_mse": obs.get("mean_median_filter_mse"),
            "mean_single_mode_mse": obs.get("mean_single_mode_mse"),
            "xi_snr_min": found.get("xi_snr_min"),
            "xi_snr_max": found.get("xi_snr_max"),
            "efficiency_accuracy_ok": eff_agg.get("accuracy_ok_across_n"),
        },
        "markdown": "\n".join(md_lines),
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("docs/benchmark-results"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("docs/braket-campaign-report.md"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("docs/benchmark-results/braket-campaign/campaign_report.json"),
    )
    args = parser.parse_args(argv)
    root = args.results_root
    report = build_report(
        results_root=root,
        lock_path=root / "e1_colab_lock.json",
        advantage_path=root / "braket-campaign" / "advantage" / "advantage_summary.json",
        efficiency_path=root / "braket-campaign" / "efficiency" / "efficiency_summary.json",
        smoke_path=root / "braket-live" / "sv1_smoke.json",
        hybrid_path=root / "braket-live" / "hybrid_summary.json",
        spend_log_path=Path("docs/compute-unit-spend-log.md"),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps({k: report[k] for k in report if k != "markdown"}, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(report["markdown"], encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "markdown"}, indent=2))
    print(f"wrote {args.output_md}")
    return 0 if report["campaign_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
