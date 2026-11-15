"""The 15 November project report quotes its logs and its own arithmetic."""
from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path("docs/project-report-15nov2026")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_soak_logs_agree_with_their_own_samples():
    soak = _load("check_soak")
    one = soak.load("v8_enterprise_40min_soak_B1.json")
    eight = soak.load("v8_enterprise_40min_soak_B8.json")
    soak.check(one)
    soak.check(eight)
    ratio = soak.fleet_ratio()
    assert abs(ratio - 7.990683) < 1e-4
    assert abs(ratio / 8 - 0.998835) < 1e-4


def test_the_ask_adds_up_and_the_report_quotes_it():
    projection = _load("projection")
    report = (HERE / "project-report.md").read_text(encoding="utf-8")
    assert projection.use_total() == projection.ASK == 2_000_000
    for _name, amount in projection.USE_OF_FUNDS:
        assert f"${amount:,}" in report
    assert "5.40" in report and "9.60" in report and "21.60" in report
