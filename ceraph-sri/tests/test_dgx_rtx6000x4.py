"""The four-GPU runner still writes the CPU evidence pack with no CUDA device."""
from __future__ import annotations

import json
from pathlib import Path

from experiments.dgx_rtx6000x4 import run


def test_cpu_session_writes_report_notes_and_figures(tmp_path: Path):
    report = run(tmp_path, allow_fewer=True)
    saved = json.loads((tmp_path / "session_report.json").read_text(encoding="utf-8"))
    assert saved["cpu"]["quality"][0]["flux"] == "SUPPORTED"
    assert saved["cpu"]["quality"][1]["raw"] == "NOT_MET"
    assert saved["cpu"]["algebra"]["paired_cancellation"] == "SUPPORTED"
    assert saved["cpu"]["algebra"]["sign_xor"] == "SUPPORTED"
    assert (tmp_path / "tech_spec_notes.md").exists()
    assert (tmp_path / "patent_spec_notes.md").exists()
    assert Path(report["figures"]["quality"]).exists()
    assert Path(report["figures"]["abstain"]).exists()
    assert report["gpu"].get("status") in {"NO_CUDA", "NEED_4_GPUS", "RAN"}
