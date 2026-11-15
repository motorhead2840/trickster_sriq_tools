"""Focused tests for Phase 3 vertical slice (CPU, observable-only)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.phase3_vertical_slice import (
    ALGEBRA_RTOL,
    NOISE_GRID,
    PASS_RELATIVE_REDUCTION,
    SEEDS,
    SHOTS,
    decide,
    run,
    run_cell,
)
from sri.cv.estimator import SensingConfig, estimate_signal, observable_yoga_buddhi, simulate_measurements


def test_preregistered_constants_match_spec_doc():
    text = Path("docs/phase3-vertical-slice-spec.md").read_text(encoding="utf-8")
    assert "201, 202, 203, 204, 205" in text or "`201, 202, 203, 204, 205`" in text
    assert "0.08" in text
    assert SEEDS == [201, 202, 203, 204, 205]
    assert SHOTS == 20_000
    assert NOISE_GRID == [0.5, 1.0, 2.0]
    assert PASS_RELATIVE_REDUCTION == 0.08


def test_observable_path_never_needs_signal_or_noise_arrays():
    cfg = SensingConfig(
        signal_variance=1.0,
        noise_variance=1.0,
        coupling=1.0,
        efficiency=0.8,
        signal_r=0.0,
        reference_r=0.7,
    )
    rng = np.random.default_rng(7)
    _s, _n, y_a, y_b = simulate_measurements(cfg, 128, rng)
    # Callers can discard s,n — estimator APIs only take y_a,y_b,config
    sri = observable_yoga_buddhi(y_a, y_b, cfg)
    plain = estimate_signal(y_a, y_b, cfg)
    assert sri.projected.shape == y_a.shape
    assert np.allclose(sri.projected, plain.estimate, rtol=ALGEBRA_RTOL, atol=ALGEBRA_RTOL)


def test_run_cell_flags_observable_only_and_algebra():
    cell = run_cell(seed=201, noise_variance=1.0, shots=2000)
    assert cell["observable_only"] is True
    assert cell["algebra_ok"] is True
    assert cell["oracle_paired_mse"] < 1e-20


def test_decide_thresholds():
    base = {
        "algebra_ok": True,
        "mmse_within_5sigma": True,
        "observable_only": True,
        "relative_reduction_vs_single": 0.10,
    }
    assert decide([base, {**base, "relative_reduction_vs_single": 0.09}])["verdict"] == "pass"
    assert decide([base, {**base, "relative_reduction_vs_single": 0.01}])["verdict"] == "inconclusive"
    # Fail uses mean R < 0 (preregistered), not a single negative cell.
    assert decide([
        {**base, "relative_reduction_vs_single": -0.02},
        {**base, "relative_reduction_vs_single": -0.01},
    ])["verdict"] == "fail"
    bad = {**base, "algebra_ok": False}
    assert decide([bad])["verdict"] == "fail"


@pytest.mark.slow
def test_full_phase3_runner_writes_artifacts(tmp_path: Path):
    summary = run(tmp_path)
    assert (tmp_path / "phase3_summary.json").exists()
    assert (tmp_path / "phase3_cells.json").exists()
    assert summary["decision"]["verdict"] in {"pass", "fail", "inconclusive"}
    loaded = json.loads((tmp_path / "phase3_summary.json").read_text(encoding="utf-8"))
    assert loaded["experiment"] == "phase3_vertical_slice_cv_sensing"
    assert len(loaded["cells"]) == len(SEEDS) * len(NOISE_GRID)
