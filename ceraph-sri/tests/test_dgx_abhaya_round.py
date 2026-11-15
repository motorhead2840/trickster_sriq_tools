"""The DGX round's decisions stay tied to the written protocol."""
from __future__ import annotations

from pathlib import Path

from experiments.dgx_abhaya_round import (
    HALL_RATIO,
    LAM,
    SLOPE_MAX,
    cpu_boundary,
    cpu_preview,
    decide_fusion,
    decide_hallucination,
    decide_product_speed,
    decide_scaling,
    decide_thermal,
)

DOC = Path("docs/dgx-abhaya-round.md")
FRAMEWORK = Path("docs/theoretical-framework.md")


def test_protocol_constants_are_in_the_plan_and_the_framework():
    plan = DOC.read_text(encoding="utf-8")
    framework = FRAMEWORK.read_text(encoding="utf-8")
    assert f"{HALL_RATIO}" in plan
    assert f"{LAM}" in plan
    assert f"{SLOPE_MAX}" in plan
    assert "0.2379" in plan
    assert "NOT evidence" in plan or "not evidence" in plan
    assert "Abhaya-gate" in framework
    assert "dgx-abhaya-round.md" in framework


def test_cpu_preview_withholds_the_latent_value_and_records_a_decision():
    preview = cpu_preview()
    assert preview["xi_definition"].startswith("sliding window variance")
    assert preview["hallucination"] == decide_hallucination(
        preview["abhaya_mse"], preview["identity_mse"], preview["moving_average_mse"]
    )
    assert preview["hallucination"] == "SUPPORTED"
    assert preview["flux_hallucination"] == "SUPPORTED"
    assert preview["flux_variance"] == "SUPPORTED"
    boundary = cpu_boundary()
    assert boundary["role"].startswith("boundary")
    assert boundary["hallucination"] == "NOT_MET"
    assert boundary["abhaya_mse"] > boundary["moving_average_mse"]
    assert boundary["flux_hallucination"] == "SUPPORTED"


def test_speed_scaling_and_thermal_decisions():
    assert decide_fusion(0.8, 1.0) == "SUPPORTED"
    assert decide_fusion(0.95, 1.0) == "NOT_MET"
    assert decide_product_speed(0.5, 1.0, "SUPPORTED") == "SUPPORTED"
    assert decide_product_speed(0.5, 1.0, "NOT_MET") == "NOT_MET"
    assert decide_product_speed(1.5, 1.0, "SUPPORTED") == "NOT_MET"
    flat = decide_scaling([1024, 2048, 4096, 8192], [1.0, 1.01, 1.02, 1.03])
    growing = decide_scaling([1024, 2048, 4096, 8192], [1.0, 2.0, 4.0, 8.0])
    assert flat["scaling"] == "SUPPORTED"
    assert flat["sublinear"] == "SUPPORTED"
    assert growing["scaling"] == "NOT_MET"
    assert growing["sublinear"] == "NOT_MET"
    assert growing["log_log_slope"] == 1.0 or abs(growing["log_log_slope"] - 1.0) < 1e-9
    assert decide_thermal(0.4, 0.5, 40.0, 42.0) == "SUPPORTED"
    assert decide_thermal(0.8, 0.5, 40.0, 42.0) == "NOT_MET"
    assert decide_thermal(0.4, 0.5, 50.0, 42.0) == "NOT_MET"
