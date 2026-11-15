"""The hybrid session reports missing tiers instead of pretending they ran."""
from __future__ import annotations

from sri.omlang.hybrid_stack import run_stack


def test_missing_tiers_do_not_count_as_a_real_stack():
    report = run_stack(n=128, lam=1.0, seed=11)
    assert report["numpy"]["status"] == "RAN"
    assert report["numpy"]["schedule"]["block"] in {128, 256, 512, 1024}
    assert report["stack_real"] is False
    assert report["julia"]["status"] in {"MISSING", "RAN", "ERROR"}
    assert report["jax"]["status"] in {"MISSING", "RAN", "ERROR"}
    assert report["triton"]["status"] in {"MISSING", "NO_CUDA", "RAN", "ERROR"}
    if report["julia"]["status"] != "RAN" or report["jax"]["status"] != "RAN" or report["triton"]["status"] != "RAN":
        assert "not real" in report["sentence"]
