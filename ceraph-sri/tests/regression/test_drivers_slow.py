"""Published-size driver reproductions.

Marked slow so the default CPU suite can skip them. Compare headline Monte
Carlo numbers with a relative tolerance of 1e-6.
"""
from __future__ import annotations

import pytest

from tests.regression.support import (
    SIM_ATOL,
    SIM_RTOL,
    V2_FIT_RTOL,
    assert_close,
    load_snapshot,
    load_suite,
    quiet,
)

pytestmark = pytest.mark.slow


def test_v1_small_reproduces_gauge_and_xi(tmp_path):
    suite = load_suite("v1")
    snap = load_snapshot("v1_key_numbers.json")
    with quiet():
        payload = suite.run_suite(suite.SimConfig.from_size("small"), tmp_path / "v1", n_track_walks=5)
    gauge = payload["gauge"]
    assert_close(
        {
            "amp": gauge["rms_amp_ungauged"],
            "phase": gauge["rms_phase_ungauged"],
            "kappa": gauge["rms_kappa_gauged"],
            "kappa_true_scaled": gauge["rms_kappa_gauged_true_scaled"],
            "kappa_hat_scaled": gauge["rms_kappa_gauged_hat_scaled"],
        },
        {
            "amp": snap["gauge_rms_amp_ungauged"],
            "phase": snap["gauge_rms_phase_ungauged"],
            "kappa": snap["gauge_rms_kappa"],
            "kappa_true_scaled": snap["gauge_rms_kappa_true_scaled"],
            "kappa_hat_scaled": snap["gauge_rms_kappa_hat_scaled"],
        },
        rtol=SIM_RTOL,
        atol=SIM_ATOL,
        path="gauge",
    )
    got = {
        graph: payload["xi"]["summary"][graph]["xi_after"]["mean"]
        for graph in snap["xi_after_mean"]
    }
    assert_close(got, snap["xi_after_mean"], rtol=SIM_RTOL, atol=SIM_ATOL, path="xi_after")


def test_v2_small_reproduces_fit_and_ring_mse(tmp_path):
    suite = load_suite("v2")
    snap = load_snapshot("v2_key_numbers.json")
    with quiet():
        payload = suite.run_suite(suite.SimConfig.from_size("small"), tmp_path / "v2")
    fit = payload["law"]["fits"]["global"]
    assert fit["n"] == snap["global_fit"]["n"]
    assert_close(fit["c_floor"], snap["global_fit"]["c_floor"], rtol=SIM_RTOL, atol=SIM_ATOL, path="c_floor")
    assert_close(fit["r2"], snap["global_fit"]["r2"], rtol=SIM_RTOL, atol=SIM_ATOL, path="r2")
    assert_close(
        fit["c_layout"],
        snap["global_fit"]["c_layout"],
        rtol=V2_FIT_RTOL,
        atol=SIM_ATOL,
        path="c_layout",
    )
    for snr, by_k in snap["nstar_measured"].items():
        for k, published in by_k.items():
            got = payload["law"]["nstar"][float(snr)][int(k)]["from_formula_floor"]["nstar"]
            assert_close(got, published, rtol=5e-6, atol=SIM_ATOL, path=f"nstar[{snr}][{k}]")
    block = payload["law"]["summary"][20.0][64]["ring"]
    index = list(block["N"]).index(64)
    assert_close(
        block["mse_kappa"]["mean"][index],
        snap["ring_mse_snr20_k64_n64"],
        rtol=SIM_RTOL,
        atol=SIM_ATOL,
        path="ring_mse",
    )
    for snr, theory in snap["nstar_theory"].items():
        assert_close(
            payload["law"]["nstar"][float(snr)]["theory"],
            theory,
            rtol=SIM_RTOL,
            atol=0.0,
            path=f"nstar[{snr}]",
        )


def test_v3_small_reproduces_measured_floor_and_oracle(tmp_path):
    suite = load_suite("v3")
    snap = load_snapshot("v3_key_numbers.json")
    with quiet():
        payload = suite.run_suite(suite.SimConfig.from_size("small"), tmp_path / "v3")
    meta = payload["meta"]
    assert_close(meta["r_meas_20"], snap["r_meas_20"], rtol=SIM_RTOL, atol=SIM_ATOL, path="r_meas_20")
    assert_close(meta["r_meas_10"], snap["r_meas_10"], rtol=SIM_RTOL, atol=SIM_ATOL, path="r_meas_10")
    assert meta["r_formula_20"] == snap["r_formula_20"]
    got = payload["results"]["stationary"]["late"]["oracle Kalman"]["rms_kappa"]
    assert_close(
        got,
        snap["stationary_oracle_rms_kappa"],
        rtol=SIM_RTOL,
        atol=SIM_ATOL,
        path="oracle_rms",
    )


def test_v4_small_reproduces_static_mse(tmp_path):
    suite = load_suite("v4")
    snap = load_snapshot("v4_key_numbers.json")
    with quiet():
        payload = suite.run_suite(
            suite.SimConfig.from_size("small"),
            tmp_path / "v4",
            backend="numpy",
            do_agreement=False,
            do_timing=False,
        )
    got = {}
    for key, row in payload["experiment_a"]["static_r"].items():
        name = "|".join(str(part) for part in key) if isinstance(key, tuple) else str(key)
        got[name] = row["mse_kappa"]["mean"]
    assert_close(got, snap["static_mse_kappa_mean"], rtol=SIM_RTOL, atol=SIM_ATOL, path="static_mse")
