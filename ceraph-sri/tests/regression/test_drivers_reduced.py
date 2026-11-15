"""Driver runs at reduced size.

Deterministic fields (closed forms, resistances, steady populations) are
compared to the published snapshots. Monte Carlo grids that only match at
the published size are covered by the slow tests.
"""
from __future__ import annotations

from sri.graphs import ring_mean_resistance, torus_mean_resistance
from tests.regression.support import (
    SIM_ATOL,
    SIM_RTOL,
    assert_close,
    load_snapshot,
    load_suite,
    quiet,
)


def _static_key(key) -> str:
    if isinstance(key, tuple):
        return "|".join(str(part) for part in key)
    return str(key)


def test_v1_reduced_driver_keeps_theoretical_xi(tmp_path):
    suite = load_suite("v1")
    snap = load_snapshot("v1_key_numbers.json")
    cfg = suite.SimConfig.from_size(
        "small",
        n_list=(16,),
        n_fixed=16,
        n_trials=2,
        k_default=16,
        k_list=(16,),
        snr_list_db=(20.0,),
        t_iter=30,
        t_track=40,
    )
    with quiet():
        payload = suite.run_suite(cfg, tmp_path / "v1", n_track_walks=1)
    assert payload["theoretical_xi"] == snap["theoretical_xi"]
    assert payload["xi"]["xi_theory"] == snap["xi_theory"]
    gauge = payload["gauge"]
    assert_close(
        gauge["rms_kappa_gauged_true_scaled"],
        gauge["rms_kappa_gauged"],
        rtol=0.0,
        atol=1e-12,
        path="gauge_true_scaled",
    )
    assert_close(
        gauge["rms_kappa_gauged_hat_scaled"],
        gauge["rms_kappa_gauged"],
        rtol=0.0,
        atol=1e-12,
        path="gauge_hat_scaled",
    )


def test_v2_reduced_driver_matches_published_law_constants(tmp_path):
    suite = load_suite("v2")
    snap = load_snapshot("v2_key_numbers.json")
    cfg = suite.SimConfig.from_size(
        "small",
        n_list=(16, 36),
        n_trials=2,
        k_law=(16,),
        snr_law_db=(20.0,),
        n_list_overlap=(25,),
        rho_list=(1.0,),
        snr_overlap_db=(20.0,),
        n_walks=1,
        t_track=80,
        t_iter=40,
        t_conv=40,
        n_conv=16,
        n_fixed=16,
        n_track=16,
    )
    with quiet():
        payload = suite.run_suite(cfg, tmp_path / "v2")
    assert_close(payload["moments"], snap["moments"], rtol=SIM_RTOL, atol=0.0, path="moments")
    assert_close(
        payload["law"]["nstar"][20.0]["theory"],
        snap["nstar_theory"]["20.0"],
        rtol=SIM_RTOL,
        atol=0.0,
        path="nstar20",
    )
    for n_key, published in snap["ring_rbar"].items():
        n = int(n_key)
        if n not in payload["law"]["rbar"]:
            assert ring_mean_resistance(n) == published or abs(
                ring_mean_resistance(n) - published
            ) <= SIM_ATOL + SIM_RTOL * abs(published)
            continue
        assert_close(
            payload["law"]["rbar"][n]["ring"],
            published,
            rtol=SIM_RTOL,
            atol=0.0,
            path=f"ring_rbar[{n}]",
        )
    for n_key, published in (("16", snap["torus_rbar"]["16"]), ("36", snap["torus_rbar"]["36"])):
        assert_close(
            payload["law"]["rbar"][int(n_key)]["torus"],
            published,
            rtol=SIM_RTOL,
            atol=0.0,
            path=f"torus_rbar[{n_key}]",
        )
        assert_close(torus_mean_resistance(int(n_key)), published, rtol=SIM_RTOL, atol=0.0)


def test_v3_reduced_driver_matches_measurement_floor(tmp_path):
    suite = load_suite("v3")
    snap = load_snapshot("v3_key_numbers.json")
    cfg = suite.SimConfig.from_size(
        "small",
        n_track=16,
        k_track=16,
        t_track=80,
        n_walks=1,
        n_static=1,
        forget_factors=(0.97,),
    )
    with quiet():
        payload = suite.run_suite(cfg, tmp_path / "v3")
    assert payload["meta"]["r_formula_20"] == snap["r_formula_20"]
    assert payload["meta"]["r_formula_10"] == snap["r_formula_10"]
    assert payload["cfg"]["seed"] == snap["seed"]


def test_v4_reduced_driver_matches_process_noise_and_floors(tmp_path):
    suite = load_suite("v4")
    snap = load_snapshot("v4_key_numbers.json")
    cfg = suite.SimConfig.from_size(
        "small",
        n_track=16,
        t_track=80,
        n_walks=1,
        n_static=1,
        n_gauge=16,
        n_gauge_trials=1,
        bench_steps=4,
        bench_walks=1,
        bench_n=16,
    )
    with quiet():
        payload = suite.run_suite(
            cfg, tmp_path / "v4", backend="numpy", do_agreement=False, do_timing=False
        )
    assert payload["xi_theory"] == snap["xi_theory"]
    assert_close(payload["process_q"], snap["process_q"], rtol=0.0, atol=0.0, path="process_q")
    seen = set()
    for key, row in payload["experiment_a"]["static_r"].items():
        name = _static_key(key)
        seen.add(name)
        assert_close(
            row["formula"],
            snap["static_formula"][name],
            rtol=0.0,
            atol=0.0,
            path=f"formula[{name}]",
        )
    assert seen == set(snap["static_formula"])


def test_v5_driver_reproduces_selection_table(tmp_path):
    suite = load_suite("v5")
    snap = load_snapshot("v5_key_numbers.json")
    cfg = suite.SimConfig.from_size("small")
    with quiet():
        payload = suite.run_suite(
            cfg, tmp_path / "v5", backend="numpy", do_agreement=False, do_timing=False
        )
    for row in payload["exp_selection"]["theory"]:
        key = str(row["d"])
        assert_close(row["r_star"], snap["r_star"][key], rtol=0.0, atol=0.0, path=f"r_star[{key}]")
        assert_close(
            row["r_empirical"],
            snap["r_empirical"][key],
            rtol=SIM_RTOL,
            atol=SIM_ATOL,
            path=f"r_empirical[{key}]",
        )
        assert row["f_at_rstar"] == snap["f_at_rstar"][key]
    for row in payload["exp_selection"]["markov"]:
        if abs(float(row["r"]) - 2.0) >= 1e-12:
            continue
        key = str(row["d"])
        assert_close(
            {"F_ss_theory": row["F_ss_theory"], "F_ss_numeric": row["F_ss_numeric"]},
            snap["f_ss_at_r2"][key],
            rtol=SIM_RTOL,
            atol=SIM_ATOL,
            path=f"f_ss[{key}]",
        )


def test_v6_driver_reproduces_closed_form_slice(tmp_path):
    suite = load_suite("v6")
    snap = load_snapshot("v6_key_numbers.json")
    cfg = suite.SimConfig.from_size("small")
    with quiet():
        payload = suite.run_suite(
            cfg, tmp_path / "v6", backend="numpy", do_agreement=False, do_timing=False
        )
    interval = list(payload["exp_squeezing"]["interval0"])
    # One ulp of libm noise showed up on the CI runner (delta 1e-16).
    assert_close(interval, snap["interval0"], rtol=0.0, atol=1e-12, path="interval0")
    assert_close(payload["exp_squeezing"]["vs_k"], snap["vs_k"], rtol=0.0, atol=1e-12, path="vs_k")
    assert_close(
        payload["exp_wiring"]["ipr_below"]["ipr_mean"],
        snap["ipr_below_mean"],
        rtol=SIM_RTOL,
        atol=SIM_ATOL,
        path="ipr_below",
    )
    assert_close(
        payload["exp_wiring"]["ipr_above"]["ipr_mean"],
        snap["ipr_above_mean"],
        rtol=SIM_RTOL,
        atol=SIM_ATOL,
        path="ipr_above",
    )
