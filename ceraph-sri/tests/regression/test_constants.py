"""Closed forms that the published runs record."""
from __future__ import annotations

from sri.errorlaw import expected_veil_moment, ring_nstar_theory, snr_linear
from sri.selection import f_ss_depolarizing, r_threshold
from tests.regression.support import assert_close, load_snapshot


class _Moments:
    def e_tau_inv_pow(self, power: int) -> float:
        return expected_veil_moment(0.25, power)


def test_selection_steady_population_and_threshold():
    published = {2: 4.0, 3: 5.667, 4: 6.5, 8: 7.75}
    for dimension, shown in published.items():
        value = r_threshold(0.9, dimension)
        assert round(value, 3) == shown
        assert abs(value - (0.9 - 1.0 / dimension) / 0.1) <= 1e-12
    for dimension in (2, 3, 4, 8):
        for rate in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
            assert f_ss_depolarizing(rate, dimension) == (rate + 1.0 / dimension) / (rate + 1.0)


def test_selection_snapshot_matches_the_closed_form():
    snap = load_snapshot("v5_key_numbers.json")
    assert snap["xi_crit"] == 0.9
    for dimension, shown in {2: 4.0, 3: 5.667, 4: 6.5, 8: 7.75}.items():
        key = str(dimension)
        assert round(snap["r_star"][key], 3) == shown
        assert snap["f_at_rstar"][key] == 0.9
        row = snap["f_ss_at_r2"][key]
        assert row["F_ss_theory"] == f_ss_depolarizing(2.0, dimension)
        assert row["F_ss_theory"] == (2.0 + 1.0 / dimension) / 3.0


def test_error_law_crossover_is_about_twelve_times_snr():
    snap = load_snapshot("v2_key_numbers.json")
    cfg = _Moments()
    for snr_db in (0.0, 5.0, 10.0, 20.0):
        theory = ring_nstar_theory(cfg, snr_db)
        published = snap["nstar_theory"][str(snr_db)]
        assert theory["nstar_12snr"] == 12.0 * snr_linear(snr_db)
        assert_close(theory, published, rtol=1e-6, atol=0.0, path=f"nstar[{snr_db}]")
    assert snap["moments"]["e_inv2"] == expected_veil_moment(0.25, 2)
    assert snap["moments"]["e_inv4"] == expected_veil_moment(0.25, 4)
    assert snap["moments"]["factor"] == snap["moments"]["e_inv2"] / snap["moments"]["e_inv4"]
