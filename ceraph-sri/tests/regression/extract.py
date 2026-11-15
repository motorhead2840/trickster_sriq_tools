"""Pull small key-number records out of the published JSON artifacts.

The large reference files stay in place. These records are what the
regression tests compare against.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _statuses(payload: dict) -> dict[str, str]:
    return {item["conjecture_id"]: item["status"] for item in payload["conjectures"]}


def snapshot_v1() -> dict:
    data = _load("experiments/unveiling/v1/reference_results.json")
    gauge = data["gauge"]
    return {
        "source": "experiments/unveiling/v1/reference_results.json",
        "size": data["size"],
        "theoretical_xi": data["theoretical_xi"],
        "xi_theory": data["xi"]["xi_theory"],
        "gauge_rms_amp_ungauged": gauge["rms_amp_ungauged"],
        "gauge_rms_phase_ungauged": gauge["rms_phase_ungauged"],
        "gauge_rms_kappa": gauge["rms_kappa_gauged"],
        "gauge_rms_kappa_true_scaled": gauge["rms_kappa_gauged_true_scaled"],
        "gauge_rms_kappa_hat_scaled": gauge["rms_kappa_gauged_hat_scaled"],
        "xi_after_mean": {
            graph: data["xi"]["summary"][graph]["xi_after"]["mean"]
            for graph in ("ring", "torus", "expander", "svd")
        },
    }


def snapshot_v2() -> dict:
    data = _load("experiments/unveiling/v2/reference_results.json")
    theory = {}
    for snr, block in data["law"]["nstar"].items():
        theory[str(snr)] = block["theory"]
    ring = {}
    torus = {}
    for n, kinds in data["law"]["rbar"].items():
        ring[str(n)] = kinds["ring"]
        if "torus" in kinds:
            torus[str(n)] = kinds["torus"]
    fit = data["law"]["fits"]["global"]
    ring_block = data["law"]["summary"]["20.0"]["64"]["ring"]
    n_values = ring_block["N"]
    index = n_values.index(64)
    measured = {}
    for snr, block in data["law"]["nstar"].items():
        crossings = {}
        for key, row in block.items():
            if key == "theory" or not isinstance(row, dict):
                continue
            nstar = row.get("from_formula_floor", {}).get("nstar")
            if isinstance(nstar, (int, float)):
                crossings[str(key)] = nstar
        if crossings:
            measured[str(snr)] = crossings
    return {
        "source": "experiments/unveiling/v2/reference_results.json",
        "theoretical_xi": data["theoretical_xi"],
        "moments": data["moments"],
        "nstar_theory": theory,
        "nstar_measured": measured,
        "ring_rbar": ring,
        "torus_rbar": torus,
        "global_fit": {
            "c_floor": fit["c_floor"],
            "c_layout": fit["c_layout"],
            "r2": fit["r2"],
            "n": fit["n"],
        },
        "ring_mse_snr20_k64_n64": ring_block["mse_kappa"]["mean"][index],
    }


def snapshot_v3() -> dict:
    data = _load("experiments/unveiling/v3/reference_results.json")
    meta = data["meta"]
    oracle = data["results"]["stationary"]["late"]["oracle Kalman"]["rms_kappa"]
    return {
        "source": "experiments/unveiling/v3/reference_results.json",
        "seed": data["cfg"]["seed"],
        "r_formula_20": meta["r_formula_20"],
        "r_formula_10": meta["r_formula_10"],
        "r_meas_20": meta["r_meas_20"],
        "r_meas_10": meta["r_meas_10"],
        "stationary_oracle_rms_kappa": oracle,
    }


def snapshot_v4() -> dict:
    data = _load("experiments/unveiling/v4/reference_results.json")
    formulas = {}
    measured = {}
    for key, row in data["experiment_a"]["static_r"].items():
        formulas[key] = row["formula"]
        measured[key] = row["mse_kappa"]["mean"]
    return {
        "source": "experiments/unveiling/v4/reference_results.json",
        "xi_theory": data["xi_theory"],
        "process_q": data["process_q"],
        "static_formula": formulas,
        "static_mse_kappa_mean": measured,
    }


def snapshot_v5() -> dict:
    data = _load("experiments/unveiling/v5/reference_results.json")
    theory = data["exp_selection"]["theory"]
    at_r2 = {}
    for row in data["exp_selection"]["markov"]:
        if abs(float(row["r"]) - 2.0) < 1e-12:
            at_r2[str(row["d"])] = {
                "F_ss_theory": row["F_ss_theory"],
                "F_ss_numeric": row["F_ss_numeric"],
            }
    return {
        "source": "experiments/unveiling/v5/reference_results.json",
        "xi_crit": data["config"]["xi_crit"],
        "r_star": {str(row["d"]): row["r_star"] for row in theory},
        "r_empirical": {str(row["d"]): row["r_empirical"] for row in theory},
        "f_at_rstar": {str(row["d"]): row["f_at_rstar"] for row in theory},
        "f_ss_at_r2": at_r2,
    }


def snapshot_v6() -> dict:
    data = _load("experiments/unveiling/v6/reference_results.json")
    squeezing = data["exp_squeezing"]
    k_grid = list(squeezing["k_grid"])
    index = k_grid.index(0.5)
    return {
        "source": "experiments/unveiling/v6/reference_results.json",
        "interval0": list(squeezing["interval0"]),
        "vs_k": squeezing["vs_k"],
        "vs_k_at_0_5": {name: values[index] for name, values in squeezing["vs_k"].items()},
        "ipr_below_mean": data["exp_wiring"]["ipr_below"]["ipr_mean"],
        "ipr_above_mean": data["exp_wiring"]["ipr_above"]["ipr_mean"],
    }


def snapshot_observable() -> dict:
    data = _load("docs/benchmark-results/cv-observable/observable_only_summary.json")
    return {
        "source": "docs/benchmark-results/cv-observable/observable_only_summary.json",
        "analytic_mse": data["per_seed"][0]["mmse_two_mode_analytic_mse"],
        "weights": data["per_seed"][0]["weights"],
        "aggregate": data["aggregate"],
        "sample_mse": {
            str(row["seed"]): row["mmse_two_mode_sample_mse"] for row in data["per_seed"]
        },
        "oracle_paired_mse": {
            str(row["seed"]): row["oracle_paired_mse"] for row in data["per_seed"]
        },
        "statuses": _statuses(data),
    }


def snapshot_foundation() -> dict:
    data = _load("docs/benchmark-results/foundation/foundation_axioms_summary.json")
    return {
        "source": "docs/benchmark-results/foundation/foundation_axioms_summary.json",
        "aggregate": data["aggregate"],
        "statuses": _statuses(data),
        "records_count": data["records_count"],
    }


def snapshot_e1() -> dict:
    data = _load("docs/benchmark-results/e1_colab_lock.json")
    return {
        "source": "docs/benchmark-results/e1_colab_lock.json",
        "circuit_b_research_default": data["circuit_b_research_default"],
        "shots": data["configuration"]["shots"],
        "seeds": data["configuration"]["seeds"],
        "dimension": data["configuration"]["dimension"],
        "exit": data["colab_exit_criteria"],
        "observable_aggregate": data["aggregates"]["observable"],
        "foundation_aggregate": data["aggregates"]["foundation"],
        "statuses": _statuses(data),
    }


def snapshot_conjectures() -> dict:
    data = _load("docs/benchmark-results/conjecture_status.json")
    return {
        "source": "docs/benchmark-results/conjecture_status.json",
        "statuses": _statuses(data),
    }


SNAPSHOTS = {
    "v1_key_numbers.json": snapshot_v1,
    "v2_key_numbers.json": snapshot_v2,
    "v3_key_numbers.json": snapshot_v3,
    "v4_key_numbers.json": snapshot_v4,
    "v5_key_numbers.json": snapshot_v5,
    "v6_key_numbers.json": snapshot_v6,
    "observable_only.json": snapshot_observable,
    "foundation_axioms.json": snapshot_foundation,
    "e1_colab_lock.json": snapshot_e1,
    "conjecture_status.json": snapshot_conjectures,
}


def write_snapshots(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, builder in SNAPSHOTS.items():
        payload = builder()
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        (directory / name).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "snapshots"
    write_snapshots(out)
    print(f"wrote {len(SNAPSHOTS)} snapshots to {out}")
