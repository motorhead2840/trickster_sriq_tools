"""Reproduction driver for the v2 round.

Numerical kernels come from the sri package. This file keeps the
experiment loop, the seeds, and the published defaults. Plots and the
optional torch path are not run.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from sri.errorlaw import (
    estimate_nstar_from_curve,
    fit_two_term_law,
    predicted_terms,
    ring_nstar_theory,
    snr_linear,
)
from sri.graphs import (
    is_connected,
    is_perfect_square,
    laplacian_eigbounds,
    make_graph,
    mean_resistance,
    ring_mean_resistance,
)
from sri.tracking import steady_state_kalman_gain
from sri.unveiling import (
    _connection_matrix,
    _symmetrized_log_amp,
    amplitude_chebyshev,
    assign_windows,
    edge_ratios_from_windows,
    estimate_centralized_svd,
    estimate_decentralized,
    gauge_align,
    generate_observations,
    generate_windowed_observations,
    kappa_errors,
    relative_veil_matrix,
    residual_veils,
    sample_veils,
    used_adjacency,
    xi_empirical,
    xi_from_consistency,
)

def _plot_stub(*_args, **_kwargs):
    """Plots are not written by the reproduction driver."""
    return {}

plot_convergence = _plot_stub

plot_error_law = _plot_stub

plot_partial_overlap = _plot_stub

plot_tracking = _plot_stub

ComplexArr = NDArray[np.complex128]
FloatArr = NDArray[np.float64]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# SIZE = "small" is meant to finish on a CPU in well under 5 minutes.
# SIZE = "large" is a denser sweep for a longer CPU run.
SIZE_PRESETS = {
    "small": dict(
        n_list=(16, 36, 64, 100, 144, 225, 400, 576, 784, 1024),
        n_trials=6,
        k_law=(16, 64, 256),
        snr_law_db=(0.0, 5.0, 10.0, 20.0),
        k_overlap=64,
        rho_list=(1.0, 0.75, 0.5, 0.25),
        n_list_overlap=(25, 36, 64, 100, 144, 225, 400, 576),
        snr_overlap_db=(10.0, 20.0),
        n_fixed=64,
        n_track=64,
        k_track=16,
        t_track=250,
        n_walks=6,
        t_iter=400,
        n_conv=64,
        t_conv=800,
    ),
    "large": dict(
        n_list=(
            16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225,
            256, 324, 400, 484, 576, 676, 784, 900, 1024,
        ),
        n_trials=16,
        k_law=(16, 64, 256),
        snr_law_db=(0.0, 5.0, 10.0, 20.0),
        k_overlap=64,
        rho_list=(1.0, 0.75, 0.5, 0.25),
        n_list_overlap=(25, 36, 64, 100, 144, 225, 400, 576, 784, 1024),
        snr_overlap_db=(0.0, 10.0, 20.0),
        n_fixed=64,
        n_track=64,
        k_track=16,
        t_track=400,
        n_walks=12,
        t_iter=800,
        n_conv=144,
        t_conv=2000,
    ),
}

DEFAULT_S_AMP = 0.25
DEFAULT_S_PHASE = 0.40
EXPANDER_DEGREE = 4
BASE_SEED = 7
OU_RHO = 0.97  # same as v1

GRAPH_ORDER = ("ring", "torus", "expander")
GRAPH_LABELS = {
    "ring": "ring (1-D cycle)",
    "torus": "2-D torus",
    "expander": "4-regular expander",
    "svd": "centralized SVD",
}
GRAPH_COLORS = {
    "ring": "#1f77b4",
    "torus": "#ff7f0e",
    "expander": "#2ca02c",
    "svd": "#111111",
}
RHO_COLORS = {
    1.0: "#1f77b4",
    0.75: "#ff7f0e",
    0.5: "#2ca02c",
    0.25: "#d62728",
}


@dataclass
class SimConfig:
    size: str = "small"
    s_amp: float = DEFAULT_S_AMP
    s_phase: float = DEFAULT_S_PHASE
    expander_degree: int = EXPANDER_DEGREE
    seed: int = BASE_SEED
    n_list: tuple[int, ...] = SIZE_PRESETS["small"]["n_list"]
    n_trials: int = 6
    k_law: tuple[int, ...] = SIZE_PRESETS["small"]["k_law"]
    snr_law_db: tuple[float, ...] = SIZE_PRESETS["small"]["snr_law_db"]
    k_overlap: int = 64
    rho_list: tuple[float, ...] = SIZE_PRESETS["small"]["rho_list"]
    n_list_overlap: tuple[int, ...] = SIZE_PRESETS["small"]["n_list_overlap"]
    snr_overlap_db: tuple[float, ...] = SIZE_PRESETS["small"]["snr_overlap_db"]
    n_fixed: int = 64
    n_track: int = 64
    k_track: int = 16
    t_track: int = 250
    n_walks: int = 6
    t_iter: int = 400
    n_conv: int = 64
    t_conv: int = 800

    @classmethod
    def from_size(cls, size: str, **overrides) -> "SimConfig":
        if size not in SIZE_PRESETS:
            raise ValueError(f"SIZE must be 'small' or 'large', got {size!r}")
        data = dict(SIZE_PRESETS[size])
        data.update(overrides)
        return cls(size=size, **data)

    def theoretical_xi(self) -> float:
        return float(np.exp(-(self.s_amp**2 + self.s_phase**2)))

    def e_tau_inv_pow(self, p: int) -> float:
        """E[|τ|^(-p)] for τ = exp(η + iξ), η ~ N(0, s_a²). Equals exp(p² s_a² / 2).

        |τ|^{-p} = exp(-p η), so E = exp(p² s_a² / 2).
        For p=2: exp(2 s_a²). For p=4: exp(8 s_a²).
        """
        return float(np.exp((p**2) * self.s_amp**2 / 2.0))


# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------





















# ---------------------------------------------------------------------------
# Effective resistance  R̄(G) = trace(L⁺)/N
# ---------------------------------------------------------------------------









# ---------------------------------------------------------------------------
# Observation model (full overlap and windowed partial overlap)
# ---------------------------------------------------------------------------



























# ---------------------------------------------------------------------------
# Relative veils (full overlap, fast path)
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Estimators (sparse LS / eigensolves for large N)
# ---------------------------------------------------------------------------



















# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------













def _summarize(values) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return dict(mean=float("nan"), std=float("nan"), sem=float("nan"), median=float("nan"), n=0)
    return dict(
        mean=float(arr.mean()),
        std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        sem=float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0,
        median=float(np.median(arr)),
        n=int(arr.size),
    )


# ---------------------------------------------------------------------------
# Error-law helpers
# ---------------------------------------------------------------------------












# ---------------------------------------------------------------------------
# Graph cache (one expander per N, so R̄ is computed once)
# ---------------------------------------------------------------------------

def build_graph_cache(n_values, cfg: SimConfig, rng: Generator) -> dict:
    """For each N, one ring, one torus (if square), one expander; plus R̄ and eigbounds."""
    cache = {}
    for n in n_values:
        cache[n] = {}
        for kind in GRAPH_ORDER:
            if kind == "torus" and not is_perfect_square(n):
                continue
            g_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))
            A = make_graph(kind, n, g_rng, degree=cfg.expander_degree)
            rbar = mean_resistance(A, kind=kind)
            bounds = laplacian_eigbounds(A)
            cache[n][kind] = dict(A=A, rbar=rbar, bounds=bounds)
    return cache


# ---------------------------------------------------------------------------
# Experiment A — error-law test
# ---------------------------------------------------------------------------

def _trial_errors_full_overlap(tau, Y, cache_n, cfg) -> dict:
    """One Y, all graphs in the cache plus SVD."""
    out = {}
    tau_hat_svd = estimate_centralized_svd(Y)
    out["svd"] = kappa_errors(tau_hat_svd, tau)
    for kind, info in cache_n.items():
        A = info["A"]
        r = relative_veil_matrix(Y, A)
        tau_hat = estimate_decentralized(r, A, mode="converged")
        out[kind] = kappa_errors(tau_hat, tau)
        out[kind]["xi_after"] = xi_empirical(out[kind]["residual"])
    out["svd"]["xi_after"] = xi_empirical(out["svd"]["residual"])
    out["xi_true"] = xi_empirical(tau)
    return out


def experiment_error_law(cfg: SimConfig, rng: Generator, progress: Callable | None = None) -> dict:
    """Exp A: MSE vs N, SNR, K for ring/torus/expander/SVD, vs the predicted two-term law."""
    n_list = tuple(cfg.n_list)
    k_list = tuple(cfg.k_law)
    snr_list = tuple(cfg.snr_law_db)
    cache = build_graph_cache(n_list, cfg, np.random.default_rng(rng.integers(0, 2**63 - 1)))
    rbar = {n: {kind: cache[n][kind]["rbar"] for kind in cache[n]} for n in n_list}

    # records[snr][k][kind][n] = list of mse/rms
    metrics = ("mse_kappa", "mse_amp", "mse_phase", "rms_kappa", "rms_amp", "rms_phase", "xi_after")
    records = {}
    for snr in snr_list:
        records[snr] = {}
        for k in k_list:
            records[snr][k] = {}
            kinds = ("ring", "torus", "expander", "svd")
            for kind in kinds:
                records[snr][k][kind] = {n: {m: [] for m in metrics} for n in n_list}

    n_jobs = len(n_list) * len(k_list) * len(snr_list) * cfg.n_trials
    done = 0
    t0 = time.perf_counter()
    for n in n_list:
        for k in k_list:
            for snr in snr_list:
                for _ in range(cfg.n_trials):
                    trial_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))
                    tau, _ = sample_veils(n, cfg.s_amp, cfg.s_phase, trial_rng)
                    Y, _psi = generate_observations(tau, k, snr, trial_rng)
                    errs = _trial_errors_full_overlap(tau, Y, cache[n], cfg)
                    for kind in records[snr][k]:
                        if kind not in errs:
                            continue
                        e = errs[kind]
                        rec = records[snr][k][kind][n]
                        for m in ("mse_kappa", "mse_amp", "mse_phase", "rms_kappa", "rms_amp", "rms_phase"):
                            rec[m].append(e[m])
                        rec["xi_after"].append(e.get("xi_after", np.nan))
                    done += 1
                    if progress and (done % 20 == 0 or done == n_jobs):
                        progress(done, n_jobs, time.perf_counter() - t0)

    # summarize + attach predicted terms
    summary = {}
    rows_for_fit = []  # global fit over graphs (not SVD) and all SNR, K, N
    for snr in snr_list:
        summary[snr] = {}
        for k in k_list:
            summary[snr][k] = {}
            for kind in records[snr][k]:
                N_used = []
                block = {"N": [], "rbar": []}
                for m in metrics:
                    block[m] = {"mean": [], "sem": []}
                block["floor"] = []
                block["layout"] = []
                block["predicted_mse"] = []
                for n in n_list:
                    if kind != "svd" and kind not in cache[n]:
                        continue
                    rec = records[snr][k][kind][n]
                    if not rec["mse_kappa"]:
                        continue
                    N_used.append(n)
                    block["N"].append(n)
                    rb = 0.0 if kind == "svd" else cache[n][kind]["rbar"]
                    block["rbar"].append(rb)
                    terms = predicted_terms(cfg, k, snr, rb)
                    block["floor"].append(terms["floor"])
                    block["layout"].append(terms["layout"])
                    block["predicted_mse"].append(terms["predicted_mse"])
                    for m in metrics:
                        st = _summarize(rec[m])
                        block[m]["mean"].append(st["mean"])
                        block[m]["sem"].append(st["sem"])
                    if kind != "svd":
                        st = _summarize(rec["mse_kappa"])
                        rows_for_fit.append(dict(
                            snr=snr, k=k, n=n, kind=kind,
                            x1=terms["floor"], x2=terms["layout"],
                            y=st["mean"], w=1.0 / max(st["sem"] ** 2, 1e-18),
                        ))
                summary[snr][k][kind] = block

    # fits: global, per-SNR, per-K, per-graph
    def _fit_from(pred):
        if not pred:
            return dict(c_floor=np.nan, c_layout=np.nan, r2=np.nan, n=0)
        x1 = np.array([r["x1"] for r in pred])
        x2 = np.array([r["x2"] for r in pred])
        y = np.array([r["y"] for r in pred])
        w = np.array([r["w"] for r in pred])
        return fit_two_term_law(x1, x2, y, w)

    fits = {"global": _fit_from(rows_for_fit), "by_snr": {}, "by_k": {}, "by_kind": {}}
    for snr in snr_list:
        fits["by_snr"][snr] = _fit_from([r for r in rows_for_fit if r["snr"] == snr])
    for k in k_list:
        fits["by_k"][k] = _fit_from([r for r in rows_for_fit if r["k"] == k])
    for kind in GRAPH_ORDER:
        fits["by_kind"][kind] = _fit_from([r for r in rows_for_fit if r["kind"] == kind])

    # N* from data vs theory, per SNR and K (ring)
    nstar = {}
    for snr in snr_list:
        nstar[snr] = {"theory": ring_nstar_theory(cfg, snr)}
        for k in k_list:
            ring_block = summary[snr][k].get("ring")
            if not ring_block or not ring_block["N"]:
                continue
            # floor from the formula (same for all graphs); also empirical floor from expander / SVD
            floor_formula = float(np.mean(ring_block["floor"]))
            floor_svd = float(np.mean(summary[snr][k]["svd"]["mse_kappa"]["mean"])) if "svd" in summary[snr][k] else floor_formula
            floor_exp = None
            if "expander" in summary[snr][k] and summary[snr][k]["expander"]["N"]:
                floor_exp = float(np.mean(summary[snr][k]["expander"]["mse_kappa"]["mean"]))
            est_formula = estimate_nstar_from_curve(
                np.array(ring_block["N"]), np.array(ring_block["mse_kappa"]["mean"]), floor_formula
            )
            est_svd = estimate_nstar_from_curve(
                np.array(ring_block["N"]), np.array(ring_block["mse_kappa"]["mean"]), floor_svd
            )
            # also from fitted coefficients: set c1 floor = c2 layout(N)
            # layout = e4/(K SNR²) * (N²-1)/(12N)
            terms0 = predicted_terms(cfg, k, snr, 1.0)
            c1 = fits["global"]["c_floor"]
            c2 = fits["global"]["c_layout"]
            nstar_from_fit = float("nan")
            if np.isfinite(c1) and np.isfinite(c2) and c2 > 0 and terms0["layout"] > 0:
                # c1 * e2/(K SNR) = c2 * e4/(K SNR²) * (N²-1)/(12N)
                # (N²-1)/N = 12 * SNR * (c1/c2) * (e2/e4)
                fac = terms0["e_inv2"] / terms0["e_inv4"]
                c = 12.0 * snr_linear(snr) * (c1 / c2) * fac
                nstar_from_fit = float(0.5 * (c + np.sqrt(c * c + 4.0)))
            nstar[snr][k] = dict(
                floor_formula=floor_formula,
                floor_svd=floor_svd,
                floor_expander=floor_exp,
                from_formula_floor=est_formula,
                from_svd_floor=est_svd,
                from_fitted_c=nstar_from_fit,
            )

    elapsed = time.perf_counter() - t0
    return dict(
        n_list=list(n_list),
        k_list=list(k_list),
        snr_list=list(snr_list),
        rbar={int(n): {k: float(v) for k, v in rbar[n].items()} for n in n_list},
        summary=summary,
        fits=fits,
        nstar=nstar,
        moments=dict(
            e_inv2=cfg.e_tau_inv_pow(2),
            e_inv4=cfg.e_tau_inv_pow(4),
            factor=cfg.e_tau_inv_pow(2) / cfg.e_tau_inv_pow(4),
            xi_theory=cfg.theoretical_xi(),
        ),
        elapsed_s=elapsed,
        n_trials=cfg.n_trials,
    )


# ---------------------------------------------------------------------------
# Experiment B — partial overlap
# ---------------------------------------------------------------------------

def experiment_partial_overlap(cfg: SimConfig, rng: Generator, progress: Callable | None = None) -> dict:
    """Exp B: sliding-window (or greedy) overlap ρ, error vs N per layout.

    Prediction: at ρ<1 the first-order noise a_n^{(nm)} depends on the edge, so
    it is no longer a pure coboundary, and layout should matter at smaller N.
    """
    n_list = tuple(cfg.n_list_overlap)
    rho_list = tuple(cfg.rho_list)
    snr_list = tuple(cfg.snr_overlap_db)
    k = int(cfg.k_overlap)
    cache = build_graph_cache(n_list, cfg, np.random.default_rng(rng.integers(0, 2**63 - 1)))

    metrics = ("mse_kappa", "rms_kappa", "rms_amp", "rms_phase", "min_shared", "mean_shared")
    # records[snr][rho][kind][n]
    records = {}
    for snr in snr_list:
        records[snr] = {}
        for rho in rho_list:
            records[snr][rho] = {kind: {n: {m: [] for m in metrics} for n in n_list} for kind in GRAPH_ORDER}

    # pre-assign windows once per (kind, n, rho) so the overlap design is fixed
    windows = {}
    w_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))
    for n in n_list:
        windows[n] = {}
        for kind in cache[n]:
            windows[n][kind] = {}
            for rho in rho_list:
                idx, L, info = assign_windows(kind, n, k, rho, cache[n][kind]["A"], w_rng)
                windows[n][kind][rho] = dict(idx=idx, L=L, info=info)

    n_jobs = len(n_list) * len(rho_list) * len(snr_list) * cfg.n_trials
    done = 0
    t0 = time.perf_counter()
    skipped = []
    for n in n_list:
        for rho in rho_list:
            for snr in snr_list:
                for _ in range(cfg.n_trials):
                    trial_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))
                    tau, _ = sample_veils(n, cfg.s_amp, cfg.s_phase, trial_rng)
                    for kind in cache[n]:
                        win = windows[n][kind][rho]
                        A = cache[n][kind]["A"]
                        Y, _stream = generate_windowed_observations(
                            tau, win["idx"], win["L"], snr, trial_rng
                        )
                        if rho >= 1.0 - 1e-12:
                            r = relative_veil_matrix(Y, A)
                            A_use = A
                            counts = np.full(A.shape, int(k), dtype=np.int32) * (A > 0)
                        else:
                            r, counts = edge_ratios_from_windows(Y, win["idx"], A)
                            A_use = used_adjacency(A, counts, min_count=1)
                        if not is_connected(A_use):
                            skipped.append(dict(n=n, rho=rho, snr=snr, kind=kind, reason="disconnected"))
                            continue
                        tau_hat = estimate_decentralized(r, A_use, mode="converged")
                        err = kappa_errors(tau_hat, tau)
                        rec = records[snr][rho][kind][n]
                        rec["mse_kappa"].append(err["mse_kappa"])
                        rec["rms_kappa"].append(err["rms_kappa"])
                        rec["rms_amp"].append(err["rms_amp"])
                        rec["rms_phase"].append(err["rms_phase"])
                        edge_counts = counts[np.triu(A, 1) > 0]
                        rec["min_shared"].append(int(edge_counts.min()) if edge_counts.size else 0)
                        rec["mean_shared"].append(float(edge_counts.mean()) if edge_counts.size else 0.0)
                    done += 1
                    if progress and (done % 10 == 0 or done == n_jobs):
                        progress(done, n_jobs, time.perf_counter() - t0)

    # summarize + N* vs ρ
    summary = {}
    nstar = {}
    for snr in snr_list:
        summary[snr] = {}
        nstar[snr] = {}
        for rho in rho_list:
            summary[snr][rho] = {}
            nstar[snr][rho] = {}
            for kind in GRAPH_ORDER:
                block = {"N": []}
                for m in metrics:
                    block[m] = {"mean": [], "sem": []}
                for n in n_list:
                    rec = records[snr][rho][kind][n]
                    if not rec["mse_kappa"]:
                        continue
                    block["N"].append(n)
                    for m in metrics:
                        st = _summarize(rec[m])
                        block[m]["mean"].append(st["mean"])
                        block[m]["sem"].append(st["sem"])
                summary[snr][rho][kind] = block
            # N* for the ring, floor from expander (or torus) at the same ρ
            ring_block = summary[snr][rho].get("ring")
            if ring_block and ring_block["N"]:
                # empirical floor: expander if present else torus, else smallest-N ring (weak)
                floor = None
                for src in ("expander", "torus"):
                    b = summary[snr][rho].get(src)
                    if b and b["N"] and b["mse_kappa"]["mean"]:
                        floor = float(np.mean(b["mse_kappa"]["mean"]))
                        break
                if floor is None:
                    floor = float(ring_block["mse_kappa"]["mean"][0])
                # also a formula floor that uses effective samples ρK
                floor_rhoK = predicted_terms(cfg, max(int(round(rho * k)), 1), snr, 0.0)["floor"]
                floor_fullK = predicted_terms(cfg, k, snr, 0.0)["floor"]
                est = estimate_nstar_from_curve(
                    np.array(ring_block["N"]), np.array(ring_block["mse_kappa"]["mean"]), floor
                )
                nstar[snr][rho] = dict(
                    floor_empirical=floor,
                    floor_rhoK=floor_rhoK,
                    floor_fullK=floor_fullK,
                    ring=est,
                )

    # pack window-info (no arrays)
    info_slim = {}
    for n in n_list:
        info_slim[n] = {}
        for kind in windows[n]:
            info_slim[n][kind] = {}
            for rho in rho_list:
                info_slim[n][kind][rho] = windows[n][kind][rho]["info"]

    return dict(
        n_list=list(n_list),
        rho_list=list(rho_list),
        snr_list=list(snr_list),
        k=k,
        summary=summary,
        nstar=nstar,
        window_info=info_slim,
        skipped=skipped[:50],
        n_skipped=len(skipped),
        elapsed_s=time.perf_counter() - t0,
        n_trials=cfg.n_trials,
        first_order_note=(
            "On shared samples S_nm = S_n ∩ S_m, the first-order term in log r_nm is "
            "a_n^{(nm)} - a_m^{(nm)}, with a_n^{(nm)} = ⟨u_n, ψ⟩_{S_nm} / ‖ψ‖_{S_nm}². "
            "This is a coboundary of a single node potential iff a_n^{(nm)} does not "
            "depend on the neighbour m, which holds when every neighbour pair sees the "
            "same samples (ρ=1). For a sliding window, S_nm is a different subset for "
            "each edge, so the 1-cochain generally has a nonzero curl around plaquettes."
        ),
    )


# ---------------------------------------------------------------------------
# Experiment C — Kalman vs fixed gains vs adaptive Ξ, including a q jump
# ---------------------------------------------------------------------------



def _static_measurement_variance(cfg: SimConfig, A: FloatArr, rng: Generator, n_trials: int = 8) -> dict:
    """Empirical gauge-fixed MSE of the instantaneous estimator, no drift, for r in the Kalman filter."""
    n = A.shape[0]
    acc = dict(mse_kappa=[], mse_amp=[], mse_phase=[])
    for _ in range(n_trials):
        tr = np.random.default_rng(rng.integers(0, 2**63 - 1))
        tau, _ = sample_veils(n, cfg.s_amp, cfg.s_phase, tr)
        Y, _ = generate_observations(tau, cfg.k_track, cfg.snr_law_db[-1] if False else 20.0, tr)
        # use the same SNR as tracking (passed in via a temporary attribute, default 20)
        r = relative_veil_matrix(Y, A)
        tau_hat = estimate_decentralized(r, A, mode="converged")
        err = kappa_errors(tau_hat, tau)
        acc["mse_kappa"].append(err["mse_kappa"])
        acc["mse_amp"].append(err["mse_amp"])
        acc["mse_phase"].append(err["mse_phase"])
    return {m: _summarize(acc[m]) for m in acc}


def experiment_tracking_kalman(
    cfg: SimConfig,
    rng: Generator,
    snr_db: float = 20.0,
    jump: bool = False,
    jump_factor: float = 10.0,
    n_walks: int | None = None,
) -> dict:
    """OU drift of κ_n(t) on a torus, tracked in the log domain after gauge alignment.

    Methods: fixed gains (0.05, 0.20, 0.60), adaptive g(Ξ) as in v1,
    Kalman with true (q, r), Kalman with q × 0.25 and q × 4.
    If jump=True, the process-noise variance q is multiplied by jump_factor
    at the midpoint; the 'Kalman (true q)' filter keeps the *initial* gain,
    and an oracle Kalman (retuned to the new q) is added for reference.
    """
    n = cfg.n_track
    k = cfg.k_track
    t_steps = cfg.t_track
    n_walks = int(n_walks if n_walks is not None else cfg.n_walks)
    rho_ou = OU_RHO
    graph_kind = "torus"
    fixed_gains = (0.05, 0.20, 0.60)
    g_min, g_max = 0.05, 0.60
    burn = min(50, t_steps // 5)
    t_jump = t_steps // 2 if jump else None

    A = make_graph(graph_kind, n, rng, degree=cfg.expander_degree)

    # measurement variance: formula floor and a short static measurement
    terms = predicted_terms(cfg, k, snr_db, 0.0)
    r_formula = terms["floor"]  # E|κ error|²
    static = _static_measurement_variance_at(cfg, A, snr_db, k, rng, n_trials=max(6, cfg.n_trials))
    r_meas = static["mse_kappa"]["mean"]
    r_amp = static["mse_amp"]["mean"]
    r_phase = static["mse_phase"]["mean"]
    # process noise of the OU: κ_{t+1} = ρ κ_t + w,  Var(w_η)=s_a²(1-ρ²), etc.
    q_amp = (cfg.s_amp**2) * (1.0 - rho_ou**2)
    q_phase = (cfg.s_phase**2) * (1.0 - rho_ou**2)
    q_kappa = q_amp + q_phase

    def gains_from_q(q_a, q_p, r_a, r_p):
        ka = steady_state_kalman_gain(rho_ou, q_a, max(r_a, 1e-18))
        kp = steady_state_kalman_gain(rho_ou, q_p, max(r_p, 1e-18))
        return ka, kp

    kalman_true = gains_from_q(q_amp, q_phase, r_amp, r_phase)
    kalman_q025 = gains_from_q(0.25 * q_amp, 0.25 * q_phase, r_amp, r_phase)
    kalman_q4 = gains_from_q(4.0 * q_amp, 4.0 * q_phase, r_amp, r_phase)
    kalman_jump = gains_from_q(jump_factor * q_amp, jump_factor * q_phase, r_amp, r_phase)
    # also a formula-r variant of true q (split r equally if needed)
    # split r_formula in the empirical amp/phase proportion
    tot = max(r_amp + r_phase, 1e-18)
    r_amp_f = r_formula * (r_amp / tot)
    r_phase_f = r_formula * (r_phase / tot)
    kalman_formula = gains_from_q(q_amp, q_phase, r_amp_f, r_phase_f)

    names = [f"fixed g={g:.2f}" for g in fixed_gains] + [
        "adaptive g(Ξ)",
        "Kalman true q,r",
        "Kalman q×0.25",
        "Kalman q×4",
    ]
    if jump:
        names = names + ["Kalman initial q (stuck)", "Kalman oracle after jump"]
        # "Kalman true q,r" will use initial q throughout as well in the jump
        # experiment; we keep both names: true-initial, and oracle.

    rms_kappa_all = {name: np.zeros((n_walks, t_steps)) for name in names}
    rms_amp_all = {name: np.zeros((n_walks, t_steps)) for name in names}
    rms_phase_all = {name: np.zeros((n_walks, t_steps)) for name in names}
    g_used_all = {name: np.zeros((n_walks, t_steps)) for name in names}

    t0 = time.perf_counter()
    for walk in range(n_walks):
        w_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))
        tau, kappa = sample_veils(n, cfg.s_amp, cfg.s_phase, w_rng)
        kappa_hat = {name: np.zeros(n, dtype=np.complex128) for name in names}
        for t in range(t_steps):
            if t > 0:
                scale = np.sqrt(jump_factor) if (jump and t_jump is not None and t >= t_jump) else 1.0
                innov_a = scale * cfg.s_amp * np.sqrt(1.0 - rho_ou**2) * w_rng.standard_normal(n)
                innov_p = scale * cfg.s_phase * np.sqrt(1.0 - rho_ou**2) * w_rng.standard_normal(n)
                kappa = rho_ou * kappa + innov_a + 1j * innov_p
                tau = np.exp(kappa)
            Y, _ = generate_observations(tau, k, snr_db, w_rng)
            rmat = relative_veil_matrix(Y, A)
            tau_inst = estimate_decentralized(rmat, A, mode="converged")

            for name in names:
                tau_held = np.exp(kappa_hat[name])
                inst_aligned, _ = gauge_align(tau_inst, tau_held if t > 0 else tau_inst)
                kappa_inst = np.log(inst_aligned)
                # method-specific predict + gain
                if name.startswith("fixed"):
                    g = float(name.split("=")[1])
                    # v1: no OU predict, pure IIR. Keep that for the fixed/adaptive baselines.
                    if t == 0:
                        kappa_hat[name] = kappa_inst
                    else:
                        kappa_hat[name] = (1.0 - g) * kappa_hat[name] + g * kappa_inst
                    g_used_all[name][walk, t] = g
                elif name.startswith("adaptive"):
                    proxy = xi_from_consistency(rmat, A, tau_held) if t > 0 else 0.5
                    g = g_min + (g_max - g_min) * float(np.clip(proxy, 0.0, 1.0))
                    if t == 0:
                        kappa_hat[name] = kappa_inst
                    else:
                        kappa_hat[name] = (1.0 - g) * kappa_hat[name] + g * kappa_inst
                    g_used_all[name][walk, t] = g
                else:
                    # Kalman in (amp, phase) separately, with an OU predict step
                    if "oracle" in name:
                        ka, kp = kalman_jump if (t_jump is not None and t >= t_jump) else kalman_true
                    elif "q×0.25" in name or "q x 0.25" in name or "q×0.25" in name:
                        ka, kp = kalman_q025
                    elif "q×4" in name:
                        ka, kp = kalman_q4
                    elif "initial q" in name or "stuck" in name:
                        ka, kp = kalman_true
                    elif "formula" in name:
                        ka, kp = kalman_formula
                    else:
                        # "Kalman true q,r": in the jump experiment this is *initial* true q
                        ka, kp = kalman_true
                    g_a, g_p = ka["g"], kp["g"]
                    if t == 0:
                        kappa_hat[name] = kappa_inst
                    else:
                        pred = rho_ou * kappa_hat[name]
                        eta_new = (1.0 - g_a) * pred.real + g_a * kappa_inst.real
                        xi_new = (1.0 - g_p) * pred.imag + g_p * kappa_inst.imag
                        kappa_hat[name] = eta_new + 1j * xi_new
                    g_used_all[name][walk, t] = 0.5 * (g_a + g_p)

                tau_hat = np.exp(kappa_hat[name])
                err = kappa_errors(tau_hat, tau)
                rms_amp_all[name][walk, t] = err["rms_amp"]
                rms_phase_all[name][walk, t] = err["rms_phase"]
                rms_kappa_all[name][walk, t] = err["rms_kappa"]

    def mean_sem(arr):
        return arr.mean(axis=0), (
            arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 else np.zeros(arr.shape[1])
        )

    rms_kappa_mean, rms_kappa_sem = {}, {}
    for name in names:
        rms_kappa_mean[name], rms_kappa_sem[name] = mean_sem(rms_kappa_all[name])

    def late_stats(name, sl):
        # s.e.m. across walks of the per-walk late mean
        per = rms_kappa_all[name][:, sl].mean(axis=1)
        amp = rms_amp_all[name][:, sl].mean(axis=1)
        ph = rms_phase_all[name][:, sl].mean(axis=1)
        return dict(
            rms_kappa=float(per.mean()),
            rms_kappa_sem=float(per.std(ddof=1) / np.sqrt(n_walks)) if n_walks > 1 else 0.0,
            rms_amp=float(amp.mean()),
            rms_amp_sem=float(amp.std(ddof=1) / np.sqrt(n_walks)) if n_walks > 1 else 0.0,
            rms_phase=float(ph.mean()),
            rms_phase_sem=float(ph.std(ddof=1) / np.sqrt(n_walks)) if n_walks > 1 else 0.0,
            g_mean=float(g_used_all[name][:, sl].mean()),
        )

    if jump and t_jump is not None:
        late_pre = {name: late_stats(name, slice(burn, t_jump)) for name in names}
        late_post = {name: late_stats(name, slice(t_jump + 20, t_steps)) for name in names}
        late = late_post  # primary "late" is after the jump
    else:
        late_pre = None
        late_post = None
        late = {name: late_stats(name, slice(burn, t_steps)) for name in names}

    return dict(
        n=n,
        k=k,
        t_steps=t_steps,
        snr_db=snr_db,
        graph=graph_kind,
        rho_ou=rho_ou,
        n_walks=n_walks,
        jump=jump,
        jump_factor=jump_factor if jump else 1.0,
        t_jump=t_jump,
        names=names,
        rms_kappa=rms_kappa_mean,
        rms_kappa_sem=rms_kappa_sem,
        rms_amp={name: rms_amp_all[name].mean(axis=0) for name in names},
        rms_phase={name: rms_phase_all[name].mean(axis=0) for name in names},
        g_used={name: g_used_all[name].mean(axis=0) for name in names},
        late=late,
        late_pre=late_pre,
        late_post=late_post,
        burn=burn,
        r_formula=r_formula,
        r_measured=r_meas,
        r_amp=r_amp,
        r_phase=r_phase,
        q_amp=q_amp,
        q_phase=q_phase,
        q_kappa=q_kappa,
        kalman_gains=dict(
            true=dict(amp=kalman_true[0], phase=kalman_true[1]),
            q025=dict(amp=kalman_q025[0], phase=kalman_q025[1]),
            q4=dict(amp=kalman_q4[0], phase=kalman_q4[1]),
            jump=dict(amp=kalman_jump[0], phase=kalman_jump[1]),
            formula=dict(amp=kalman_formula[0], phase=kalman_formula[1]),
        ),
        elapsed_s=time.perf_counter() - t0,
    )


def _static_measurement_variance_at(cfg, A, snr_db, k, rng, n_trials=8) -> dict:
    n = A.shape[0]
    acc = dict(mse_kappa=[], mse_amp=[], mse_phase=[])
    for _ in range(n_trials):
        tr = np.random.default_rng(rng.integers(0, 2**63 - 1))
        tau, _ = sample_veils(n, cfg.s_amp, cfg.s_phase, tr)
        Y, _ = generate_observations(tau, k, snr_db, tr)
        r = relative_veil_matrix(Y, A)
        tau_hat = estimate_decentralized(r, A, mode="converged")
        err = kappa_errors(tau_hat, tau)
        acc["mse_kappa"].append(err["mse_kappa"])
        acc["mse_amp"].append(err["mse_amp"])
        acc["mse_phase"].append(err["mse_phase"])
    return {m: _summarize(acc[m]) for m in acc}


# ---------------------------------------------------------------------------
# Experiment D — convergence / settling vs 1/λ₂  (brief)
# ---------------------------------------------------------------------------

def experiment_convergence(cfg: SimConfig, rng: Generator) -> dict:
    """Iterative Jacobi (and Chebyshev on amplitude) vs the converged LS / eigenvector.

    Reports distance-to-converged vs iteration, and the predicted settling scale
    ~ 1 / (λ₂/d) Jacobi steps.
    """
    n = cfg.n_conv
    t_max = cfg.t_conv
    checkpoints = np.unique(np.clip(np.geomspace(1, t_max, 20).astype(int), 1, t_max))
    tau, _ = sample_veils(n, cfg.s_amp, cfg.s_phase, rng)
    Y, _ = generate_observations(tau, 64, 20.0, rng)
    out = {}
    for g in GRAPH_ORDER:
        if g == "torus" and not is_perfect_square(n):
            continue
        A = make_graph(g, n, rng, degree=cfg.expander_degree)
        bounds = laplacian_eigbounds(A)
        r = relative_veil_matrix(Y, A)
        tau_inf = estimate_decentralized(r, A, mode="converged")
        # Jacobi + power iteration
        def run_iter(cheby: bool):
            eta = np.zeros(n, dtype=np.float64)
            u = np.ones(n, dtype=np.complex128)
            dist, rms, t_list = [], [], []
            # we need per-checkpoint snapshots; run with increasing t would redo work,
            # so step one-by-one and record
            z = _symmetrized_log_amp(r, A)
            b = (A * z).sum(axis=1)
            deg = np.maximum(A.sum(axis=1), 1.0)
            H = _connection_matrix(r, A, unit_modulus=True)
            # For Chebyshev we call the whole amplitude solver at checkpoints only
            # (cheap at N=64). Phase is always power iteration.
            if cheby:
                for t in checkpoints:
                    eta_t = amplitude_chebyshev(r, A, int(t), bounds["lam2"], bounds["lam_max"])
                    # t power-iteration steps from scratch
                    uu = np.ones(n, dtype=np.complex128)
                    for _ in range(int(t)):
                        uu = H @ uu
                        uu = uu / np.maximum(np.abs(uu), 1e-15)
                    tau_t = np.exp(eta_t) * uu
                    residual = residual_veils(tau_t, tau_inf)
                    dist.append(float(np.sqrt(np.mean(np.abs(np.log(residual)) ** 2))))
                    rms.append(kappa_errors(tau_t, tau)["rms_kappa"])
                    t_list.append(int(t))
                return dict(t=t_list, dist_to_converged=dist, rms_to_truth=rms)

            eta = np.zeros(n, dtype=np.float64)
            u = np.ones(n, dtype=np.complex128)
            chk = set(int(t) for t in checkpoints)
            for t in range(1, t_max + 1):
                eta = (A @ eta + b) / deg
                eta = eta - eta.mean()
                v = H @ u
                u = v / np.maximum(np.abs(v), 1e-15)
                if t in chk or t == t_max:
                    tau_t = np.exp(eta) * u
                    residual = residual_veils(tau_t, tau_inf)
                    dist.append(float(np.sqrt(np.mean(np.abs(np.log(residual)) ** 2))))
                    rms.append(kappa_errors(tau_t, tau)["rms_kappa"])
                    t_list.append(t)
            return dict(t=t_list, dist_to_converged=dist, rms_to_truth=rms)

        jac = run_iter(False)
        chb = run_iter(True)

        def steps_to(dist_t, t, thresh=1e-3):
            for ti, d in zip(t, dist_t):
                if d <= thresh:
                    return int(ti)
            return None

        out[g] = dict(
            bounds=bounds,
            jacobi=jac,
            chebyshev=chb,
            rms_converged=kappa_errors(tau_inf, tau)["rms_kappa"],
            jacobi_steps_to_1em3=steps_to(jac["dist_to_converged"], jac["t"], 1e-3),
            chebyshev_steps_to_1em3=steps_to(chb["dist_to_converged"], chb["t"], 1e-3),
            predicted_jacobi_steps=bounds["steps_1_over_gap"],
        )
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------







def _plot_tracking_panel(result: dict, outdir: Path, name: str, title: str) -> Path:
    plt = _setup_mpl()
    t = np.arange(result["t_steps"])
    names = result["names"]
    # colour cycle
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, 1, figsize=(10.6, 7.4), sharex=True)
    ax = axes[0]
    for i, nm in enumerate(names):
        mu = result["rms_kappa"][nm]
        se = result["rms_kappa_sem"][nm]
        ax.plot(t, mu, color=cmap(i), lw=1.5, label=nm)
        ax.fill_between(t, mu - se, mu + se, color=cmap(i), alpha=0.15, lw=0)
    if result.get("t_jump"):
        ax.axvline(result["t_jump"], color="k", ls="--", lw=1, alpha=0.7, label="q jump")
    ax.set_ylabel(r"gauge-fixed RMS of $\kappa$")
    ax.set_title(title)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.set_yscale("log")
    ax = axes[1]
    for i, nm in enumerate(names):
        ax.plot(t, result["g_used"][nm], color=cmap(i), lw=1.4, label=nm)
    if result.get("t_jump"):
        ax.axvline(result["t_jump"], color="k", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("time step")
    ax.set_ylabel("gain used (amp/phase mean for Kalman)")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    path = outdir / name
    fig.savefig(path)
    plt.close(fig)
    return path






# ---------------------------------------------------------------------------
# Conclusions (generated from the actual numbers)
# ---------------------------------------------------------------------------

def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NA"
    return f"{x:.{nd}f}"


def _fmt_nstar(est: dict) -> str:
    if not est:
        return "NA"
    st = est.get("status", "")
    if st == "ok":
        return f"{est['nstar']:.1f}"
    if st == "above_range":
        return f"> {est['nstar_lo']:.0f} (no crossing up to N_max)"
    if st == "below_range":
        return f"< {est['nstar_hi']:.0f} (already above 2×floor at N_min)"
    return f"({st})"


def format_conclusions(cfg: SimConfig, law, overlap, track, jump, conv) -> str:
    lines = []
    lines.append("WHAT THIS RUN ACTUALLY SHOWED")
    lines.append("==============================")
    lines.append(
        f"SIZE={cfg.size},  s_a={cfg.s_amp}, s_p={cfg.s_phase},  "
        f"theoretical Ξ = exp(-(s_a²+s_p²)) = {cfg.theoretical_xi():.4f}."
    )
    mom = law["moments"]
    lines.append(
        f"E|τ|^{{-2}} = exp(2 s_a²) = {mom['e_inv2']:.4f},  "
        f"E|τ|^{{-4}} = exp(8 s_a²) = {mom['e_inv4']:.4f},  "
        f"factor E|τ|^{{-2}}/E|τ|^{{-4}} = {mom['factor']:.4f}."
    )
    lines.append(
        f"Error-law grid: N={law['n_list']},  K={law['k_list']},  "
        f"SNR={law['snr_list']} dB,  {law['n_trials']} trials."
    )
    lines.append("")

    # ----- A: fitted constants -----
    lines.append("A. Error-law test")
    lines.append("-----------------")
    lines.append(
        "Proposed law:  MSE(κ̂) ≈ c_floor · E|τ|^{-2}/(K·SNR)  +  c_layout · E|τ|^{-4}/(K·SNR²) · R̄(G),"
    )
    lines.append("with predicted c_floor ≈ 1, c_layout ≈ 1  (other second-order terms exist, some node-only).")
    gfit = law["fits"]["global"]
    lines.append(
        f"   Global weighted LS fit (ring+torus+expander, all SNR, K, N):  "
        f"c_floor = { _fmt(gfit.get('c_floor'), 3) },  c_layout = { _fmt(gfit.get('c_layout'), 3) },  "
        f"R² = { _fmt(gfit.get('r2'), 3) }  (n={gfit.get('n', 0)} points)."
    )
    if np.isfinite(gfit.get("intercept", np.nan)):
        lines.append(
            f"   With an intercept: intercept={_fmt(gfit['intercept'], 5)},  "
            f"c_floor={_fmt(gfit['c_floor_with_intercept'], 3)},  "
            f"c_layout={_fmt(gfit['c_layout_with_intercept'], 3)}."
        )
    # how close to 1
    cf, cl = gfit.get("c_floor"), gfit.get("c_layout")
    if np.isfinite(cf) and np.isfinite(cl):
        def _close(c, name):
            if 0.5 <= c <= 2.0:
                return f"{name} is within a factor of two of the predicted 1"
            return f"{name} is NOT close to the predicted 1 (this is a genuine mismatch with the leading-term theory)"
        lines.append(f"   {_close(cf, 'c_floor')}.  {_close(cl, 'c_layout')}.")
    lines.append("   Fits by SNR:")
    for snr, ft in law["fits"]["by_snr"].items():
        lines.append(
            f"     SNR={float(snr):.0f} dB:  c_floor={_fmt(ft.get('c_floor'), 3)},  "
            f"c_layout={_fmt(ft.get('c_layout'), 3)},  R²={_fmt(ft.get('r2'), 3)}"
        )
    lines.append("   Fits by K:")
    for k, ft in law["fits"]["by_k"].items():
        lines.append(
            f"     K={k}:  c_floor={_fmt(ft.get('c_floor'), 3)},  "
            f"c_layout={_fmt(ft.get('c_layout'), 3)},  R²={_fmt(ft.get('r2'), 3)}"
        )
    lines.append("   Fits by graph:")
    for kind, ft in law["fits"]["by_kind"].items():
        lines.append(
            f"     {kind}:  c_floor={_fmt(ft.get('c_floor'), 3)},  "
            f"c_layout={_fmt(ft.get('c_layout'), 3)},  R²={_fmt(ft.get('r2'), 3)}"
        )

    # Rbar check
    lines.append("   Mean effective resistance R̄(G)=trace(L⁺)/N used in the overlay:")
    for n in law["n_list"]:
        rb = law["rbar"][n]
        ring_mean_resistance(n)
        extra = f"  (ring exact {(n**2-1)/(12*n):.4f})"
        bits = ", ".join(f"{k}={v:.4f}" for k, v in rb.items())
        lines.append(f"     N={n}: {bits}{extra}")

    # floor sanity vs 1-Ξ at v1 point if present (K=64, 20 dB)
    if 20.0 in law["summary"] and 64 in law["summary"][20.0]:
        b = law["summary"][20.0][64]
        floor = b["ring"]["floor"][0] if b["ring"]["floor"] else np.nan
        # 1-Ξ ≈ mse at small N?  leftover 1-Ξ from residual after correction
        # report measured mse and 1-mean(xi_after)
        for kind in list(GRAPH_ORDER) + ["svd"]:
            if kind not in b or not b[kind]["N"]:
                continue
            # pick N closest to 64
            Ns = np.array(b[kind]["N"])
            j = int(np.argmin(np.abs(Ns - 64)))
            mse = b[kind]["mse_kappa"]["mean"][j]
            xi_a = b[kind]["xi_after"]["mean"][j] if b[kind]["xi_after"]["mean"] else np.nan
            one_m = (1.0 - xi_a) if np.isfinite(xi_a) else np.nan
            lines.append(
                f"   Sanity at SNR=20 dB, K=64, N={int(Ns[j])}, {kind}:  "
                f"MSE={_fmt(mse, 6)},  1−Ξ={_fmt(one_m, 6)},  formula floor={_fmt(floor, 6)}  "
                f"(v1 reported floor 1.77e-4 vs measured 1−Ξ 1.9e-4)."
            )

    lines.append("")
    lines.append("   Ring crossover N* (where the layout term equals the floor, i.e. MSE ≈ 2×floor):")
    lines.append("   Theory is independent of K.  We report the formula-floor crossing from the measured ring curve.")
    for snr in law["snr_list"]:
        th = law["nstar"][snr]["theory"]
        lines.append(
            f"     SNR={snr:.0f} dB:  theory 12·SNR={th['nstar_12snr']:.1f},  "
            f"12·SNR·factor={th['nstar_12snr_factor']:.1f},  exact-R̄ root={th['nstar_exact_rbar']:.1f}."
        )
        for k in law["k_list"]:
            rec = law["nstar"][snr].get(k)
            if not rec:
                continue
            lines.append(
                f"       K={k}:  measured N* (formula floor) = {_fmt_nstar(rec['from_formula_floor'])};  "
                f"using SVD floor = {_fmt_nstar(rec['from_svd_floor'])};  "
                f"from fitted c = {_fmt(rec.get('from_fitted_c'), 1)}."
            )
    # K-independence verdict
    lines.append("   K-independence check (measured N* should agree across K at each SNR, if a crossing exists):")
    for snr in law["snr_list"]:
        vals = []
        for k in law["k_list"]:
            rec = law["nstar"][snr].get(k)
            if rec and rec["from_formula_floor"].get("status") == "ok":
                vals.append((k, rec["from_formula_floor"]["nstar"]))
        if len(vals) >= 2:
            ns = [v[1] for v in vals]
            rel = (max(ns) - min(ns)) / max(np.mean(ns), 1e-9)
            bit = ", ".join(f"K={k}: {n:.1f}" for k, n in vals)
            verdict = "consistent with K-independence" if rel < 0.35 else "NOT clearly K-independent in this run"
            lines.append(f"     SNR={snr:.0f} dB: {bit}.  Relative spread {rel:.2f}.  {verdict}.")
        elif len(vals) == 1:
            lines.append(f"     SNR={snr:.0f} dB: only one K produced a crossing ({vals[0][0]}: {vals[0][1]:.1f}); cannot test K-independence.")
        else:
            lines.append(f"     SNR={snr:.0f} dB: no in-range crossing at any K.")

    lines.append("")

    # ----- B -----
    lines.append("B. Partial overlap")
    lines.append("------------------")
    lines.append(overlap["first_order_note"])
    lines.append(
        f"   Grid: ρ={overlap['rho_list']},  N={overlap['n_list']},  K={overlap['k']},  "
        f"SNR={overlap['snr_list']} dB,  {overlap['n_trials']} trials.  "
        f"Skipped disconnected estimates: {overlap['n_skipped']}."
    )
    # window construction notes
    # report a representative info
    info = overlap.get("window_info", {})
    for kind in GRAPH_ORDER:
        # pick mid N, each rho
        ns = overlap["n_list"]
        mid = ns[len(ns) // 2]
        if mid in info and kind in info.get(mid, {}):
            bits = []
            for rho in overlap["rho_list"]:
                inf = info[mid][kind].get(rho, {})
                bits.append(
                    f"ρ={rho:g}: method={inf.get('method')}, min_share={inf.get('min_overlap')}, "
                    f"mean_share={inf.get('mean_overlap')}, ok={inf.get('ok')}"
                )
            lines.append(f"   Window construction at N={mid}, {kind}: " + " | ".join(bits))

    for snr in overlap["snr_list"]:
        lines.append(f"   SNR={snr:.0f} dB, ring N* vs ρ (empirical floor = mean expander/torus MSE at that ρ):")
        nstar_by_rho = []
        for rho in overlap["rho_list"]:
            rec = overlap["nstar"][snr].get(rho, {})
            est = rec.get("ring")
            # also report MSE at largest N for ring vs expander, as a layout-effect diagnostic
            ring_b = overlap["summary"][snr][rho]["ring"]
            exp_b = overlap["summary"][snr][rho].get("expander") or overlap["summary"][snr][rho].get("torus")
            mse_r = ring_b["mse_kappa"]["mean"][-1] if ring_b["N"] else np.nan
            mse_e = exp_b["mse_kappa"]["mean"][-1] if exp_b and exp_b["N"] else np.nan
            ratio = mse_r / mse_e if (np.isfinite(mse_r) and np.isfinite(mse_e) and mse_e > 0) else np.nan
            lines.append(
                f"     ρ={rho:g}:  N* = {_fmt_nstar(est) if est else 'NA'};  "
                f"MSE_ring / MSE_ref at N_max = {_fmt(ratio, 2)}  "
                f"(ring MSE={_fmt(mse_r, 5)}, ref MSE={_fmt(mse_e, 5)})."
            )
            if est and est.get("status") == "ok":
                nstar_by_rho.append((rho, est["nstar"]))
        if len(nstar_by_rho) >= 2:
            # does N* shrink as ρ drops?
            # compare ρ=1 vs smallest ρ that has a crossing
            by = dict(nstar_by_rho)
            if 1.0 in by:
                others = [(r, n) for r, n in nstar_by_rho if r < 1.0]
                if others:
                    smaller = all(n < 0.8 * by[1.0] for _, n in others)
                    any_smaller = any(n < 0.9 * by[1.0] for _, n in others)
                    if smaller:
                        lines.append("     Verdict: N* dropped as ρ dropped, consistent with the 'layout matters earlier' prediction.")
                    elif any_smaller:
                        lines.append("     Verdict: N* dropped for at least some ρ<1, but not uniformly. The prediction is only partly supported.")
                    else:
                        lines.append("     Verdict: N* did NOT clearly drop as ρ dropped. This contradicts a strong reading of the prediction (or the N range is too coarse).")
            else:
                lines.append("     Verdict: no in-range N* at ρ=1 to compare against.")
        else:
            # fall back to MSE ratio trend
            ratios = []
            for rho in overlap["rho_list"]:
                ring_b = overlap["summary"][snr][rho]["ring"]
                exp_b = overlap["summary"][snr][rho].get("expander") or overlap["summary"][snr][rho].get("torus")
                if ring_b["N"] and exp_b and exp_b["N"]:
                    ratios.append((rho, ring_b["mse_kappa"]["mean"][-1] / max(exp_b["mse_kappa"]["mean"][-1], 1e-18)))
            if len(ratios) >= 2:
                # does the ring/ref ratio at N_max grow as ρ drops?
                r1 = dict(ratios).get(1.0)
                rmin = ratios[-1][1]  # smallest ρ is last if rho_list is decreasing? it's (1, 0.75, 0.5, 0.25)
                if r1 is not None and rmin > 1.3 * r1:
                    lines.append("     Verdict: at N_max, the ring-vs-ref MSE ratio grew as ρ dropped, which is the qualitative layout-earlier signature even without a clean N* crossing.")
                elif r1 is not None and rmin <= 1.1 * r1:
                    lines.append("     Verdict: at N_max, the ring-vs-ref MSE ratio did not grow clearly as ρ dropped. The 'layout matters earlier' prediction is not supported by this diagnostic.")
                else:
                    lines.append("     Verdict: the ring-vs-ref MSE ratio changed only modestly with ρ; the evidence is inconclusive.")

    lines.append("")

    # ----- C -----
    lines.append("C. Kalman baseline for the drifting veil")
    lines.append("----------------------------------------")
    lines.append(
        f"   OU ρ={track['rho_ou']}, N={track['n']}, K={track['k']}, SNR={track['snr_db']:.0f} dB, "
        f"{track['n_walks']} walks, torus, t={track['t_steps']}, burn-in={track['burn']}."
    )
    lines.append(
        f"   Process q:  q_amp={track['q_amp']:.5f}, q_phase={track['q_phase']:.5f}, q_κ={track['q_kappa']:.5f}."
    )
    lines.append(
        f"   Measurement r:  formula floor={track['r_formula']:.6f},  "
        f"measured MSE={track['r_measured']:.6f}  (amp={track['r_amp']:.6f}, phase={track['r_phase']:.6f})."
    )
    kt = track["kalman_gains"]["true"]
    lines.append(
        f"   Steady-state Kalman gains (true q, measured r):  "
        f"g_amp={kt['amp']['g']:.3f}, g_phase={kt['phase']['g']:.3f}."
    )
    lines.append("   Stationary-OU late-window RMS(κ) (s.e.m. across walks):")
    for name, st in track["late"].items():
        lines.append(
            f"     - {name:<28}  RMS(κ)={st['rms_kappa']:.4f} ± {st['rms_kappa_sem']:.4f},  "
            f"amp={st['rms_amp']:.4f} ± {st.get('rms_amp_sem', 0):.4f},  "
            f"phase={st['rms_phase']:.4f} ± {st.get('rms_phase_sem', 0):.4f},  mean g={st['g_mean']:.3f}"
        )
    winner = min(track["late"].items(), key=lambda kv: kv[1]["rms_kappa"])
    lines.append(f"   Lowest late error (stationary): {winner[0]} at RMS(κ)={winner[1]['rms_kappa']:.4f}.")
    # compare Kalman true vs adaptive vs best fixed vs misspecified
    def _get(res, key_substr):
        for name, st in res["late"].items():
            if key_substr in name:
                return name, st
        return None, None
    k_true_name, k_true = _get(track, "Kalman true")
    adapt_name, adapt = _get(track, "adaptive")
    k025_name, k025 = _get(track, "q×0.25")
    k4_name, k4 = _get(track, "q×4")
    if k_true and adapt:
        if k_true["rms_kappa"] + k_true["rms_kappa_sem"] < adapt["rms_kappa"] - adapt["rms_kappa_sem"]:
            lines.append("   Kalman with true (q,r) beat the adaptive Ξ rule on the stationary OU (outside combined s.e.m.).")
        elif adapt["rms_kappa"] + adapt["rms_kappa_sem"] < k_true["rms_kappa"] - k_true["rms_kappa_sem"]:
            lines.append("   The adaptive Ξ rule beat Kalman with true (q,r) on the stationary OU (outside combined s.e.m.). That is unexpected if the Kalman model is well specified; it may reflect gauge-alignment or amp/phase split details, and is reported as a result, not a claim of superiority.")
        else:
            lines.append("   On the stationary OU, Kalman with true (q,r) and the adaptive Ξ rule are not clearly separated beyond s.e.m.")
    if k_true and k025 and k4:
        lines.append(
            f"   Misspecified q:  true {k_true['rms_kappa']:.4f},  q×0.25 {k025['rms_kappa']:.4f},  "
            f"q×4 {k4['rms_kappa']:.4f}."
        )

    lines.append("")
    lines.append(
        f"   Jump experiment: q → {jump['jump_factor']:.0f} q at t={jump['t_jump']}, "
        f"{jump['n_walks']} walks.  Late-window figures below are AFTER the jump (plus a 20-step buffer)."
    )
    if jump.get("late_pre"):
        lines.append("   Before the jump:")
        for name, st in jump["late_pre"].items():
            lines.append(
                f"     - {name:<28}  RMS(κ)={st['rms_kappa']:.4f} ± {st['rms_kappa_sem']:.4f},  "
                f"amp={st['rms_amp']:.4f}, phase={st['rms_phase']:.4f}"
            )
    lines.append("   After the jump:")
    for name, st in jump["late"].items():
        lines.append(
            f"     - {name:<28}  RMS(κ)={st['rms_kappa']:.4f} ± {st['rms_kappa_sem']:.4f},  "
            f"amp={st['rms_amp']:.4f}, phase={st['rms_phase']:.4f}"
        )
    _, adapt_j = _get(jump, "adaptive")
    _, stuck = _get(jump, "stuck")
    if stuck is None:
        _, stuck = _get(jump, "initial q")
    _, oracle = _get(jump, "oracle")
    if adapt_j and stuck:
        if adapt_j["rms_kappa"] + adapt_j["rms_kappa_sem"] < stuck["rms_kappa"] - stuck["rms_kappa_sem"]:
            lines.append("   After the jump, the adaptive Ξ rule beat a Kalman filter stuck at the initial q.")
        elif stuck["rms_kappa"] + stuck["rms_kappa_sem"] < adapt_j["rms_kappa"] - adapt_j["rms_kappa_sem"]:
            lines.append(
                "   After the jump, a Kalman filter stuck at the initial q still beat (or was no worse than, within noise) "
                "the adaptive Ξ rule. The v1 adaptive rule raises g when neighbour consistency is *high*; a sudden "
                "increase in drift can make the held estimate inconsistent and *lower* g, which is the wrong direction. "
                "That limitation is visible here if it occurs."
            )
        else:
            lines.append("   After the jump, the adaptive rule and the stuck-initial-q Kalman are not clearly separated beyond s.e.m.")
    if oracle:
        lines.append(f"   Oracle Kalman retuned to the post-jump q reached RMS(κ)={oracle['rms_kappa']:.4f} ± {oracle['rms_kappa_sem']:.4f}.")

    lines.append("")

    # ----- D -----
    lines.append("D. Convergence / settling of local iterations")
    lines.append("---------------------------------------------")
    for g, d in conv.items():
        b = d["bounds"]
        lines.append(
            f"   {g}:  λ₂={b['lam2']:.5f}, λ_max={b['lam_max']:.4f}, deg={b['deg']:.2f},  "
            f"predicted Jacobi steps d/λ₂ ≈ {b['steps_1_over_gap']:.1f}.  "
            f"Jacobi steps to 1e-3: {d.get('jacobi_steps_to_1em3')},  "
            f"Chebyshev steps to 1e-3: {d.get('chebyshev_steps_to_1em3')}.  "
            f"Converged RMS(κ)={d['rms_converged']:.4f}."
        )
        js, cs, pred = d.get("jacobi_steps_to_1em3"), d.get("chebyshev_steps_to_1em3"), b["steps_1_over_gap"]
        jdist = d.get("jacobi", {}).get("dist_to_converged", [])
        cdist = d.get("chebyshev", {}).get("dist_to_converged", [])
        j_last = float(jdist[-1]) if jdist else float("nan")
        c_last = float(cdist[-1]) if cdist else float("nan")
        j_tlast = d.get("jacobi", {}).get("t", [None])[-1] if d.get("jacobi", {}).get("t") else None
        if js is None:
            lines.append(
                f"     Jacobi did not reach 1e-3 in {j_tlast} steps (final dist={j_last:.4f}).  "
                f"Chebyshev ended at {c_last:.4f}.  Because Chebyshev only accelerates the amplitude "
                f"update and tracks Jacobi here, the leftover is in the phase-sync iteration, not in "
                f"the Laplacian mixing time."
            )
        elif np.isfinite(pred) and pred > 0:
            ratio = js / pred
            lines.append(
                f"     Jacobi settling / (d/λ₂) = {ratio:.2f}.  "
                f"The predicted scale is a 1/e-style mixing time, so a factor of a few to reach 1e-3 is expected; "
                f"a factor of 20+ would mean the prediction is not useful."
            )
        if js and cs:
            if cs < 0.7 * js:
                lines.append(f"     Chebyshev reduced settling steps ({cs} vs {js}).")
            elif cs > 1.3 * js:
                lines.append(f"     Chebyshev did NOT help on this graph in this run ({cs} vs Jacobi {js}); reported honestly.")
            else:
                lines.append(f"     Chebyshev was comparable to Jacobi ({cs} vs {js}).")

    lines.append("")
    lines.append("WHAT THIS DOES NOT SHOW")
    lines.append("=======================")
    lines.append(
        "- This is a classical numerical simulation of complex arrays. Using i and a "
        "Hilbert-space notation for a veil does not make the simulation quantum, and it "
        "does not demonstrate entanglement, a quantum advantage, or a new physical chip."
    )
    lines.append(
        "- There is no CMOS, timing-error, voltage, or energy model here. The notebook "
        "does not compare a tile array to Razor-style detection or adaptive voltage scaling. "
        "That measured-result claim still needs an FPGA or ASIC experiment."
    )
    lines.append(
        "- The two-term error law is a leading-term calculation with approximate constants. "
        "A fitted c_floor or c_layout far from 1 is a limitation of that calculation, not a "
        "licence to treat the fitted numbers as a theorem. N* extracted from a finite N grid "
        "has interpolation error, and a 'no crossing' result is only a bound."
    )
    lines.append(
        "- Partial overlap on the expander is not a sliding window (an expander has no "
        "geometry). Ring and torus use exact windows that guarantee ρK shared samples on "
        "every edge. If the expander construction failed to reach the target overlap, that "
        "is stated in the window-info lines above, and those points should be read with care."
    )
    lines.append(
        "- A programming model in which the programmer declares Ξ_crit(c) is not implemented here."
    )
    lines.append("")
    lines.append("RUNTIME")
    lines.append(
        f"  expA {law.get('elapsed_s', float('nan')):.1f}s,  "
        f"expB {overlap.get('elapsed_s', float('nan')):.1f}s,  "
        f"expC {track.get('elapsed_s', float('nan')):.1f}s,  "
        f"expC-jump {jump.get('elapsed_s', float('nan')):.1f}s,  "
        f"expD  (included in wall time)."
    )
    return "\n".join(lines)


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        if obj.size > 8000:
            return dict(_type="ndarray", shape=list(obj.shape), dtype=str(obj.dtype), omitted=True)
        if np.iscomplexobj(obj):
            return dict(_type="complex_array", real=obj.real.tolist(), imag=obj.imag.tolist())
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return val if np.isfinite(val) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.complexfloating, complex)):
        return dict(_type="complex", real=float(np.real(obj)), imag=float(np.imag(obj)))
    if isinstance(obj, Path):
        return str(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj
    return str(obj)


def run_suite(cfg: SimConfig, outdir: Path, progress: Callable | None = None) -> dict:
    """Run experiments A–D, write PNGs and results.json, return the payload."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    wall0 = time.perf_counter()

    def _prog(done, total, elapsed):
        print(f"  {done}/{total}  ({elapsed:.1f}s)", flush=True)
        if progress is not None:
            progress(done, total, elapsed)

    print(f"SIZE={cfg.size}")
    print(f"theoretical Ξ = {cfg.theoretical_xi():.6f}")
    print(f"E|τ|^{{-2}}={cfg.e_tau_inv_pow(2):.4f}, E|τ|^{{-4}}={cfg.e_tau_inv_pow(4):.4f}")
    print(f"N list = {cfg.n_list}")
    print(f"trials = {cfg.n_trials}")

    print("\n== experiment A: error-law test ==")
    law = experiment_error_law(cfg, np.random.default_rng(rng.integers(0, 2**63 - 1)), progress=_prog)
    print(f"  elapsed {law['elapsed_s']:.1f}s")
    gfit = law["fits"]["global"]
    print(f"  fitted c_floor={gfit.get('c_floor')}, c_layout={gfit.get('c_layout')}, R²={gfit.get('r2')}")
    for snr in law["snr_list"]:
        th = law["nstar"][snr]["theory"]
        print(f"  SNR={snr:.0f}: theory N* (12·SNR·factor)={th['nstar_12snr_factor']:.1f}")
        for k in law["k_list"]:
            rec = law["nstar"][snr].get(k)
            if rec:
                print(f"    K={k}: measured {_fmt_nstar(rec['from_formula_floor'])}")
    paths = plot_error_law(law, cfg, outdir)
    print("  wrote", {k: str(v) for k, v in paths.items()})

    print("\n== experiment B: partial overlap ==")
    overlap = experiment_partial_overlap(cfg, np.random.default_rng(rng.integers(0, 2**63 - 1)), progress=_prog)
    print(f"  elapsed {overlap['elapsed_s']:.1f}s, skipped={overlap['n_skipped']}")
    paths.update(plot_partial_overlap(overlap, cfg, outdir))

    print("\n== experiment C: Kalman tracking (stationary OU) ==")
    track = experiment_tracking_kalman(
        cfg, np.random.default_rng(rng.integers(0, 2**63 - 1)), snr_db=20.0, jump=False
    )
    print(f"  elapsed {track['elapsed_s']:.1f}s")
    for name, st in track["late"].items():
        print(f"  {name}: RMS(κ)={st['rms_kappa']:.4f} ± {st['rms_kappa_sem']:.4f}")

    print("\n== experiment C: Kalman tracking (q jump ×10) ==")
    jump = experiment_tracking_kalman(
        cfg, np.random.default_rng(rng.integers(0, 2**63 - 1)), snr_db=20.0, jump=True, jump_factor=10.0
    )
    print(f"  elapsed {jump['elapsed_s']:.1f}s")
    for name, st in jump["late"].items():
        print(f"  after jump  {name}: RMS(κ)={st['rms_kappa']:.4f} ± {st['rms_kappa_sem']:.4f}")
    paths.update(plot_tracking(track, jump, outdir))

    print("\n== experiment D: convergence / settling ==")
    conv = experiment_convergence(cfg, np.random.default_rng(rng.integers(0, 2**63 - 1)))
    for g, d in conv.items():
        print(
            f"  {g}: pred d/λ₂={d['bounds']['steps_1_over_gap']:.1f}, "
            f"Jacobi to 1e-3={d.get('jacobi_steps_to_1em3')}, "
            f"Chebyshev to 1e-3={d.get('chebyshev_steps_to_1em3')}"
        )
    paths["expD"] = plot_convergence(conv, outdir)

    wall = time.perf_counter() - wall0
    print(f"\nTOTAL WALL TIME {wall:.1f}s")
    conclusions = format_conclusions(cfg, law, overlap, track, jump, conv)
    (outdir / "conclusions.txt").write_text(conclusions)

    # slim payload (drop nothing huge; arrays in tracking are t_steps long)
    payload = dict(
        size=cfg.size,
        theoretical_xi=cfg.theoretical_xi(),
        wall_time_s=wall,
        moments=law["moments"],
        law=dict(
            n_list=law["n_list"],
            k_list=law["k_list"],
            snr_list=law["snr_list"],
            rbar=law["rbar"],
            summary=law["summary"],
            fits=law["fits"],
            nstar=law["nstar"],
            moments=law["moments"],
            elapsed_s=law["elapsed_s"],
            n_trials=law["n_trials"],
        ),
        overlap=dict(
            n_list=overlap["n_list"],
            rho_list=overlap["rho_list"],
            snr_list=overlap["snr_list"],
            k=overlap["k"],
            summary=overlap["summary"],
            nstar=overlap["nstar"],
            window_info=overlap["window_info"],
            n_skipped=overlap["n_skipped"],
            skipped=overlap["skipped"],
            first_order_note=overlap["first_order_note"],
            elapsed_s=overlap["elapsed_s"],
            n_trials=overlap["n_trials"],
        ),
        track=track,
        jump=jump,
        conv=conv,
        conclusions=conclusions,
        paths={k: str(v) for k, v in paths.items()},
    )
    with (outdir / "results.json").open("w") as f:
        json.dump(_json_safe(payload), f, indent=2)
    print("wrote", outdir / "results.json")
    print("wrote", outdir / "conclusions.txt")
    return payload


# Names retained so the NumPy driver imports. They are not on the seeded path.
def _setup_mpl(*_args, **_kwargs):
    raise RuntimeError('_setup_mpl' + " is not part of the NumPy reproduction driver")
