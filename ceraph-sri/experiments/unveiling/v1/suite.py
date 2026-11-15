"""Reproduction driver for the v1 round.

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
    fit_log_n,
    fit_power_law,
)
from sri.graphs import make_graph
from sri.unveiling import (
    _connection_matrix,
    _symmetrized_log_amp,
    complex_normal,
    estimate_centralized_svd,
    estimate_decentralized,
    gauge_align,
    independent_edge_ratios,
    kappa_errors,
    noise_std_from_snr,
    relative_veil_matrix,
    residual_veils,
    sample_veils,
    xi_empirical,
    xi_from_consistency,
)

def _plot_stub(*_args, **_kwargs):
    """Plots are not written by the reproduction driver."""
    return []

plot_convergence = _plot_stub

plot_scaling = _plot_stub

plot_scaling_edge_noise = _plot_stub

plot_snr_k = _plot_stub

plot_tracking = _plot_stub

plot_xi = _plot_stub

ComplexArr = NDArray[np.complex128]
FloatArr = NDArray[np.float64]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# SIZE = "small" is meant to finish on a CPU in well under 5 minutes.
# SIZE = "large" is a denser sweep for a Colab T4/L4 (or a longer CPU run).
SIZE_PRESETS = {
    "small": dict(
        n_list=(16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225),
        n_fixed=64,
        n_trials=10,
        k_default=64,
        snr_default_db=20.0,
        snr_list_db=(0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0),
        k_list=(8, 16, 32, 64, 128),
        t_iter=500,
        t_track=250,
        k_track=16,
    ),
    "large": dict(
        n_list=(
            16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225,
            256, 324, 400, 484, 576, 676, 784, 900, 1024,
        ),
        n_fixed=256,
        n_trials=24,
        k_default=128,
        snr_default_db=20.0,
        snr_list_db=(0.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0),
        k_list=(4, 8, 16, 32, 64, 128, 256),
        t_iter=800,
        t_track=400,
        k_track=16,
    ),
}

DEFAULT_S_AMP = 0.25  # std of η (log-amplitude veil)
DEFAULT_S_PHASE = 0.40  # std of ξ (phase veil, radians)
EXPANDER_DEGREE = 4  # match the 2-D torus so topology, not degree, is the contrast
BASE_SEED = 7

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


@dataclass
class SimConfig:
    size: str = "small"
    s_amp: float = DEFAULT_S_AMP
    s_phase: float = DEFAULT_S_PHASE
    expander_degree: int = EXPANDER_DEGREE
    seed: int = BASE_SEED
    # filled from SIZE_PRESETS
    n_list: tuple[int, ...] = SIZE_PRESETS["small"]["n_list"]
    n_fixed: int = 64
    n_trials: int = 10
    k_default: int = 64
    snr_default_db: float = 20.0
    snr_list_db: tuple[float, ...] = SIZE_PRESETS["small"]["snr_list_db"]
    k_list: tuple[int, ...] = SIZE_PRESETS["small"]["k_list"]
    t_iter: int = 500
    t_track: int = 250
    k_track: int = 16

    @classmethod
    def from_size(cls, size: str, **overrides) -> "SimConfig":
        if size not in SIZE_PRESETS:
            raise ValueError(f"SIZE must be 'small' or 'large', got {size!r}")
        data = dict(SIZE_PRESETS[size])
        data.update(overrides)
        return cls(size=size, **data)

    def theoretical_xi(self) -> float:
        # τ = exp(η + iξ), η ~ N(0, s_a²), ξ ~ N(0, s_p²) independent
        # Ξ = |E τ|² / E|τ|² = exp(-(s_a² + s_p²))
        return float(np.exp(-(self.s_amp**2 + self.s_phase**2)))


# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------



















# ---------------------------------------------------------------------------
# Observation model
# ---------------------------------------------------------------------------







def generate_observations(
    tau: ComplexArr,
    k: int,
    snr_db: float,
    rng: Generator,
    psi: ComplexArr | None = None,
    overlap: float = 1.0,
) -> tuple[ComplexArr, ComplexArr, FloatArr]:
    """y_{n,k} = τ_n ψ_k + w_{n,k}.

    If overlap < 1, each tile observes a random subset of the K samples and
    missing entries are marked in `mask` (1 = observed). Inner products must
    then be taken only on jointly observed indices.

    Returns Y (N×K), psi (K,), mask (N×K).
    """
    n = tau.shape[0]
    if psi is None:
        psi = complex_normal(rng, k)
    sigma = noise_std_from_snr(snr_db, signal_power=1.0)
    W = sigma * complex_normal(rng, (n, k))
    Y = tau[:, None] * psi[None, :] + W
    if overlap >= 1.0:
        mask = np.ones((n, k), dtype=np.float64)
    else:
        # independent observation pattern per tile, keep at least one sample
        keep = rng.random((n, k)) < overlap
        keep[np.arange(n), rng.integers(0, k, size=n)] = True
        mask = keep.astype(np.float64)
        Y = Y * mask
    return Y.astype(np.complex128), psi.astype(np.complex128), mask


# ---------------------------------------------------------------------------
# Relative veils
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

















# ---------------------------------------------------------------------------
# Metrics: gauge alignment, RMS error, Ξ
# ---------------------------------------------------------------------------















# ---------------------------------------------------------------------------
# Single-trial and experiment runners
# ---------------------------------------------------------------------------

def run_one_trial(
    n: int,
    k: int,
    snr_db: float,
    graph_kind: str,
    cfg: SimConfig,
    rng: Generator,
    mode: str = "converged",
    overlap: float = 1.0,
) -> dict:
    tau, _kappa = sample_veils(n, cfg.s_amp, cfg.s_phase, rng)
    A = make_graph(graph_kind, n, rng, degree=cfg.expander_degree)
    Y, _psi, mask = generate_observations(tau, k, snr_db, rng, overlap=overlap)
    r = relative_veil_matrix(Y, A, mask=mask if overlap < 1.0 else None)
    tau_hat_loc = estimate_decentralized(r, A, mode=mode, t_iter=cfg.t_iter)
    tau_hat_svd = estimate_centralized_svd(Y, mask=mask if overlap < 1.0 else None)
    err_loc = kappa_errors(tau_hat_loc, tau)
    err_svd = kappa_errors(tau_hat_svd, tau)
    residual_loc = err_loc["residual"]
    residual_svd = err_svd["residual"]
    return dict(
        n=n,
        k=k,
        snr_db=snr_db,
        graph=graph_kind,
        xi_true=xi_empirical(tau),
        xi_after_local=xi_empirical(residual_loc),
        xi_after_svd=xi_empirical(residual_svd),
        xi_proxy=xi_from_consistency(r, A, tau_hat_loc),
        local=err_loc,
        svd=err_svd,
        tau=tau,
        tau_hat_local=tau_hat_loc,
        tau_hat_svd=tau_hat_svd,
        residual_local=residual_loc,
        residual_svd=residual_svd,
    )


def _empty_series():
    keys = (
        "rms_amp",
        "rms_phase",
        "rms_kappa",
        "mse_amp",
        "mse_phase",
        "mse_kappa",
        "xi_true",
        "xi_after_local",
        "xi_after_svd",
    )
    return {k: [] for k in keys}


def _append_trial(series_local, series_svd, trial):
    for name in ("rms_amp", "rms_phase", "rms_kappa", "mse_amp", "mse_phase", "mse_kappa"):
        series_local[name].append(trial["local"][name])
        series_svd[name].append(trial["svd"][name])
    series_local["xi_true"].append(trial["xi_true"])
    series_local["xi_after_local"].append(trial["xi_after_local"])
    series_local["xi_after_svd"].append(trial["xi_after_svd"])
    series_svd["xi_true"].append(trial["xi_true"])
    series_svd["xi_after_local"].append(trial["xi_after_local"])
    series_svd["xi_after_svd"].append(trial["xi_after_svd"])


def _summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return dict(
        mean=float(arr.mean()),
        std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        sem=float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0,
        median=float(np.median(arr)),
    )






def experiment_xi(cfg: SimConfig, rng: Generator) -> dict:
    """Exp 3: Ξ before and after correction, plus residual scatter at fixed N."""
    n = cfg.n_fixed
    t0 = time.perf_counter()
    # keep one representative trial for scatter plots, and aggregates for bars
    aggregates = {g: dict(xi_true=[], xi_after=[], rms_kappa=[]) for g in GRAPH_ORDER + ("svd",)}
    showcase = None
    for trial_idx in range(cfg.n_trials):
        trial_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))
        tau, _ = sample_veils(n, cfg.s_amp, cfg.s_phase, trial_rng)
        Y, _psi, mask = generate_observations(tau, cfg.k_default, cfg.snr_default_db, trial_rng)
        tau_hat_svd = estimate_centralized_svd(Y)
        err_svd = kappa_errors(tau_hat_svd, tau)
        aggregates["svd"]["xi_true"].append(xi_empirical(tau))
        aggregates["svd"]["xi_after"].append(xi_empirical(err_svd["residual"]))
        aggregates["svd"]["rms_kappa"].append(err_svd["rms_kappa"])
        hats = {}
        for g in GRAPH_ORDER:
            A = make_graph(g, n, np.random.default_rng(trial_rng.integers(0, 2**63 - 1)), cfg.expander_degree)
            r = relative_veil_matrix(Y, A)
            tau_hat = estimate_decentralized(r, A, mode="converged")
            err = kappa_errors(tau_hat, tau)
            aggregates[g]["xi_true"].append(xi_empirical(tau))
            aggregates[g]["xi_after"].append(xi_empirical(err["residual"]))
            aggregates[g]["rms_kappa"].append(err["rms_kappa"])
            hats[g] = (tau_hat, err)
        if trial_idx == 0:
            showcase = dict(
                tau=tau,
                tau_hat_svd=tau_hat_svd,
                residual_svd=err_svd["residual"],
                hats={g: dict(tau_hat=hats[g][0], residual=hats[g][1]["residual"], err=hats[g][1]) for g in GRAPH_ORDER},
            )
    summary = {}
    for g, agg in aggregates.items():
        summary[g] = {name: _summarize(vals) for name, vals in agg.items()}
    return dict(
        n=n,
        k=cfg.k_default,
        snr_db=cfg.snr_default_db,
        xi_theory=cfg.theoretical_xi(),
        summary=summary,
        showcase=showcase,
        elapsed_s=time.perf_counter() - t0,
    )




def experiment_convergence(cfg: SimConfig, rng: Generator) -> dict:
    """Diagnostic: iterative local updates versus the converged LS/eigenvector."""
    n = cfg.n_fixed
    t_max = cfg.t_iter
    checkpoints = np.unique(np.clip(np.geomspace(1, t_max, 16).astype(int), 1, t_max))
    tau, _ = sample_veils(n, cfg.s_amp, cfg.s_phase, rng)
    Y, _psi, mask = generate_observations(tau, cfg.k_default, cfg.snr_default_db, rng)
    out = {}
    for g in GRAPH_ORDER:
        A = make_graph(g, n, rng, degree=cfg.expander_degree)
        r = relative_veil_matrix(Y, A)
        tau_inf = estimate_decentralized(r, A, mode="converged")
        # run iterations and record distance to the converged estimate and to truth
        eta = np.zeros(n, dtype=np.float64)
        u = np.ones(n, dtype=np.complex128)
        z = _symmetrized_log_amp(r, A)
        b = (A * z).sum(axis=1)
        deg = np.maximum(A.sum(axis=1), 1.0)
        H = _connection_matrix(r, A, unit_modulus=True)
        dist_inf = []
        rms_truth = []
        t_list = []
        for t in range(1, t_max + 1):
            eta = (A @ eta + b) / deg
            eta = eta - eta.mean()
            v = H @ u
            u = v / np.maximum(np.abs(v), 1e-15)
            if t in set(checkpoints) or t == t_max:
                tau_t = np.exp(eta) * u
                # distance to the converged estimator, after a relative gauge
                residual = residual_veils(tau_t, tau_inf)
                dist_inf.append(float(np.sqrt(np.mean(np.abs(np.log(residual)) ** 2))))
                rms_truth.append(kappa_errors(tau_t, tau)["rms_kappa"])
                t_list.append(t)
        out[g] = dict(
            t=t_list,
            dist_to_converged=dist_inf,
            rms_to_truth=rms_truth,
            rms_converged=kappa_errors(tau_inf, tau)["rms_kappa"],
        )
    return out


def experiment_gauge_demo(cfg: SimConfig, rng: Generator) -> dict:
    """Show that a common complex factor is invisible to cross-checks."""
    n = cfg.n_fixed
    tau, _ = sample_veils(n, cfg.s_amp, cfg.s_phase, rng)
    Y, psi, mask = generate_observations(tau, cfg.k_default, cfg.snr_default_db, rng)
    A = make_graph("torus", n, rng, cfg.expander_degree)
    r = relative_veil_matrix(Y, A)
    tau_hat = estimate_decentralized(r, A, mode="converged")
    # apply an arbitrary extra gauge to the true veils and to the estimates
    alpha_true = np.exp(0.35 + 1j * 1.10)
    err_raw_amp = np.sqrt(np.mean((np.log(np.abs(tau_hat)) - np.log(np.abs(tau))) ** 2))
    err_raw_phase = np.sqrt(np.mean(np.angle(tau / tau_hat) ** 2))
    err_fixed = kappa_errors(tau_hat, tau)
    err_fixed_scaled = kappa_errors(tau_hat, tau * alpha_true)
    err_fixed_hat_scaled = kappa_errors(tau_hat * (2.0 - 0.3j), tau)
    return dict(
        alpha_applied=complex(alpha_true),
        rms_amp_ungauged=float(err_raw_amp),
        rms_phase_ungauged=float(err_raw_phase),
        rms_kappa_gauged=err_fixed["rms_kappa"],
        rms_kappa_gauged_true_scaled=err_fixed_scaled["rms_kappa"],
        rms_kappa_gauged_hat_scaled=err_fixed_hat_scaled["rms_kappa"],
        mean_tau_hat=complex(np.mean(tau_hat)),
        mean_tau=complex(np.mean(tau)),
    )


# ---------------------------------------------------------------------------
# Serialization helpers (drop bulky arrays before JSON save)
# ---------------------------------------------------------------------------

def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        if obj.size > 4000:
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


def slim_scaling(summary: dict) -> dict:
    slim = {k: v for k, v in summary.items() if k != "raw"}
    return slim


def slim_xi(result: dict) -> dict:
    show = result.get("showcase")
    slim_show = None
    if show is not None:
        slim_show = {
            "tau": show["tau"],
            "residual_svd": show["residual_svd"],
            "residuals": {g: show["hats"][g]["residual"] for g in GRAPH_ORDER},
        }
    return dict(
        n=result["n"],
        k=result["k"],
        snr_db=result["snr_db"],
        xi_theory=result["xi_theory"],
        summary=result["summary"],
        showcase=slim_show,
        elapsed_s=result["elapsed_s"],
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------















# ===========================================================================
# Additional experiments and improved plots (override earlier definitions)
# ===========================================================================

EDGE_NOISE_SIGMA = 0.08  # independent per-edge log-amp and phase noise (B&H setting)




def _summarize_by_n(records_by_n: dict, n_list: tuple[int, ...]) -> dict:
    out = {"N": list(n_list)}
    metric_names = (
        "rms_amp",
        "rms_phase",
        "rms_kappa",
        "mse_amp",
        "mse_phase",
        "mse_kappa",
        "xi_true",
        "xi_after_local",
        "xi_after_svd",
    )
    for metric in metric_names:
        means, sems = [], []
        for n in n_list:
            st = _summarize(records_by_n[n][metric])
            means.append(st["mean"])
            sems.append(st["sem"])
        out[metric] = {"mean": means, "sem": sems}
    x = np.array(n_list, dtype=np.float64)
    out["fits"] = {
        "rms_amp_power": fit_power_law(x, np.array(out["rms_amp"]["mean"])),
        "rms_phase_power": fit_power_law(x, np.array(out["rms_phase"]["mean"])),
        "rms_kappa_power": fit_power_law(x, np.array(out["rms_kappa"]["mean"])),
        "mse_kappa_power": fit_power_law(x, np.array(out["mse_kappa"]["mean"])),
        "mse_kappa_logN": fit_log_n(x, np.array(out["mse_kappa"]["mean"])),
    }
    return out


def experiment_scaling_at_snr(
    cfg: SimConfig, rng: Generator, snr_db: float, progress: Callable | None = None
) -> dict:
    """Shared-input model, one SNR: gauge-fixed error vs N for each graph + SVD."""
    records = {g: {n: _empty_series() for n in cfg.n_list} for g in GRAPH_ORDER}
    records["svd"] = {n: _empty_series() for n in cfg.n_list}
    n_jobs = len(cfg.n_list) * cfg.n_trials
    done = 0
    t0 = time.perf_counter()
    for n in cfg.n_list:
        for _ in range(cfg.n_trials):
            trial_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))
            tau, _ = sample_veils(n, cfg.s_amp, cfg.s_phase, trial_rng)
            Y, _psi, _mask = generate_observations(tau, cfg.k_default, snr_db, trial_rng)
            tau_hat_svd = estimate_centralized_svd(Y)
            err_svd = kappa_errors(tau_hat_svd, tau)
            svd_trial = dict(
                xi_true=xi_empirical(tau),
                xi_after_local=np.nan,
                xi_after_svd=xi_empirical(err_svd["residual"]),
                local={
                    k: np.nan
                    for k in ("rms_amp", "rms_phase", "rms_kappa", "mse_amp", "mse_phase", "mse_kappa")
                },
                svd=err_svd,
            )
            _append_trial(_empty_series(), records["svd"][n], svd_trial)
            for g in GRAPH_ORDER:
                g_rng = np.random.default_rng(trial_rng.integers(0, 2**63 - 1))
                A = make_graph(g, n, g_rng, degree=cfg.expander_degree)
                r = relative_veil_matrix(Y, A)
                tau_hat = estimate_decentralized(r, A, mode="converged")
                err = kappa_errors(tau_hat, tau)
                trial = dict(
                    xi_true=xi_empirical(tau),
                    xi_after_local=xi_empirical(err["residual"]),
                    xi_after_svd=xi_empirical(err_svd["residual"]),
                    local=err,
                    svd=err_svd,
                )
                _append_trial(records[g][n], _empty_series(), trial)
            done += 1
            if progress and (done % 10 == 0 or done == n_jobs):
                progress(done, n_jobs, time.perf_counter() - t0)
    summary = {g: _summarize_by_n(by_n, cfg.n_list) for g, by_n in records.items()}
    summary["elapsed_s"] = time.perf_counter() - t0
    summary["snr_db"] = snr_db
    summary["model"] = "shared-input"
    return summary


def experiment_scaling(cfg: SimConfig, rng: Generator, progress: Callable | None = None) -> dict:
    """Exp 1, shared-input model, at the default SNR and at 0 dB.

    The 0 dB sweep is included because, in this model, topology only starts
    to matter once the ratios r_nm are quite noisy. See the notebook text.
    """
    t0 = time.perf_counter()
    snr_list = (cfg.snr_default_db, 0.0)
    by_snr = {}
    for snr in snr_list:
        by_snr[float(snr)] = experiment_scaling_at_snr(cfg, rng, snr_db=float(snr), progress=progress)
    elapsed = time.perf_counter() - t0
    # keep a backward-compatible top-level alias at the default SNR so older
    # plot code that expects summary['ring'] still works if called that way
    default = by_snr[float(cfg.snr_default_db)]
    out = dict(default)
    out["by_snr"] = by_snr
    out["elapsed_s"] = elapsed
    return out


def experiment_scaling_edge_noise(
    cfg: SimConfig, rng: Generator, sigma: float = EDGE_NOISE_SIGMA
) -> dict:
    """Exp 1b: independent per-edge noise (Barooah–Hespanha assumptions)."""
    records = {g: {n: _empty_series() for n in cfg.n_list} for g in GRAPH_ORDER}
    t0 = time.perf_counter()
    for n in cfg.n_list:
        for _ in range(cfg.n_trials):
            trial_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))
            tau, _ = sample_veils(n, cfg.s_amp, cfg.s_phase, trial_rng)
            for g in GRAPH_ORDER:
                g_rng = np.random.default_rng(trial_rng.integers(0, 2**63 - 1))
                A = make_graph(g, n, g_rng, degree=cfg.expander_degree)
                r = independent_edge_ratios(tau, A, sigma, g_rng)
                tau_hat = estimate_decentralized(r, A, mode="converged")
                err = kappa_errors(tau_hat, tau)
                trial = dict(
                    xi_true=xi_empirical(tau),
                    xi_after_local=xi_empirical(err["residual"]),
                    xi_after_svd=np.nan,
                    local=err,
                    svd={
                        k: np.nan
                        for k in ("rms_amp", "rms_phase", "rms_kappa", "mse_amp", "mse_phase", "mse_kappa")
                    },
                )
                _append_trial(records[g][n], _empty_series(), trial)
    summary = {g: _summarize_by_n(records[g], cfg.n_list) for g in GRAPH_ORDER}
    summary["elapsed_s"] = time.perf_counter() - t0
    summary["sigma"] = sigma
    summary["model"] = "independent-edge-noise"
    return summary


def experiment_snr_and_k(cfg: SimConfig, rng: Generator) -> dict:
    """Exp 2: error vs SNR and vs K at fixed N; now includes the ring."""
    n = cfg.n_fixed
    graphs = GRAPH_ORDER  # ring, torus, expander
    t0 = time.perf_counter()

    def sweep(param_name, values, fixed: dict):
        out = {
            g: {
                m: {"mean": [], "sem": []}
                for m in ("rms_amp", "rms_phase", "rms_kappa", "mse_kappa", "xi_after_local")
            }
            for g in graphs + ("svd",)
        }
        out["values"] = list(values)
        for val in values:
            kwargs = dict(fixed)
            kwargs[param_name] = val
            bucket = {g: _empty_series() for g in graphs + ("svd",)}
            for _ in range(cfg.n_trials):
                trial_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))
                tau, _ = sample_veils(n, cfg.s_amp, cfg.s_phase, trial_rng)
                Y, _psi, _mask = generate_observations(tau, kwargs["k"], kwargs["snr_db"], trial_rng)
                tau_hat_svd = estimate_centralized_svd(Y)
                err_svd = kappa_errors(tau_hat_svd, tau)
                svd_trial = dict(
                    xi_true=xi_empirical(tau),
                    xi_after_local=np.nan,
                    xi_after_svd=xi_empirical(err_svd["residual"]),
                    local={
                        k: np.nan
                        for k in ("rms_amp", "rms_phase", "rms_kappa", "mse_amp", "mse_phase", "mse_kappa")
                    },
                    svd=err_svd,
                )
                _append_trial(_empty_series(), bucket["svd"], svd_trial)
                for g in graphs:
                    A = make_graph(
                        g,
                        n,
                        np.random.default_rng(trial_rng.integers(0, 2**63 - 1)),
                        cfg.expander_degree,
                    )
                    r = relative_veil_matrix(Y, A)
                    tau_hat = estimate_decentralized(r, A, mode="converged")
                    err = kappa_errors(tau_hat, tau)
                    trial = dict(
                        xi_true=xi_empirical(tau),
                        xi_after_local=xi_empirical(err["residual"]),
                        xi_after_svd=xi_empirical(err_svd["residual"]),
                        local=err,
                        svd=err_svd,
                    )
                    _append_trial(bucket[g], _empty_series(), trial)
            for g in graphs + ("svd",):
                src = bucket[g]
                for metric in ("rms_amp", "rms_phase", "rms_kappa", "mse_kappa"):
                    st = _summarize(src[metric])
                    out[g][metric]["mean"].append(st["mean"])
                    out[g][metric]["sem"].append(st["sem"])
                if g == "svd":
                    st = _summarize(src["xi_after_svd"])
                else:
                    st = _summarize(src["xi_after_local"])
                out[g]["xi_after_local"]["mean"].append(st["mean"])
                out[g]["xi_after_local"]["sem"].append(st["sem"])
        return out

    vs_snr = sweep("snr_db", cfg.snr_list_db, dict(k=cfg.k_default, snr_db=cfg.snr_default_db))
    vs_k = sweep("k", cfg.k_list, dict(k=cfg.k_default, snr_db=cfg.snr_default_db))
    return dict(n=n, vs_snr=vs_snr, vs_k=vs_k, elapsed_s=time.perf_counter() - t0)


def experiment_tracking(cfg: SimConfig, rng: Generator, n_walks: int = 5) -> dict:
    """Exp 4: slow Ornstein–Uhlenbeck drift of κ_n(t), several walks.

    An OU process keeps the stationary variance of the veil at (s_a², s_p²),
    so the problem does not silently get harder over time (a plain random
    walk would). ρ close to 1 gives a long memory.
    """
    n = cfg.n_fixed
    k = cfg.k_track
    t_steps = cfg.t_track
    snr_db = cfg.snr_default_db
    graph_kind = "torus"
    rho = 0.97
    fixed_gains = (0.05, 0.20, 0.60)
    g_min, g_max = 0.05, 0.60
    names = [f"fixed g={g:.2f}" for g in fixed_gains] + ["adaptive g(Ξ)"]
    burn = min(50, t_steps // 5)

    rms_kappa_all = {name: np.zeros((n_walks, t_steps)) for name in names}
    rms_amp_all = {name: np.zeros((n_walks, t_steps)) for name in names}
    rms_phase_all = {name: np.zeros((n_walks, t_steps)) for name in names}
    xi_true_all = np.zeros((n_walks, t_steps))
    xi_after_all = {name: np.zeros((n_walks, t_steps)) for name in names}
    xi_proxy_all = {name: np.zeros((n_walks, t_steps)) for name in names}
    g_used_all = {name: np.zeros((n_walks, t_steps)) for name in names}

    t0 = time.perf_counter()
    for walk in range(n_walks):
        w_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))
        A = make_graph(graph_kind, n, w_rng, degree=cfg.expander_degree)
        tau, kappa = sample_veils(n, cfg.s_amp, cfg.s_phase, w_rng)
        kappa_hat = {name: np.zeros(n, dtype=np.complex128) for name in names}
        for t in range(t_steps):
            if t > 0:
                innov_a = cfg.s_amp * np.sqrt(1.0 - rho**2) * w_rng.standard_normal(n)
                innov_p = cfg.s_phase * np.sqrt(1.0 - rho**2) * w_rng.standard_normal(n)
                kappa = rho * kappa + innov_a + 1j * innov_p
                tau = np.exp(kappa)
            Y, _psi, _mask = generate_observations(tau, k, snr_db, w_rng)
            r = relative_veil_matrix(Y, A)
            tau_inst = estimate_decentralized(r, A, mode="converged")
            xi_true_all[walk, t] = xi_empirical(tau)
            for name in names:
                if name.startswith("fixed"):
                    g = float(name.split("=")[1])
                else:
                    tau_held = np.exp(kappa_hat[name])
                    proxy = xi_from_consistency(r, A, tau_held) if t > 0 else 0.5
                    g = g_min + (g_max - g_min) * float(np.clip(proxy, 0.0, 1.0))
                g_used_all[name][walk, t] = g
                tau_held = np.exp(kappa_hat[name])
                inst_aligned, _ = gauge_align(tau_inst, tau_held if t > 0 else tau_inst)
                kappa_inst = np.log(inst_aligned)
                if t == 0:
                    kappa_hat[name] = kappa_inst
                else:
                    kappa_hat[name] = (1.0 - g) * kappa_hat[name] + g * kappa_inst
                tau_hat = np.exp(kappa_hat[name])
                err = kappa_errors(tau_hat, tau)
                rms_amp_all[name][walk, t] = err["rms_amp"]
                rms_phase_all[name][walk, t] = err["rms_phase"]
                rms_kappa_all[name][walk, t] = err["rms_kappa"]
                xi_after_all[name][walk, t] = xi_empirical(err["residual"])
                xi_proxy_all[name][walk, t] = xi_from_consistency(r, A, tau_hat)

    def mean_sem(arr):
        return arr.mean(axis=0), (arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 else np.zeros(arr.shape[1]))

    rms_kappa_mean, rms_kappa_sem = {}, {}
    xi_after_mean, g_used_mean = {}, {}
    for name in names:
        rms_kappa_mean[name], rms_kappa_sem[name] = mean_sem(rms_kappa_all[name])
        xi_after_mean[name], _ = mean_sem(xi_after_all[name])
        g_used_mean[name], _ = mean_sem(g_used_all[name])
    xi_true_mean, _ = mean_sem(xi_true_all)
    xi_proxy_mean, _ = mean_sem(xi_proxy_all["adaptive g(Ξ)"][None, ...] if False else xi_proxy_all["adaptive g(Ξ)"])

    late = {}
    for name in names:
        late[name] = dict(
            rms_kappa=float(rms_kappa_all[name][:, burn:].mean()),
            rms_amp=float(rms_amp_all[name][:, burn:].mean()),
            rms_phase=float(rms_phase_all[name][:, burn:].mean()),
            rms_kappa_sem=float(rms_kappa_all[name][:, burn:].mean(axis=1).std(ddof=1) / np.sqrt(n_walks)) if n_walks > 1 else 0.0,
            g_mean=float(g_used_all[name][:, burn:].mean()),
        )
    return dict(
        n=n,
        k=k,
        t_steps=t_steps,
        snr_db=snr_db,
        graph=graph_kind,
        rho=rho,
        n_walks=n_walks,
        names=names,
        rms_kappa=rms_kappa_mean,
        rms_kappa_sem=rms_kappa_sem,
        rms_kappa_example={name: rms_kappa_all[name][0] for name in names},
        rms_amp={name: rms_amp_all[name].mean(axis=0) for name in names},
        rms_phase={name: rms_phase_all[name].mean(axis=0) for name in names},
        xi_true=xi_true_mean,
        xi_after=xi_after_mean,
        xi_proxy=xi_proxy_mean,
        g_used=g_used_mean,
        late=late,
        burn=burn,
        elapsed_s=time.perf_counter() - t0,
    )


# ----- improved plots -------------------------------------------------------











def format_conclusions(cfg: SimConfig, scaling, snr_k, xi_res, track, gauge, conv, edge=None) -> str:
    """Honest, number-backed write-up from this run."""
    lines = []

    def slope_line(block, g):
        fp = block[g]["fits"]["rms_kappa_power"]
        fm = block[g]["fits"]["mse_kappa_power"]
        fl = block[g]["fits"]["mse_kappa_logN"]
        return (
            f"   - {GRAPH_LABELS[g]:<20}  "
            f"RMS ∝ N^{fp['slope']:.3f} (r={fp['r_value']:.3f}),  "
            f"MSE ∝ N^{fm['slope']:.3f} (r={fm['r_value']:.3f}),  "
            f"MSE vs log N: slope={fl['slope']:.4f} (r={fl['r_value']:.3f})"
        )

    def rms_span(block, g):
        y0 = block[g]["rms_kappa"]["mean"][0]
        y1 = block[g]["rms_kappa"]["mean"][-1]
        n0, n1 = block[g]["N"][0], block[g]["N"][-1]
        return f"   - {GRAPH_LABELS[g]:<20}  {y0:.4f} → {y1:.4f}    (N={n0} → {n1})"

    lines.append("WHAT THIS RUN ACTUALLY SHOWED")
    lines.append("==============================")
    lines.append(
        f"SIZE={cfg.size},  s_a={cfg.s_amp}, s_p={cfg.s_phase},  "
        f"theoretical Ξ = exp(-(s_a²+s_p²)) = {cfg.theoretical_xi():.4f}."
    )
    lines.append(
        f"Default K={cfg.k_default}, default SNR={cfg.snr_default_db:.0f} dB, "
        f"{cfg.n_trials} trials, N up to {max(cfg.n_list)}."
    )
    lines.append("")
    lines.append("1. Gauge ambiguity is real, not a bookkeeping choice.")
    lines.append(
        f"   On the N={cfg.n_fixed} torus demo, the ungauged RMS errors were "
        f"amp={gauge['rms_amp_ungauged']:.4f}, phase={gauge['rms_phase_ungauged']:.4f}."
    )
    lines.append(
        f"   After removing the best common complex factor, RMS(κ) = {gauge['rms_kappa_gauged']:.4f}."
    )
    lines.append(
        f"   Scaling the true τ by an extra factor {gauge['alpha_applied']} left that "
        f"gauged error at {gauge['rms_kappa_gauged_true_scaled']:.4f}; "
        f"scaling τ̂ instead left it at {gauge['rms_kappa_gauged_hat_scaled']:.4f}."
    )
    lines.append(
        "   Cross-checks cannot see a common gain or a common phase. That is the "
        "mathematical form of Axiom 1 for this model: an absolute veil is not observable, "
        "only the relative balance between tiles."
    )
    lines.append("")
    lines.append("2. In the *shared-input* model, graph topology barely matters at high SNR.")
    by_snr = scaling.get("by_snr", {float(cfg.snr_default_db): scaling})
    for snr, block in sorted(by_snr.items(), key=lambda kv: -kv[0]):
        lines.append(f"   Shared inputs, SNR={snr:.0f} dB, K={cfg.k_default}. Fitted exponents:")
        keys = GRAPH_ORDER + (("svd",) if "svd" in block else ())
        for g in keys:
            lines.append(slope_line(block, g))
        lines.append("   Mean RMS(κ):")
        for g in keys:
            lines.append(rms_span(block, g))
    lines.append(
        "   Why the high-SNR curves are flat: each tile observes y_n = τ_n ψ + w_n with "
        "the *same* unknown ψ. The leading noise in log r_nm is then of the form ε_n − ε_m, "
        "a coboundary (a discrete gradient of a *node* potential). Least-squares on any "
        "connected graph recovers η + ε, so the residual does not accumulate along long "
        "paths. This is a different measurement model from Barooah–Hespanha, whose theorems "
        "assume *independent* noise on each edge."
    )
    lines.append(
        "   At 0 dB the higher-order terms (products of noises, and the division by "
        "⟨y_m, y_m⟩) break that coboundary structure. The ring, with poor connectivity, "
        "then loses ground against the torus, the expander, and the centralized SVD."
    )
    lines.append("")
    if edge is not None:
        lines.append("3. When we give the estimator independent per-edge noise (the B&H setting),")
        lines.append("   the literature scaling appears clearly.")
        lines.append(f"   Edge-noise σ = {edge.get('sigma', EDGE_NOISE_SIGMA):.2f}. Fitted exponents:")
        for g in GRAPH_ORDER:
            lines.append(slope_line(edge, g))
        lines.append("   Mean RMS(κ):")
        for g in GRAPH_ORDER:
            lines.append(rms_span(edge, g))
        lines.append(
            "   Expected MSE exponents: ring ~ 1, 2-D grid ~ log N (not a power of N), "
            "expander ~ 0. Expected RMS exponents are about half of those, i.e. ring ~ 1/2, "
            "expander ~ 0. The fits should be read against that, and against the modest N range."
        )
        item_next = 4
    else:
        item_next = 3
    lines.append("")
    lines.append(f"{item_next}. More samples and a higher SNR reduce the error, as they should.")
    vs_s = snr_k["vs_snr"]
    vs_k = snr_k["vs_k"]

    def fmt_curve(root, g, metric):
        return ", ".join(f"{v:.4f}" for v in root[g][metric]["mean"])

    for g in GRAPH_ORDER + ("svd",):
        if g in vs_s:
            lines.append(
                f"   {GRAPH_LABELS[g]} RMS(κ) vs SNR {list(vs_s['values'])}: {fmt_curve(vs_s, g, 'rms_kappa')}"
            )
    for g in GRAPH_ORDER + ("svd",):
        if g in vs_k:
            lines.append(
                f"   {GRAPH_LABELS[g]} RMS(κ) vs K   {list(vs_k['values'])}: {fmt_curve(vs_k, g, 'rms_kappa')}"
            )
    item_next += 1
    lines.append("")
    lines.append(f"{item_next}. Correction raises Ξ toward 1, but does not reach 1.")
    lines.append(
        f"   Theoretical Ξ of the drawn veils is {xi_res['xi_theory']:.4f}. "
        f"Empirical Ξ before correction was {xi_res['summary']['svd']['xi_true']['mean']:.4f} "
        f"± {xi_res['summary']['svd']['xi_true']['sem']:.4f}  "
        f"(so 1−Ξ ≈ {1-xi_res['summary']['svd']['xi_true']['mean']:.4f})."
    )
    for g in GRAPH_ORDER + ("svd",):
        a = xi_res["summary"][g]["xi_after"]
        e = xi_res["summary"][g]["rms_kappa"]
        lines.append(
            f"   - after {GRAPH_LABELS[g]}:  Ξ = {a['mean']:.6f} ± {a['sem']:.6f}  "
            f"(1−Ξ ≈ {1-a['mean']:.6f}),  RMS(κ) = {e['mean']:.4f} ± {e['sem']:.4f}"
        )
    lines.append(
        "   The leftover is measurement noise and finite K, not a bug in the estimator. "
        "It is consistent with Axiom 1: the veil can be balanced, not erased."
    )
    item_next += 1
    lines.append("")
    lines.append(f"{item_next}. A slowly drifting veil can be tracked with a first-order gain.")
    n_walks = track.get("n_walks", 1)
    lines.append(f"   Late-window mean RMS(κ) over {n_walks} OU walk(s), after the burn-in:")
    for name, st in track["late"].items():
        extra = f"  ± {st['rms_kappa_sem']:.4f}" if "rms_kappa_sem" in st else ""
        lines.append(
            f"   - {name:<18}  RMS(κ)={st['rms_kappa']:.4f}{extra},  "
            f"amp={st['rms_amp']:.4f}, phase={st['rms_phase']:.4f},  mean g={st['g_mean']:.3f}"
        )
    winner = min(track["late"].items(), key=lambda kv: kv[1]["rms_kappa"])
    adaptive = track["late"]["adaptive g(Ξ)"]["rms_kappa"]
    lines.append(
        f"   Lowest late error in this run: {winner[0]} at RMS(κ)={winner[1]['rms_kappa']:.4f}. "
        f"The adaptive rule ended at {adaptive:.4f}."
    )
    lines.append(
        "   That comparison is for a simple g(Ξ_proxy) rule and a modest number of walks; "
        "it is not a general claim that adaptive gain beats a well-tuned fixed g."
    )
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
        "- This is not a theorem. The N-scaling we fit is empirical over a modest range of N. "
        "The LS theory on graphs with independent edge noise is already known (Barooah & "
        "Hespanha). What would be new is a theorem about unveiling without a reference under "
        "the *shared-input* (or partial-overlap) model — including a precise statement of "
        "when the coboundary structure holds and when topology takes over."
    )
    lines.append(
        "- The main model gives every tile the same K samples ψ_k. If neighboring tiles never "
        "process related inputs, the ratio r_nm is not an estimate of τ_n/τ_m, and the method "
        "as written does not apply. A careful partial-overlap model (neighboring tiles sharing "
        "a window of a data stream) is left for a later notebook."
    )
    lines.append(
        "- A programming model in which the programmer declares Ξ_crit(c) is not implemented here."
    )
    lines.append(
        "- The local Jacobi / phase-sync iteration on a ring needs many more than a few hundred "
        "steps to reach the LS/eigenvector solution (see the convergence diagnostic). The "
        "scaling plots report that *converged* decentralized estimator, not a short iteration."
    )
    lines.append("")
    lines.append("RUNTIME")
    edge_t = f",  exp1b {edge.get('elapsed_s', float('nan')):.1f}s" if edge is not None else ""
    lines.append(
        f"  exp1 {scaling.get('elapsed_s', float('nan')):.1f}s{edge_t},  "
        f"exp2 {snr_k.get('elapsed_s', float('nan')):.1f}s,  "
        f"exp3 {xi_res.get('elapsed_s', float('nan')):.1f}s,  "
        f"exp4 {track.get('elapsed_s', float('nan')):.1f}s."
    )
    return "\n".join(lines)


def run_suite(cfg: SimConfig, outdir: Path, n_track_walks: int = 5, progress: Callable | None = None) -> dict:
    """Run every experiment, write PNGs and results.json, return the payload."""
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
    print(f"N list = {cfg.n_list}")
    print(f"trials = {cfg.n_trials}, K = {cfg.k_default}, SNR = {cfg.snr_default_db} dB")

    print("\n== gauge demo ==")
    gauge = experiment_gauge_demo(cfg, np.random.default_rng(rng.integers(0, 2**63 - 1)))
    print(
        f"  ungauged RMS amp={gauge['rms_amp_ungauged']:.4f}, "
        f"phase={gauge['rms_phase_ungauged']:.4f};  "
        f"gauged RMS(κ)={gauge['rms_kappa_gauged']:.4f}"
    )

    print("\n== convergence diagnostic ==")
    conv = experiment_convergence(cfg, np.random.default_rng(rng.integers(0, 2**63 - 1)))
    for g, d in conv.items():
        print(
            f"  {g}: last dist to converged = {d['dist_to_converged'][-1]:.3e}, "
            f"converged RMS = {d['rms_converged']:.4f}"
        )
    paths = {"convergence": plot_convergence(conv, outdir)}

    print("\n== experiment 1: shared-input scaling vs N (20 dB and 0 dB) ==")
    scaling = experiment_scaling(
        cfg, np.random.default_rng(rng.integers(0, 2**63 - 1)), progress=_prog
    )
    print(f"  elapsed {scaling['elapsed_s']:.1f}s")
    for snr, block in sorted(scaling["by_snr"].items(), key=lambda kv: -kv[0]):
        print(f"  -- SNR={snr:.0f} dB --")
        for g in tuple(GRAPH_ORDER) + (("svd",) if "svd" in block else ()):
            f = block[g]["fits"]
            print(
                f"  {g:10s}  RMS slope={f['rms_kappa_power']['slope']:.3f}  "
                f"MSE slope={f['mse_kappa_power']['slope']:.3f} "
                f"(r={f['mse_kappa_power']['r_value']:.3f})"
            )
    paths["exp1"] = plot_scaling(scaling, cfg, outdir)

    print("\n== experiment 1b: independent edge-noise scaling ==")
    edge = experiment_scaling_edge_noise(cfg, np.random.default_rng(rng.integers(0, 2**63 - 1)))
    print(f"  elapsed {edge['elapsed_s']:.1f}s")
    for g in GRAPH_ORDER:
        f = edge[g]["fits"]
        print(
            f"  {g:10s}  RMS slope={f['rms_kappa_power']['slope']:.3f} "
            f"(r={f['rms_kappa_power']['r_value']:.3f})  "
            f"MSE slope={f['mse_kappa_power']['slope']:.3f} "
            f"(r={f['mse_kappa_power']['r_value']:.3f})"
        )
    paths["exp1b"] = plot_scaling_edge_noise(edge, cfg, outdir)

    print("\n== experiment 2: SNR and K ==")
    snr_k = experiment_snr_and_k(cfg, np.random.default_rng(rng.integers(0, 2**63 - 1)))
    print(f"  elapsed {snr_k['elapsed_s']:.1f}s")
    paths["exp2"] = plot_snr_k(snr_k, cfg, outdir)

    print("\n== experiment 3: Ξ before/after ==")
    xi_res = experiment_xi(cfg, np.random.default_rng(rng.integers(0, 2**63 - 1)))
    print(f"  elapsed {xi_res['elapsed_s']:.1f}s")
    print(
        f"  Ξ theory={xi_res['xi_theory']:.4f}, "
        f"before={xi_res['summary']['svd']['xi_true']['mean']:.4f}"
    )
    paths["exp3"] = plot_xi(xi_res, cfg, outdir)

    print("\n== experiment 4: tracking ==")
    track = experiment_tracking(
        cfg, np.random.default_rng(rng.integers(0, 2**63 - 1)), n_walks=n_track_walks
    )
    print(f"  elapsed {track['elapsed_s']:.1f}s")
    for name, st in track["late"].items():
        print(f"  {name}: RMS(κ)={st['rms_kappa']:.4f}")
    paths["exp4"] = plot_tracking(track, cfg, outdir)

    wall = time.perf_counter() - wall0
    print(f"\nTOTAL WALL TIME {wall:.1f}s")
    conclusions = format_conclusions(cfg, scaling, snr_k, xi_res, track, gauge, conv, edge=edge)
    (outdir / "conclusions.txt").write_text(conclusions)

    def _slim_block(block: dict) -> dict:
        return {k: v for k, v in block.items() if k != "raw"}

    scaling_slim = dict(elapsed_s=scaling["elapsed_s"], by_snr={})
    for snr, block in scaling["by_snr"].items():
        scaling_slim["by_snr"][str(snr)] = _slim_block(block)

    payload = dict(
        size=cfg.size,
        theoretical_xi=cfg.theoretical_xi(),
        wall_time_s=wall,
        gauge=gauge,
        conv={g: dict(d) for g, d in conv.items()},
        scaling=scaling_slim,
        edge=_slim_block(edge),
        snr_k=snr_k,
        xi=slim_xi(xi_res),
        track={
            k: track[k]
            for k in (
                "n", "k", "t_steps", "snr_db", "graph", "rho", "n_walks",
                "names", "rms_kappa", "rms_kappa_sem", "rms_amp", "rms_phase",
                "xi_true", "xi_after", "xi_proxy", "g_used", "late", "burn", "elapsed_s",
            )
        },
        conclusions=conclusions,
        paths={k: str(v) for k, v in paths.items()},
    )
    with (outdir / "results.json").open("w") as f:
        json.dump(_json_safe(payload), f, indent=2)
    print("wrote", outdir / "results.json")
    print("wrote", outdir / "conclusions.txt")
    return payload
