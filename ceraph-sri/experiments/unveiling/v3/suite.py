"""Reproduction driver for the v3 round.

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

from sri.errorlaw import snr_linear
from sri.graphs import torus_adjacency
from sri.tracking import (
    WalkFilter,
    method_specs,
    q_multiplier,
    split_r,
    steady_state_kalman_gain,
)
from sri.unveiling import (
    estimate_decentralized,
    gauge_align,
    generate_observations,
    kappa_errors,
    relative_veil_matrix,
    sample_veils,
)

def _plot_stub(*_args, **_kwargs):
    """Plots are not written by the reproduction driver."""
    return []

plot_scenario = _plot_stub

plot_sensitivity = _plot_stub

plot_summary = _plot_stub

plot_theory = _plot_stub

ComplexArr = NDArray[np.complex128]
FloatArr = NDArray[np.float64]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# SIZE = "small" is meant to finish on a CPU in well under 5 minutes.
# SIZE = "large" uses more walks and a longer track; still CPU work.
SIZE_PRESETS = {
    "small": dict(
        n_track=64,
        k_track=16,
        t_track=300,
        n_walks=8,
        snr_db=20.0,
        assumed_snr_db=20.0,
        forget_factors=(0.90, 0.95, 0.97, 0.99),
        default_forget=0.97,
        n_static=8,
    ),
    "large": dict(
        n_track=64,
        k_track=16,
        t_track=400,
        n_walks=16,
        snr_db=20.0,
        assumed_snr_db=20.0,
        forget_factors=(0.90, 0.95, 0.97, 0.98, 0.99),
        default_forget=0.97,
        n_static=12,
    ),
}

DEFAULT_S_AMP = 0.25
DEFAULT_S_PHASE = 0.40
BASE_SEED = 7
OU_RHO = 0.97
FAULT_TILE = 0
FAULT_STEP = 0.5  # added to Re(κ) of FAULT_TILE
STAY_PROB = 0.95  # IMM mode self-transition
G_FIXED = (0.05, 0.20, 0.60, 0.90)
G_MIN_V1, G_MAX_V1 = 0.05, 0.60
Q_MIN_FACTOR = 1e-4  # q_min = this times the nominal q0 (per channel)
R_MIN_FACTOR = 1e-4
ADAPT_DELAY = 8  # steps of frozen q,r before adaptation starts
RECOVERY_FACTOR = 1.5
POST_EVENT_BUFFER = 20

METHOD_ORDER = [
    "oracle Kalman",
    "Kalman initial q",
    "fixed g=0.05",
    "fixed g=0.20",
    "fixed g=0.60",
    "fixed g=0.90",
    "v1 g(Ξ)",
    "Mehra AKF",
    "Sage–Husa AKF",
    "corrected g(Ξ_ν)",
    "IMM (q, 10q)",
]

PLOT_METHODS = [
    "oracle Kalman",
    "Kalman initial q",
    "fixed g=0.90",
    "v1 g(Ξ)",
    "Mehra AKF",
    "Sage–Husa AKF",
    "corrected g(Ξ_ν)",
    "IMM (q, 10q)",
]

METHOD_STYLE = {
    "oracle Kalman": dict(color="#111111", ls="-", lw=2.0),
    "Kalman initial q": dict(color="#555555", ls="--", lw=1.6),
    "fixed g=0.05": dict(color="#bbbbbb", ls=":", lw=1.0),
    "fixed g=0.20": dict(color="#999999", ls=":", lw=1.0),
    "fixed g=0.60": dict(color="#777777", ls="-.", lw=1.1),
    "fixed g=0.90": dict(color="#8c564b", ls="-.", lw=1.4),
    "v1 g(Ξ)": dict(color="#d62728", ls="--", lw=1.8),
    "Mehra AKF": dict(color="#1f77b4", ls="-", lw=1.8),
    "Sage–Husa AKF": dict(color="#2ca02c", ls="-", lw=1.8),
    "corrected g(Ξ_ν)": dict(color="#ff7f0e", ls="-", lw=1.8),
    "IMM (q, 10q)": dict(color="#9467bd", ls="-", lw=1.8),
}

# Documented flop counts per tile per step (both amplitude and phase channels).
# Arithmetic only; no unveiling / graph LS, which is shared across methods.
# These are engineering estimates, not profiled cycle counts.
FLOPS_ESTIMATE = {
    "oracle Kalman": 20,
    "Kalman initial q": 20,
    "fixed g=0.05": 8,
    "fixed g=0.20": 8,
    "fixed g=0.60": 8,
    "fixed g=0.90": 8,
    "v1 g(Ξ)": 40,  # IIR plus O(degree) consistency statistic (~degree 4 torus)
    "Mehra AKF": 36,
    "Sage–Husa AKF": 44,
    "corrected g(Ξ_ν)": 34,
    "IMM (q, 10q)": 72,
}

SCENARIO_KIND_ORDER = (
    "stationary",
    "jump_up",
    "jump_down",
    "ramp",
    "fault",
    "mismatch",
)


@dataclass
class SimConfig:
    size: str = "small"
    s_amp: float = DEFAULT_S_AMP
    s_phase: float = DEFAULT_S_PHASE
    seed: int = BASE_SEED
    n_track: int = 64
    k_track: int = 16
    t_track: int = 300
    n_walks: int = 8
    snr_db: float = 20.0
    assumed_snr_db: float = 20.0
    forget_factors: tuple[float, ...] = (0.90, 0.95, 0.97, 0.99)
    default_forget: float = 0.97
    n_static: int = 8
    rho_ou: float = OU_RHO

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
        """E[|τ|^(-p)] for τ = exp(η + iξ), η ~ N(0, s_a²). Equals exp(p² s_a² / 2)."""
        return float(np.exp((p**2) * self.s_amp**2 / 2.0))

    def process_q(self) -> dict[str, float]:
        fac = 1.0 - self.rho_ou**2
        q_amp = (self.s_amp**2) * fac
        q_phase = (self.s_phase**2) * fac
        return dict(q_amp=q_amp, q_phase=q_phase, q_kappa=q_amp + q_phase)

    def r_formula(self, snr_db: float | None = None, k: int | None = None) -> float:
        """Leading-term floor  E|τ|^{-2} / (K · SNR).  Total MSE(κ) units."""
        snr_db = self.snr_db if snr_db is None else snr_db
        k = self.k_track if k is None else k
        return self.e_tau_inv_pow(2) / (k * snr_linear(snr_db))


# ---------------------------------------------------------------------------
# Shared primitives (subset of v2, behaviour preserved)
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


def predicted_terms(cfg: SimConfig, k: int, snr_db: float) -> dict[str, float]:
    snr = snr_linear(snr_db)
    e2 = cfg.e_tau_inv_pow(2)
    floor = e2 / (k * snr)
    return dict(e_inv2=e2, floor=floor, snr=snr, k=k, snr_db=snr_db)


def static_measurement_variance(cfg: SimConfig, A: FloatArr, snr_db: float, k: int, rng: Generator, n_trials: int = 8) -> dict:
    """Empirical gauge-fixed MSE of the instantaneous estimator, no drift."""
    n = A.shape[0]
    acc = dict(mse_kappa=[], mse_amp=[], mse_phase=[])
    for _ in range(n_trials):
        tr = np.random.default_rng(rng.integers(0, 2**63 - 1))
        tau, _ = sample_veils(n, cfg.s_amp, cfg.s_phase, tr)
        Y, _ = generate_observations(tau, k, snr_db, tr)
        r = relative_veil_matrix(Y, A)
        tau_hat = estimate_decentralized(r, A)
        err = kappa_errors(tau_hat, tau)
        acc["mse_kappa"].append(err["mse_kappa"])
        acc["mse_amp"].append(err["mse_amp"])
        acc["mse_phase"].append(err["mse_phase"])
    return {m: _summarize(acc[m]) for m in acc}



# ---------------------------------------------------------------------------
# Scalar OU Kalman: Riccati, closed-form gain, channel update
# ---------------------------------------------------------------------------







# ---------------------------------------------------------------------------
# Adaptive filters (per tile, two channels, log domain)
# ---------------------------------------------------------------------------








# ---------------------------------------------------------------------------
# Scenarios, traces, metrics
# ---------------------------------------------------------------------------



def scenario_catalog(cfg: SimConfig) -> list[dict]:
    t_steps = cfg.t_track
    t_event = t_steps // 2
    assumed = cfg.assumed_snr_db
    true = cfg.snr_db
    return [
        dict(
            kind="stationary",
            title="stationary OU",
            true_snr_db=true,
            assumed_snr_db=assumed,
            t_event=None,
            has_event=False,
        ),
        dict(
            kind="jump_up",
            title="q × 10 jump up",
            true_snr_db=true,
            assumed_snr_db=assumed,
            t_event=t_event,
            has_event=True,
        ),
        dict(
            kind="jump_down",
            title="q ÷ 10 jump down",
            true_snr_db=true,
            assumed_snr_db=assumed,
            t_event=t_event,
            has_event=True,
        ),
        dict(
            kind="ramp",
            title="slowly ramping q (×1 → ×10)",
            true_snr_db=true,
            assumed_snr_db=assumed,
            t_event=t_event,
            has_event=True,
        ),
        dict(
            kind="fault",
            title=f"sudden step of {FAULT_STEP} in Re(κ) of one tile",
            true_snr_db=true,
            assumed_snr_db=assumed,
            t_event=t_event,
            has_event=True,
        ),
        dict(
            kind="mismatch",
            title="measurement-noise mismatch (true 10 dB, r̂ assumes 20 dB)",
            true_snr_db=10.0,
            assumed_snr_db=20.0,
            t_event=None,
            has_event=False,
        ),
    ]


def generate_walk_trace(
    cfg: SimConfig,
    A: FloatArr,
    rng: Generator,
    kind: str,
    true_snr_db: float,
    t_event: int | None,
) -> dict:
    """Simulate one walk: true κ(t), instantaneous unveiled τ(t), ratio matrix.

    The expensive unveiling is done once; every tracking method then sees the
    same measurements.
    """
    n = cfg.n_track
    k = cfg.k_track
    t_steps = cfg.t_track
    rho = cfg.rho_ou
    q = cfg.process_q()
    tau, kappa = sample_veils(n, cfg.s_amp, cfg.s_phase, rng)
    kappa_true = np.zeros((t_steps, n), dtype=np.complex128)
    tau_inst = np.zeros((t_steps, n), dtype=np.complex128)
    rmat = np.zeros((t_steps, n, n), dtype=np.complex128)
    q_amp_t = np.zeros(t_steps)
    q_phase_t = np.zeros(t_steps)
    for t in range(t_steps):
        scale = np.sqrt(q_multiplier(kind, t, t_event if t_event is not None else t_steps + 1, t_steps))
        if t > 0:
            innov_a = scale * cfg.s_amp * np.sqrt(1.0 - rho**2) * rng.standard_normal(n)
            innov_p = scale * cfg.s_phase * np.sqrt(1.0 - rho**2) * rng.standard_normal(n)
            kappa = rho * kappa + innov_a + 1j * innov_p
            if kind == "fault" and t_event is not None and t == t_event:
                kappa = kappa.copy()
                kappa[FAULT_TILE] = kappa[FAULT_TILE] + FAULT_STEP
            tau = np.exp(kappa)
        q_amp_t[t] = (scale**2) * q["q_amp"]
        q_phase_t[t] = (scale**2) * q["q_phase"]
        kappa_true[t] = kappa
        Y, _ = generate_observations(tau, k, true_snr_db, rng)
        R = relative_veil_matrix(Y, A)
        inst = estimate_decentralized(R, A)
        rmat[t] = R
        tau_inst[t] = inst
    return dict(
        kappa_true=kappa_true,
        tau_inst=tau_inst,
        rmat=rmat,
        q_amp_t=q_amp_t,
        q_phase_t=q_phase_t,
    )


def recovery_time(method_rms: FloatArr, oracle_rms: FloatArr, t_event: int, factor: float = RECOVERY_FACTOR) -> dict:
    """Earliest lag k >= 0 such that the method stays within factor × the oracle
    from t_event+k through the end of the track.

    A brief dip below the threshold that later blows up does *not* count as recovery.
    method_rms, oracle_rms have shape (n_walks, t_steps) or (t_steps,).
    Returns per-walk times (nan if never), plus summary stats in *steps after the event*.
    """
    m = np.asarray(method_rms, dtype=np.float64)
    o = np.asarray(oracle_rms, dtype=np.float64)
    if m.ndim == 1:
        m = m[None, :]
        o = o[None, :]
    n_walks, t_steps = m.shape
    times = np.full(n_walks, np.nan)
    for w in range(n_walks):
        ok = m[w, t_event:] <= factor * np.maximum(o[w, t_event:], 1e-18)
        # stay-within: ok[k:] all True
        stay = np.cumprod(ok[::-1]).astype(bool)[::-1]
        idx = np.flatnonzero(stay)
        if idx.size:
            times[w] = float(idx[0])
    finite = times[np.isfinite(times)]
    return dict(
        times=times,
        mean=float(finite.mean()) if finite.size else float("nan"),
        median=float(np.median(finite)) if finite.size else float("nan"),
        sem=float(finite.std(ddof=1) / np.sqrt(finite.size)) if finite.size > 1 else 0.0,
        frac_recovered=float(finite.size / n_walks),
        n_recovered=int(finite.size),
        n_walks=int(n_walks),
    )


def late_window_slice(kind: str, t_event: int | None, t_steps: int, burn: int) -> slice:
    if t_event is None:
        return slice(burn, t_steps)
    return slice(t_event + POST_EVENT_BUFFER, t_steps)


def run_filters_on_trace(
    cfg: SimConfig,
    A: FloatArr,
    trace: dict,
    specs: list[dict],
    r_amp_true: float,
    r_phase_true: float,
) -> dict:
    """Run every method on one precomputed walk. Returns per-method time series."""
    t_steps, n = trace["kappa_true"].shape
    out = {}
    for spec in specs:
        filt = WalkFilter(spec["name"], n, cfg, spec)
        rms_k = np.zeros(t_steps)
        rms_a = np.zeros(t_steps)
        rms_p = np.zeros(t_steps)
        g_t = np.zeros(t_steps)
        qhat_t = np.zeros(t_steps)
        rhat_t = np.zeros(t_steps)
        xi_t = np.zeros(t_steps)
        nis_t = np.zeros(t_steps)
        fault_err = np.zeros(t_steps)
        for t in range(t_steps):
            tau_true = np.exp(trace["kappa_true"][t])
            inst_raw = trace["tau_inst"][t]
            if t == 0:
                inst_aligned, _ = gauge_align(inst_raw, inst_raw)
            else:
                inst_aligned, _ = gauge_align(inst_raw, np.exp(filt.kappa))
            z = np.log(inst_aligned)
            filt.step(
                z,
                t,
                rmat=trace["rmat"][t],
                A=A,
                q_amp_true=float(trace["q_amp_t"][t]),
                q_phase_true=float(trace["q_phase_t"][t]),
                r_amp_true=r_amp_true,
                r_phase_true=r_phase_true,
            )
            tau_hat = np.exp(filt.kappa)
            err = kappa_errors(tau_hat, tau_true)
            rms_k[t] = err["rms_kappa"]
            rms_a[t] = err["rms_amp"]
            rms_p[t] = err["rms_phase"]
            g_t[t] = filt.last_g
            qhat_t[t] = filt.last_qhat
            rhat_t[t] = filt.last_rhat
            xi_t[t] = filt.last_xi_nu
            nis_t[t] = filt.last_nis
            # faulted-tile |log residual|
            res = err["residual"]
            fault_err[t] = float(np.abs(np.log(res[FAULT_TILE])))
        out[spec["name"]] = dict(
            rms_kappa=rms_k,
            rms_amp=rms_a,
            rms_phase=rms_p,
            g=g_t,
            qhat=qhat_t,
            rhat=rhat_t,
            xi=xi_t,
            nis=nis_t,
            fault_err=fault_err,
        )
    return out


def experiment_scenario(
    cfg: SimConfig,
    A: FloatArr,
    rng: Generator,
    scenario: dict,
    r_amp_assumed: float,
    r_phase_assumed: float,
    r_amp_true: float,
    r_phase_true: float,
    specs: list[dict] | None = None,
    n_walks: int | None = None,
    progress: Callable | None = None,
) -> dict:
    n_walks = int(n_walks if n_walks is not None else cfg.n_walks)
    t_steps = cfg.t_track
    burn = min(50, t_steps // 5)
    kind = scenario["kind"]
    t_event = scenario["t_event"]
    if specs is None:
        specs = method_specs(cfg, r_amp_assumed, r_phase_assumed)
    names = [s["name"] for s in specs]

    series = {
        name: dict(
            rms_kappa=np.zeros((n_walks, t_steps)),
            rms_amp=np.zeros((n_walks, t_steps)),
            rms_phase=np.zeros((n_walks, t_steps)),
            g=np.zeros((n_walks, t_steps)),
            qhat=np.zeros((n_walks, t_steps)),
            rhat=np.zeros((n_walks, t_steps)),
            xi=np.zeros((n_walks, t_steps)),
            nis=np.zeros((n_walks, t_steps)),
            fault_err=np.zeros((n_walks, t_steps)),
        )
        for name in names
    }
    q_amp_mean = np.zeros(t_steps)
    q_phase_mean = np.zeros(t_steps)

    t0 = time.perf_counter()
    for w in range(n_walks):
        w_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))
        trace = generate_walk_trace(cfg, A, w_rng, kind, scenario["true_snr_db"], t_event)
        q_amp_mean += trace["q_amp_t"]
        q_phase_mean += trace["q_phase_t"]
        ran = run_filters_on_trace(cfg, A, trace, specs, r_amp_true, r_phase_true)
        for name in names:
            for key in series[name]:
                series[name][key][w] = ran[name][key]
        if progress is not None:
            progress(w + 1, n_walks, time.perf_counter() - t0)

    q_amp_mean /= n_walks
    q_phase_mean /= n_walks

    def mean_sem(arr):
        mu = arr.mean(axis=0)
        if arr.shape[0] > 1:
            se = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
        else:
            se = np.zeros(arr.shape[1])
        return mu, se

    curves = {}
    for name in names:
        curves[name] = {}
        for key in ("rms_kappa", "rms_amp", "rms_phase", "g", "qhat", "rhat", "xi", "nis", "fault_err"):
            mu, se = mean_sem(series[name][key])
            curves[name][key] = mu
            curves[name][key + "_sem"] = se

    sl = late_window_slice(kind, t_event, t_steps, burn)
    sl_pre = slice(burn, t_event) if t_event is not None else None

    def late_stats_from(name, slc):
        per_k = series[name]["rms_kappa"][:, slc].mean(axis=1)
        per_a = series[name]["rms_amp"][:, slc].mean(axis=1)
        per_p = series[name]["rms_phase"][:, slc].mean(axis=1)
        per_f = series[name]["fault_err"][:, slc].mean(axis=1)
        return dict(
            rms_kappa=float(per_k.mean()),
            rms_kappa_sem=float(per_k.std(ddof=1) / np.sqrt(n_walks)) if n_walks > 1 else 0.0,
            rms_amp=float(per_a.mean()),
            rms_amp_sem=float(per_a.std(ddof=1) / np.sqrt(n_walks)) if n_walks > 1 else 0.0,
            rms_phase=float(per_p.mean()),
            rms_phase_sem=float(per_p.std(ddof=1) / np.sqrt(n_walks)) if n_walks > 1 else 0.0,
            fault_err=float(per_f.mean()),
            fault_err_sem=float(per_f.std(ddof=1) / np.sqrt(n_walks)) if n_walks > 1 else 0.0,
            g_mean=float(series[name]["g"][:, slc].mean()),
            qhat_mean=float(series[name]["qhat"][:, slc].mean()),
            rhat_mean=float(series[name]["rhat"][:, slc].mean()),
            per_walk_rms=per_k,
        )

    late = {name: late_stats_from(name, sl) for name in names}
    late_pre = {name: late_stats_from(name, sl_pre) for name in names} if sl_pre is not None else None

    oracle = "oracle Kalman"
    regret = {}
    for name in names:
        if name == oracle:
            regret[name] = dict(late=0.0, late_sem=0.0)
            continue
        # per-walk late regret vs the oracle on the same walks
        d = late[name]["per_walk_rms"] - late[oracle]["per_walk_rms"]
        regret[name] = dict(
            late=float(d.mean()),
            late_sem=float(d.std(ddof=1) / np.sqrt(n_walks)) if n_walks > 1 else 0.0,
        )

    recovery = {}
    overshoot = {}
    if t_event is not None:
        for name in names:
            if kind == "fault":
                recovery[name] = recovery_time(
                    series[name]["fault_err"], series[oracle]["fault_err"], t_event
                )
                # peak faulted-tile error after the event, mean over walks
                peak = series[name]["fault_err"][:, t_event:].max(axis=1)
                base = series[name]["fault_err"][:, max(t_event - 5, 0) : t_event].mean(axis=1)
                overshoot[name] = dict(
                    peak=float(peak.mean()),
                    peak_sem=float(peak.std(ddof=1) / np.sqrt(n_walks)) if n_walks > 1 else 0.0,
                    baseline=float(base.mean()),
                    peak_over_baseline=float(np.mean(peak / np.maximum(base, 1e-12))),
                )
            else:
                recovery[name] = recovery_time(
                    series[name]["rms_kappa"], series[oracle]["rms_kappa"], t_event
                )

    # drop bulky per-walk arrays from late (keep a compact copy)
    late_compact = {}
    per_walk_late = {}
    for name in names:
        rec = dict(late[name])
        per_walk_late[name] = rec.pop("per_walk_rms").tolist()
        late_compact[name] = rec
    late_pre_compact = None
    if late_pre is not None:
        late_pre_compact = {}
        for name in names:
            rec = dict(late_pre[name])
            rec.pop("per_walk_rms", None)
            late_pre_compact[name] = rec

    return dict(
        kind=kind,
        title=scenario["title"],
        n=cfg.n_track,
        k=cfg.k_track,
        t_steps=t_steps,
        n_walks=n_walks,
        rho_ou=cfg.rho_ou,
        true_snr_db=scenario["true_snr_db"],
        assumed_snr_db=scenario["assumed_snr_db"],
        t_event=t_event,
        burn=burn,
        names=names,
        curves=curves,
        late=late_compact,
        late_pre=late_pre_compact,
        regret=regret,
        recovery=recovery,
        overshoot=overshoot,
        q_amp_t=q_amp_mean,
        q_phase_t=q_phase_mean,
        r_amp_assumed=r_amp_assumed,
        r_phase_assumed=r_phase_assumed,
        r_amp_true=r_amp_true,
        r_phase_true=r_phase_true,
        elapsed_s=time.perf_counter() - t0,
        per_walk_late=per_walk_late,
    )





# ---------------------------------------------------------------------------
# Forgetting-factor sensitivity (reuses traces)
# ---------------------------------------------------------------------------

def experiment_forget_sensitivity(
    cfg: SimConfig,
    A: FloatArr,
    rng: Generator,
    r_amp_assumed: float,
    r_phase_assumed: float,
    r_by_snr: dict,
    n_walks: int | None = None,
) -> dict:
    """Mehra / Sage–Husa / corrected Ξ_ν vs λ, on *shared* traces.

    Restricted to stationary and jump_up.  Traces are generated once per
    (scenario, walk) and every λ is run on the same measurements, so the
    comparison is not confounded by a fresh unveiling draw.
    """
    n_walks = int(n_walks if n_walks is not None else cfg.n_walks)
    catalog = {s["kind"]: s for s in scenario_catalog(cfg)}
    kinds = ("stationary", "jump_up")
    adapt_kinds = ("mehra", "sage_husa", "xi_corrected")
    t0 = time.perf_counter()
    out = {}
    for kind in kinds:
        scen = catalog[kind]
        rtrue = r_by_snr[scen["true_snr_db"]]
        traces = []
        for _ in range(n_walks):
            w_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))
            traces.append(
                generate_walk_trace(cfg, A, w_rng, kind, scen["true_snr_db"], scen["t_event"])
            )
        block = {}
        oracle_spec = [s for s in method_specs(cfg, r_amp_assumed, r_phase_assumed) if s["name"] == "oracle Kalman"]
        # oracle once (independent of λ)
        oracle_series = dict(rms_kappa=np.zeros((n_walks, cfg.t_track)))
        oracle_late = []
        for w, tr in enumerate(traces):
            ran = run_filters_on_trace(cfg, A, tr, oracle_spec, rtrue["amp"], rtrue["phase"])
            oracle_series["rms_kappa"][w] = ran["oracle Kalman"]["rms_kappa"]
        sl = late_window_slice(kind, scen["t_event"], cfg.t_track, min(50, cfg.t_track // 5))
        oracle_late = oracle_series["rms_kappa"][:, sl].mean(axis=1)

        for lam in cfg.forget_factors:
            specs = [
                s
                for s in method_specs(cfg, r_amp_assumed, r_phase_assumed, forget=float(lam))
                if s["kind"] in adapt_kinds
            ]
            names = [s["name"] for s in specs]
            late_walks = {name: np.zeros(n_walks) for name in names}
            rec_store = {name: np.zeros((n_walks, cfg.t_track)) for name in names}
            for w, tr in enumerate(traces):
                ran = run_filters_on_trace(cfg, A, tr, specs, rtrue["amp"], rtrue["phase"])
                for name in names:
                    rec_store[name][w] = ran[name]["rms_kappa"]
                    late_walks[name][w] = ran[name]["rms_kappa"][sl].mean()
            entry = {}
            for name in names:
                per = late_walks[name]
                d = per - oracle_late
                entry[name] = dict(
                    rms_kappa=float(per.mean()),
                    rms_kappa_sem=float(per.std(ddof=1) / np.sqrt(n_walks)) if n_walks > 1 else 0.0,
                    regret=float(d.mean()),
                    regret_sem=float(d.std(ddof=1) / np.sqrt(n_walks)) if n_walks > 1 else 0.0,
                )
                if scen["t_event"] is not None:
                    rec = recovery_time(rec_store[name], oracle_series["rms_kappa"], scen["t_event"])
                    # drop the raw times array for a compact record
                    entry[name]["recovery_median"] = rec["median"]
                    entry[name]["frac_recovered"] = rec["frac_recovered"]
            block[str(lam)] = entry
        out[kind] = block
    return dict(
        by_scenario=out,
        forget_factors=list(cfg.forget_factors),
        n_walks=n_walks,
        elapsed_s=time.perf_counter() - t0,
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------



def _style(name: str) -> dict:
    st = METHOD_STYLE.get(name, dict(color="#333333", ls="-", lw=1.2))
    return dict(color=st["color"], ls=st["ls"], lw=st["lw"])


def _methods_for_plot(names) -> list[str]:
    return [m for m in PLOT_METHODS if m in names] or list(names)




def _shade(ax, t, mu, se, **style):
    ax.plot(t, mu, **style)
    if se is not None:
        ax.fill_between(t, mu - se, mu + se, color=style.get("color", "C0"), alpha=0.18, linewidth=0)








# ---------------------------------------------------------------------------
# Conclusions, JSON, suite
# ---------------------------------------------------------------------------

def _fmt(x, nd=4):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "NA"
        return f"{float(x):.{nd}f}"
    except Exception:
        return "NA"


def _beats(a: dict, b: dict, key="rms_kappa", sem_key=None) -> str:
    """'beats' / 'loses to' / 'ties' using combined s.e.m. of the late means."""
    sem_key = sem_key or (key + "_sem")
    da = a[key]
    db = b[key]
    sa = a.get(sem_key, 0.0)
    sb = b.get(sem_key, 0.0)
    comb = float(np.sqrt(sa * sa + sb * sb))
    if not np.isfinite(da) or not np.isfinite(db):
        return "is not comparable to"
    if da + 1e-15 < db and (db - da) > comb:
        return "beats"
    if db + 1e-15 < da and (da - db) > comb:
        return "loses to"
    return "ties (within combined s.e.m.)"


def _best_fixed(late: dict) -> tuple[str, dict]:
    cands = [(n, late[n]) for n in late if n.startswith("fixed g=")]
    cands.sort(key=lambda kv: kv[1]["rms_kappa"])
    return cands[0]


def format_table(result: dict, which="late") -> str:
    block = result[which]
    lines = []
    header = (
        f"{'method':<22}  {'RMS(κ)':>14}  {'amp':>14}  {'phase':>14}  "
        f"{'regret':>14}  {'g':>6}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for name in result["names"]:
        st = block[name]
        reg = result["regret"].get(name, {})
        lines.append(
            f"{name:<22}  "
            f"{st['rms_kappa']:.4f} ± {st['rms_kappa_sem']:.4f}  "
            f"{st['rms_amp']:.4f} ± {st['rms_amp_sem']:.4f}  "
            f"{st['rms_phase']:.4f} ± {st['rms_phase_sem']:.4f}  "
            f"{reg.get('late', 0.0):+.4f} ± {reg.get('late_sem', 0.0):.4f}  "
            f"{st['g_mean']:.3f}"
        )
    return "\n".join(lines)


def format_conclusions(cfg: SimConfig, results: dict, sens: dict, meta: dict) -> str:
    lines = []
    q = cfg.process_q()
    lines.append("WHAT THIS RUN ACTUALLY SHOWED")
    lines.append("=" * 30)
    lines.append(
        f"SIZE={cfg.size},  s_a={cfg.s_amp}, s_p={cfg.s_phase},  "
        f"theoretical Ξ = exp(-(s_a²+s_p²)) = {cfg.theoretical_xi():.4f}."
    )
    lines.append(
        f"OU ρ={cfg.rho_ou}, N={cfg.n_track}, K={cfg.k_track},  "
        f"t={cfg.t_track},  n_walks={cfg.n_walks},  torus,  seed={cfg.seed}."
    )
    lines.append(
        f"Process q0:  q_amp={q['q_amp']:.5f}, q_phase={q['q_phase']:.5f}, q_κ={q['q_kappa']:.5f}."
    )
    lines.append(
        f"Measurement r at 20 dB:  formula floor={meta['r_formula_20']:.6e},  "
        f"measured MSE={meta['r_meas_20']:.6e}  "
        f"(amp={meta['r_amp_meas_20']:.6e}, phase={meta['r_phase_meas_20']:.6e})."
    )
    lines.append(
        f"Measurement r at 10 dB:  formula floor={meta['r_formula_10']:.6e},  "
        f"measured MSE={meta['r_meas_10']:.6e}  "
        f"(amp={meta['r_amp_meas_10']:.6e}, phase={meta['r_phase_meas_10']:.6e})."
    )
    lines.append(
        f"Chip methods used r̂ from the 20 dB formula floor, split in the measured 20 dB "
        f"amp/phase proportion:  r_amp={meta['r_amp_assumed']:.6e}, r_phase={meta['r_phase_assumed']:.6e}."
    )
    g0a = steady_state_kalman_gain(cfg.rho_ou, q["q_amp"], meta["r_amp_assumed"])
    g0p = steady_state_kalman_gain(cfg.rho_ou, q["q_phase"], meta["r_phase_assumed"])
    lines.append(
        f"Steady-state Kalman gains at (q0, r̂):  g_amp={g0a['g']:.3f}, g_phase={g0p['g']:.3f}."
    )
    lines.append(
        f"Adaptive default forgetting λ={cfg.default_forget}; "
        f"adaptation frozen for the first {ADAPT_DELAY} steps.  "
        f"Sage–Husa uses constant d=1-λ (not the shrinking textbook schedule)."
    )
    lines.append("")

    adaptive = ["Mehra AKF", "Sage–Husa AKF", "corrected g(Ξ_ν)", "IMM (q, 10q)"]
    mean_regret = {m: [] for m in adaptive}
    collapse = {m: [] for m in adaptive}

    for kind in SCENARIO_KIND_ORDER:
        if kind not in results:
            continue
        res = results[kind]
        lines.append(f"{kind.upper().replace('_', ' ')}  —  {res['title']}")
        lines.append("-" * 40)
        extra = f"   true SNR={res['true_snr_db']:.0f} dB, assumed SNR={res['assumed_snr_db']:.0f} dB"
        if res["t_event"] is not None:
            extra += f", event at t={res['t_event']}"
        lines.append(extra + f",  elapsed {res['elapsed_s']:.1f}s.")
        if res.get("late_pre") is not None:
            lines.append("   Before the event:")
            for name in res["names"]:
                st = res["late_pre"][name]
                lines.append(
                    f"     - {name:<22}  RMS(κ)={st['rms_kappa']:.4f} ± {st['rms_kappa_sem']:.4f},  "
                    f"amp={st['rms_amp']:.4f}, phase={st['rms_phase']:.4f},  mean g={st['g_mean']:.3f}"
                )
            lines.append("   After the event (late window starts 20 steps later):")
        else:
            lines.append("   Late-window RMS(κ) (s.e.m. across walks):")
        for name in res["names"]:
            st = res["late"][name]
            reg = res["regret"].get(name, {})
            rec_txt = ""
            if name in res.get("recovery", {}):
                rec = res["recovery"][name]
                if rec["frac_recovered"] >= 1.0 - 1e-12:
                    rec_txt = f",  recovery {rec['median']:.0f} steps (all walks)"
                elif rec["frac_recovered"] <= 0.0:
                    rec_txt = ",  never recovered (0 walks within 1.5× oracle)"
                else:
                    rec_txt = (
                        f",  recovery median {rec['median']:.0f} steps "
                        f"({rec['n_recovered']}/{rec['n_walks']} walks)"
                    )
            ov_txt = ""
            if name in res.get("overshoot", {}):
                ov = res["overshoot"][name]
                ov_txt = (
                    f",  faulted-tile peak {ov['peak']:.4f} ± {ov['peak_sem']:.4f} "
                    f"({ov['peak_over_baseline']:.2f}× pre-event baseline)"
                )
            lines.append(
                f"     - {name:<22}  RMS(κ)={st['rms_kappa']:.4f} ± {st['rms_kappa_sem']:.4f},  "
                f"amp={st['rms_amp']:.4f} ± {st['rms_amp_sem']:.4f},  "
                f"phase={st['rms_phase']:.4f} ± {st['rms_phase_sem']:.4f},  "
                f"regret={reg.get('late', 0.0):+.4f} ± {reg.get('late_sem', 0.0):.4f},  "
                f"mean g={st['g_mean']:.3f}{rec_txt}{ov_txt}"
            )
        oracle = res["late"]["oracle Kalman"]
        stuck = res["late"]["Kalman initial q"]
        v1 = res["late"]["v1 g(Ξ)"]
        best_name, best_st = _best_fixed(res["late"])
        lines.append(
            f"   Best fixed IIR gain in this scenario: {best_name} at RMS(κ)={best_st['rms_kappa']:.4f} ± {best_st['rms_kappa_sem']:.4f}."
        )
        lines.append(
            f"   Oracle {_beats(oracle, stuck)} Kalman-at-initial-q,  "
            f"and {_beats(oracle, v1)} the v1 Ξ rule."
        )
        lines.append(
            f"   Kalman-at-initial-q {_beats(stuck, v1)} the v1 Ξ rule."
        )
        for m in adaptive:
            if m not in res["late"]:
                continue
            st = res["late"][m]
            lines.append(
                f"   {m} {_beats(st, stuck)} Kalman-at-initial-q,  "
                f"{_beats(st, v1)} v1 Ξ,  "
                f"{_beats(st, best_st)} {best_name},  "
                f"{_beats(st, oracle)} the oracle."
            )
            mean_regret[m].append(res["regret"][m]["late"])
            # collapse heuristic: late RMS more than 5× the oracle, or never recovers when an event exists
            collapsed = st["rms_kappa"] > 5.0 * max(oracle["rms_kappa"], 1e-8)
            if kind in ("jump_up", "jump_down", "ramp", "fault") and m in res.get("recovery", {}):
                if res["recovery"][m]["frac_recovered"] < 0.5:
                    collapsed = True
            if kind == "jump_up" and st["rms_kappa"] > 0.3:
                collapsed = True
            collapse[m].append((kind, collapsed))
        lines.append("")

    lines.append("FORGETTING-FACTOR SENSITIVITY")
    lines.append("-" * 40)
    lines.append(
        f"   Shared traces, n_walks={sens.get('n_walks', cfg.n_walks)},  "
        f"λ in {list(sens.get('forget_factors', cfg.forget_factors))},  "
        f"elapsed {sens.get('elapsed_s', float('nan')):.1f}s."
    )
    default_is_ok = True
    for kind, block in sens.get("by_scenario", {}).items():
        lines.append(f"   {kind}:")
        for name in adaptive[:3]:
            rows = []
            for lam in sens.get("forget_factors", []):
                rec = block.get(str(lam), {}).get(name)
                if rec is None:
                    continue
                rows.append((float(lam), rec["rms_kappa"], rec["rms_kappa_sem"], rec.get("regret", float("nan"))))
            if not rows:
                continue
            best = min(rows, key=lambda r: r[1])
            default = next((r for r in rows if abs(r[0] - cfg.default_forget) < 1e-12), None)
            row_txt = ",  ".join(f"λ={r[0]:.2f}: {r[1]:.4f}±{r[2]:.4f}" for r in rows)
            lines.append(f"     {name}:  {row_txt}.  Best λ={best[0]:.2f}.")
            if default is not None and best[1] > 0 and (default[1] - best[1]) / max(best[1], 1e-8) > 0.25:
                default_is_ok = False
                lines.append(
                    f"       Default λ={cfg.default_forget} is more than 25% worse than the best λ in this cell."
                )
    if default_is_ok:
        lines.append(
            f"   The default λ={cfg.default_forget} was within 25% of the best λ in every reported cell, "
            "so it is a reasonable single setting for this suite, not a fully tuned optimum."
        )
    lines.append("")

    lines.append("WHICH RULE TO ADOPT FOR THE SRI TILE")
    lines.append("-" * 40)
    # score: mean late regret across scenarios; reject methods that collapsed on jump_up
    scored = []
    for m in adaptive:
        regs = mean_regret[m]
        if not regs:
            continue
        n_collapse = sum(1 for _, c in collapse[m] if c)
        jump_collapse = any(k == "jump_up" and c for k, c in collapse[m])
        scored.append(dict(name=m, mean_regret=float(np.mean(regs)), n_collapse=n_collapse, jump_collapse=jump_collapse))
    # prefer no jump_up collapse, then fewer collapses, then lower mean regret
    scored.sort(key=lambda d: (d["jump_collapse"], d["n_collapse"], d["mean_regret"]))
    if scored:
        recm = scored[0]
        lines.append(
            f"   Recommended rule on this evidence: {recm['name']}.  "
            f"Mean late regret vs the oracle, averaged across the {len(mean_regret[recm['name']])} scenarios, "
            f"is {recm['mean_regret']:.4f}.  "
            f"Collapse flags (5× oracle, or recovery on fewer than half the walks, or jump-up RMS>0.3): "
            f"{recm['n_collapse']} scenario(s)."
        )
        # supporting comparisons on jump_up and mismatch, the two design-critical tests
        if "jump_up" in results:
            ju = results["jump_up"]
            st = ju["late"][recm["name"]]
            lines.append(
                f"   On the q×10 jump, {recm['name']} reached RMS(κ)={st['rms_kappa']:.4f} ± {st['rms_kappa_sem']:.4f} "
                f"versus oracle {ju['late']['oracle Kalman']['rms_kappa']:.4f}, "
                f"Kalman-at-initial-q {ju['late']['Kalman initial q']['rms_kappa']:.4f}, "
                f"and v1 Ξ {ju['late']['v1 g(Ξ)']['rms_kappa']:.4f}."
            )
            rec = ju.get("recovery", {}).get(recm["name"], {})
            if rec:
                if rec["frac_recovered"] <= 0:
                    lines.append("   It did not reach 1.5× the oracle's contemporaneous error after the jump in any walk.")
                else:
                    lines.append(
                        f"   Median recovery to within 1.5× the oracle was {rec['median']:.0f} steps "
                        f"({rec['n_recovered']}/{rec['n_walks']} walks)."
                    )
        if "mismatch" in results and recm["name"] in results["mismatch"]["late"]:
            mm = results["mismatch"]
            st = mm["late"][recm["name"]]
            lines.append(
                f"   On the r-mismatch test (true 10 dB, r̂ from a 20 dB formula), {recm['name']} "
                f"reached RMS(κ)={st['rms_kappa']:.4f} ± {st['rms_kappa_sem']:.4f} "
                f"versus oracle {mm['late']['oracle Kalman']['rms_kappa']:.4f} "
                f"and Kalman-at-initial-q {mm['late']['Kalman initial q']['rms_kappa']:.4f}."
            )
            # Mehra-specific caveat if Mehra lost badly here
            if "Mehra AKF" in mm["late"]:
                mh = mm["late"]["Mehra AKF"]
                sh = mm["late"].get("Sage–Husa AKF")
                if sh is not None and _beats(mh, sh) == "loses to":
                    lines.append(
                        "   Note: Mehra attributes all excess innovation power to q, so an underestimated r "
                        "can push its gain *up*, which is the wrong direction for extra measurement noise. "
                        "Sage–Husa adapts r as well; that difference is visible here if Sage–Husa wins this cell."
                    )
        # if the recommended method is not uniformly best, say so
        not_best = []
        for kind in SCENARIO_KIND_ORDER:
            if kind not in results:
                continue
            late = results[kind]["late"]
            best_ad = min(adaptive, key=lambda m: late[m]["rms_kappa"] if m in late else 1e9)
            if best_ad != recm["name"]:
                not_best.append((kind, best_ad, late[best_ad]["rms_kappa"], late[recm["name"]]["rms_kappa"]))
        # ignore ties at the reported precision
        not_best = [(k, o, ov, rv) for k, o, ov, rv in not_best if ov + 5e-5 < rv]
        if not_best:
            lines.append("   It is not uniformly best in every scenario:")
            for kind, other, ov, rv in not_best:
                lines.append(
                    f"     {kind}: {other} was lower ({ov:.4f} vs {rv:.4f}).  "
                    "The recommendation is a robustness compromise, not a per-cell optimum."
                )
        # Beating the oracle, when it happens, is not a claim of super-efficiency.
        # v2 already saw a Kalman filter with q×4 beat the filter with the true q;
        # a slightly higher gain can help in this finite-K, gauge-aligned model.
        beat_oracle = []
        for kind in SCENARIO_KIND_ORDER:
            if kind not in results or recm["name"] not in results[kind]["late"]:
                continue
            if results[kind]["regret"][recm["name"]]["late"] < -1e-4:
                beat_oracle.append(kind)
        if beat_oracle:
            lines.append(
                "   Where this rule has a slightly *lower* late RMS than the oracle "
                f"({', '.join(beat_oracle)}), that is consistent with v2, where a Kalman filter "
                "with q×4 beat the filter that used the true q.  A slightly higher gain can help "
                "in this finite-K, gauge-aligned model.  It is not evidence that the adaptive "
                "rule is more efficient than a well-specified Kalman filter in the large-sample limit."
            )
        # flop cost
        fl = FLOPS_ESTIMATE.get(recm["name"], None)
        if fl is not None:
            lines.append(
                f"   Documented filter cost is about {fl} flops per tile per step (both channels), "
                "versus ~20 for a fixed-q Kalman and ~72 for the two-model IMM.  "
                "These are engineering counts, not measured cycles.  The shared unveiling step "
                "(graph least-squares on the neighbour ratios) dominates any of these filter costs."
            )
        # v1
        v1_jump = results.get("jump_up", {}).get("late", {}).get("v1 g(Ξ)")
        if v1_jump is not None:
            lines.append(
                f"   The v1 Ξ-consistency rule is not recommended: after the q×10 jump it reached "
                f"RMS(κ)={v1_jump['rms_kappa']:.4f} ± {v1_jump['rms_kappa_sem']:.4f}.  "
                "That rule still lowers the gain when neighbour consistency falls, which is the wrong sign."
            )
    else:
        lines.append("   No adaptive method produced a usable score.  This is a failure of the suite or the estimators.")
    lines.append("")

    lines.append("WHAT REMAINS OPEN")
    lines.append("-" * 40)
    lines.append(
        "   - Identifiability.  Innovation variance alone cannot separate q from r:  "
        "E[ν²] = a²P + q + r.  The Mehra estimator is unbiased for q in steady state when r̂ = r "
        "and P equals the true posterior; otherwise the bias in r̂ is absorbed into q̂.  "
        "A lag-1 Mehra correction, or a Cramér–Rao bound for this blind model, was not derived here."
    )
    lines.append(
        "   - The forgetting factor is still a tuning knob.  Sensitivity is reported, but there is no "
        "automatic λ scheduler in the recommended rule."
    )
    lines.append(
        "   - Partial overlap.  v2 showed that layout matters as soon as neighbouring tiles share only "
        "part of their inputs.  All tracking walks here used full overlap on a 64-tile torus.  "
        "The theorem-plus-filter story under sliding-window overlap is untested."
    )
    lines.append(
        "   - The gauge is still a single complex factor for the whole network, computed from the "
        "held estimate.  A chip that cannot do that global step would need a reference tile or a "
        "different pin.  That choice was left open in v2 and is still open."
    )
    lines.append(
        "   - IMM used two models at {q0, 10 q0}.  The q÷10 jump is therefore a mismatched-model test, "
        "not a matched one.  A three-model bank, or a different pair, was not tried."
    )
    lines.append(
        "   - The fault model is a single real-part step of 0.5 on one tile.  Other faults "
        "(phase steps, multiple tiles, sticking) were not tried."
    )
    lines.append(
        "   - Settling time of the *unveiling* iteration (v2 experiment D) is not revisited; "
        "every filter here was fed the converged least-squares / leading-eigenvector estimate."
    )
    lines.append("")
    lines.append("WHAT THIS DOES NOT SHOW")
    lines.append("=" * 23)
    lines.append(
        "- This is a classical numerical simulation of complex arrays.  Using i and a Hilbert-space "
        "notation for a veil does not make the simulation quantum, and it does not demonstrate "
        "entanglement, a quantum advantage, or a new physical chip."
    )
    lines.append(
        "- There is no CMOS, timing-error, voltage, or energy model here.  The notebook does not "
        "compare a tile array to Razor-style detection or adaptive voltage scaling.  That measured-result "
        "claim still needs an FPGA or ASIC experiment."
    )
    lines.append(
        "- A programming model in which the programmer declares Ξ_crit(c) is not implemented here."
    )
    lines.append(
        "- Filter flop counts are documented estimates of arithmetic in the *filter*, not a chip "
        "area or clock-cycle model, and they exclude the shared unveiling solve."
    )
    lines.append("")
    lines.append("RUNTIME")
    parts = [f"  {k} {results[k]['elapsed_s']:.1f}s" for k in SCENARIO_KIND_ORDER if k in results]
    parts.append(f"sensitivity {sens.get('elapsed_s', float('nan')):.1f}s")
    parts.append(f"wall {meta.get('wall_time_s', float('nan')):.1f}s")
    lines.append("  " + ",  ".join(p.strip() for p in parts) + ".")
    return "\n".join(lines) + "\n"


def _json_safe(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj
    return str(obj)


def run_suite(cfg: SimConfig, outdir: Path, progress: Callable | None = None) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    wall0 = time.perf_counter()
    rng = np.random.default_rng(cfg.seed)
    A = torus_adjacency(cfg.n_track)
    print(f"SIZE={cfg.size}, N={cfg.n_track}, K={cfg.k_track}, t={cfg.t_track}, walks={cfg.n_walks}")
    print(f"theoretical Ξ={cfg.theoretical_xi():.6f},  default λ={cfg.default_forget}")

    print("\n== static measurement variance (for r) ==")
    static_20 = static_measurement_variance(
        cfg, A, 20.0, cfg.k_track, np.random.default_rng(rng.integers(0, 2**63 - 1)), n_trials=cfg.n_static
    )
    static_10 = static_measurement_variance(
        cfg, A, 10.0, cfg.k_track, np.random.default_rng(rng.integers(0, 2**63 - 1)), n_trials=cfg.n_static
    )
    r_formula_20 = cfg.r_formula(20.0)
    r_formula_10 = cfg.r_formula(10.0)
    r_amp_assumed, r_phase_assumed = split_r(
        r_formula_20, static_20["mse_amp"]["mean"], static_20["mse_phase"]["mean"]
    )
    print(
        f"  20 dB: formula={r_formula_20:.6e}, measured={static_20['mse_kappa']['mean']:.6e} "
        f"(amp={static_20['mse_amp']['mean']:.6e}, phase={static_20['mse_phase']['mean']:.6e})"
    )
    print(
        f"  10 dB: formula={r_formula_10:.6e}, measured={static_10['mse_kappa']['mean']:.6e} "
        f"(amp={static_10['mse_amp']['mean']:.6e}, phase={static_10['mse_phase']['mean']:.6e})"
    )
    print(f"  assumed r (formula 20 dB, measured split): amp={r_amp_assumed:.6e}, phase={r_phase_assumed:.6e}")

    r_by_snr = {
        20.0: dict(amp=static_20["mse_amp"]["mean"], phase=static_20["mse_phase"]["mean"], total=static_20["mse_kappa"]["mean"]),
        10.0: dict(amp=static_10["mse_amp"]["mean"], phase=static_10["mse_phase"]["mean"], total=static_10["mse_kappa"]["mean"]),
    }

    results = {}
    for scen in scenario_catalog(cfg):
        print(f"\n== scenario: {scen['kind']}  ({scen['title']}) ==")
        rtrue = r_by_snr[scen["true_snr_db"]]
        res = experiment_scenario(
            cfg,
            A,
            np.random.default_rng(rng.integers(0, 2**63 - 1)),
            scen,
            r_amp_assumed,
            r_phase_assumed,
            rtrue["amp"],
            rtrue["phase"],
        )
        results[scen["kind"]] = res
        print(f"  elapsed {res['elapsed_s']:.1f}s")
        for name, st in res["late"].items():
            print(
                f"  {name:<22}  RMS(κ)={st['rms_kappa']:.4f} ± {st['rms_kappa_sem']:.4f}  "
                f"g={st['g_mean']:.3f}"
            )

    print("\n== forgetting-factor sensitivity (shared traces) ==")
    sens = experiment_forget_sensitivity(
        cfg,
        A,
        np.random.default_rng(rng.integers(0, 2**63 - 1)),
        r_amp_assumed,
        r_phase_assumed,
        r_by_snr,
    )
    print(f"  elapsed {sens['elapsed_s']:.1f}s")

    print("\n== plots ==")
    paths = {}
    paths["theory_gain"] = str(plot_theory(cfg, r_amp_assumed, r_phase_assumed, outdir))
    for kind, res in results.items():
        paths[f"exp_{kind}"] = str(plot_scenario(res, outdir))
    paths["exp_summary"] = str(plot_summary(results, outdir))
    paths["exp_sensitivity"] = str(plot_sensitivity(sens, outdir))
    for k, pth in paths.items():
        print(f"  {k}: {pth}")

    wall = time.perf_counter() - wall0
    meta = dict(
        r_formula_20=r_formula_20,
        r_formula_10=r_formula_10,
        r_meas_20=static_20["mse_kappa"]["mean"],
        r_meas_10=static_10["mse_kappa"]["mean"],
        r_amp_meas_20=static_20["mse_amp"]["mean"],
        r_phase_meas_20=static_20["mse_phase"]["mean"],
        r_amp_meas_10=static_10["mse_amp"]["mean"],
        r_phase_meas_10=static_10["mse_phase"]["mean"],
        r_amp_assumed=r_amp_assumed,
        r_phase_assumed=r_phase_assumed,
        static_20=static_20,
        static_10=static_10,
        wall_time_s=wall,
        flops=FLOPS_ESTIMATE,
    )
    conclusions = format_conclusions(cfg, results, sens, meta)
    (outdir / "conclusions.txt").write_text(conclusions)
    payload = dict(
        cfg=dict(
            size=cfg.size,
            s_amp=cfg.s_amp,
            s_phase=cfg.s_phase,
            n_track=cfg.n_track,
            k_track=cfg.k_track,
            t_track=cfg.t_track,
            n_walks=cfg.n_walks,
            snr_db=cfg.snr_db,
            assumed_snr_db=cfg.assumed_snr_db,
            default_forget=cfg.default_forget,
            forget_factors=list(cfg.forget_factors),
            seed=cfg.seed,
            rho_ou=cfg.rho_ou,
        ),
        results=results,
        sensitivity=sens,
        meta=meta,
        paths=paths,
        conclusions=conclusions,
        wall_time_s=wall,
    )
    (outdir / "results.json").write_text(json.dumps(_json_safe(payload)))
    print(f"\nTOTAL WALL TIME {wall:.1f}s")
    print("wrote", outdir / "results.json")
    print("wrote", outdir / "conclusions.txt")
    return payload
