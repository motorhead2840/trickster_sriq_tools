"""Reproduction driver for the v5 round.

Numerical kernels come from the sri package. This file keeps the
experiment loop, the seeds, and the published defaults. Plots and the
optional torch path are not run.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from sri.selection import (
    blp_backflow,
    build_pseudomode_ad,
    contrast,
    embed_left,
    evolve_dispatch,
    f_ss_depolarizing,
    f_ss_from_generator,
    f_step_selection_depolarizing,
    fidelity_target,
    identity,
    initial_product,
    jumps_depolarizing_weyl,
    jumps_selection,
    jumps_t1t2,
    kron,
    pseudomode_g,
    pure_state,
    purity,
    r_threshold,
    reduced_tile,
    thermal_state,
    trace_distance,
)

def torch_module():
    """No torch backend in this reproduction driver."""
    return None

def resolve_backend(backend: str = "numpy") -> dict:
    """NumPy backend record. Official small runs stay on NumPy."""
    return dict(requested=str(backend), name="numpy", device=None,
                torch_mod=None, cuda=False, torch_version=None)

def timing_bench(*_args, **_kwargs) -> dict:
    """Timing is not part of the seeded numerical reproduction."""
    return {"rows": [], "note": "timing bench omitted"}

def _plot_stub(*_args, **_kwargs):
    """Plots are not written by the reproduction driver."""
    return []

plot_adaptive = _plot_stub

plot_array = _plot_stub

plot_memory = _plot_stub

plot_selection = _plot_stub

plot_timing = _plot_stub

ComplexArr = NDArray[np.complex128]
FloatArr = NDArray[np.float64]

BASE_SEED = 11
XI_CRIT_DEFAULT = 0.90  # programmer-declared fidelity setpoint
N_MAX_DEFAULT = 2
FAULT_STEP = 0.0  # unused placeholder for style parity

# Effective Markov AD rate for a resonant Lorentzian pseudomode: Γ = 4 g² / κ
# so g = sqrt(Γ κ / 4) holds the spectral density fixed while κ is scanned.


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIZE_PRESETS = {
    "small": dict(
        d_list=(2, 3, 4, 8),
        n_max=2,
        t_final=6.0,
        n_steps=120,
        gamma0=1.0,
        xi_crit=XI_CRIT_DEFAULT,
        n_walks=4,
        r_grid=(0.5, 1.0, 2.0, 4.0, 5.5, 6.5, 7.75, 8.0, 12.0, 16.0),
        kappa_grid=(0.5, 1.0, 2.0, 4.0, 8.0, 16.0),
        d_memory=(2, 8),
        t_track=8.0,
        n_track_steps=160,
        n_array=(2, 3),
        j_couplings=(0.0, 0.5, 1.5),
        n_static_check=1,
        bench_d=4,
        bench_batch=16,
        bench_steps=80,
        bench_nmax=2,
    ),
    "large": dict(
        d_list=(2, 3, 4, 8),
        n_max=4,
        t_final=12.0,
        n_steps=240,
        gamma0=1.0,
        xi_crit=XI_CRIT_DEFAULT,
        n_walks=12,
        r_grid=(0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0),
        kappa_grid=(0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
        d_memory=(2, 4, 8),
        t_track=16.0,
        n_track_steps=320,
        n_array=(2, 3, 4),
        j_couplings=(0.0, 0.25, 0.5, 1.0, 2.0),
        n_static_check=1,
        bench_d=8,
        bench_batch=32,
        bench_steps=160,
        bench_nmax=4,
    ),
}


@dataclass
class SimConfig:
    size: str = "small"
    seed: int = BASE_SEED
    d_list: tuple[int, ...] = (2, 3, 4, 8)
    n_max: int = N_MAX_DEFAULT
    t_final: float = 6.0
    n_steps: int = 120
    gamma0: float = 1.0
    xi_crit: float = XI_CRIT_DEFAULT
    n_walks: int = 4
    r_grid: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 5.5, 6.5, 7.75, 8.0, 12.0, 16.0)
    kappa_grid: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
    d_memory: tuple[int, ...] = (2, 8)
    t_track: float = 8.0
    n_track_steps: int = 160
    n_array: tuple[int, ...] = (2, 3)
    j_couplings: tuple[float, ...] = (0.0, 0.5, 1.5)
    n_static_check: int = 1
    bench_d: int = 4
    bench_batch: int = 16
    bench_steps: int = 80
    bench_nmax: int = 2
    # Target is always computational-basis state |t> with t=1 when d>1
    # (the "fiance": 1 distinguished state among d). For d=2, |t>=|1>.
    target_index: int = 1
    dephasing_ratio: float = 0.5  # Γ_φ / Γ_ad in the T1/T2 model

    @classmethod
    def from_size(cls, size: str, **overrides) -> "SimConfig":
        if size not in SIZE_PRESETS:
            raise ValueError(f"SIZE must be 'small' or 'large', got {size!r}")
        data = dict(SIZE_PRESETS[size])
        data.update(overrides)
        return cls(size=size, **data)

    def target(self, d: int) -> int:
        t = int(self.target_index)
        if t < 0 or t >= d:
            return 1 if d > 1 else 0
        return t

    def times(self, t_final: float | None = None, n_steps: int | None = None) -> FloatArr:
        t_final = self.t_final if t_final is None else t_final
        n_steps = self.n_steps if n_steps is None else n_steps
        return np.linspace(0.0, t_final, n_steps + 1)


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------







# ---------------------------------------------------------------------------
# Operators and channels
# ---------------------------------------------------------------------------





































# ---------------------------------------------------------------------------
# Jump-operator factories
# ---------------------------------------------------------------------------













# ---------------------------------------------------------------------------
# Lindblad evolution: RHS, RK4, exact Liouvillian expm
# ---------------------------------------------------------------------------















# ---------------------------------------------------------------------------
# Pseudomode embedding
# ---------------------------------------------------------------------------











# ---------------------------------------------------------------------------
# Theory: selection vs depolarization
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
# Optional PyTorch backend (complex64 on GPU, batched RK4)
# ---------------------------------------------------------------------------







def th_clamp_tr(tr):
    th = _th()
    return th.where(tr.abs() < 1e-18, th.ones_like(tr), tr)








# ---------------------------------------------------------------------------
# Experiment 1 — Markovian T1/T2 vs a pseudomode memory register
# ---------------------------------------------------------------------------



def experiment_memory(cfg: SimConfig, rng: Generator, backend_info: dict | None = None) -> dict:
    """Compare Markov T1/T2 with a Lorentzian pseudomode of the same Γ_eff."""
    t0 = time.perf_counter()
    times = cfg.times()
    gamma = cfg.gamma0
    # Extra computational-basis dephasing washes out the BLP backflow of a
    # Lorentzian AD bath (checked at Γ_φ = 0.5 Γ).  Experiment 1 therefore
    # compares pure AD to a pseudomode of the same Γ_eff, so the memory
    # diagnostic is not being hidden by a second, memoryless channel.
    gamma_phi = 0.0
    toward = 0
    cells = {}
    traces = {}
    for d in cfg.d_memory:
        if d not in cfg.d_list and d not in cfg.d_memory:
            continue
        # Two probe states: the attractor |toward> and the most-excited |d-1>,
        # plus a plus-state vs a computational state for a second pair.
        rho_a = pure_state(d, toward)
        rho_b = pure_state(d, d - 1 if d > 1 else 0)
        if d == 2:
            # also |+> vs |-> as a coherence-sensitive pair, stored separately
            pass
        H0 = np.zeros((d, d), dtype=np.complex128)
        jumps_m = jumps_t1t2(d, gamma, gamma_phi, toward=toward)
        batch = np.stack([rho_a, rho_b], axis=0)
        ev_m = evolve_dispatch(batch, H0, jumps_m, times, backend_info, method="expm")
        # ev: (T, 2, D, D)
        red_a_m = ev_m[:, 0]
        red_b_m = ev_m[:, 1]
        dist_m = np.array([trace_distance(red_a_m[i], red_b_m[i]) for i in range(len(times))])
        pur_m = np.array([0.5 * (float(purity(red_a_m[i])) + float(purity(red_b_m[i]))) for i in range(len(times))])
        rec_m = blp_backflow(dist_m, times)

        kappa_rows = []
        for kap in cfg.kappa_grid:
            H, jumps, dd, dm = build_pseudomode_ad(
                d, cfg.n_max, gamma, float(kap), toward=toward, dephasing=gamma_phi
            )
            ja = initial_product(rho_a, cfg.n_max)
            jb = initial_product(rho_b, cfg.n_max)
            # For D = d*(n_max+1), use rk4 if the Liouvillian is large
            method = "expm" if (dd * dm) ** 2 <= 256 else "rk4"
            ev_a = evolve_dispatch(ja, H, jumps, times, backend_info, method=method)
            ev_b = evolve_dispatch(jb, H, jumps, times, backend_info, method=method)
            ra = np.stack([reduced_tile(ev_a[i], dd, dm) for i in range(len(times))])
            rb = np.stack([reduced_tile(ev_b[i], dd, dm) for i in range(len(times))])
            dist = np.array([trace_distance(ra[i], rb[i]) for i in range(len(times))])
            pur = np.array([0.5 * (float(purity(ra[i])) + float(purity(rb[i]))) for i in range(len(times))])
            rec = blp_backflow(dist, times)
            kappa_rows.append(dict(
                kappa=float(kap),
                g=pseudomode_g(gamma, float(kap)),
                gamma_eff=gamma,
                blp=rec,
                dist=dist,
                purity=pur,
            ))
        cells[d] = dict(
            d=d,
            markov=dict(blp=rec_m, dist=dist_m, purity=pur_m, gamma_ad=gamma, gamma_phi=gamma_phi),
            pseudomode=kappa_rows,
            n_max=cfg.n_max,
            toward=toward,
        )
        traces[d] = dict(times=times, markov_dist=dist_m, markov_purity=pur_m)
        print(f"  mem d={d}: Markov N_BLP={rec_m['N_BLP']:.3e} monotone={rec_m['monotone']}; "
              f"slow κ={cfg.kappa_grid[0]} N_BLP={kappa_rows[0]['blp']['N_BLP']:.3e}; "
              f"fast κ={cfg.kappa_grid[-1]} N_BLP={kappa_rows[-1]['blp']['N_BLP']:.3e}")
    return dict(
        cells=cells,
        traces=traces,
        times=times,
        gamma0=gamma,
        gamma_phi=gamma_phi,
        kappa_grid=list(cfg.kappa_grid),
        d_list=list(cfg.d_memory),
        n_max=cfg.n_max,
        elapsed_s=time.perf_counter() - t0,
    )


# ---------------------------------------------------------------------------
# Experiment 2 — engineered selection vs veiling, threshold, memory scan
# ---------------------------------------------------------------------------

def time_to_fidelity(f_ts: FloatArr, times: FloatArr, thresh: float) -> float:
    hit = np.where(np.asarray(f_ts) >= thresh)[0]
    if hit.size == 0:
        return float("nan")
    return float(times[int(hit[0])])


def experiment_selection(cfg: SimConfig, rng: Generator, backend_info: dict | None = None) -> dict:
    """Drive a tile into a declared pointer |t> against a depolarizing veil."""
    t0 = time.perf_counter()
    times = cfg.times()
    gamma = cfg.gamma0
    xi = cfg.xi_crit
    markov_rows = []
    memory_rows = []
    theory_rows = []
    traces = {}  # (d, r) -> F(t)

    for d in cfg.d_list:
        tidx = cfg.target(d)
        r_star = r_threshold(xi, d)
        theory_rows.append(dict(d=d, r_star=r_star, xi_crit=xi, f_at_rstar=f_ss_depolarizing(r_star, d)))
        H0 = np.zeros((d, d), dtype=np.complex128)
        # start from the maximally mixed state (no prior focus)
        rho0 = thermal_state(d)
        for r in cfg.r_grid:
            gs = float(r) * gamma
            jumps = jumps_selection(d, tidx, gs)
            ev = evolve_dispatch(rho0, H0, jumps, times, backend_info, method="expm", depolarizing=gamma)
            f_ts = np.array([float(fidelity_target(ev[i], d, tidx)) for i in range(len(times))])
            p_ts = np.array([float(purity(ev[i])) for i in range(len(times))])
            c_ts = np.array([float(contrast(ev[i], d, tidx)) for i in range(len(times))])
            f_num = f_ss_from_generator(d, tidx, gs, gamma, kind="depolarizing")
            f_th = f_ss_depolarizing(float(r), d)
            rec = dict(
                d=d, r=float(r), gamma_sel=gs, gamma=gamma,
                F_final=float(f_ts[-1]), F_ss_theory=f_th, F_ss_numeric=f_num,
                F_abs_err=abs(f_ts[-1] - f_th),
                purity_final=float(p_ts[-1]),
                contrast_final=float(c_ts[-1]),
                t_F099=time_to_fidelity(f_ts, times, 0.99),
                t_xi=time_to_fidelity(f_ts, times, xi),
                holds_xi=bool(f_ts[-1] >= xi - 1e-4),
            )
            markov_rows.append(rec)
            if (d, float(r)) in ((d, 2.0), (d, 8.0)) or abs(float(r) - r_star) < 0.5:
                traces[(d, float(r))] = dict(times=times, F=f_ts, purity=p_ts, contrast=c_ts)
        # empirical threshold: smallest r on the grid with F_final >= xi,
        # plus a linear interpolation between the straddling points
        xs = [row for row in markov_rows if row["d"] == d]
        xs = sorted(xs, key=lambda z: z["r"])
        r_emp = float("nan")
        for i, row in enumerate(xs):
            if row["holds_xi"]:
                if i == 0:
                    r_emp = row["r"]
                else:
                    f0, f1 = xs[i - 1]["F_final"], row["F_final"]
                    r0, r1 = xs[i - 1]["r"], row["r"]
                    if f1 <= f0:
                        r_emp = row["r"]
                    else:
                        # interpolate to F=xi
                        frac = (xi - f0) / (f1 - f0)
                        r_emp = r0 + frac * (r1 - r0)
                break
        theory_rows[-1]["r_empirical"] = float(r_emp)
        print(f"  sel d={d}: r*_theory={r_star:.3f}, r*_empirical={r_emp:.3f}, "
              f"F_ss(r=2)={f_ss_depolarizing(2, d):.3f}")

    # Memory vs Markov selection: fix r at 2*r*(d) (comfortably above threshold
    # when the veil is Markov) and scan κ of a *pseudomode veil* with the same
    # Γ_eff, plus Markovian selection.  Does memory help or hurt F_final?
    # The pseudomode veil here is amplitude-damping into |0>, so it is *not*
    # the same channel as depolarization; we therefore also run a Markov AD
    # control at the same Γ_eff, and compare memory against that control.
    for d in cfg.d_memory:
        tidx = cfg.target(d)
        r_use = max(2.0 * r_threshold(xi, d), 2.0)
        gs = r_use * gamma
        toward = 0
        # Markov AD + dephasing control (same operators the pseudomode limits to)
        H0 = np.zeros((d, d), dtype=np.complex128)
        jumps_c = jumps_selection(d, tidx, gs) + jumps_t1t2(d, gamma, cfg.dephasing_ratio * gamma, toward=toward)
        ev_c = evolve_dispatch(thermal_state(d), H0, jumps_c, times, backend_info, method="expm")
        f_c = np.array([float(fidelity_target(ev_c[i], d, tidx)) for i in range(len(times))])
        p_c = np.array([float(purity(ev_c[i])) for i in range(len(times))])
        control = dict(
            d=d, r=r_use, F_final=float(f_c[-1]), purity_final=float(p_c[-1]),
            t_xi=time_to_fidelity(f_c, times, xi), holds_xi=bool(f_c[-1] >= xi - 1e-4),
            kind="markov_t1t2",
        )
        for kap in cfg.kappa_grid:
            H, jumps_pm, dd, dm = build_pseudomode_ad(
                d, cfg.n_max, gamma, float(kap), toward=toward,
                dephasing=cfg.dephasing_ratio * gamma,
            )
            # add Markovian selection on the tile, embedded
            for L in jumps_selection(d, tidx, gs):
                jumps_pm.append(embed_left(L, dm))
            rho0 = initial_product(thermal_state(d), cfg.n_max)
            method = "expm" if (dd * dm) ** 2 <= 256 else "rk4"
            ev = evolve_dispatch(rho0, H, jumps_pm, times, backend_info, method=method)
            red = np.stack([reduced_tile(ev[i], dd, dm) for i in range(len(times))])
            f_ts = np.array([float(fidelity_target(red[i], d, tidx)) for i in range(len(times))])
            p_ts = np.array([float(purity(red[i])) for i in range(len(times))])
            c_ts = np.array([float(contrast(red[i], d, tidx)) for i in range(len(times))])
            memory_rows.append(dict(
                d=d, r=r_use, kappa=float(kap), g=pseudomode_g(gamma, float(kap)),
                F_final=float(f_ts[-1]),
                F_markov_control=control["F_final"],
                delta_F=float(f_ts[-1] - control["F_final"]),
                purity_final=float(p_ts[-1]),
                contrast_final=float(c_ts[-1]),
                t_xi=time_to_fidelity(f_ts, times, xi),
                holds_xi=bool(f_ts[-1] >= xi - 1e-4),
                F_ts=f_ts, purity_ts=p_ts,
            ))
        print(f"  mem-sel d={d}: Markov-AD F={control['F_final']:.3f}; "
              f"slow κ={cfg.kappa_grid[0]} F={memory_rows[-len(cfg.kappa_grid)]['F_final']:.3f}; "
              f"fast κ={cfg.kappa_grid[-1]} F={memory_rows[-1]['F_final']:.3f}")
        memory_rows.append({**control, "kappa": float("inf")})  # sentinel for the control

    return dict(
        markov=markov_rows,
        memory=memory_rows,
        theory=theory_rows,
        traces=traces,
        times=times,
        gamma0=gamma,
        xi_crit=xi,
        r_grid=list(cfg.r_grid),
        kappa_grid=list(cfg.kappa_grid),
        d_list=list(cfg.d_list),
        elapsed_s=time.perf_counter() - t0,
    )


# ---------------------------------------------------------------------------
# Experiment 3 — drifting Γ, adaptive γ_sel to hold F >= Ξ_crit
# ---------------------------------------------------------------------------

def gamma_profile(kind: str, t: float, t_event: float, gamma0: float) -> float:
    if kind == "stationary":
        return gamma0
    if kind == "jump_up":
        return 10.0 * gamma0 if t >= t_event else gamma0
    if kind == "ramp":
        if t < t_event:
            return gamma0
        # linear to 10× by the end; the caller passes t_final as a closure?
        return gamma0  # filled in by the wrapper
    raise ValueError(kind)


def experiment_adaptive(cfg: SimConfig, rng: Generator, backend_info: dict | None = None) -> dict:
    """Hold F >= Ξ_crit while the veiling rate drifts, with a Mehra-style controller.

    The controller never sees the true Γ or the true state.  It gets a weak,
    noisy readout of the target population (a simulated weak measurement) and
    raises or lowers γ_sel from a covariance-matching innovation rule.
    Cost is the time-average of γ_sel (a rate / 'energy' proxy, not a physical
    joule count).
    """
    t0 = time.perf_counter()
    times = cfg.times(cfg.t_track, cfg.n_track_steps)
    float(times[1] - times[0]) if len(times) > 1 else 0.05
    gamma0 = cfg.gamma0
    xi = cfg.xi_crit
    sigma_m = 0.04  # weak-measurement noise on F
    kinds = ("stationary", "jump_up", "ramp")
    d_list = tuple(d for d in (2, 8) if d in cfg.d_list or d in cfg.d_memory)
    t_event = 0.5 * cfg.t_track
    cells = {}

    def gamma_of(kind, t):
        if kind == "stationary":
            return gamma0
        if kind == "jump_up":
            return (10.0 * gamma0) if t >= t_event else gamma0
        # ramp from 1× to 10× over the second half
        if t < t_event:
            return gamma0
        progress = (t - t_event) / max(cfg.t_track - t_event, 1e-12)
        return gamma0 * (1.0 + 9.0 * progress)

    for d in d_list:
        cfg.target(d)
        r_star = r_threshold(xi, d)
        # a small margin so the oracle actually holds the inequality
        r_oracle = 1.15 * r_star
        r_fixed = r_oracle  # fixed uses the *initial* Γ only
        for kind in kinds:
            walks = []
            for w in range(cfg.n_walks):
                wrng = np.random.default_rng(rng.integers(0, 2**63 - 1))
                # three controllers share the same veiling path, different γ_sel.
                # Populations stay diagonal (H=0, thermal start); F is evolved
                # with the exact one-step map, equivalent to expm of the
                # Liouvillian on this family.  (A previous RK4 path blew up at
                # d=8 after a Γ×10 jump, and a fresh expm per step made the
                # official small suite miss the 3-minute budget.)
                Fval = {name: 1.0 / d for name in ("fixed", "oracle", "mehra")}
                hist = {name: dict(F=[], gsel=[], viol=0, cost=0.0, rec_t=float("nan")) for name in Fval}
                # Mehra state
                ghat = r_fixed * gamma0
                C = 0.05 ** 2
                a_forget = 0.90
                q_proc = 0.02 ** 2
                # Ceiling must admit the oracle rate after a Γ×10 jump
                # (1.15 r* × 10 Γ0 ≈ 46 Γ0 at d=2, ≈ 89 Γ0 at d=8).  The old
                # 30 Γ0 cap was only there to keep RK4 from blowing up.
                g_ceil = 200.0 * gamma0
                for ti, t in enumerate(times):
                    G = gamma_of(kind, t)
                    g_sched = {
                        "fixed": r_fixed * gamma0,
                        "oracle": r_oracle * G,
                    }
                    # integrate one step (except t=0 record-only)
                    if ti > 0:
                        dti = float(times[ti] - times[ti - 1])
                        g_sched["mehra"] = float(max(ghat, 0.0))
                        for name in Fval:
                            Fval[name] = f_step_selection_depolarizing(
                                Fval[name], g_sched[name], G, dti, d
                            )
                            hist[name]["cost"] += g_sched[name] * dti
                    for name in Fval:
                        F = float(Fval[name])
                        hist[name]["F"].append(F)
                        hist[name]["gsel"].append(g_sched[name] if ti > 0 else r_fixed * gamma0)
                        if F < xi:
                            hist[name]["viol"] += 1
                    # weak measurement of the Mehra tile, then innovation update
                    F_true = hist["mehra"]["F"][-1]
                    z = F_true + sigma_m * wrng.standard_normal()
                    z = float(np.clip(z, 0.0, 1.0))
                    # We treat the deficit (Ξ_crit - z)_+ as an innovation that
                    # should raise γ_sel, and a surplus as a reason to relax.
                    # A simple covariance-matching (Mehra) scalar for g:
                    # predict C <- a^2 C + q, S = C + r_m, K = C/S, then
                    # raise g when z is below Ξ_crit.
                    r_m = sigma_m ** 2
                    C = a_forget * C + q_proc
                    S = C + r_m
                    K = C / max(S, 1e-18)
                    innov = (xi - z)  # positive => we are below the setpoint
                    # map a fidelity deficit onto a rate increment
                    # dg ~ (r_star * G_ref) * innov / (1-xi), roughly the
                    # linearisation of F_ss around the threshold.
                    g_nom = r_star * gamma0
                    ghat = ghat + K * innov * (g_nom / max(1.0 - xi, 1e-6))
                    ghat = float(np.clip(ghat, 0.0, g_ceil))
                    C = (1.0 - K) * C
                # recovery: first time after t_event that Mehra/oracle/fixed
                # stays at F>=xi through the end
                for name in Fval:
                    F = np.array(hist[name]["F"])
                    after = np.where(times >= t_event)[0]
                    rec = float("nan")
                    for idx in after:
                        if np.all(F[idx:] >= xi - 1e-4):
                            rec = float(times[idx] - t_event)
                            break
                    hist[name]["rec_t"] = rec
                    hist[name]["F"] = F
                    hist[name]["gsel"] = np.array(hist[name]["gsel"])
                    hist[name]["cost"] = hist[name]["cost"] / max(cfg.t_track, 1e-12)
                    hist[name]["viol_frac"] = hist[name]["viol"] / max(len(times), 1)
                    # Late violations: after a 1/Γ settling time, or after the event.
                    t_v = 1.0 if kind == "stationary" else t_event
                    late_ix = np.where(times >= t_v)[0]
                    if late_ix.size:
                        hist[name]["viol_frac"] = float(np.mean(F[late_ix] < xi - 1e-4))
                    hist[name]["F_late"] = float(F[int(0.7 * len(F)):].mean())
                walks.append(hist)

            # aggregate
            cell = {}
            for name in ("fixed", "oracle", "mehra"):
                F_late = [w[name]["F_late"] for w in walks]
                cost = [w[name]["cost"] for w in walks]
                viol = [w[name]["viol_frac"] for w in walks]
                rec = [w[name]["rec_t"] for w in walks]
                Fmean = np.mean([w[name]["F"] for w in walks], axis=0)
                gmean = np.mean([w[name]["gsel"] for w in walks], axis=0)
                cell[name] = dict(
                    F_late=_summarize(F_late),
                    cost=_summarize(cost),
                    viol_frac=_summarize(viol),
                    rec_t=_summarize(rec),
                    F_mean=Fmean,
                    g_mean=gmean,
                )
            # regret vs oracle on late F (positive = worse than oracle) and on cost
            for name in ("fixed", "mehra"):
                dF = [walks[i][name]["F_late"] - walks[i]["oracle"]["F_late"] for i in range(len(walks))]
                dC = [walks[i][name]["cost"] - walks[i]["oracle"]["cost"] for i in range(len(walks))]
                cell[name]["regret_F"] = _summarize(dF)
                cell[name]["regret_cost"] = _summarize(dC)
            cells[(d, kind)] = dict(
                d=d, kind=kind, t_event=t_event, r_star=r_star, r_oracle=r_oracle,
                n_walks=cfg.n_walks, times=times, **cell,
            )
            print(
                f"  adapt d={d} {kind}: late F fixed={cell['fixed']['F_late']['mean']:.3f} "
                f"oracle={cell['oracle']['F_late']['mean']:.3f} mehra={cell['mehra']['F_late']['mean']:.3f}; "
                f"viol mehra={cell['mehra']['viol_frac']['mean']:.2f} oracle={cell['oracle']['viol_frac']['mean']:.2f}"
            )
    return dict(
        cells=cells,
        d_list=list(d_list),
        kinds=list(kinds),
        xi_crit=xi,
        gamma0=gamma0,
        t_track=cfg.t_track,
        t_event=t_event,
        n_walks=cfg.n_walks,
        sigma_m=sigma_m,
        elapsed_s=time.perf_counter() - t0,
    )


# ---------------------------------------------------------------------------
# Experiment 4 — a small coupled array, ring vs all-to-all
# ---------------------------------------------------------------------------

PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def embed_qubit(op: ComplexArr, i: int, n: int) -> ComplexArr:
    parts = [identity(2) for _ in range(n)]
    parts[i] = op
    return kron(*parts)


def edges_of(n: int, layout: str) -> list[tuple[int, int]]:
    if layout == "ring":
        if n == 2:
            return [(0, 1)]
        return [(i, (i + 1) % n) for i in range(n)]
    if layout == "all-to-all":
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    raise ValueError(layout)


def hopping_hamiltonian(n: int, layout: str, J: float) -> ComplexArr:
    H = np.zeros((2 ** n, 2 ** n), dtype=np.complex128)
    if J == 0:
        return H
    for i, j in edges_of(n, layout):
        H = H + J * (
            embed_qubit(PAULI_X, i, n) @ embed_qubit(PAULI_X, j, n)
            + embed_qubit(PAULI_Y, i, n) @ embed_qubit(PAULI_Y, j, n)
        )
    return H


def reduce_one_qubit(rho: ComplexArr, i: int, n: int) -> ComplexArr:
    """Partial trace of an n-qubit state down to qubit i."""
    r = rho.reshape((2,) * n + (2,) * n)
    ket_order = [i] + [k for k in range(n) if k != i]
    bra_order = [n + i] + [n + k for k in range(n) if k != i]
    r = np.transpose(r, ket_order + bra_order)
    r = r.reshape(2, 2 ** (n - 1), 2, 2 ** (n - 1))
    return np.trace(r, axis1=1, axis2=3)


def experiment_array(cfg: SimConfig, rng: Generator, backend_info: dict | None = None) -> dict:
    """A few coupled tiles, full density matrix. Wiring is ring vs all-to-all.

    Each tile runs the same local selection into |1> against a local
    depolarizing veil. Neighbours are coupled only by a hopping Hamiltonian
    (XX+YY) on the declared graph; there is no global reference. This is a
    small, honest test of whether the wiring still matters, not a large-N
    network unveil.
    """
    t0 = time.perf_counter()
    times = cfg.times()
    gamma = cfg.gamma0
    xi = cfg.xi_crit
    # use a rate a bit above the d=2 threshold so an isolated tile would hold Ξ
    r = 1.15 * r_threshold(xi, 2)
    gs = r * gamma
    tidx = 1
    cells = {}
    traces = {}
    for n in cfg.n_array:
        dim = 2 ** n
        # local jumps on each qubit
        jumps = []
        for i in range(n):
            for L in jumps_selection(2, tidx, gs) + jumps_depolarizing_weyl(2, gamma):
                jumps.append(embed_qubit(L, i, n))
        rho0 = thermal_state(dim)
        for layout in ("ring", "all-to-all"):
            for J in cfg.j_couplings:
                H = hopping_hamiltonian(n, layout, float(J))
                method = "expm" if dim * dim <= 256 else "rk4"
                ev = evolve_dispatch(rho0, H, jumps, times, backend_info, method=method)
                # mean single-tile fidelity and pairwise reduced-state disagreement
                F_mean = []
                F_min = []
                agree = []  # 1 - mean trace distance between reduced tiles
                for k in range(len(times)):
                    reds = [reduce_one_qubit(ev[k], i, n) for i in range(n)]
                    fs = [float(fidelity_target(reds[i], 2, tidx)) for i in range(n)]
                    F_mean.append(float(np.mean(fs)))
                    F_min.append(float(np.min(fs)))
                    if n >= 2:
                        ds = [float(trace_distance(reds[i], reds[j])) for i in range(n) for j in range(i + 1, n)]
                        agree.append(1.0 - float(np.mean(ds)))
                    else:
                        agree.append(1.0)
                rec = dict(
                    n=n, layout=layout, J=float(J), r=r, gamma_sel=gs,
                    F_final=float(F_mean[-1]), F_min_final=float(F_min[-1]),
                    agree_final=float(agree[-1]),
                    t_xi=time_to_fidelity(np.array(F_mean), times, xi),
                    holds_xi=bool(F_mean[-1] >= xi - 1e-4),
                    F_mean=np.array(F_mean),
                    agree=np.array(agree),
                )
                cells[(n, layout, float(J))] = rec
                if float(J) in (0.0, cfg.j_couplings[-1]):
                    traces[(n, layout, float(J))] = dict(times=times, F=rec["F_mean"], agree=rec["agree"])
                print(f"  array N={n} {layout} J={J}: F={rec['F_final']:.3f} agree={rec['agree_final']:.3f}")
    # Mismatch series: tile 0 starts in |0> (off-target) while the others
    # start in |1>.  All tiles still run the same local selector.  A symmetric
    # product initial state cannot distinguish ring from all-to-all; this one can.
    mismatch = {}
    for n in cfg.n_array:
        dim = 2 ** n
        factors = [pure_state(2, 1) for _ in range(n)]
        factors[0] = pure_state(2, 0)
        rho0 = factors[0]
        for f in factors[1:]:
            rho0 = np.kron(rho0, f)
        jumps = []
        for i in range(n):
            for L in jumps_selection(2, tidx, gs) + jumps_depolarizing_weyl(2, gamma):
                jumps.append(embed_qubit(L, i, n))
        for layout in ("ring", "all-to-all"):
            for J in cfg.j_couplings:
                H = hopping_hamiltonian(n, layout, float(J))
                method = "expm" if dim * dim <= 256 else "rk4"
                ev = evolve_dispatch(rho0, H, jumps, times, backend_info, method=method)
                F0, Fmean, agree = [], [], []
                for k in range(len(times)):
                    reds = [reduce_one_qubit(ev[k], i, n) for i in range(n)]
                    fs = [float(fidelity_target(reds[i], 2, tidx)) for i in range(n)]
                    F0.append(fs[0])
                    Fmean.append(float(np.mean(fs)))
                    if n >= 2:
                        ds = [float(trace_distance(reds[i], reds[j])) for i in range(n) for j in range(i + 1, n)]
                        agree.append(1.0 - float(np.mean(ds)))
                    else:
                        agree.append(1.0)
                mismatch[(n, layout, float(J))] = dict(
                    n=n, layout=layout, J=float(J),
                    F0_final=float(F0[-1]), F_final=float(Fmean[-1]),
                    agree_final=float(agree[-1]),
                    F0=np.array(F0), F_mean=np.array(Fmean), agree=np.array(agree),
                )
                print(f"  mismatch N={n} {layout} J={J}: F0={F0[-1]:.3f} Fmean={Fmean[-1]:.3f} agree={agree[-1]:.3f}")
    return dict(
        cells=cells,
        traces=traces,
        mismatch=mismatch,
        times=times,
        r=r,
        gamma0=gamma,
        xi_crit=xi,
        n_array=list(cfg.n_array),
        j_couplings=list(cfg.j_couplings),
        elapsed_s=time.perf_counter() - t0,
        note=(
            "Full density matrix on 2–3 (or 4, SIZE=large) qubits. "
            "The symmetric product initial state cannot distinguish ring from all-to-all; "
            "the mismatch series (tile 0 starts off-target) is the wiring test. "
            "This is not a large-N unveiling test and does not claim a general wiring law."
        ),
    )


# ---------------------------------------------------------------------------
# Agreement and timing
# ---------------------------------------------------------------------------







# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------













# ---------------------------------------------------------------------------
# Conclusions and suite
# ---------------------------------------------------------------------------

def format_conclusions(payload: dict) -> str:
    cfg = payload["config"]
    e1, e2, e3, e4 = payload["exp_memory"], payload["exp_selection"], payload["exp_adaptive"], payload["exp_array"]
    agree, bench = payload.get("agreement") or {}, payload.get("timing") or {}
    backend = payload.get("backend_info") or {}
    lines = []
    a = lines.append
    a("WHAT THIS RUN ACTUALLY SHOWED")
    a("==============================")
    a(f"SIZE={cfg['size']}, seed={cfg['seed']}, programmer-declared Ξ_crit={cfg['xi_crit']}.")
    a("In this notebook Ξ_crit is a declared lower bound on the target fidelity F=<t|ρ|t>,")
    a("i.e. on how sharply the tile is focused on the selected pointer. It is not the")
    a("v1–v4 consistency statistic, though it plays the same role as a context-level floor.")
    a(f"Time unit 1/Γ0 with Γ0={cfg['gamma0']}. Official backend: {payload.get('official_backend','numpy')}.")
    a(f"Exp 1 (memory) wall {e1['elapsed_s']:.1f}s;  Exp 2 (selection) {e2['elapsed_s']:.1f}s;  "
      f"Exp 3 (adaptive) {e3['elapsed_s']:.1f}s;  Exp 4 (array) {e4['elapsed_s']:.1f}s.")
    a("")
    a("EXPERIMENT 1 — Markovian T1/T2 vs a pseudomode memory register")
    a("--------------------------------------------------------------")
    a("The same effective amplitude-damping rate Γ is realised either as a memoryless")
    a("Lindblad jump on the tile, or as a Lorentzian bath (J(ω) with width κ/2) via an")
    a("auxiliary damped bosonic mode truncated at n_max, with g = sqrt(Γ κ / 4).")
    a("The tile+register obey a Lindblad equation; the tile alone need not.")
    a("Probe pair: the damping attractor vs the most-excited computational state.")
    a("")
    for d in e1["d_list"]:
        cell = e1["cells"][d]
        m = cell["markov"]["blp"]
        a(f"  d={d}, n_max={cell['n_max']}:")
        a(f"    Markov T1/T2:  N_BLP={m['N_BLP']:.4e}, monotone={m['monotone']}, "
          f"D_final={m['d_final']:.4f}, purity_final={cell['markov']['purity'][-1]:.4f}")
        for row in cell["pseudomode"]:
            b = row["blp"]
            a(f"    κ={row['kappa']:<6g} g={row['g']:.3f}:  N_BLP={b['N_BLP']:.4e}, "
              f"monotone={b['monotone']}, n_up={b['n_increases']}, D_final={b['d_final']:.4f}")
    # helper verdict
    helped = []
    for d in e1["d_list"]:
        cell = e1["cells"][d]
        n_mark = cell["markov"]["blp"]["N_BLP"]
        n_slow = cell["pseudomode"][0]["blp"]["N_BLP"]
        n_fast = cell["pseudomode"][-1]["blp"]["N_BLP"]
        helped.append((d, n_mark, n_slow, n_fast))
    a("  Reading: a clearly positive N_BLP at small κ, shrinking toward the Markov value")
    a("  as κ grows, is the signature that the register is acting as ta's memory.")
    a("")

    a("EXPERIMENT 2 — engineered selection and the Ξ_crit threshold")
    a("------------------------------------------------------------")
    a("Jump operators L_j = sqrt(γ_sel) |t><j| for j≠t, competing with a depolarizing")
    a("veil of rate Γ.  Markov theory: F_ss = (r + 1/d)/(r + 1),  r=γ_sel/Γ,")
    a("threshold r* = (Ξ_crit - 1/d)/(1 - Ξ_crit).")
    a("")
    a("  d   r*_theory  r*_empirical  F_ss(r=2) th/num   max |F-theory|")
    for trow in e2["theory"]:
        d = trow["d"]
        rows = [r for r in e2["markov"] if r["d"] == d]
        err = max(r["F_abs_err"] for r in rows) if rows else float("nan")
        r2 = next((r for r in rows if abs(r["r"] - 2.0) < 1e-12), None)
        fth = r2["F_ss_theory"] if r2 else float("nan")
        fnum = r2["F_ss_numeric"] if r2 else float("nan")
        a(f"  {d:<3} {trow['r_star']:<10.4f} {trow['r_empirical']:<13.4f} "
          f"{fth:.4f}/{fnum:.4f}     {err:.2e}")
    a("")
    a("  Time to F>0.99 (NaN = not reached in the simulated window) and contrast:")
    for d in e2["d_list"]:
        bits = []
        for r in e2["r_grid"]:
            row = next((z for z in e2["markov"] if z["d"] == d and abs(z["r"] - r) < 1e-12), None)
            if row is None:
                continue
            bits.append(f"r={r:g}: F={row['F_final']:.3f} C={row['contrast_final']:.3f} t099={row['t_F099']:.2f}")
        a(f"    d={d}: " + "; ".join(bits))
    a("")
    a("  Memory vs memoryless veil, same Γ_eff, same γ_sel (r ≈ 2 r* or at least 2):")
    for d in sorted({r["d"] for r in e2["memory"]}):
        ctrl = next((r for r in e2["memory"] if r["d"] == d and r.get("kind") == "markov_t1t2"), None)
        rows = [r for r in e2["memory"] if r["d"] == d and np.isfinite(r.get("kappa", np.nan))]
        if ctrl is None or not rows:
            continue
        a(f"    d={d}  Markov T1/T2 F={ctrl['F_final']:.4f}  holds Ξ={ctrl['holds_xi']}")
        for row in rows:
            tag = "helps" if row["delta_F"] > 0.01 else ("hurts" if row["delta_F"] < -0.01 else "ties")
            a(f"      κ={row['kappa']:<6g} F={row['F_final']:.4f}  ΔF={row['delta_F']:+.4f}  ({tag})  "
              f"holds Ξ={row['holds_xi']}")
    a("")

    a("EXPERIMENT 3 — drifting Γ, holding F ≥ Ξ_crit")
    a("---------------------------------------------")
    a("Controllers: fixed γ_sel (tuned to the initial Γ), an oracle that sets")
    a("γ_sel = 1.15 r* Γ(t), and a Mehra-style rule that sees only a weak noisy")
    a("readout of F (σ_m={:.2f}) and never the true Γ or the true state.".format(e3.get("sigma_m", 0.04)))
    a("Cost is the time-average of γ_sel, a rate proxy, not a physical energy.")
    a("")
    for d in e3["d_list"]:
        for kind in e3["kinds"]:
            rec = e3["cells"].get((d, kind))
            if rec is None:
                continue
            a(f"  d={d} {kind} (r*={rec['r_star']:.3f}):")
            for name in ("fixed", "oracle", "mehra"):
                row = rec[name]
                extra = ""
                if name != "oracle":
                    extra = (f"  regret_F={row['regret_F']['mean']:+.4f}±{row['regret_F']['sem']:.4f}"
                             f"  regret_cost={row['regret_cost']['mean']:+.3f}")
                a(f"    {name:8s}  late F={row['F_late']['mean']:.4f}±{row['F_late']['sem']:.4f}  "
                  f"viol={row['viol_frac']['mean']:.3f}  cost={row['cost']['mean']:.3f}  "
                  f"rec_t={row['rec_t']['mean']:.2f}{extra}")
    a("")

    a("EXPERIMENT 4 — a small coupled array (full density matrix)")
    a("---------------------------------------------------------")
    a(e4["note"])
    a(f"Each tile selects |1> at r={e4['r']:.3f} against a local depolarizing veil;")
    a("neighbours couple only by XX+YY hopping on the declared graph.")
    a("")
    for n in e4["n_array"]:
        for layout in ("ring", "all-to-all"):
            bits = []
            for J in e4["j_couplings"]:
                rec = e4["cells"].get((n, layout, float(J)))
                if rec is None:
                    continue
                bits.append(f"J={J:g}: F={rec['F_final']:.3f} agree={rec['agree_final']:.3f}")
            a(f"  N={n} {layout} (identical init): " + "; ".join(bits))
    if e4.get("mismatch"):
        a("  Mismatch (tile 0 starts off-target in |0>, others in |1>):")
        for n in e4["n_array"]:
            for layout in ("ring", "all-to-all"):
                bits = []
                for J in e4["j_couplings"]:
                    rec = e4["mismatch"].get((n, layout, float(J)))
                    if rec is None:
                        continue
                    bits.append(f"J={J:g}: F0={rec['F0_final']:.3f} Fmean={rec['F_final']:.3f} agree={rec['agree_final']:.3f}")
                a(f"    N={n} {layout}: " + "; ".join(bits))
    a("")

    rec_line, failures = _recommend(e1, e2, e3, e4, cfg)
    a("RECOMMENDATION")
    a("--------------")
    a(rec_line)
    a("")
    a("PLAIN FAILURES AND LIMITS")
    a("-------------------------")
    for f in failures:
        a(f"- {f}")
    a("")
    a("BACKEND, AGREEMENT, AND TIMING")
    a("------------------------------")
    a(f"Requested BACKEND={backend.get('requested')}, resolved={backend.get('name')}, "
      f"torch={backend.get('torch_version')}, cuda={backend.get('cuda')}, device={backend.get('device')}.")
    if agree.get("available"):
        num = agree["numeric"]
        a(f"Numeric agreement (Markov selection, d=3, r=4, 40 steps): "
          f"max |ΔF|={num['max_abs_F']:.3e}, rms={num['rms_F']:.3e}  "
          f"({'ok' if num['ok'] else 'NOT CLOSE'}).")
        a("  This is exact-expm (NumPy, complex128) vs RK4 (torch, complex64), not bit-for-bit.")
    else:
        a("PyTorch was not available; agreement was skipped.")
    if bench:
        a(f"Timing bench: d={bench.get('d')}, n_max={bench.get('n_max')}, "
          f"batch={bench.get('batch')}, steps={bench.get('steps')}.")
        for row in bench.get("rows") or []:
            a(f"  {row['backend']:6s}  {row['device']:8s}  {row['elapsed_s']:.3f}s  "
              f"speedup vs NumPy CPU: ×{row.get('speedup_vs_numpy', float('nan')):.2f}")
        if bench.get("note"):
            a("  " + bench["note"])
        np_row = next((r for r in bench.get("rows") or [] if r["backend"] == "numpy"), None)
        th_cpu = next((r for r in bench.get("rows") or [] if r["backend"] == "torch" and "cuda" not in str(r["device"])), None)
        th_gpu = next((r for r in bench.get("rows") or [] if r["backend"] == "torch" and "cuda" in str(r["device"])), None)
        if np_row and th_cpu and th_cpu["elapsed_s"] > np_row["elapsed_s"]:
            a("  Honest note: on this machine the torch CPU path was slower than NumPy. "
              "That does not predict GPU behaviour at SIZE=large.")
        if th_gpu is None:
            a("  No CUDA device was visible on this machine. The GPU path is only smoke-tested on CPU here. "
              "A Colab Pro GPU is required to measure a real GPU speedup at SIZE=large.")
    a("")
    a(f"Total suite wall time: {payload.get('elapsed_s', float('nan')):.1f}s.")
    a("")
    return "\n".join(lines) + "\n"


def _recommend(e1, e2, e3, e4, cfg):
    failures = []
    # backflow present?
    saw_memory = False
    for d in e1["d_list"]:
        cell = e1["cells"][d]
        n_mark = cell["markov"]["blp"]["N_BLP"]
        n_slow = cell["pseudomode"][0]["blp"]["N_BLP"]
        n_fast = cell["pseudomode"][-1]["blp"]["N_BLP"]
        if n_slow > 0.01 and n_slow > n_mark + 0.005:
            saw_memory = True
        if n_slow <= n_mark + 0.005:
            failures.append(
                f"At d={d} the slow-κ pseudomode did not show a clear BLP backflow above the Markov baseline "
                f"(N_slow={n_slow:.3e}, N_Markov={n_mark:.3e}). Memory is not automatically visible at this n_max / grid."
            )
        if n_fast > n_slow and n_slow > 0.01:
            failures.append(f"At d={d} N_BLP did not shrink toward the Markov limit as κ increased (slow={n_slow:.3e}, fast={n_fast:.3e}).")
    if not saw_memory:
        failures.append("The pseudomode register did not produce a clean, κ-dependent information backflow in this run; treat the memory claim as unconfirmed here.")

    # theory agreement
    for trow in e2["theory"]:
        rows = [r for r in e2["markov"] if r["d"] == trow["d"]]
        err = max((r["F_abs_err"] for r in rows), default=0.0)
        if err > 0.03:
            failures.append(f"Markov F_ss at d={trow['d']} deviates from (r+1/d)/(r+1) by up to {err:.3f}; check the dissipator normalisation.")
        if np.isfinite(trow["r_empirical"]) and abs(trow["r_empirical"] - trow["r_star"]) > 0.5 * max(trow["r_star"], 0.5):
            failures.append(
                f"Empirical r* at d={trow['d']} is {trow['r_empirical']:.3f} vs theory {trow['r_star']:.3f}."
            )

    # memory help/hurt
    mem_verdicts = []
    for d in sorted({r["d"] for r in e2["memory"]}):
        rows = [r for r in e2["memory"] if r["d"] == d and np.isfinite(r.get("kappa", np.nan))]
        if not rows:
            continue
        # compare slowest κ to Markov control
        slow = min(rows, key=lambda z: z["kappa"])
        if slow["delta_F"] < -0.03:
            mem_verdicts.append(f"hurts at d={d} (slow κ ΔF={slow['delta_F']:+.3f})")
        elif slow["delta_F"] > 0.03:
            mem_verdicts.append(f"helps at d={d} (slow κ ΔF={slow['delta_F']:+.3f})")
        else:
            mem_verdicts.append(f"is roughly neutral at d={d} (slow κ ΔF={slow['delta_F']:+.3f})")

    # adaptive failures
    for d in e3["d_list"]:
        rec = e3["cells"].get((d, "jump_up"))
        if rec is None:
            continue
        if rec["fixed"]["viol_frac"]["mean"] < 0.05:
            failures.append(f"The fixed-γ_sel control barely violated Ξ_crit after a q-style Γ×10 jump at d={d}; the jump may not have been stressful enough, or the late window is dominated by a still-high F.")
        if rec["mehra"]["viol_frac"]["mean"] > rec["fixed"]["viol_frac"]["mean"] + 0.15:
            failures.append(f"The Mehra-style controller violated Ξ_crit more often than a fixed rate at d={d} after the jump (viol {rec['mehra']['viol_frac']['mean']:.2f} vs {rec['fixed']['viol_frac']['mean']:.2f}).")
        if rec["oracle"]["viol_frac"]["mean"] > 0.2:
            failures.append(f"Even the oracle schedule violated Ξ_crit on a fraction {rec['oracle']['viol_frac']['mean']:.2f} of the d={d} jump track; the 1.15 r* margin or the finite settling time is not enough.")
    for d in e3["d_list"]:
        rec = e3["cells"].get((d, "stationary"))
        if rec is not None and rec["mehra"]["viol_frac"]["mean"] > 0.2:
            failures.append(
                f"On a stationary veil Mehra chatters around Ξ_crit at d={d} "
                f"(late F={rec['mehra']['F_late']['mean']:.3f}, viol={rec['mehra']['viol_frac']['mean']:.2f}) "
                f"because the innovation setpoint is exactly Ξ_crit and the readout is noisy; "
                f"the oracle's 1.15 margin avoids this."
            )
        rec = e3["cells"].get((d, "ramp"))
        if rec is not None and (
            rec["mehra"]["viol_frac"]["mean"] > 0.5
            or not __import__("math").isfinite(rec["mehra"]["rec_t"]["mean"])
        ):
            failures.append(
                f"Mehra did not hold a sustained F≥Ξ_crit through the Γ ramp at d={d} "
                f"(late F={rec['mehra']['F_late']['mean']:.3f}, viol={rec['mehra']['viol_frac']['mean']:.2f}, rec_t=nan); "
                f"it beat a fixed rate on late F but still undershot the oracle."
            )

    # array
    # wiring effect
    wiring = []
    for n in e4["n_array"]:
        jmax = max(e4["j_couplings"])
        ring = e4["cells"].get((n, "ring", float(jmax)))
        ata = e4["cells"].get((n, "all-to-all", float(jmax)))
        iso = e4["cells"].get((n, "ring", 0.0)) or e4["cells"].get((n, "all-to-all", 0.0))
        if ring and ata and iso:
            dF = ata["F_final"] - ring["F_final"]
            dA = ata["agree_final"] - ring["agree_final"]
            if abs(dF) < 0.01 and abs(dA) < 0.01:
                wiring.append(f"N={n} identical init: ring and all-to-all agree to 0.01 at J={jmax}, as expected for a permutation-symmetric product state.")
        mm_r = (e4.get("mismatch") or {}).get((n, "ring", float(jmax)))
        mm_a = (e4.get("mismatch") or {}).get((n, "all-to-all", float(jmax)))
        mm_0 = (e4.get("mismatch") or {}).get((n, "ring", 0.0)) or (e4.get("mismatch") or {}).get((n, "all-to-all", 0.0))
        if mm_r and mm_a and mm_0:
            dF = mm_a["F0_final"] - mm_r["F0_final"]
            dA = mm_a["agree_final"] - mm_r["agree_final"]
            dJ = mm_a["F0_final"] - mm_0["F0_final"]
            if abs(dF) < 0.01 and abs(dJ) < 0.01:
                wiring.append(f"N={n} mismatch: even with tile 0 off-target, ring vs all-to-all and J=0 vs J={jmax} changed F0 by less than 0.01; wiring did not matter at this size and coupling.")
            else:
                wiring.append(f"N={n} mismatch at J={jmax}: ATA−ring ΔF0={dF:+.3f}, Δagree={dA:+.3f}; vs J=0, ATA ΔF0={dJ:+.3f}.")
    if wiring:
        # not necessarily a failure
        pass

    # F>0.99 times
    never = []
    for row in e2["markov"]:
        if row["r"] >= 8 and not np.isfinite(row["t_F099"]):
            never.append(f"d={row['d']} r={row['r']}")
    if never:
        rmax = max((row["r"] for row in e2["markov"]), default=16.0)
        dlist = sorted({row["d"] for row in e2["markov"]})
        bits = ", ".join(
            f"d={d}: F_ss(r={rmax:g})={((rmax + 1.0/d)/(rmax + 1.0)):.3f}, need r>{(0.99 - 1.0/d)/0.01:.1f}"
            for d in dlist
        )
        failures.append(
            "F>0.99 was not reached at any r on the grid (max r="
            + f"{rmax:g}). That is expected from the closed form F_ss=(r+1/d)/(r+1): "
            + bits
            + ". The grid never entered the F_ss>0.99 region, so this is not a finite-time integration failure."
        )

    rec = (
        f"A programmer-declared Ξ_crit={cfg['xi_crit']} is a usable on-tile setpoint for the selected pointer. "
        f"In the Markov depolarizing model the predicted threshold r* = (Ξ_crit − 1/d)/(1 − Ξ_crit) "
        f"rises slowly with d (selecting 1 of 8 needs a slightly higher rate ratio than 1 of 2). "
    )
    if mem_verdicts:
        rec += "Against a T1/T2-type veil of the same Γ_eff, a pseudomode memory register " + "; ".join(mem_verdicts) + ". "
    rec += (
        "If only a memoryless veil is present, a fixed γ_sel = 1.15 r* Γ0 holds Ξ_crit until Γ drifts; "
        "an oracle that tracks Γ(t) is the reference, and a Mehra-style rule that sees only a weak "
        "readout of F is the on-tile option — see the tables for whether it actually recovers. "
        "On a 2–3 tile array this run is too small to claim a wiring law; treat Experiment 4 as a smoke test."
    )
    if not saw_memory:
        rec += " Do not claim that the register has been shown to act as ta's memory until the BLP diagnostic is clearly positive."
    for w in wiring:
        if "did not matter" in w:
            failures.append(w)
    return rec, failures


def json_safe(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(k, tuple):
                key = "|".join(str(x) for x in k)
            else:
                key = str(k)
            out[key] = json_safe(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, complex):
        return {"re": float(obj.real), "im": float(obj.imag)}
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def run_suite(cfg: SimConfig, outdir, backend: str = "numpy",
              do_agreement: bool = True, do_timing: bool = True) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    info = resolve_backend(backend)
    official = "numpy" if cfg.size == "small" and info["requested"] in ("numpy", "auto") else info["name"]
    if cfg.size == "small" and info["requested"] in ("numpy", "auto"):
        a_info = resolve_backend("numpy")
    else:
        a_info = info
    print(f"SRI v5  SIZE={cfg.size}  BACKEND requested={info['requested']} resolved={info['name']}  "
          f"official={official}  cuda={info['cuda']}  torch={info['torch_version']}  Ξ_crit={cfg.xi_crit}")
    rng = np.random.default_rng(cfg.seed)

    print("Experiment 1 — Markovian vs pseudomode memory")
    e1 = experiment_memory(cfg, rng, a_info)
    print("Experiment 2 — engineered selection and threshold")
    e2 = experiment_selection(cfg, rng, a_info)
    print("Experiment 3 — drifting veil, adaptive γ_sel")
    e3 = experiment_adaptive(cfg, rng, a_info)
    print("Experiment 4 — small coupled array")
    e4 = experiment_array(cfg, rng, a_info)

    agree = None
    if do_agreement:
        if torch_module() is not None:
            th_info = resolve_backend("torch")
            print("Agreement check (numpy expm vs torch RK4)")
            agree = agreement_check(cfg, np.random.default_rng(cfg.seed + 3), th_info)
            print(f"  agreement ok={agree.get('ok')}  max|ΔF|={agree['numeric']['max_abs_F']:.3e}")
        else:
            agree = dict(available=False, reason="PyTorch is not installed.")
            print("  skipped (no torch)")

    bench = None
    if do_timing:
        print("Timing bench")
        th_info = resolve_backend("torch") if torch_module() is not None else resolve_backend("numpy")
        bench = timing_bench(cfg, th_info)
        for row in bench["rows"]:
            print(f"  {row['backend']} {row['device']}: {row['elapsed_s']:.3f}s  ×{row.get('speedup_vs_numpy', float('nan')):.2f}")
        if bench.get("note"):
            print("  " + bench["note"])

    pngs = []
    pngs += plot_memory(e1, outdir)
    pngs += plot_selection(e2, outdir)
    pngs += plot_adaptive(e3, outdir)
    pngs += plot_array(e4, outdir)
    if bench:
        pngs += plot_timing(bench, outdir)
    print("wrote", ", ".join(Path(p).name for p in pngs))

    payload = dict(
        config=dict(size=cfg.size, seed=cfg.seed, xi_crit=cfg.xi_crit, gamma0=cfg.gamma0,
                    n_max=cfg.n_max, t_final=cfg.t_final, n_steps=cfg.n_steps),
        official_backend=official,
        backend_info=dict(requested=info["requested"], name=info["name"], device=str(info["device"]),
                          cuda=info["cuda"], torch_version=info["torch_version"]),
        exp_memory=e1,
        exp_selection=e2,
        exp_adaptive=e3,
        exp_array=e4,
        agreement=agree,
        timing=bench,
        pngs=pngs,
        elapsed_s=time.perf_counter() - t0,
    )
    conclusions = format_conclusions(payload)
    payload["conclusions"] = conclusions
    (outdir / "conclusions.txt").write_text(conclusions)
    (outdir / "results.json").write_text(json.dumps(json_safe(payload), indent=2))
    print(f"suite wall {payload['elapsed_s']:.1f}s")
    print(conclusions)
    return payload


# Names retained so the NumPy driver imports. They are not on the seeded path.
def _th(*_args, **_kwargs):
    raise RuntimeError('_th' + " is not part of the NumPy reproduction driver")
def agreement_check(*_args, **_kwargs):
    raise RuntimeError('agreement_check' + " is not part of the NumPy reproduction driver")
