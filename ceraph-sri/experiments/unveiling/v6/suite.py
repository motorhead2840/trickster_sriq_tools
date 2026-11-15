"""Reproduction driver for the v6 round.

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

from sri.cv import (
    eta_for_relative_advantage,
    k_interval_b_beats_a,
    monte_carlo_mse,
    mse_optimal,
    relative_advantage,
    squeezing_photons,
    strategies,
    total_photons,
)
from sri.errorlaw import predicted_mse
from sri.graphs import (
    is_perfect_square,
    laplacian_eigbounds,
    mean_resistance,
)
from sri.unveiling import (
    estimate_decentralized,
    generate_observations as _generate_observations_pair,
    mse_kappa,
    relative_veil_matrix,
    sample_veils,
)
from sri.wiring_quasi import (
    aa_onsite_ipr,
    layout_catalogue,
    make_layout,
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

plot_squeezing = _plot_stub

plot_timing = _plot_stub

plot_wiring = _plot_stub

def generate_observations(tau, k, snr_db, rng):
    """Shared-input observations. Returns Y; the shared input is still drawn."""
    observed, _psi = _generate_observations_pair(tau, k, snr_db, rng)
    return observed

ComplexArr = np.ndarray
FloatArr = np.ndarray

BASE_SEED = 13
HBAR_CONV = 2  # ħ = 2: vacuum variance = 1 per quadrature (Weedbrook 2012)
VACUUM_VAR = 1.0


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SIZE_PRESETS = {
    "small": dict(
        k_grid=(0.15, 0.25, 0.4, 0.5, 0.65, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0),
        eta_grid=(0.3, 0.5, 0.65, 0.75, 0.85, 0.95, 1.0),
        r_grid=(0.5, 1.0, 1.5),
        ratio_grid=(0.5, 1.0, 4.0, 16.0),
        v_s=1.0,
        r_default=1.0,
        ratio_default=16.0,
        eta_default=0.85,
        n_mc=20000,
        n_mc_points=6,
        n_list=(36, 52, 64, 100),
        n_list_nonsquare=(36, 52, 64, 100),
        k_samples=16,
        snr_db=20.0,
        n_trials=8,
        s_amp=0.25,
        s_phase=0.4,
        degree=4,
        ws_p=0.10,
        aa_lambda_below=0.40,
        aa_lambda_above=0.95,
        fib_weak=0.25,
        bench_batch=8,
        bench_n=64,
    ),
    "large": dict(
        k_grid=tuple(np.round(np.linspace(0.05, 2.0, 40), 5)),
        eta_grid=tuple(np.round(np.linspace(0.2, 1.0, 17), 5)),
        r_grid=(0.4, 0.8, 1.2, 1.6, 2.0),
        ratio_grid=(0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
        v_s=1.0,
        r_default=1.2,
        ratio_default=16.0,
        eta_default=0.85,
        n_mc=80000,
        n_mc_points=10,
        n_list=(36, 52, 64, 100, 144, 196),
        n_list_nonsquare=(36, 52, 64, 100, 144, 196),
        k_samples=32,
        snr_db=20.0,
        n_trials=16,
        s_amp=0.25,
        s_phase=0.4,
        degree=4,
        ws_p=0.10,
        aa_lambda_below=0.40,
        aa_lambda_above=0.95,
        fib_weak=0.25,
        bench_batch=16,
        bench_n=100,
    ),
}


@dataclass
class SimConfig:
    size: str = "small"
    seed: int = BASE_SEED
    k_grid: tuple[float, ...] = SIZE_PRESETS["small"]["k_grid"]
    eta_grid: tuple[float, ...] = SIZE_PRESETS["small"]["eta_grid"]
    r_grid: tuple[float, ...] = SIZE_PRESETS["small"]["r_grid"]
    ratio_grid: tuple[float, ...] = SIZE_PRESETS["small"]["ratio_grid"]
    v_s: float = 1.0
    r_default: float = 1.0
    ratio_default: float = 16.0
    eta_default: float = 0.85
    n_mc: int = 20000
    n_mc_points: int = 6
    n_list: tuple[int, ...] = (36, 64, 100)
    n_list_nonsquare: tuple[int, ...] = (36, 64, 100)
    k_samples: int = 16
    snr_db: float = 20.0
    n_trials: int = 8
    s_amp: float = 0.25
    s_phase: float = 0.4
    degree: int = 4
    ws_p: float = 0.10
    aa_lambda_below: float = 0.40
    aa_lambda_above: float = 0.95
    fib_weak: float = 0.25
    bench_batch: int = 8
    bench_n: int = 64

    @classmethod
    def from_size(cls, size: str, **overrides) -> "SimConfig":
        if size not in SIZE_PRESETS:
            raise ValueError(f"SIZE must be 'small' or 'large', got {size!r}")
        data = dict(SIZE_PRESETS[size])
        data.update(overrides)
        return cls(size=size, **data)

    def e_tau_inv_pow(self, p: int) -> float:
        """E|τ|^{-p} for τ=exp(η+iξ), η~N(0,s_amp^2). Only even p in the real exponent."""
        # |τ|^{-p} = exp(-p η), E exp(-p η) = exp((p^2/2) s_amp^2)? 
        # η ~ N(0, σ²), E e^{t η} = exp(t² σ² / 2). Here t = -p, E = exp(p² s_amp² / 2).
        # For p=2: exp(2 s_amp²). For p=4: exp(8 s_amp²).
        # Wait: E e^{t η} = exp(t² σ² / 2), t=-p, t²=p², exp(p² σ² / 2).
        # p=2: exp(4/2 σ²)=exp(2 σ²). p=4: exp(16/2 σ²)=exp(8 σ²). Yes.
        return float(np.exp((p ** 2) * 0.5 * self.s_amp ** 2))


# ---------------------------------------------------------------------------
# Exp 1 — Gaussian CV estimation
# ---------------------------------------------------------------------------



































# ---------------------------------------------------------------------------
# Exp 2 — graphs and static unveiling (v2 error law)
# ---------------------------------------------------------------------------

























GOLDEN_CONJ = 0.5 * (np.sqrt(5.0) - 1.0)  # (√5-1)/2














































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
# Layout catalogue
# ---------------------------------------------------------------------------







# ---------------------------------------------------------------------------
# Experiment 1
# ---------------------------------------------------------------------------

def experiment_squeezing(cfg: SimConfig, rng: Generator) -> dict:
    t0 = time.perf_counter()
    v_s = cfg.v_s
    r0 = cfg.r_default
    ratio0 = cfg.ratio_default
    v_n0 = ratio0 * v_s
    eta0 = cfg.eta_default
    strats = strategies(r0)
    k_vals = np.asarray(cfg.k_grid, dtype=np.float64)
    eta_vals = np.asarray(cfg.eta_grid, dtype=np.float64)
    ratio_vals = np.asarray(cfg.ratio_grid, dtype=np.float64)

    # --- slice 1: MSE vs k at (eta0, r0, strong noise) ---
    vs_k = {}
    for name, (ra, rb) in strats.items():
        vs_k[name] = [mse_optimal(v_s, v_n0, float(k), eta0, ra, rb) for k in k_vals]
    interval0 = k_interval_b_beats_a(v_n0, eta0, r0)

    # --- slice 2: relative advantage vs eta at several k, strong noise ---
    vs_eta = {name: {} for name in strats}
    vanish_1pct = {name: {} for name in strats}
    vanish_5pct = {name: {} for name in strats}
    k_eta_scan = (0.25, 0.5, 0.8, 1.0, 1.5)
    for name, (ra, rb) in strats.items():
        for k in k_eta_scan:
            rels = []
            mses = []
            for eta in eta_vals:
                m0 = mse_optimal(v_s, v_n0, k, float(eta), 0.0, 0.0)
                m = mse_optimal(v_s, v_n0, k, float(eta), ra, rb)
                mses.append(m)
                rels.append(relative_advantage(m, m0))
            vs_eta[name][k] = dict(mse=mses, rel=rels)
            vanish_1pct[name][k] = eta_for_relative_advantage(v_s, v_n0, k, r0, 0.01, name)
            vanish_5pct[name][k] = eta_for_relative_advantage(v_s, v_n0, k, r0, 0.05, name)

    # advantage of B over classical at eta=0.85 and just below, strong noise
    hold_rows = []
    for eta in (0.95, 0.85, 0.75, 0.65, 0.50, 0.30):
        for k in k_eta_scan:
            m0 = mse_optimal(v_s, v_n0, k, eta, 0.0, 0.0)
            mB = mse_optimal(v_s, v_n0, k, eta, *strats["squeeze_B"])
            mA = mse_optimal(v_s, v_n0, k, eta, *strats["squeeze_A"])
            hold_rows.append(dict(
                eta=eta, k=k,
                mse_none=m0, mse_A=mA, mse_B=mB,
                rel_B=relative_advantage(mB, m0),
                rel_A=relative_advantage(mA, m0),
                B_beats_A=bool(mB < mA - 1e-15),
                B_beats_classical=bool(mB < m0 - 1e-15),
            ))

    # --- region map in (k, eta) at several noise ratios, fixed r0 ---
    regions = []
    k_fine = np.linspace(0.02, 1.98, 80)
    eta_fine = np.linspace(0.15, 1.0, 50)
    for ratio in ratio_vals:
        v_n = ratio * v_s
        mask = np.zeros((len(eta_fine), len(k_fine)), dtype=bool)
        intervals = []
        for i, eta in enumerate(eta_fine):
            iv = k_interval_b_beats_a(v_n, float(eta), r0)
            intervals.append(iv)
            if iv is None:
                continue
            lo, hi = iv
            mask[i, :] = (k_fine >= lo) & (k_fine <= hi)
        regions.append(dict(
            ratio=float(ratio), v_n=float(v_n),
            k_fine=k_fine, eta_fine=eta_fine, mask=mask,
            interval_at_eta0=k_interval_b_beats_a(v_n, eta0, r0),
        ))

    # --- intervals vs noise ratio at eta0 ---
    interval_vs_ratio = []
    for ratio in list(ratio_vals) + [32.0, 64.0]:
        iv = k_interval_b_beats_a(ratio * v_s, eta0, r0)
        interval_vs_ratio.append(dict(ratio=float(ratio), interval=iv))

    # --- Monte Carlo check ---
    mc_specs = [
        dict(k=0.5, eta=0.85, r_a=0.0, r_b=r0, tag="B, k=0.5, eta=0.85"),
        dict(k=0.5, eta=0.85, r_a=r0, r_b=0.0, tag="A, k=0.5, eta=0.85"),
        dict(k=1.0, eta=0.85, r_a=0.0, r_b=r0, tag="B, k=1, eta=0.85"),
        dict(k=1.5, eta=0.85, r_a=0.0, r_b=r0, tag="B, k=1.5, eta=0.85"),
        dict(k=0.5, eta=0.5, r_a=0.0, r_b=r0, tag="B, k=0.5, eta=0.5"),
        dict(k=0.5, eta=1.0, r_a=0.0, r_b=r0, tag="B, k=0.5, eta=1"),
    ][: cfg.n_mc_points]
    mc_rows = []
    for spec in mc_specs:
        rec = monte_carlo_mse(v_s, v_n0, spec["k"], spec["eta"], spec["r_a"], spec["r_b"], cfg.n_mc, rng)
        rec.update(spec)
        rec["ok"] = bool(rec["rel_err"] < 0.08)  # MC finite-sample tolerance
        mc_rows.append(rec)
        print(f"  MC {spec['tag']}: theory={rec['mse_theory']:.5f}  mc={rec['mse_mc']:.5f}  rel={rec['rel_err']:.3e}  {'ok' if rec['ok'] else 'MISMATCH'}")

    # photons at the default slice
    photon_rows = []
    for name, (ra, rb) in strats.items():
        ph = total_photons(v_s, v_n0, 0.5, ra, rb)
        photon_rows.append(dict(name=name, r_a=ra, r_b=rb, **ph))

    # exact identities to record
    # 1) unsqueezed σ² = 1 for all eta
    # 2) B vs classical: always strictly better for k≠0, η∈(0,1], r>0
    always_B_vs_class = all(row["B_beats_classical"] or row["k"] == 0 for row in hold_rows)

    print(
        f"  B-beats-A interval at η={eta0}, Vn/Vs={ratio0}, r={r0}: "
        + (f"k ∈ ({interval0[0]:.3f}, {interval0[1]:.3f})" if interval0 else "empty")
    )
    relB_085 = next(r for r in hold_rows if abs(r["eta"] - 0.85) < 1e-12 and abs(r["k"] - 0.5) < 1e-12)
    print(f"  squeeze-B vs unsqueezed at k=0.5, η=0.85, Vn/Vs={ratio0}: rel. adv.={relB_085['rel_B']:.4f}")

    return dict(
        v_s=v_s, v_n=v_n0, r=r0, eta0=eta0, ratio0=ratio0,
        k_grid=k_vals, eta_grid=eta_vals, ratio_grid=ratio_vals,
        strategies={n: dict(r_a=ra, r_b=rb) for n, (ra, rb) in strats.items()},
        vs_k=vs_k,
        interval0=interval0,
        vs_eta=vs_eta,
        vanish_1pct=vanish_1pct,
        vanish_5pct=vanish_5pct,
        hold_rows=hold_rows,
        regions=regions,
        interval_vs_ratio=interval_vs_ratio,
        mc_rows=mc_rows,
        photon_rows=photon_rows,
        always_B_vs_class=always_B_vs_class,
        convention="ħ=2, vacuum variance=1 per quadrature (Weedbrook 2012)",
        elapsed_s=time.perf_counter() - t0,
        note=(
            "The exact comparison squeeze-B vs unsqueezed is strictly favourable "
            "for every η in (0,1] when k≠0, V_n>0, r>0 (the relative gap shrinks as η→0). "
            "The interesting boundary is squeeze-B vs squeeze-A, an interval inside (0,1) "
            "that opens under strong noise."
        ),
    )


# ---------------------------------------------------------------------------
# Experiment 2
# ---------------------------------------------------------------------------

def experiment_wiring(cfg: SimConfig, rng: Generator) -> dict:
    t0 = time.perf_counter()
    catalogue = layout_catalogue(cfg)
    cells = {}
    ipr_below = aa_onsite_ipr(max(cfg.n_list), 0.5)
    ipr_above = aa_onsite_ipr(max(cfg.n_list), 1.5)
    print(f"  on-site AA IPR diagnostic (N={max(cfg.n_list)}): λ=0.5 mean IPR={ipr_below['ipr_mean']:.4f}; "
          f"λ=1.5 mean IPR={ipr_above['ipr_mean']:.4f} (transition of the standard model is at λ=1)")

    for n in cfg.n_list:
        for item in catalogue:
            if item["kind"] == "torus" and not is_perfect_square(n):
                continue
            A = make_layout(item["kind"], n, cfg, rng)
            bounds = laplacian_eigbounds(A)
            rbar = mean_resistance(A) if bounds["connected"] else float("inf")
            pred = predicted_mse(cfg, cfg.k_samples, cfg.snr_db, rbar) if np.isfinite(rbar) else None
            mses = []
            if bounds["connected"]:
                for _ in range(cfg.n_trials):
                    tau, _ = sample_veils(n, cfg.s_amp, cfg.s_phase, rng)
                    Y = generate_observations(tau, cfg.k_samples, cfg.snr_db, rng)
                    rmat = relative_veil_matrix(Y, A)
                    tau_hat = estimate_decentralized(rmat, A)
                    mses.append(mse_kappa(tau_hat, tau))
            summ = _summarize(mses)
            rec = dict(
                n=n, key=item["key"], title=item["title"], family=item["family"],
                kind=item["kind"], rbar=rbar, predicted=(pred or {}).get("predicted", float("nan")),
                floor=(pred or {}).get("floor", float("nan")),
                layout_term=(pred or {}).get("layout", float("nan")),
                mse=summ, **bounds,
            )
            cells[(n, item["key"])] = rec
            flag = "" if bounds["connected"] else "  DISCONNECTED"
            mse_s = f"{summ['mean']:.4e}±{summ['sem']:.1e}" if summ["n"] else "n/a"
            print(f"  N={n:3d} {item['key']:12s}  R̄={rbar if np.isfinite(rbar) else float('inf'):.4f}  "
                  f"λ2={bounds['lam2']:.4f}  deg={bounds['deg']:.2f}  edges={bounds['n_edges']:.0f}  "
                  f"MSE={mse_s}{flag}")

    # verdict: among unweighted degree-matched layouts, does any quasiperiodic
    # wiring beat the torus / expander on R_bar or MSE at the largest N?
    return dict(
        cells=cells,
        catalogue=catalogue,
        n_list=list(cfg.n_list),
        k_samples=cfg.k_samples,
        snr_db=cfg.snr_db,
        n_trials=cfg.n_trials,
        degree=cfg.degree,
        ipr_below=ipr_below,
        ipr_above=ipr_above,
        elapsed_s=time.perf_counter() - t0,
    )


# ---------------------------------------------------------------------------
# Agreement / timing
# ---------------------------------------------------------------------------







# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------









# ---------------------------------------------------------------------------
# Conclusions
# ---------------------------------------------------------------------------

def _recommend_wiring(e2: dict) -> tuple[str, list[str]]:
    failures = []
    # pick a torus N if possible for fair unweighted compare
    n_fair = max(n for n in e2["n_list"] if True)
    # require all compared layouts present at n_fair; torus only if square
    def rec_at(key):
        return e2["cells"].get((n_fair, key))

    ring = rec_at("ring4")
    rec_at("torus")
    rec_at("expander")
    rec_at("fib_circ")
    rec_at("mod13")
    m13n = rec_at("mod13_naive")
    fibw = rec_at("fib_w")
    aalo = rec_at("aa_below")
    aahi = rec_at("aa_above")

    def mse_of(rec):
        return rec["mse"]["mean"] if rec and rec["mse"]["n"] else float("nan")

    def rbar_of(rec):
        return rec["rbar"] if rec else float("nan")

    bits = []
    # unweighted quasiperiodic vs ring / torus / expander
    baselines = [(k, rec_at(k)) for k in ("ring4", "torus", "expander") if rec_at(k)]
    best_base = min(baselines, key=lambda kv: mse_of(kv[1])) if baselines else None

    for key, title in (("fib_circ", "Fibonacci circulant"), ("mod13", "mod-13 circulant")):
        rec = rec_at(key)
        if rec is None or not rec["connected"]:
            failures.append(f"{title} at N={n_fair} was disconnected or missing.")
            continue
        if best_base:
            bname, brec = best_base
            dm = mse_of(rec) - mse_of(brec)
            dr = rbar_of(rec) - rbar_of(brec)
            if np.isfinite(dr) and dr > 0.15 * max(rbar_of(brec), 1e-18):
                bits.append(
                    f"{title} is a worse layout than {bname} at N={n_fair} by effective resistance "
                    f"(ΔR̄={dr:+.3f}); measured ΔMSE={dm:+.2e} at 20 dB is floor-dominated."
                )
            elif np.isfinite(dr) and dr < -0.15 * max(rbar_of(brec), 1e-18):
                bits.append(
                    f"{title} is a better layout than {bname} at N={n_fair} by effective resistance "
                    f"(ΔR̄={dr:+.3f}); measured ΔMSE={dm:+.2e} at 20 dB is floor-dominated."
                )
            else:
                bits.append(f"{title} is roughly even with {bname} at N={n_fair} (ΔR̄={dr:+.3f}, ΔMSE={dm:+.2e}).")

    if m13n is not None and not m13n["connected"]:
        failures.append(
            f"Naive mod-13-only routing (no ring backbone) is disconnected at N={n_fair} "
            f"(gcd(13,{n_fair})={np.gcd(13, n_fair)}); a modulo jump alone is not a layout."
        )
    elif m13n is not None and np.isfinite(rbar_of(m13n)) and ring is not None:
        if rbar_of(m13n) > 2 * rbar_of(ring) or mse_of(m13n) > 2 * mse_of(ring):
            bits.append(
                f"Naive mod-13-only routing is a much worse layout than a degree-4 ring at N={n_fair} "
                f"(R̄={rbar_of(m13n):.3f} vs {rbar_of(ring):.3f})."
            )

    # weighted / localization
    if aahi and aalo and ring:
        if rbar_of(aahi) > 1.5 * rbar_of(aalo) or (aahi["lam2"] < 0.5 * aalo["lam2"]):
            bits.append(
                f"AA-weighted hoppings near vanishing (λ={aahi and e2['catalogue'][-1]}) "
                f"raised R̄ from {rbar_of(aalo):.3f} to {rbar_of(aahi):.3f} and cut λ₂ from "
                f"{aalo['lam2']:.3f} to {aahi['lam2']:.3f} at N={n_fair}; weak links hurt consensus."
            )
        if mse_of(aahi) > mse_of(aalo) * 1.15:
            bits.append(f"Measured MSE was higher in the near-vanishing AA ring ({mse_of(aahi):.3e} vs {mse_of(aalo):.3e}).")

    if aahi and aalo:
        bits.append(
            f"AA-weighted hoppings at N={n_fair}: near-vanishing λ raised R̄ from "
            f"{rbar_of(aalo):.3f} to {rbar_of(aahi):.3f} and cut λ₂ from "
            f"{aalo['lam2']:.4f} to {aahi['lam2']:.4f}. Weak links worsen the layout term; "
            f"at 20 dB the measured MSE is still mostly the floor "
            f"({mse_of(aahi):.2e} vs {mse_of(aalo):.2e}) and does not cleanly rank them."
        )
    if fibw and ring:
        bits.append(
            f"Fibonacci-weighted hoppings at N={n_fair} raised R̄ vs the uniform degree-4 ring "
            f"({rbar_of(fibw):.3f} vs {rbar_of(ring):.3f}, λ₂ {fibw['lam2']:.4f} vs {ring['lam2']:.4f}); "
            f"measured MSE at 20 dB was {mse_of(fibw):.2e} vs {mse_of(ring):.2e}, floor-dominated."
        )
    bits.append(
        "At 20 dB the 1/SNR² layout term is small next to the floor, so R̄ and λ₂ are the "
        "reliable layout comparison; measured MSE with 8 trials is too floor-heavy to declare a winner among well-connected graphs."
    )

    # disconnected anywhere
    for (n, key), rec in e2["cells"].items():
        if not rec["connected"]:
            failures.append(f"Layout {key} is disconnected at N={n} (λ₂={rec['lam2']:.2e}).")

    rec_line = " ".join(bits) if bits else "See the tables; no strong wiring contrast was recorded."
    return rec_line, failures


def format_conclusions(payload: dict) -> str:
    cfg = payload["config"]
    e1 = payload["exp_squeezing"]
    e2 = payload["exp_wiring"]
    backend = payload["backend_info"]
    agree = payload.get("agreement") or {}
    bench = payload.get("timing") or {}
    lines: list[str] = []
    a = lines.append
    a("WHAT THIS RUN ACTUALLY SHOWED")
    a("==============================")
    a(f"SIZE={cfg['size']}, seed={cfg['seed']}. Official backend: {payload.get('official_backend')}.")
    a("Convention: ħ=2, vacuum variance = 1 per quadrature (Weedbrook 2012).")
    a("Squeezing resource is counted as sinh² r photons. Tiles are IC computing units.")
    a(f"Exp 1 (CV squeezing) wall {e1['elapsed_s']:.1f}s;  Exp 2 (wiring) {e2['elapsed_s']:.1f}s.")
    a("")
    a("EXPERIMENT 1 — reference-mode squeezing")
    a("---------------------------------------")
    a("Mode A carries s+n, mode B carries k·n; both see a lossy beam splitter of")
    a("transmission η and are homodyned. The estimator is the optimal linear (Wiener)")
    a("map of the two x-quadratures onto s. Closed-form MSE as in the theory cell.")
    a(f"Default slice: V_s={e1['v_s']:g}, V_n/V_s={e1['ratio0']:g}, r={e1['r']:g} ")
    a(f"(sinh² r={squeezing_photons(e1['r']):.3f} photons), η_default={e1['eta0']}.")
    a("")
    a("MSE vs k at the default slice:")
    header = f"  {'k':>6}  {'unsqueezed':>12}  {'squeeze A':>12}  {'squeeze B':>12}  {'split':>12}  B>A?"
    a(header)
    iv = e1["interval0"]
    for i, k in enumerate(e1["k_grid"]):
        m0 = e1["vs_k"]["none"][i]
        mA = e1["vs_k"]["squeeze_A"][i]
        mB = e1["vs_k"]["squeeze_B"][i]
        mS = e1["vs_k"]["split"][i]
        flag = "yes" if mB < mA - 1e-15 else "no"
        a(f"  {k:6.3f}  {m0:12.5f}  {mA:12.5f}  {mB:12.5f}  {mS:12.5f}  {flag}")
    if iv:
        a(f"Exact B-beats-A interval at this slice: {iv[0]:.4f} < k < {iv[1]:.4f}.")
    else:
        a("Exact B-beats-A interval at this slice: empty (noise is not strong enough, or η,r too weak).")
    a("This interval lives inside (0,1) and opens as V_n grows. The slogan")
    a("'B beats A when k<1 under strong noise' is the large-V_n limit of that interval")
    a("(the endpoints approach 0 and 1). It is not literally every k<1 at finite noise.")
    a("")
    a("Advantage over the unsqueezed coherent baseline (same V_s, V_n, k, η;")
    a("a known extra coherent displacement does not change this Wiener MSE, so a")
    a("'matched-photon' classical displacement baseline coincides with unsqueezed):")
    a(f"  {'eta':>6}  {'k':>5}  {'rel.adv. B':>12}  {'rel.adv. A':>12}  B>class  B>A")
    for row in e1["hold_rows"]:
        a(f"  {row['eta']:6.2f}  {row['k']:5.2f}  {row['rel_B']:12.4f}  {row['rel_A']:12.4f}  "
          f"{'yes' if row['B_beats_classical'] else 'no ':3s}      {'yes' if row['B_beats_A'] else 'no'}")
    a("Exact result: squeeze-B (and squeeze-A) strictly beat unsqueezed for every")
    a("η in (0,1] when k≠0, V_n>0, r>0. The advantage does not change sign at η=0.85;")
    a("it only shrinks toward 0 as η→0, because loss replaces squeezed noise with vacuum.")
    a("The η=0.85 success criterion, read strictly, therefore holds. The practical size")
    a("of the advantage is in the table; we also record the η at which relative")
    a("advantage first reaches 1% and 5% (from the low-η side):")
    a(f"  {'k':>5}  {'η (1% B)':>10}  {'η (5% B)':>10}  {'η (1% A)':>10}  {'η (5% A)':>10}")
    for k in (0.25, 0.5, 0.8, 1.0, 1.5):
        a(f"  {k:5.2f}  {e1['vanish_1pct']['squeeze_B'][k]:10.3f}  {e1['vanish_5pct']['squeeze_B'][k]:10.3f}  "
          f"{e1['vanish_1pct']['squeeze_A'][k]:10.3f}  {e1['vanish_5pct']['squeeze_A'][k]:10.3f}")
    a("")
    a("Monte Carlo homodyne check (theoretical Wiener weights, sampled quadratures):")
    for rec in e1["mc_rows"]:
        a(f"  {rec['tag']}: theory={rec['mse_theory']:.6f}  MC={rec['mse_mc']:.6f}  "
          f"rel.err={rec['rel_err']:.2e}  {'ok' if rec['ok'] else 'MISMATCH'}")
    a("")
    a("Photon numbers at k=0.5, default slice (displacement + squeezing):")
    for row in e1["photon_rows"]:
        a(f"  {row['name']:10s}  r_A={row['r_a']:.3f}  r_B={row['r_b']:.3f}  "
          f"n_squeeze={row['n_squeeze']:.3f}  n_tot={row['n_tot']:.3f}")
    a("")

    a("EXPERIMENT 2 — quasiperiodic and other wirings, v2 error law")
    a("----------------------------------------------------------")
    a(f"Static unveiling, full overlap, K={e2['k_samples']}, SNR={e2['snr_db']:g} dB, "
      f"{e2['n_trials']} trials, relative (gauge-fixed) MSE(κ̂).")
    a("Unweighted layouts are aimed at degree 4 (same edge budget where possible).")
    a("Weighted layouts use the same ±1,±2 hop set with modulated hoppings.")
    a("The standard on-site Aubry–André transition (λ=1 in the 2λ-cos, t=1 convention)")
    a("is used only as an IPR diagnostic; it is not the unveiling graph.")
    a(f"  on-site AA N={e2['ipr_below']['n']}: λ=0.5 mean IPR={e2['ipr_below']['ipr_mean']:.4f}, "
      f"λ=1.5 mean IPR={e2['ipr_above']['ipr_mean']:.4f}.")
    a("")
    a(f"  {'N':>4}  {'layout':<28}  {'R̄':>8}  {'λ2':>8}  {'deg':>6}  {'edges':>5}  "
      f"{'MSE':>10}  {'pred':>10}  conn")
    for n in e2["n_list"]:
        for item in e2["catalogue"]:
            rec = e2["cells"].get((n, item["key"]))
            if rec is None:
                continue
            rbar = rec["rbar"]
            rbar_s = f"{rbar:8.3f}" if np.isfinite(rbar) else "     inf"
            mse_s = f"{rec['mse']['mean']:10.3e}" if rec["mse"]["n"] else "       n/a"
            pred_s = f"{rec['predicted']:10.3e}" if np.isfinite(rec["predicted"]) else "       n/a"
            a(f"  {n:4d}  {item['title']:<28}  {rbar_s}  {rec['lam2']:8.4f}  {rec['deg']:6.2f}  "
              f"{rec['n_edges']:5.0f}  {mse_s}  {pred_s}  {'Y' if rec['connected'] else 'N'}")
    a("")
    wire_line, wire_fail = _recommend_wiring(e2)
    a("Wiring verdict: " + wire_line)
    a("")

    a("RECOMMENDATION")
    a("--------------")
    if iv:
        rec_s = (
            f"Under strong technical noise (V_n/V_s={e1['ratio0']:g}) at η={e1['eta0']} and r={e1['r']:g}, "
            f"put the squeezing on the reference mode when k is inside ({iv[0]:.3f}, {iv[1]:.3f}); "
            "put it on the signal mode outside that interval, in particular for k≥1. "
        )
    else:
        rec_s = (
            "At this slice the B-beats-A interval is empty; squeezing the signal mode is at least as good. "
        )
    rec_s += (
        "Advantage over the unsqueezed baseline is strictly positive at every η in (0,1] "
        "but becomes small at high loss; quote the relative-advantage table, not a slogan. "
        "This is a Gaussian-estimator fact in the stated model, not a hardware result. "
        "On wiring: do not assume that a quasiperiodic or localizing layout helps unveiling; "
        "weak links raise R̄(G) and can hurt the consensus used to recover relative veils. "
        + wire_line
    )
    a(rec_s)
    a("")
    a("PLAIN FAILURES AND LIMITS")
    a("-------------------------")
    failures = list(wire_fail)
    if not all(r["ok"] for r in e1["mc_rows"]):
        failures.append("One or more Monte Carlo points missed the closed form at the 8% relative tolerance.")
    if iv is None:
        failures.append("No (k) interval where squeeze-B beats squeeze-A was open at the default slice.")
    # practical advantage at low eta
    rels_low = [r["rel_B"] for r in e1["hold_rows"] if abs(r["eta"] - 0.30) < 1e-12]
    if rels_low and max(rels_low) < 0.05:
        failures.append(
            f"At η=0.30 the relative advantage of squeeze-B over unsqueezed is small "
            f"(max {max(rels_low):.3f} on the scanned k); a strictly positive gap is not a large one."
        )
    if not failures:
        a("- None beyond the ordinary limits of a SIZE=small grid and a Gaussian linear model.")
    else:
        for f in failures:
            a(f"- {f}")
    a("- Localization in the on-site Aubry–André Hamiltonian is not the same object as a")
    a("  poorly wired comparison graph; we used weighted hoppings as the unveiling analogue")
    a("  and kept the on-site IPR as a separate diagnostic.")
    a("- A known extra coherent displacement at matched total photon number does not change")
    a("  the Wiener MSE in this model; we therefore did not invent a brighter-classical")
    a("  baseline that rescales s and n.")
    a("")
    a("BACKEND, AGREEMENT, AND TIMING")
    a("------------------------------")
    a(f"Requested BACKEND={backend.get('requested')}, resolved={backend.get('name')}, "
      f"torch={backend.get('torch_version')}, cuda={backend.get('cuda')}, device={backend.get('device')}.")
    if agree.get("available"):
        num = agree["numeric"]
        a(f"Numeric agreement (closed-form NumPy vs torch, ħ=2 formula): "
          f"max |ΔMSE|={num['max_abs_mse']:.3e}  ({'ok' if num['ok'] else 'NOT CLOSE'}).")
    else:
        a("PyTorch was not available; agreement was skipped.")
    if bench:
        a(f"Timing bench: closed-form grid {bench.get('n_k')}×{bench.get('n_eta')}.")
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
    if obj is None or isinstance(obj, (str, float, int, bool)):
        return obj
    return str(obj)


def run_suite(cfg: SimConfig, outdir, backend: str = "numpy",
              do_agreement: bool = True, do_timing: bool = True) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    info = resolve_backend(backend)
    # official small stays numpy even under auto
    official = "numpy" if (cfg.size == "small" and info["requested"] in ("auto", "numpy")) else info["name"]
    if cfg.size == "small" and info["requested"] in ("auto", "numpy"):
        info = dict(info)
        info["name"] = "numpy"
        info["device"] = None
    rng = np.random.default_rng(cfg.seed)
    print(
        f"SRI v6  SIZE={cfg.size}  BACKEND requested={info['requested']} resolved={info['name']}  "
        f"official={official}  cuda={info['cuda']}  torch={info['torch_version']}  "
        f"convention=ħ=2 vacuum={VACUUM_VAR}"
    )
    print("Experiment 1 — CV reference squeezing")
    e1 = experiment_squeezing(cfg, rng)
    print("Experiment 2 — quasiperiodic and other wirings")
    e2 = experiment_wiring(cfg, rng)

    agree = None
    if do_agreement:
        if torch_module() is not None:
            th_info = resolve_backend("torch")
            print("Agreement check (numpy vs torch closed form)")
            agree = agreement_check(cfg, np.random.default_rng(cfg.seed + 3), th_info)
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
    pngs += plot_squeezing(e1, outdir)
    pngs += plot_wiring(e2, outdir)
    if bench:
        pngs += plot_timing(bench, outdir)
    print("wrote", ", ".join(Path(p).name for p in pngs))

    payload = dict(
        config=dict(size=cfg.size, seed=cfg.seed, v_s=cfg.v_s, r_default=cfg.r_default,
                    ratio_default=cfg.ratio_default, eta_default=cfg.eta_default,
                    k_samples=cfg.k_samples, snr_db=cfg.snr_db, n_trials=cfg.n_trials),
        official_backend=official,
        backend_info=dict(requested=info["requested"], name=info["name"], device=str(info["device"]),
                          cuda=info["cuda"], torch_version=info["torch_version"]),
        exp_squeezing=e1,
        exp_wiring=e2,
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
def agreement_check(*_args, **_kwargs):
    raise RuntimeError('agreement_check' + " is not part of the NumPy reproduction driver")
