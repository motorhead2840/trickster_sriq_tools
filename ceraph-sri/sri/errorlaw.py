"""Two-term error law for a gauge-fixed veil estimate.

MSE(κ̂) ≈ E|τ|^{-2} / (K · SNR) + E|τ|^{-4} / (K · SNR^2) · R̄(G),
with R̄(G) = tr(L+)/n. On a ring the second term catches the first near
N* ≈ 12 · SNR. The formula is a leading-term calculation with coefficients
near 1, not an identity.

``xp`` is NumPy. The closed forms are plain array arithmetic so a later
optional backend can rebind ``xp``. This module does not import torch.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

xp = np

def expected_veil_moment(s_amp: float, power: int) -> float:
    """E|τ|^{-p} for τ = exp(η + iξ) with η ~ N(0, s_amp^2).

    The phase does not enter. The value is exp(p^2 s_amp^2 / 2).
    """
    return float(np.exp((power ** 2) * 0.5 * float(s_amp) ** 2))

def snr_linear(snr_db: float) -> float:
    """Convert an SNR in dB to a linear power ratio."""
    return float(10.0 ** (snr_db / 10.0))

def predicted_terms(cfg: Any, k: int, snr_db: float, rbar: float) -> dict[str, float]:
    """The two terms of the proposed law, with theoretical coefficients 1 and 1."""
    snr = snr_linear(snr_db)
    e2 = cfg.e_tau_inv_pow(2)  # E|τ|^{-2}
    e4 = cfg.e_tau_inv_pow(4)  # E|τ|^{-4}
    floor = e2 / (k * snr)
    layout = (e4 / (k * snr * snr)) * rbar
    return dict(
        e_inv2=e2,
        e_inv4=e4,
        floor=floor,
        layout=layout,
        predicted_mse=floor + layout,
        factor_e2_over_e4=e2 / e4,
    )

def predicted_mse(cfg: Any, k: int, snr_db: float, rbar: float) -> dict[str, float]:
    """Two-term veil-error prediction at a given sample count, SNR, and mean resistance."""
    snr = 10.0 ** (snr_db / 10.0)
    e2 = cfg.e_tau_inv_pow(2)
    e4 = cfg.e_tau_inv_pow(4)
    floor = e2 / (k * snr)
    layout = (e4 / (k * snr * snr)) * rbar
    return dict(e_inv2=e2, e_inv4=e4, floor=floor, layout=layout, predicted=floor + layout)

def ring_nstar_theory(cfg: Any, snr_db: float, use_exact_rbar: bool = True) -> dict[str, float]:
    """N* where the ring layout term equals the floor.

    Leading approximation: N* ≈ 12 · SNR · (E|τ|^{-2}/E|τ|^{-4}), independent of K.
    With the exact ring R̄ = (N²-1)/(12N), solve (N²-1)/N = 12 · SNR · factor.
    """
    snr = snr_linear(snr_db)
    factor = cfg.e_tau_inv_pow(2) / cfg.e_tau_inv_pow(4)
    n_approx_unit = 12.0 * snr  # the "12·SNR" shorthand (factor treated as 1)
    n_approx = 12.0 * snr * factor
    # exact: N - 1/N = 12 * SNR * factor =: c
    c = 12.0 * snr * factor
    # N² - c N - 1 = 0  →  N = (c + sqrt(c²+4))/2
    n_exact = 0.5 * (c + np.sqrt(c * c + 4.0))
    return dict(
        snr_db=float(snr_db),
        snr_lin=snr,
        factor=float(factor),
        nstar_12snr=float(n_approx_unit),
        nstar_12snr_factor=float(n_approx),
        nstar_exact_rbar=float(n_exact),
    )

def estimate_nstar_from_curve(
    N: np.ndarray,
    mse: np.ndarray,
    floor: float,
    sem: np.ndarray | None = None,
) -> dict:
    """Interpolate N where mse(N) = 2*floor (layout term equals floor).

    Returns a dict with nstar (or bounds if the crossing is not inside the range).
    """
    N = np.asarray(N, dtype=np.float64)
    mse = np.asarray(mse, dtype=np.float64)
    ok = np.isfinite(N) & np.isfinite(mse) & (N > 0) & (mse > 0)
    N, mse = N[ok], mse[ok]
    out = dict(
        floor=float(floor),
        target=float(2.0 * floor),
        nstar=float("nan"),
        nstar_lo=float("nan"),
        nstar_hi=float("nan"),
        status="empty",
        mse_first=float(mse[0]) if mse.size else float("nan"),
        mse_last=float(mse[-1]) if mse.size else float("nan"),
        N_first=float(N[0]) if N.size else float("nan"),
        N_last=float(N[-1]) if N.size else float("nan"),
    )
    if mse.size < 2 or not np.isfinite(floor) or floor <= 0:
        out["status"] = "bad-floor"
        return out
    target = 2.0 * floor
    above = mse >= target
    if not np.any(above):
        out["status"] = "above_range"  # N* > N_max (layout never catches the floor)
        out["nstar_lo"] = float(N[-1])
        return out
    if np.all(above):
        # already above at smallest N
        out["status"] = "below_range"
        out["nstar_hi"] = float(N[0])
        return out
    # first crossing: last False then first True (assume roughly increasing)
    # use linear interpolation on N (not log) of mse vs N
    # find all sign changes of mse-target
    d = mse - target
    sign = np.sign(d)
    crossings = []
    for i in range(len(d) - 1):
        if sign[i] == 0:
            crossings.append(float(N[i]))
        elif sign[i] * sign[i + 1] < 0:
            # linear interp
            t = d[i] / (d[i] - d[i + 1]) if d[i] != d[i + 1] else 0.0
            crossings.append(float(N[i] + t * (N[i + 1] - N[i])))
    if not crossings:
        out["status"] = "no_crossing"
        return out
    out["nstar"] = float(crossings[0])
    out["all_crossings"] = crossings
    out["status"] = "ok"
    return out

def fit_two_term_law(x1: np.ndarray, x2: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> dict:
    """Weighted LS: y ≈ c1 x1 + c2 x2  (no intercept). Also reports R² and a with-intercept fit."""
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x1) & np.isfinite(x2) & np.isfinite(y)
    x1, x2, y = x1[ok], x2[ok], y[ok]
    if w is not None:
        w = np.asarray(w, dtype=np.float64)[ok]
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    else:
        w = np.ones_like(y)
    if y.size < 3:
        return dict(c_floor=np.nan, c_layout=np.nan, r2=np.nan, n=int(y.size))
    sw = np.sqrt(w)
    X = np.column_stack([x1, x2]) * sw[:, None]
    yw = y * sw
    try:
        c, _, _, _ = np.linalg.lstsq(X, yw, rcond=None)
    except np.linalg.LinAlgError:
        return dict(c_floor=np.nan, c_layout=np.nan, r2=np.nan, n=int(y.size))
    yhat = c[0] * x1 + c[1] * x2
    ss_res = float(np.sum(w * (y - yhat) ** 2))
    ybar = float(np.sum(w * y) / np.sum(w))
    ss_tot = float(np.sum(w * (y - ybar) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    # with intercept, to see leftover bias
    X3 = np.column_stack([np.ones_like(x1), x1, x2]) * sw[:, None]
    try:
        c3, _, _, _ = np.linalg.lstsq(X3, yw, rcond=None)
        intercept = float(c3[0])
        c1_i, c2_i = float(c3[1]), float(c3[2])
    except np.linalg.LinAlgError:
        intercept, c1_i, c2_i = float("nan"), float("nan"), float("nan")
    return dict(
        c_floor=float(c[0]),
        c_layout=float(c[1]),
        r2=float(r2),
        n=int(y.size),
        intercept=intercept,
        c_floor_with_intercept=c1_i,
        c_layout_with_intercept=c2_i,
        ss_res=ss_res,
        ss_tot=ss_tot,
    )

def fit_power_law(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """log y = a + b log x,  y ≈ C x^b. Drops non-positive y."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = (x > 0) & (y > 0)
    if ok.sum() < 2:
        return dict(slope=np.nan, intercept=np.nan, r_value=np.nan, intercept_log=np.nan)
    res = stats.linregress(np.log(x[ok]), np.log(y[ok]))
    return dict(
        slope=float(res.slope),
        intercept=float(np.exp(res.intercept)),
        intercept_log=float(res.intercept),
        r_value=float(res.rvalue),
    )

def fit_log_n(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """y = a + b log x  (the 2-D grid / torus prediction for MSE)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = x > 1
    if ok.sum() < 2:
        return dict(slope=np.nan, intercept=np.nan, r_value=np.nan)
    res = stats.linregress(np.log(x[ok]), y[ok])
    return dict(slope=float(res.slope), intercept=float(res.intercept), r_value=float(res.rvalue))
