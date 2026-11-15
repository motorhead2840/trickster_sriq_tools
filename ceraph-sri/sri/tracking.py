"""Adaptive trackers for a drifting per-tile veil.

The recommended rule is the innovation-based adaptive Kalman filter (Mehra):
it matches the innovation variance and never reads the true veil or the true
process variance. The corrected Ξ_ν gain is the cheaper fallback; it raises
the gain when innovations exceed the nominal model. IMM is a two-model
comparison. Sage–Husa adapts the measurement variance as well and is kept as
a comparison that loses on a rising process variance.

The v1 consistency gain raises the gain when neighbour agreement is high.
That sign is wrong after a jump in the drift, and the rule is kept only as a
baseline.

Oracle and frozen Kalman filters are for comparison. They are classical
scalar filters on the log-amplitude and phase channels.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from sri.unveiling import xi_from_consistency, xi_from_consistency_batch

ComplexArr = NDArray[np.complex128]
FloatArr = NDArray[np.float64]

STAY_PROB = 0.95
G_FIXED = (0.05, 0.20, 0.60, 0.90)
G_MIN_V1, G_MAX_V1 = 0.05, 0.60
Q_MIN_FACTOR = 1e-4
R_MIN_FACTOR = 1e-4
ADAPT_DELAY = 8

def steady_state_kalman_gain(a: float, q: float, r: float, tol: float = 1e-12, maxit: int = 10000) -> dict:
    """Scalar steady-state Kalman gain for x_{t+1} = a x_t + w, z = x + v.

    Returns the *update* gain g = P^- / (P^- + r) applied after the predict step,
    plus the implied posterior P and predicted Ppred.  Uses the closed-form
    quadratic for Ppred when q,r > 0, and falls back to iteration otherwise.
    """
    a = float(a)
    q = float(max(q, 0.0))
    r = float(max(r, 0.0))
    if r <= 0.0:
        return dict(g=1.0, P=0.0, Ppred=q, a=a, q=q, r=r)
    if q <= 0.0:
        # no process noise: gain and variances go to 0
        return dict(g=0.0, P=0.0, Ppred=0.0, a=a, q=q, r=r)
    # Ppred = Q satisfies Q² + (r(1-a²) - q) Q - q r = 0.  Positive root.
    disc = (r * (1.0 - a * a) - q) ** 2 + 4.0 * q * r
    Q = 0.5 * (q - r * (1.0 - a * a) + np.sqrt(max(disc, 0.0)))
    g = Q / (Q + r)
    P = (1.0 - g) * Q
    return dict(g=float(g), P=float(P), Ppred=float(Q), a=a, q=q, r=r)

def kalman_channel_step(x, P, z, a, q, r):
    """Vectorized scalar Kalman step.  x,P,z,q,r may be scalars or arrays."""
    xpred = a * x
    Ppred = a * a * P + q
    S = Ppred + r
    g = Ppred / np.maximum(S, 1e-18)
    nu = z - xpred
    xnew = xpred + g * nu
    Pnew = (1.0 - g) * Ppred
    return xnew, Pnew, g, nu, Ppred, S

def gauss_lik(nu, S):
    """Scalar Gaussian likelihood N(nu; 0, S), vectorized, floored for stability."""
    S = np.maximum(S, 1e-18)
    return np.exp(-0.5 * (nu * nu) / S) / np.sqrt(2.0 * np.pi * S)

def split_r(r_total: float, r_amp_ref: float | None = None, r_phase_ref: float | None = None) -> tuple[float, float]:
    """Split a total MSE(κ) measurement variance into amplitude / phase.

    If a reference split (e.g. from a static measurement) is given, keep that
    proportion; otherwise split equally.
    """
    if r_amp_ref is not None and r_phase_ref is not None:
        tot = max(r_amp_ref + r_phase_ref, 1e-18)
        return float(r_total * (r_amp_ref / tot)), float(r_total * (r_phase_ref / tot))
    return float(0.5 * r_total), float(0.5 * r_total)

def v1_xi_gain(proxy: float, g_min: float = G_MIN_V1, g_max: float = G_MAX_V1) -> float:
    """v1 consistency gain.

    The gain rises when the neighbour-consistency score rises. A sudden increase
    in veil drift makes the held estimate inconsistent and therefore lowers the
    gain, which is the wrong direction. Kept as a documented baseline.
    """
    return float(g_min + (g_max - g_min) * float(np.clip(proxy, 0.0, 1.0)))


def corrected_xi_gain(xi_nu: float, g_nom: float, g_high: float) -> float:
    """Fallback gain from the innovation score Ξ_ν = min(1, S / C).

    S is the innovation variance of the nominal model and C is a smoothed
    innovation power. The gain rises from g_nom toward g_high as Ξ_ν falls.
    """
    xi_nu = float(np.clip(xi_nu, 0.0, 1.0))
    return float(g_nom + (g_high - g_nom) * (1.0 - xi_nu))

class WalkFilter:
    """One tracking method, holding per-tile amplitude/phase state for one walk."""

    def __init__(self, name: str, n: int, cfg: Any, params: dict):
        self.name = name
        self.n = n
        self.a = float(cfg.rho_ou)
        self.params = params
        self.kind = params["kind"]
        self.kappa = np.zeros(n, dtype=np.complex128)
        # channel state
        self.xa = np.zeros(n)
        self.xp = np.zeros(n)
        self.Pa = np.zeros(n)
        self.Pp = np.zeros(n)
        self.qa = np.full(n, params["q_amp0"])
        self.qp = np.full(n, params["q_phase0"])
        self.ra = np.full(n, params["r_amp"])
        self.rp = np.full(n, params["r_phase"])
        self.Ca = np.zeros(n)  # EWMA of ν²
        self.Cp = np.zeros(n)
        self.lam = float(params.get("forget", cfg.default_forget))
        self.q_amp0 = float(params["q_amp0"])
        self.q_phase0 = float(params["q_phase0"])
        self.r_amp0 = float(params["r_amp"])
        self.r_phase0 = float(params["r_phase"])
        self.qmin_a = max(Q_MIN_FACTOR * self.q_amp0, 1e-16)
        self.qmin_p = max(Q_MIN_FACTOR * self.q_phase0, 1e-16)
        self.rmin_a = max(R_MIN_FACTOR * self.r_amp0, 1e-16)
        self.rmin_p = max(R_MIN_FACTOR * self.r_phase0, 1e-16)
        self.g_nom_a = steady_state_kalman_gain(self.a, self.q_amp0, self.r_amp0)["g"]
        self.g_nom_p = steady_state_kalman_gain(self.a, self.q_phase0, self.r_phase0)["g"]
        qhi_a = params.get("q_high_amp", 10.0 * self.q_amp0)
        qhi_p = params.get("q_high_phase", 10.0 * self.q_phase0)
        self.g_hi_a = steady_state_kalman_gain(self.a, qhi_a, self.r_amp0)["g"]
        self.g_hi_p = steady_state_kalman_gain(self.a, qhi_p, self.r_phase0)["g"]
        # IMM
        self.n_modes = 2
        self.mu = np.full((n, 2), 0.5)
        self.imm_xa = np.zeros((n, 2))
        self.imm_xp = np.zeros((n, 2))
        self.imm_Pa = np.zeros((n, 2))
        self.imm_Pp = np.zeros((n, 2))
        self.imm_q_a = np.array([self.q_amp0, qhi_a], dtype=np.float64)
        self.imm_q_p = np.array([self.q_phase0, qhi_p], dtype=np.float64)
        self.Pij = np.array(
            [[STAY_PROB, 1.0 - STAY_PROB], [1.0 - STAY_PROB, STAY_PROB]],
            dtype=np.float64,
        )
        self.last_g = 0.0
        self.last_qhat = 0.5 * (self.q_amp0 + self.q_phase0)
        self.last_rhat = 0.5 * (self.r_amp0 + self.r_phase0)
        self.last_xi_nu = 1.0
        self.last_nis = 1.0
        self._initialized = False

    def initialize(self, z: ComplexArr):
        """initialize."""
        n = self.n
        self.xa = z.real.copy()
        self.xp = z.imag.copy()
        # start P at the measurement variance (weak prior on the first sample)
        self.Pa = np.full(n, self.r_amp0)
        self.Pp = np.full(n, self.r_phase0)
        self.qa[:] = self.q_amp0
        self.qp[:] = self.q_phase0
        self.ra[:] = self.r_amp0
        self.rp[:] = self.r_phase0
        S0a = self.a * self.a * self.r_amp0 + self.q_amp0 + self.r_amp0
        S0p = self.a * self.a * self.r_phase0 + self.q_phase0 + self.r_phase0
        self.Ca[:] = S0a
        self.Cp[:] = S0p
        self.imm_xa[:, :] = z.real[:, None]
        self.imm_xp[:, :] = z.imag[:, None]
        self.imm_Pa[:, :] = self.r_amp0
        self.imm_Pp[:, :] = self.r_phase0
        self.mu[:, :] = 0.5
        self.kappa = z.copy()
        self._initialized = True
        self.last_g = 0.5 * (self.g_nom_a + self.g_nom_p)
        self.last_qhat = 0.5 * (self.q_amp0 + self.q_phase0)
        self.last_rhat = 0.5 * (self.r_amp0 + self.r_phase0)
        self.last_xi_nu = 1.0
        self.last_nis = 1.0

    def step(self, z: ComplexArr, t: int, rmat=None, A=None, q_amp_true=None, q_phase_true=None, r_amp_true=None, r_phase_true=None):
        """step."""
        if not self._initialized or t == 0:
            self.initialize(z)
            return self.kappa

        kind = self.kind
        a = self.a
        if kind == "iir_fixed":
            g = float(self.params["g"])
            self.kappa = (1.0 - g) * self.kappa + g * z
            self.last_g = g
            return self.kappa

        if kind == "iir_v1_xi":
            tau_held = np.exp(self.kappa)
            proxy = xi_from_consistency(rmat, A, tau_held) if rmat is not None else 0.5
            g = G_MIN_V1 + (G_MAX_V1 - G_MIN_V1) * float(np.clip(proxy, 0.0, 1.0))
            self.kappa = (1.0 - g) * self.kappa + g * z
            self.last_g = g
            self.last_xi_nu = float(np.clip(proxy, 0.0, 1.0))
            return self.kappa

        if kind == "kalman_oracle":
            qa = float(q_amp_true)
            qp = float(q_phase_true)
            ra = float(r_amp_true if r_amp_true is not None else self.r_amp0)
            rp = float(r_phase_true if r_phase_true is not None else self.r_phase0)
            self.xa, self.Pa, ga, *_ = kalman_channel_step(self.xa, self.Pa, z.real, a, qa, ra)
            self.xp, self.Pp, gp, *_ = kalman_channel_step(self.xp, self.Pp, z.imag, a, qp, rp)
            self.kappa = self.xa + 1j * self.xp
            self.last_g = float(0.5 * (np.mean(ga) + np.mean(gp)))
            self.last_qhat = 0.5 * (qa + qp)
            self.last_rhat = 0.5 * (ra + rp)
            return self.kappa

        if kind == "kalman_fixedq":
            self.xa, self.Pa, ga, *_ = kalman_channel_step(self.xa, self.Pa, z.real, a, self.q_amp0, self.r_amp0)
            self.xp, self.Pp, gp, *_ = kalman_channel_step(self.xp, self.Pp, z.imag, a, self.q_phase0, self.r_phase0)
            self.kappa = self.xa + 1j * self.xp
            self.last_g = float(0.5 * (np.mean(ga) + np.mean(gp)))
            self.last_qhat = 0.5 * (self.q_amp0 + self.q_phase0)
            return self.kappa

        if kind == "mehra":
            return self._step_mehra(z, t)

        if kind == "sage_husa":
            return self._step_sage_husa(z, t)

        if kind == "xi_corrected":
            return self._step_xi_corrected(z, t)

        if kind == "imm":
            return self._step_imm(z, t)

        raise ValueError(f"unknown filter kind {kind!r}")

    def _adapt_ok(self, t: int) -> bool:
        return t >= ADAPT_DELAY

    def _step_mehra(self, z: ComplexArr, t: int):
        a, lam = self.a, self.lam
        # predict/update with current q̂, then refresh q̂ from the innovation
        xa, Pa, ga, nua, Ppreda, Sa = kalman_channel_step(self.xa, self.Pa, z.real, a, self.qa, self.ra)
        xp, Pp, gp, nup, Ppredp, Sp = kalman_channel_step(self.xp, self.Pp, z.imag, a, self.qp, self.rp)
        self.Ca = lam * self.Ca + (1.0 - lam) * (nua * nua)
        self.Cp = lam * self.Cp + (1.0 - lam) * (nup * nup)
        if self._adapt_ok(t):
            # E[ν²] = a² P_prev + q + r  ⇒  q̂ = C - a² P_prev - r
            # (P_prev is self.Pa / self.Pp, still the previous posterior)
            qa = np.maximum(self.qmin_a, self.Ca - a * a * self.Pa - self.ra)
            qp = np.maximum(self.qmin_p, self.Cp - a * a * self.Pp - self.rp)
            self.qa, self.qp = qa, qp
        self.xa, self.Pa, self.xp, self.Pp = xa, Pa, xp, Pp
        self.kappa = self.xa + 1j * self.xp
        self.last_g = float(0.5 * (np.mean(ga) + np.mean(gp)))
        self.last_qhat = float(0.5 * (np.median(self.qa) + np.median(self.qp)))
        nis = 0.5 * (np.mean(nua * nua / np.maximum(Sa, 1e-18)) + np.mean(nup * nup / np.maximum(Sp, 1e-18)))
        self.last_nis = float(nis)
        # Ξ_ν = min(1, S / C)
        xi = 0.5 * (
            np.mean(np.minimum(1.0, Sa / np.maximum(self.Ca, 1e-18)))
            + np.mean(np.minimum(1.0, Sp / np.maximum(self.Cp, 1e-18)))
        )
        self.last_xi_nu = float(xi)
        return self.kappa

    def _step_sage_husa(self, z: ComplexArr, t: int):
        """Sage–Husa with *constant* forgetting d = 1-λ.

        The textbook schedule d_t = (1-b)/(1-b^{t+1}) shrinks and would be
        sluggish after a mid-run jump, which is the event we care about.
        Constant forgetting is the standard practical fix for that.
        r is also adapted, which is the intended contrast with Mehra.
        """
        a, d = self.a, (1.0 - self.lam)
        Pa_prev, Pp_prev = self.Pa, self.Pp
        xa, Pa, ga, nua, Ppreda, Sa = kalman_channel_step(self.xa, self.Pa, z.real, a, self.qa, self.ra)
        xp, Pp, gp, nup, Ppredp, Sp = kalman_channel_step(self.xp, self.Pp, z.imag, a, self.qp, self.rp)
        if self._adapt_ok(t):
            ra = (1.0 - d) * self.ra + d * (nua * nua - Ppreda)
            rp = (1.0 - d) * self.rp + d * (nup * nup - Ppredp)
            self.ra = np.maximum(self.rmin_a, ra)
            self.rp = np.maximum(self.rmin_p, rp)
            qa = (1.0 - d) * self.qa + d * (ga * ga * nua * nua + Pa - a * a * Pa_prev)
            qp = (1.0 - d) * self.qp + d * (gp * gp * nup * nup + Pp - a * a * Pp_prev)
            self.qa = np.maximum(self.qmin_a, qa)
            self.qp = np.maximum(self.qmin_p, qp)
        self.Ca = (1.0 - d) * self.Ca + d * (nua * nua)
        self.Cp = (1.0 - d) * self.Cp + d * (nup * nup)
        self.xa, self.Pa, self.xp, self.Pp = xa, Pa, xp, Pp
        self.kappa = self.xa + 1j * self.xp
        self.last_g = float(0.5 * (np.mean(ga) + np.mean(gp)))
        self.last_qhat = float(0.5 * (np.median(self.qa) + np.median(self.qp)))
        self.last_rhat = float(0.5 * (np.mean(self.ra) + np.mean(self.rp)))
        nis = 0.5 * (np.mean(nua * nua / np.maximum(Sa, 1e-18)) + np.mean(nup * nup / np.maximum(Sp, 1e-18)))
        self.last_nis = float(nis)
        return self.kappa

    def _step_xi_corrected(self, z: ComplexArr, t: int):
        """Map Ξ_ν = min(1, S/C) to a gain with the *right* sign: g rises as Ξ_ν falls.

        S is the innovation variance under the *nominal* (q0, r) model.
        C is an EWMA of ν².  The gain interpolates between g*(q0) and g*(10 q0).
        This is cheaper than identifying q, and keeps an SRI-style Ξ in [0,1].
        """
        a, lam = self.a, self.lam
        # predict with nominal q0 (the S in NIS is the nominal model's forecast)
        xpreda = a * self.xa
        xpredp = a * self.xp
        Ppreda = a * a * self.Pa + self.q_amp0
        Ppredp = a * a * self.Pp + self.q_phase0
        Sa = Ppreda + self.r_amp0
        Sp = Ppredp + self.r_phase0
        nua = z.real - xpreda
        nup = z.imag - xpredp
        self.Ca = lam * self.Ca + (1.0 - lam) * (nua * nua)
        self.Cp = lam * self.Cp + (1.0 - lam) * (nup * nup)
        xi_a = np.minimum(1.0, Sa / np.maximum(self.Ca, 1e-18))
        xi_p = np.minimum(1.0, Sp / np.maximum(self.Cp, 1e-18))
        if self._adapt_ok(t):
            ga = self.g_nom_a + (self.g_hi_a - self.g_nom_a) * (1.0 - xi_a)
            gp = self.g_nom_p + (self.g_hi_p - self.g_nom_p) * (1.0 - xi_p)
        else:
            ga = np.full(self.n, self.g_nom_a)
            gp = np.full(self.n, self.g_nom_p)
        self.xa = xpreda + ga * nua
        self.xp = xpredp + gp * nup
        # Joseph form not needed; use the applied-gain covariance
        self.Pa = (1.0 - ga) ** 2 * Ppreda + ga ** 2 * self.r_amp0
        self.Pp = (1.0 - gp) ** 2 * Ppredp + gp ** 2 * self.r_phase0
        self.kappa = self.xa + 1j * self.xp
        self.last_g = float(0.5 * (np.mean(ga) + np.mean(gp)))
        self.last_xi_nu = float(0.5 * (np.mean(xi_a) + np.mean(xi_p)))
        # implied q̂ so the plot has a comparable curve: invert the interpolation
        # q is not identified here; report a linear-in-Ξ interpolation of q0..10q0
        self.last_qhat = float(
            0.5
            * (
                np.mean(self.q_amp0 + (10.0 * self.q_amp0 - self.q_amp0) * (1.0 - xi_a))
                + np.mean(self.q_phase0 + (10.0 * self.q_phase0 - self.q_phase0) * (1.0 - xi_p))
            )
        )
        nis = 0.5 * (np.mean(nua * nua / np.maximum(Sa, 1e-18)) + np.mean(nup * nup / np.maximum(Sp, 1e-18)))
        self.last_nis = float(nis)
        return self.kappa

    def _step_imm(self, z: ComplexArr, t: int):
        """Two-model IMM with mixing, models q0 and 10 q0, shared mode per tile."""
        a = self.a
        n = self.n
        Pij = self.Pij
        mu = self.mu
        # predicted mode probs  μ̄_j = Σ_i p_ij μ_i
        mu_bar = mu @ Pij  # (n, 2)
        mu_bar = np.maximum(mu_bar, 1e-16)
        # mixing probabilities μ_{i|j} = p_ij μ_i / μ̄_j
        # mix_w[n, i, j]
        mix_w = (mu[:, :, None] * Pij[None, :, :]) / mu_bar[:, None, :]
        # mixed initials
        xa0 = np.sum(mix_w * self.imm_xa[:, :, None], axis=1)  # (n, 2) wait
        # mix_w is (n, i, j); imm_xa is (n, i)
        xa0 = np.einsum("nij,ni->nj", mix_w, self.imm_xa)
        xp0 = np.einsum("nij,ni->nj", mix_w, self.imm_xp)
        dxa = self.imm_xa[:, :, None] - xa0[:, None, :]
        dxp = self.imm_xp[:, :, None] - xp0[:, None, :]
        Pa0 = np.einsum("nij,nij->nj", mix_w, self.imm_Pa[:, :, None] + dxa * dxa)
        Pp0 = np.einsum("nij,nij->nj", mix_w, self.imm_Pp[:, :, None] + dxp * dxp)

        ga_all = np.zeros((n, 2))
        gp_all = np.zeros((n, 2))
        lik = np.ones((n, 2))
        for j in range(2):
            xa, Pa, ga, nua, _, Sa = kalman_channel_step(
                xa0[:, j], Pa0[:, j], z.real, a, self.imm_q_a[j], self.r_amp0
            )
            xp, Pp, gp, nup, _, Sp = kalman_channel_step(
                xp0[:, j], Pp0[:, j], z.imag, a, self.imm_q_p[j], self.r_phase0
            )
            self.imm_xa[:, j] = xa
            self.imm_xp[:, j] = xp
            self.imm_Pa[:, j] = Pa
            self.imm_Pp[:, j] = Pp
            ga_all[:, j] = ga
            gp_all[:, j] = gp
            # log-likelihood, summed over the two channels
            Sa = np.maximum(Sa, 1e-18)
            Sp = np.maximum(Sp, 1e-18)
            lik[:, j] = (
                -0.5 * (nua * nua / Sa + np.log(2.0 * np.pi * Sa))
                + -0.5 * (nup * nup / Sp + np.log(2.0 * np.pi * Sp))
            )

        # log μ ∝ log μ̄ + log L, with a per-tile log-sum-exp
        log_mu = np.log(np.maximum(mu_bar, 1e-16)) + lik  # lik already log
        log_mu = log_mu - np.max(log_mu, axis=1, keepdims=True)
        mu_new = np.exp(log_mu)
        mu_new = mu_new / np.maximum(mu_new.sum(axis=1, keepdims=True), 1e-18)
        bad = ~np.all(np.isfinite(mu_new), axis=1)
        if np.any(bad):
            mu_new[bad] = mu[bad]
            mu_new[bad] = mu_new[bad] / np.maximum(mu_new[bad].sum(axis=1, keepdims=True), 1e-18)
        self.mu = mu_new
        self.xa = np.sum(mu_new * self.imm_xa, axis=1)
        self.xp = np.sum(mu_new * self.imm_xp, axis=1)
        self.Pa = np.sum(mu_new * (self.imm_Pa + (self.imm_xa - self.xa[:, None]) ** 2), axis=1)
        self.Pp = np.sum(mu_new * (self.imm_Pp + (self.imm_xp - self.xp[:, None]) ** 2), axis=1)
        self.kappa = self.xa + 1j * self.xp
        g_tile = 0.5 * (np.sum(mu_new * ga_all, axis=1) + np.sum(mu_new * gp_all, axis=1))
        self.last_g = float(np.mean(g_tile))
        qhat_a = mu_new @ self.imm_q_a
        qhat_p = mu_new @ self.imm_q_p
        self.last_qhat = float(0.5 * (np.median(qhat_a) + np.median(qhat_p)))
        # mode-1 (10q) mass as a diagnostic
        self.last_nis = float(np.mean(mu_new[:, 1]))  # reuse slot: P(high-q mode)
        self.last_xi_nu = float(1.0 - np.mean(mu_new[:, 1]))
        return self.kappa

def method_specs(cfg: Any, r_amp_assumed: float, r_phase_assumed: float, forget: float | None = None) -> list[dict]:
    """Tracking methods compared on one walk: oracle, Mehra, corrected-Ξ, IMM, Sage–Husa, and the v1 baseline."""
    q = cfg.process_q()
    lam = cfg.default_forget if forget is None else float(forget)
    specs = []
    specs.append(dict(name="oracle Kalman", kind="kalman_oracle", q_amp0=q["q_amp"], q_phase0=q["q_phase"], r_amp=r_amp_assumed, r_phase=r_phase_assumed, forget=lam))
    specs.append(dict(name="Kalman initial q", kind="kalman_fixedq", q_amp0=q["q_amp"], q_phase0=q["q_phase"], r_amp=r_amp_assumed, r_phase=r_phase_assumed, forget=lam))
    for g in G_FIXED:
        specs.append(dict(name=f"fixed g={g:.2f}", kind="iir_fixed", g=g, q_amp0=q["q_amp"], q_phase0=q["q_phase"], r_amp=r_amp_assumed, r_phase=r_phase_assumed, forget=lam))
    specs.append(dict(name="v1 g(Ξ)", kind="iir_v1_xi", q_amp0=q["q_amp"], q_phase0=q["q_phase"], r_amp=r_amp_assumed, r_phase=r_phase_assumed, forget=lam))
    specs.append(dict(name="Mehra AKF", kind="mehra", q_amp0=q["q_amp"], q_phase0=q["q_phase"], r_amp=r_amp_assumed, r_phase=r_phase_assumed, forget=lam))
    specs.append(dict(name="Sage–Husa AKF", kind="sage_husa", q_amp0=q["q_amp"], q_phase0=q["q_phase"], r_amp=r_amp_assumed, r_phase=r_phase_assumed, forget=lam))
    specs.append(dict(name="corrected g(Ξ_ν)", kind="xi_corrected", q_amp0=q["q_amp"], q_phase0=q["q_phase"], r_amp=r_amp_assumed, r_phase=r_phase_assumed, forget=lam))
    specs.append(dict(name="IMM (q, 10q)", kind="imm", q_amp0=q["q_amp"], q_phase0=q["q_phase"], r_amp=r_amp_assumed, r_phase=r_phase_assumed, forget=lam,
                      q_high_amp=10.0 * q["q_amp"], q_high_phase=10.0 * q["q_phase"]))
    return specs

def q_multiplier(kind: str, t: int, t_event: int, t_steps: int) -> float:
    """Multiplier of the nominal process variance q0 at time t."""
    if kind in ("stationary", "fault", "mismatch"):
        return 1.0
    if kind == "jump_up":
        return 10.0 if t >= t_event else 1.0
    if kind == "jump_down":
        return 0.1 if t >= t_event else 1.0
    if kind == "ramp":
        if t < t_event:
            return 1.0
        denom = max(t_steps - 1 - t_event, 1)
        progress = (t - t_event) / denom
        return 1.0 + 9.0 * progress
    raise ValueError(f"unknown scenario kind {kind!r}")

_kalman_step_arr = kalman_channel_step

def _init_filter_state(kind, w, n, q_a, q_p, r_a, r_p, a, z, forget, q_hi_a, q_hi_p):
    st = dict(
        kind=kind,
        xa=z.real.copy(),
        xp=z.imag.copy(),
        Pa=np.full((w, n), r_a),
        Pp=np.full((w, n), r_p),
        qa=np.full((w, n), q_a),
        qp=np.full((w, n), q_p),
        ra=np.full((w, n), r_a),
        rp=np.full((w, n), r_p),
        Ca=np.full((w, n), a * a * r_a + q_a + r_a),
        Cp=np.full((w, n), a * a * r_p + q_p + r_p),
        kappa=z.copy(),
        last_g=np.full(w, 0.5),
        last_qhat=np.full(w, 0.5 * (q_a + q_p)),
    )
    gna = steady_state_kalman_gain(a, q_a, r_a)["g"]
    gnp = steady_state_kalman_gain(a, q_p, r_p)["g"]
    gha = steady_state_kalman_gain(a, q_hi_a, r_a)["g"]
    ghp = steady_state_kalman_gain(a, q_hi_p, r_p)["g"]
    st.update(g_nom_a=gna, g_nom_p=gnp, g_hi_a=gha, g_hi_p=ghp, forget=forget, a=a,
              q_amp0=q_a, q_phase0=q_p, r_amp0=r_a, r_phase0=r_p,
              qmin_a=max(Q_MIN_FACTOR * q_a, 1e-16), qmin_p=max(Q_MIN_FACTOR * q_p, 1e-16),
              q_hi_a=q_hi_a, q_hi_p=q_hi_p)
    if kind == "imm":
        st["mu"] = np.full((w, n, 2), 0.5)
        st["imm_xa"] = np.repeat(z.real[:, :, None], 2, axis=2)
        st["imm_xp"] = np.repeat(z.imag[:, :, None], 2, axis=2)
        st["imm_Pa"] = np.full((w, n, 2), r_a)
        st["imm_Pp"] = np.full((w, n, 2), r_p)
        st["Pij"] = np.array([[STAY_PROB, 1.0 - STAY_PROB], [1.0 - STAY_PROB, STAY_PROB]])
        st["imm_q_a"] = np.array([q_a, q_hi_a])
        st["imm_q_p"] = np.array([q_p, q_hi_p])
    return st

def filter_step(st, z, t, rmat=None, A=None, q_amp_true=None, q_phase_true=None,
                r_amp_true=None, r_phase_true=None):
    """One step for a batch state. z is (W,N) complex, already gauge-aligned to held."""
    kind = st["kind"]
    a = st["a"]
    if t == 0:
        # already initialized
        st["kappa"] = z.copy()
        st["xa"], st["xp"] = z.real.copy(), z.imag.copy()
        return st["kappa"]

    if kind == "iir_v1_xi":
        proxy = xi_from_consistency_batch(rmat, A, np.exp(st["kappa"]))
        g = G_MIN_V1 + (G_MAX_V1 - G_MIN_V1) * np.clip(proxy, 0.0, 1.0)
        st["kappa"] = (1.0 - g)[:, None] * st["kappa"] + g[:, None] * z
        st["last_g"] = g
        return st["kappa"]

    if kind == "kalman_oracle":
        qa, qp = float(q_amp_true), float(q_phase_true)
        ra = float(r_amp_true if r_amp_true is not None else st["r_amp0"])
        rp = float(r_phase_true if r_phase_true is not None else st["r_phase0"])
        st["xa"], st["Pa"], ga, *_ = _kalman_step_arr(st["xa"], st["Pa"], z.real, a, qa, ra)
        st["xp"], st["Pp"], gp, *_ = _kalman_step_arr(st["xp"], st["Pp"], z.imag, a, qp, rp)
        st["kappa"] = st["xa"] + 1j * st["xp"]
        st["last_g"] = 0.5 * (ga.mean(axis=1) + gp.mean(axis=1))
        return st["kappa"]

    if kind == "kalman_fixedq":
        st["xa"], st["Pa"], ga, *_ = _kalman_step_arr(st["xa"], st["Pa"], z.real, a, st["q_amp0"], st["r_amp0"])
        st["xp"], st["Pp"], gp, *_ = _kalman_step_arr(st["xp"], st["Pp"], z.imag, a, st["q_phase0"], st["r_phase0"])
        st["kappa"] = st["xa"] + 1j * st["xp"]
        st["last_g"] = 0.5 * (ga.mean(axis=1) + gp.mean(axis=1))
        return st["kappa"]

    if kind == "mehra":
        lam = st["forget"]
        xa, Pa, ga, nua, _Ppa, Sa = _kalman_step_arr(st["xa"], st["Pa"], z.real, a, st["qa"], st["ra"])
        xp, Pp, gp, nup, _Ppp, Sp = _kalman_step_arr(st["xp"], st["Pp"], z.imag, a, st["qp"], st["rp"])
        st["Ca"] = lam * st["Ca"] + (1.0 - lam) * (nua * nua)
        st["Cp"] = lam * st["Cp"] + (1.0 - lam) * (nup * nup)
        if t >= ADAPT_DELAY:
            st["qa"] = np.maximum(st["qmin_a"], st["Ca"] - a * a * st["Pa"] - st["ra"])
            st["qp"] = np.maximum(st["qmin_p"], st["Cp"] - a * a * st["Pp"] - st["rp"])
        st["xa"], st["Pa"], st["xp"], st["Pp"] = xa, Pa, xp, Pp
        st["kappa"] = st["xa"] + 1j * st["xp"]
        st["last_g"] = 0.5 * (ga.mean(axis=1) + gp.mean(axis=1))
        st["last_qhat"] = 0.5 * (np.median(st["qa"], axis=1) + np.median(st["qp"], axis=1))
        return st["kappa"]

    if kind == "xi_corrected":
        lam = st["forget"]
        xpreda, xpredp = a * st["xa"], a * st["xp"]
        Ppreda = a * a * st["Pa"] + st["q_amp0"]
        Ppredp = a * a * st["Pp"] + st["q_phase0"]
        Sa, Sp = Ppreda + st["r_amp0"], Ppredp + st["r_phase0"]
        nua, nup = z.real - xpreda, z.imag - xpredp
        st["Ca"] = lam * st["Ca"] + (1.0 - lam) * (nua * nua)
        st["Cp"] = lam * st["Cp"] + (1.0 - lam) * (nup * nup)
        xi_a = np.minimum(1.0, Sa / np.maximum(st["Ca"], 1e-18))
        xi_p = np.minimum(1.0, Sp / np.maximum(st["Cp"], 1e-18))
        if t >= ADAPT_DELAY:
            ga = st["g_nom_a"] + (st["g_hi_a"] - st["g_nom_a"]) * (1.0 - xi_a)
            gp = st["g_nom_p"] + (st["g_hi_p"] - st["g_nom_p"]) * (1.0 - xi_p)
        else:
            ga = np.full_like(xi_a, st["g_nom_a"])
            gp = np.full_like(xi_p, st["g_nom_p"])
        st["xa"] = xpreda + ga * nua
        st["xp"] = xpredp + gp * nup
        st["Pa"] = (1.0 - ga) ** 2 * Ppreda + ga ** 2 * st["r_amp0"]
        st["Pp"] = (1.0 - gp) ** 2 * Ppredp + gp ** 2 * st["r_phase0"]
        st["kappa"] = st["xa"] + 1j * st["xp"]
        st["last_g"] = 0.5 * (ga.mean(axis=1) + gp.mean(axis=1))
        return st["kappa"]

    if kind == "imm":
        return _imm_step(st, z)

    raise ValueError(kind)

def _imm_step(st, z):
    a = st["a"]
    w, n, _ = st["mu"].shape
    mu = st["mu"]
    Pij = st["Pij"]
    mu_bar = mu @ Pij
    mu_bar = np.maximum(mu_bar, 1e-16)
    mix_w = (mu[:, :, :, None] * Pij[None, None, :, :]) / mu_bar[:, :, None, :]
    # mix_w: (W,N,i,j)
    xa0 = np.einsum("wnij,wni->wnj", mix_w, st["imm_xa"])
    xp0 = np.einsum("wnij,wni->wnj", mix_w, st["imm_xp"])
    dxa = st["imm_xa"][:, :, :, None] - xa0[:, :, None, :]
    dxp = st["imm_xp"][:, :, :, None] - xp0[:, :, None, :]
    Pa0 = np.einsum("wnij,wnij->wnj", mix_w, st["imm_Pa"][:, :, :, None] + dxa * dxa)
    Pp0 = np.einsum("wnij,wnij->wnj", mix_w, st["imm_Pp"][:, :, :, None] + dxp * dxp)
    logL = np.zeros((w, n, 2))
    ga_all = np.zeros((w, n, 2))
    gp_all = np.zeros((w, n, 2))
    for j in range(2):
        xa, Pa, ga, nua, _, Sa = _kalman_step_arr(xa0[:, :, j], Pa0[:, :, j], z.real, a, st["imm_q_a"][j], st["r_amp0"])
        xp, Pp, gp, nup, _, Sp = _kalman_step_arr(xp0[:, :, j], Pp0[:, :, j], z.imag, a, st["imm_q_p"][j], st["r_phase0"])
        st["imm_xa"][:, :, j] = xa
        st["imm_xp"][:, :, j] = xp
        st["imm_Pa"][:, :, j] = Pa
        st["imm_Pp"][:, :, j] = Pp
        ga_all[:, :, j] = ga
        gp_all[:, :, j] = gp
        Sa = np.maximum(Sa, 1e-18)
        Sp = np.maximum(Sp, 1e-18)
        logL[:, :, j] = -0.5 * (nua * nua / Sa + np.log(2.0 * np.pi * Sa)) - 0.5 * (
            nup * nup / Sp + np.log(2.0 * np.pi * Sp)
        )
    log_mu = np.log(mu_bar) + logL
    log_mu = log_mu - np.max(log_mu, axis=2, keepdims=True)
    mu_new = np.exp(log_mu)
    mu_new = mu_new / np.maximum(mu_new.sum(axis=2, keepdims=True), 1e-18)
    bad = ~np.all(np.isfinite(mu_new), axis=2)
    if np.any(bad):
        mu_new[bad] = mu[bad]
        mu_new[bad] = mu_new[bad] / np.maximum(mu_new[bad].sum(axis=1, keepdims=True), 1e-18)
    st["mu"] = mu_new
    st["xa"] = np.sum(mu_new * st["imm_xa"], axis=2)
    st["xp"] = np.sum(mu_new * st["imm_xp"], axis=2)
    st["kappa"] = st["xa"] + 1j * st["xp"]
    g_tile = 0.5 * (np.sum(mu_new * ga_all, axis=2) + np.sum(mu_new * gp_all, axis=2))
    st["last_g"] = g_tile.mean(axis=1)
    return st["kappa"]

def make_method_states(w, n, cfg, r_amp, r_phase, z0):
    """make method states."""
    q = cfg.process_q()
    a = cfg.rho_ou
    lam = cfg.default_forget
    kinds = {
        "oracle Kalman": "kalman_oracle",
        "Kalman initial q": "kalman_fixedq",
        "v1 g(Ξ)": "iir_v1_xi",
        "Mehra AKF": "mehra",
        "corrected g(Ξ_ν)": "xi_corrected",
        "IMM (q, 10q)": "imm",
    }
    states = {}
    for name, kind in kinds.items():
        states[name] = _init_filter_state(
            kind, w, n, q["q_amp"], q["q_phase"], r_amp, r_phase, a, z0, lam,
            10.0 * q["q_amp"], 10.0 * q["q_phase"],
        )
    return states
