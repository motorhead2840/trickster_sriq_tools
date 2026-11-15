"""Selection rate and a per-tile memory register.

A tile with d levels is driven toward one declared target level by a Lindblad
jump, while a noise channel (depolarization, amplitude damping, or both) pulls
it away. In steady state the target population is

    F_ss = (r + 1/d) / (r + 1),    r = γ_sel / Γ.

The smallest r that holds F_ss at a declared floor Ξ_crit is

    r* = (Ξ_crit − 1/d) / (1 − Ξ_crit).

The memory register is a truncated Fock mode with a Lorentzian spectrum.
The tile-plus-register pair is Markovian; the tile alone need not be. The
BLP trace-distance slope measures that memory. The comparison against a
memoryless amplitude-damping channel uses no extra dephasing, because
dephasing can hide the backflow.

The population step used by the adaptive selector is the exact scalar map of
that master equation. It is not an RK4 step and it does not rebuild the
superoperator when the selection rate changes.

``xp`` is NumPy and is used by the closed forms. The density-matrix integrator
stays on NumPy. There is no torch backend in this build.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm as scipy_expm

xp = np

ComplexArr = NDArray[np.complex128]
FloatArr = NDArray[np.float64]

def ket(d: int, k: int) -> ComplexArr:
    """Computational-basis column vector."""
    v = np.zeros(d, dtype=np.complex128)
    v[k] = 1.0
    return v

def proj(d: int, k: int) -> ComplexArr:
    """Projector onto one computational-basis level."""
    p = np.zeros((d, d), dtype=np.complex128)
    p[k, k] = 1.0
    return p

def ketbra(d: int, i: int, j: int) -> ComplexArr:
    """Outer product |i><j|."""
    m = np.zeros((d, d), dtype=np.complex128)
    m[i, j] = 1.0
    return m

def identity(d: int) -> ComplexArr:
    """Identity matrix of dimension d."""
    return np.eye(d, dtype=np.complex128)

def lowering_qudit(d: int) -> ComplexArr:
    """Harmonic-like lowering: |k> -> sqrt(k+1)|k> wait, a|n> = sqrt(n)|n-1>."""
    a = np.zeros((d, d), dtype=np.complex128)
    for n in range(1, d):
        a[n - 1, n] = np.sqrt(n)
    return a

def destroy_fock(n_max: int) -> ComplexArr:
    """Truncated bosonic lowering operator on levels 0..n_max."""
    return lowering_qudit(n_max + 1)

def thermal_state(d: int) -> ComplexArr:
    """Maximally mixed state on d levels."""
    return identity(d) / d

def pure_state(d: int, k: int) -> ComplexArr:
    """Pure projector onto one level."""
    return proj(d, k)

def plus_state(d: int) -> ComplexArr:
    """Equal superposition of every level."""
    v = np.ones(d, dtype=np.complex128) / np.sqrt(d)
    return np.outer(v, v.conj())

def hermitize(rho: ComplexArr) -> ComplexArr:
    """Make a density matrix Hermitian with trace 1."""
    h = 0.5 * (rho + np.swapaxes(rho.conj(), -1, -2))
    # restore trace on the last two axes
    tr = np.trace(h, axis1=-2, axis2=-1).real
    # avoid division issues
    tr = np.where(np.abs(tr) < 1e-18, 1.0, tr)
    return h / tr[..., None, None]

def fidelity_target(rho: ComplexArr, d: int, t: int) -> FloatArr:
    """F = <t|ρ|t>, the target population. Works for (D,D) or (B,D,D)."""
    if rho.ndim == 2:
        return np.array(float(rho[t, t].real), dtype=np.float64)
    return np.real(rho[:, t, t]).astype(np.float64)

def purity(rho: ComplexArr) -> FloatArr:
    """Purity tr(ρ^2)."""
    if rho.ndim == 2:
        return np.array(float(np.real(np.trace(rho @ rho))), dtype=np.float64)
    # batched: sum_ij rho_ij * rho_ji
    val = np.einsum("...ij,...ji->...", rho, rho)
    return np.real(val).astype(np.float64)

def contrast(rho: ComplexArr, d: int, t: int) -> FloatArr:
    """Target population minus the largest other population ('focus sharpness')."""
    if rho.ndim == 2:
        pops = np.real(np.diag(rho))
        others = [pops[k] for k in range(d) if k != t]
        return np.array(float(pops[t] - max(others)), dtype=np.float64)
    pops = np.real(np.diagonal(rho, axis1=-2, axis2=-1))
    mask = np.ones(d, dtype=bool)
    mask[t] = False
    best_other = pops[:, mask].max(axis=1)
    return (pops[:, t] - best_other).astype(np.float64)

def trace_distance(rho: ComplexArr, sigma: ComplexArr) -> FloatArr:
    """(1/2) ||ρ-σ||_1 via Hermitian eigendecomposition of ρ-σ."""
    # Do not trace-normalize the difference. Use a raw Hermitian projection.
    def _td(a, b):
        dlt = 0.5 * ((a - b) + np.swapaxes((a - b).conj(), -1, -2))
        ev = np.linalg.eigvalsh(dlt)
        return 0.5 * np.sum(np.abs(ev), axis=-1)

    if rho.ndim == 2:
        return np.array(float(_td(rho, sigma)), dtype=np.float64)
    return _td(rho, sigma).astype(np.float64)

def kron(*ops) -> ComplexArr:
    """Kronecker product of several operators."""
    out = ops[0]
    for a in ops[1:]:
        out = np.kron(out, a)
    return out

def embed_left(op: ComplexArr, d_right: int) -> ComplexArr:
    """Embed a tile operator as op ⊗ I on the memory register."""
    return np.kron(op, np.eye(d_right, dtype=np.complex128))

def embed_right(d_left: int, op: ComplexArr) -> ComplexArr:
    """Embed a register operator as I ⊗ op on the tile."""
    return np.kron(np.eye(d_left, dtype=np.complex128), op)

def partial_trace_right(rho: ComplexArr, d_left: int, d_right: int) -> ComplexArr:
    """Trace out the last factor of dim d_right. rho is (D,D) or (B,D,D), D=d_left*d_right."""
    if rho.ndim == 2:
        r = rho.reshape(d_left, d_right, d_left, d_right)
        return np.trace(r, axis1=1, axis2=3)
    b = rho.shape[0]
    r = rho.reshape(b, d_left, d_right, d_left, d_right)
    return np.trace(r, axis1=2, axis2=4)

def jumps_selection(d: int, t: int, gamma_sel: float) -> list[ComplexArr]:
    """Jump operators that pull every level toward one declared target level."""
    if gamma_sel <= 0:
        return []
    s = np.sqrt(gamma_sel)
    return [s * ketbra(d, t, j) for j in range(d) if j != t]

def jumps_depolarizing_weyl(d: int, gamma: float) -> list[ComplexArr]:
    """Lindblad operators that generate L(ρ) = Γ (I/d - ρ).

    All non-identity Weyl operators W_{pq}=X^p Z^q at rate Γ/d², i.e.
    L_{pq} = sqrt(Γ)/d W_{pq}.  Then Σ_{pq} (W ρ W† - ρ) / d² gives
    (I/d)Tr(ρ) - ρ, and the prefactor Γ is exact (the identity Weyl
    operator contributes nothing and is omitted).
    """
    if gamma <= 0:
        return []
    X = np.zeros((d, d), dtype=np.complex128)
    for k in range(d):
        X[(k + 1) % d, k] = 1.0
    omega = np.exp(2j * np.pi / d)
    Z = np.diag(omega ** np.arange(d)).astype(np.complex128)
    s = np.sqrt(gamma / (d * d))
    ops = []
    Xp = identity(d)
    for p in range(d):
        Zq = identity(d)
        for q in range(d):
            if not (p == 0 and q == 0):
                ops.append(s * (Xp @ Zq))
            Zq = Zq @ Z
        Xp = Xp @ X
    return ops

def jumps_amplitude_damping(d: int, gamma_ad: float, toward: int = 0) -> list[ComplexArr]:
    """All levels decay into |toward> (a reset-like T1 model on a qudit)."""
    if gamma_ad <= 0:
        return []
    s = np.sqrt(gamma_ad)
    return [s * ketbra(d, toward, k) for k in range(d) if k != toward]

def jumps_ladder_damping(d: int, gamma_ad: float) -> list[ComplexArr]:
    """Harmonic T1: L = sqrt(Γ) a, with a the lowering operator."""
    if gamma_ad <= 0:
        return []
    return [np.sqrt(gamma_ad) * lowering_qudit(d)]

def jumps_dephasing(d: int, gamma_phi: float) -> list[ComplexArr]:
    """Local dephasing in the computational basis, L_k = sqrt(2 Γ_φ) |k><k|.

    For a qubit the single operator sqrt(Γ_φ/2) σ_z is equivalent (up to an
    identity shift) to this pair; populations are frozen, coherences decay
    at Γ_φ in the usual T2 sense when combined with 1/(2 T1).
    """
    if gamma_phi <= 0:
        return []
    s = np.sqrt(2.0 * gamma_phi)
    return [s * proj(d, k) for k in range(d)]

def jumps_t1t2(d: int, gamma_ad: float, gamma_phi: float, toward: int = 0) -> list[ComplexArr]:
    """Amplitude-damping plus dephasing jump operators."""
    return jumps_amplitude_damping(d, gamma_ad, toward=toward) + jumps_dephasing(d, gamma_phi)

def lindblad_rhs(rho: ComplexArr, H: ComplexArr, jumps: list[ComplexArr],
                 depolarizing: float = 0.0) -> ComplexArr:
    """Works for (D,D) or (B,D,D). H is (D,D) or broadcastable; each jump is (D,D).

    `depolarizing` applies the closed-form generator Γ(I/d − ρ) so we do not
    have to sum d²−1 Weyl operators at every step.
    """
    comm = H @ rho - rho @ H
    out = -1j * comm
    for L in jumps:
        Ld = L.conj().T
        Lrho = L @ rho
        out = out + Lrho @ Ld - 0.5 * (Ld @ L @ rho + rho @ (Ld @ L))
    if depolarizing:
        d = rho.shape[-1]
        out = out + depolarizing * (np.eye(d, dtype=rho.dtype) / d - rho)
    return out

def rk4_step(rho: ComplexArr, H: ComplexArr, jumps: list[ComplexArr], dt: float,
              depolarizing: float = 0.0) -> ComplexArr:
    """One RK4 step of a Lindblad equation, followed by restoring Hermiticity and trace 1."""
    k1 = lindblad_rhs(rho, H, jumps, depolarizing)
    k2 = lindblad_rhs(rho + 0.5 * dt * k1, H, jumps, depolarizing)
    k3 = lindblad_rhs(rho + 0.5 * dt * k2, H, jumps, depolarizing)
    k4 = lindblad_rhs(rho + dt * k3, H, jumps, depolarizing)
    nxt = rho + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return hermitize(nxt)

def liouvillian_matrix(H: ComplexArr, jumps: list[ComplexArr], depolarizing: float = 0.0) -> ComplexArr:
    """Dense Liouvillian superoperator, column-major on vec(ρ)."""
    d = H.shape[0]
    I = np.eye(d, dtype=np.complex128)
    superop = -1j * (np.kron(I, H) - np.kron(H.T, I))
    for L in jumps:
        LdL = L.conj().T @ L
        superop = superop + np.kron(L.conj(), L) - 0.5 * (np.kron(I, LdL) + np.kron(LdL.T, I))
    if depolarizing:
        # vec( I/d - ρ ) = (1/d) vec(I) - vec(ρ)
        D2 = d * d
        # L_dep vec(ρ) = Γ (vec(I)/d * Tr(ρ) − vec(ρ)); Tr(ρ) = vec(I)·vec(ρ) in F-order
        vecI = I.reshape(D2, order="F")
        superop = superop + depolarizing * (np.outer(vecI / d, vecI) - np.eye(D2, dtype=np.complex128))
    return superop

def expm_propagate(rho: ComplexArr, H: ComplexArr, jumps: list[ComplexArr], t: float,
                    depolarizing: float = 0.0) -> ComplexArr:
    """Exact propagator e^{t L} applied to one or a batch of DMs."""
    L = liouvillian_matrix(H, jumps, depolarizing=depolarizing)
    U = scipy_expm(t * L)
    d = H.shape[0]
    if rho.ndim == 2:
        v = rho.reshape(d * d, order="F")
        w = U @ v
        out = w.reshape((d, d), order="F")
        return hermitize(out)
    b = rho.shape[0]
    v = rho.reshape(b, d * d, order="F").T  # (D², B)
    w = U @ v
    out = w.T.reshape((b, d, d), order="F")
    return hermitize(out)

def evolve_rk4(
    rho0: ComplexArr,
    H: ComplexArr,
    jumps: list[ComplexArr],
    times: FloatArr,
    depolarizing: float = 0.0,
) -> ComplexArr:
    """Return ρ(t) for each time in `times`, shape (T, D, D) or (T, B, D, D)."""
    times = np.asarray(times, dtype=np.float64)
    rho = hermitize(rho0.copy())
    out = [rho.copy()]
    for i in range(1, len(times)):
        dt = float(times[i] - times[i - 1])
        rho = rk4_step(rho, H, jumps, dt, depolarizing=depolarizing)
        out.append(rho)
    return np.stack(out, axis=0)

def evolve_expm_piecewise(
    rho0: ComplexArr,
    H: ComplexArr,
    jumps: list[ComplexArr],
    times: FloatArr,
    depolarizing: float = 0.0,
) -> ComplexArr:
    """Exact if H, jumps are constant on the grid (one expm of L Δt, then powers)."""
    times = np.asarray(times, dtype=np.float64)
    dts = np.diff(times)
    rho = hermitize(rho0.copy())
    out = [rho.copy()]
    if len(dts) == 0:
        return np.stack(out, axis=0)
    # If the grid is uniform, compute U once.
    if np.allclose(dts, dts[0]):
        U = scipy_expm(float(dts[0]) * liouvillian_matrix(H, jumps, depolarizing=depolarizing))
        d = H.shape[0]
        if rho.ndim == 2:
            v = rho.reshape(d * d, order="F")
            for _ in dts:
                v = U @ v
                r = hermitize(v.reshape((d, d), order="F"))
                out.append(r)
                v = r.reshape(d * d, order="F")
        else:
            b = rho.shape[0]
            v = rho.reshape(b, d * d, order="F").T
            for _ in dts:
                v = U @ v
                r = hermitize(v.T.reshape((b, d, d), order="F"))
                out.append(r)
                v = r.reshape(b, d * d, order="F").T
        return np.stack(out, axis=0)
    # non-uniform: step by step
    for dt in dts:
        rho = expm_propagate(rho, H, jumps, float(dt), depolarizing=depolarizing)
        out.append(rho)
    return np.stack(out, axis=0)

def evolve(
    rho0: ComplexArr,
    H: ComplexArr,
    jumps: list[ComplexArr],
    times: FloatArr,
    method: str = "auto",
    depolarizing: float = 0.0,
) -> ComplexArr:
    """method: 'rk4', 'expm', or 'auto' (expm when D**2 <= 256, else RK4)."""
    d = H.shape[0]
    if method == "auto":
        method = "expm" if d * d <= 256 else "rk4"
    if method == "expm":
        return evolve_expm_piecewise(rho0, H, jumps, times, depolarizing=depolarizing)
    if method == "rk4":
        return evolve_rk4(rho0, H, jumps, times, depolarizing=depolarizing)
    raise ValueError(method)

def pseudomode_g(gamma_eff: float, kappa: float) -> float:
    """Coupling that realises a resonant Lorentzian with Markov rate Γ = 4 g² / κ."""
    if kappa <= 0:
        raise ValueError("kappa must be positive")
    return float(np.sqrt(max(gamma_eff, 0.0) * kappa / 4.0))

def lorentzian_J(omega, g, kappa, omega_c=0.0):
    """Lorentzian spectral density of a damped memory register."""
    return (g ** 2 / np.pi) * (kappa / 2.0) / ((omega - omega_c) ** 2 + (kappa / 2.0) ** 2)

def build_pseudomode_ad(
    d: int,
    n_max: int,
    gamma_eff: float,
    kappa: float,
    toward: int = 0,
    omega_s: float = 0.0,
    omega_c: float = 0.0,
    dephasing: float = 0.0,
) -> tuple[ComplexArr, list[ComplexArr], int, int]:
    """Tile ⊗ memory-register. The system operator is the reset lowering
    A = Σ_{k≠toward} |toward><k|, so the Markov limit is amplitude damping
    into |toward> at rate Γ_eff = 4 g² / κ (resonant).
    """
    g = pseudomode_g(gamma_eff, kappa)
    A = np.zeros((d, d), dtype=np.complex128)
    for k in range(d):
        if k != toward:
            A[toward, k] = 1.0
    a = destroy_fock(n_max)
    H_s = omega_s * np.diag(np.arange(d)).astype(np.complex128)
    H_m = omega_c * (a.conj().T @ a)
    H = embed_left(H_s, n_max + 1) + embed_right(d, H_m)
    H = H + g * (embed_left(A, n_max + 1) @ embed_right(d, a.conj().T) + embed_left(A.conj().T, n_max + 1) @ embed_right(d, a))
    jumps = [np.sqrt(kappa) * embed_right(d, a)]
    if dephasing > 0:
        for L in jumps_dephasing(d, dephasing):
            jumps.append(embed_left(L, n_max + 1))
    return H, jumps, d, n_max + 1

def initial_product(rho_s: ComplexArr, n_max: int, n_bath: int = 0) -> ComplexArr:
    """Product state of a tile density matrix and a Fock state of the memory register."""
    vac = proj(n_max + 1, n_bath)
    if rho_s.ndim == 2:
        return np.kron(rho_s, vac)
    b = rho_s.shape[0]
    return np.einsum("bij,kl->bikjl", rho_s, vac).reshape(b, rho_s.shape[1] * (n_max + 1), rho_s.shape[1] * (n_max + 1))

def reduced_tile(rho_joint: ComplexArr, d: int, d_m: int) -> ComplexArr:
    """Tile density matrix after tracing out the memory register."""
    return partial_trace_right(rho_joint, d, d_m)

def f_ss_depolarizing(r: float, d: int) -> float:
    """Steady target fidelity under γ_sel / Γ_dep = r, Markovian.

    F_ss = (r + 1/d) / (r + 1).
    """
    r = float(r)
    return (r + 1.0 / d) / (r + 1.0)  # xp is NumPy; the expression is scalar

def r_threshold(xi_crit: float, d: int) -> float:
    """Smallest r = γ_sel/Γ such that F_ss >= Ξ_crit.

    r* = (Ξ_crit - 1/d) / (1 - Ξ_crit).  Returns +inf if Ξ_crit >= 1.
    """
    if xi_crit >= 1.0:
        return float("inf")
    need = (xi_crit - 1.0 / d) / (1.0 - xi_crit)
    return float(max(need, 0.0))

def f_step_selection_depolarizing(F: float, gamma_sel: float, Gamma: float, dt: float, d: int) -> float:
    """Exact one-step map of the target population under selection + depolarization.

    With H=0 the computational-basis populations close.  The target fidelity
    obeys  dF/dt = γ_sel (1 - F) + Γ (1/d - F), whose solution on a step of
    length dt with constant rates is
        F(t+dt) = F_ss + (F(t) - F_ss) exp(-(γ_sel+Γ) dt),
    F_ss = (γ_sel + Γ/d) / (γ_sel + Γ).
    This is exactly the Liouvillian propagator on the diagonal family (thermal
    start, no coherences generated), and is used in Experiment 3 so a changing
    γ_sel does not force a fresh expm of the D²×D² generator at every step.
    """
    g = float(max(gamma_sel, 0.0))
    G = float(max(Gamma, 0.0))
    b = g + G
    if b <= 0.0 or dt == 0.0:
        return float(F)
    Fss = (g + G / float(d)) / b
    return float(Fss + (float(F) - Fss) * np.exp(-b * float(dt)))

def f_ss_from_generator(d: int, t: int, gamma_sel: float, gamma_veil: float, kind: str = "depolarizing") -> float:
    """Numerical steady state via the Liouvillian nullspace (tiny systems)."""
    H = np.zeros((d, d), dtype=np.complex128)
    jumps = jumps_selection(d, t, gamma_sel)
    dep = 0.0
    if kind == "depolarizing":
        dep = gamma_veil
    elif kind == "ad":
        jumps = jumps + jumps_amplitude_damping(d, gamma_veil, toward=0)
    elif kind == "t1t2":
        jumps = jumps + jumps_t1t2(d, gamma_veil, 0.5 * gamma_veil, toward=0)
    else:
        raise ValueError(kind)
    L = liouvillian_matrix(H, jumps, depolarizing=dep)
    # nullspace: eigenvector of L with eigenvalue closest to 0
    ev, evec = np.linalg.eig(L)
    k = int(np.argmin(np.abs(ev)))
    v = evec[:, k]
    rho = v.reshape((d, d), order="F")
    rho = hermitize(rho)
    return float(np.real(rho[t, t]))

def blp_backflow(d_ts: FloatArr, times: FloatArr) -> dict:
    """BLP information backflow from a trace-distance time series.

    σ(t) ≈ ΔD/Δt; backflow when σ>0.  N = ∫_{σ>0} σ dt is the net
    recovered distinguishability.  A Markovian CP-divisible process has N=0
    (monotone D) for every pair; a clearly positive N is evidence of memory.
    """
    d_ts = np.asarray(d_ts, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    dd = np.diff(d_ts)
    dt = np.diff(times)
    sigma = dd / np.maximum(dt, 1e-18)
    pos = np.clip(dd, 0.0, None)  # ∫_{σ>0} σ dt = sum of positive increments
    n_blp = float(pos.sum())
    # how many intervals increase
    n_up = int(np.sum(dd > 1e-8))
    return dict(
        N_BLP=n_blp,
        n_increases=n_up,
        d_min=float(d_ts.min()),
        d_max=float(d_ts.max()),
        d_final=float(d_ts[-1]),
        monotone=bool(n_up == 0),
        sigma_max=float(sigma.max()) if sigma.size else 0.0,
    )

def evolve_dispatch(rho0, H, jumps, times, backend_info=None, method="auto", depolarizing=0.0):
    """Integrate a Lindblad equation on NumPy.

    This build has no torch backend. A request whose backend name is ``torch``
    raises. Official runs pass a NumPy backend record and use ``expm`` or RK4.
    """
    if backend_info and backend_info.get("name") == "torch":
        raise RuntimeError(
            "sri.selection has no torch backend. Integrate with evolve(..., method='expm') or method='rk4'."
        )
    return evolve(rho0, H, jumps, times, method=method, depolarizing=depolarizing)
