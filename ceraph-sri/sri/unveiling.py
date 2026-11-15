"""Reference-free veil estimates from neighbour comparisons.

Each tile carries a complex veil τ. Tiles that see related inputs form the
ratio r_nm ≈ τ_n / τ_m. The least-squares log-amplitude and the leading
connection eigenvector recover τ up to one global complex factor. That factor
is not observable from cross-checks alone.

A 0/1 wiring matches the v1–v4 solver, including the sparse phase eigenvector
for more than 192 tiles. Weighted wirings put the weight into the Laplacian
and the connection, which is what the quasiperiodic comparison uses. For a
0/1 matrix the two conventions agree.

NumPy is the array backend. Sparse Laplacian solves use SciPy. There is no
torch path here.
"""
from __future__ import annotations

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import eigsh, spsolve

from sri.graphs import laplacian, laplacian_sparse

ComplexArr = NDArray[np.complex128]
FloatArr = NDArray[np.float64]

def complex_normal(rng: Generator, shape) -> ComplexArr:
    """CN(0, 1): E[|z|²] = 1, independent real/imag parts of variance 1/2."""
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)

def sample_veils(n: int, s_amp: float, s_phase: float, rng: Generator) -> tuple[ComplexArr, ComplexArr]:
    """Draw per-tile veils τ = exp(η + iξ) with independent Gaussian log-amplitude and phase."""
    eta = rng.normal(0.0, s_amp, size=n)
    xi = rng.normal(0.0, s_phase, size=n)
    kappa = eta + 1j * xi
    return np.exp(kappa).astype(np.complex128), kappa.astype(np.complex128)

def noise_std_from_snr(snr_db: float, signal_power: float = 1.0) -> float:
    """σ such that E[|ψ|²] / E[|w|²] = 10^(SNR/10), with E[|ψ|²] = signal_power."""
    return float(np.sqrt(signal_power / 10.0 ** (snr_db / 10.0)))

def generate_observations(
    tau: ComplexArr,
    k: int,
    snr_db: float,
    rng: Generator,
    psi: ComplexArr | None = None,
) -> tuple[ComplexArr, ComplexArr]:
    """Full-overlap model: y_{n,k} = τ_n ψ_k + w_{n,k}. Returns Y (N×K), psi (K,)."""
    n = tau.shape[0]
    if psi is None:
        psi = complex_normal(rng, k)
    sigma = noise_std_from_snr(snr_db, signal_power=1.0)
    W = sigma * complex_normal(rng, (n, k))
    Y = tau[:, None] * psi[None, :] + W
    return Y.astype(np.complex128), psi.astype(np.complex128)

def sliding_windows_ring(n: int, k: int, rho: float) -> tuple[NDArray[np.int64], int]:
    """Each tile observes K consecutive samples of a circular stream.

    Consecutive tiles (and the wrap-around pair) share exactly ρK samples
    when (1-ρ)K is an integer. For ρ=1 everyone sees the same K samples.
    """
    if rho >= 1.0 - 1e-12:
        idx = np.broadcast_to(np.arange(k, dtype=np.int64), (n, k)).copy()
        return idx, k
    step = int(round((1.0 - rho) * k))
    if step < 1:
        idx = np.broadcast_to(np.arange(k, dtype=np.int64), (n, k)).copy()
        return idx, k
    overlap = k - step
    if abs(overlap - rho * k) > 1e-9:
        raise ValueError(f"ring windows require (1-ρ)K integer, got ρ={rho}, K={k}")
    L = n * step
    starts = (np.arange(n, dtype=np.int64) * step)[:, None]
    idx = (starts + np.arange(k, dtype=np.int64)[None, :]) % L
    return idx, L

def sliding_windows_torus(n: int, k: int, rho: float) -> tuple[NDArray[np.int64], int]:
    """2-D sliding windows on a toroidal sample field so 4-neighbors share ρK samples.

    K must be a perfect square (window is w×w). (1-ρ)w must be an integer.
    When N is too small for a genuine torus of windows — side·(1-ρ) ≤ 1, so each
    window would cover the whole field and every pair would share all K samples —
    we fall back to a planar field plus explicit wrap-around identification
    (see `_torus_windows_planar_with_wrap`).
    """
    side = int(round(np.sqrt(n)))
    w = int(round(np.sqrt(k)))
    if side * side != n:
        raise ValueError(f"torus windows require square N, got {n}")
    if w * w != k:
        raise ValueError(f"torus windows require square K, got {k}")
    if rho >= 1.0 - 1e-12:
        idx = np.broadcast_to(np.arange(k, dtype=np.int64), (n, k)).copy()
        return idx, k
    shift = int(round((1.0 - rho) * w))
    if shift < 1:
        idx = np.broadcast_to(np.arange(k, dtype=np.int64), (n, k)).copy()
        return idx, k
    if abs((w - shift) * w - rho * k) > 1e-9:
        raise ValueError(f"torus windows require (1-ρ)√K integer, got ρ={rho}, K={k}")
    if side * shift > w:
        H = side * shift
        W = side * shift
        L = H * W
        idx = np.zeros((n, k), dtype=np.int64)
        di = np.arange(w)
        dj = np.arange(w)
        for r in range(side):
            for c in range(side):
                i = r * side + c
                rr = (r * shift + di[:, None]) % H
                cc = (c * shift + dj[None, :]) % W
                idx[i] = (rr * W + cc).ravel()
        return idx, L
    # Degenerate torus: delegate
    return _torus_windows_planar_with_wrap(side, w, shift, rho)

def _torus_windows_planar_with_wrap(
    side: int, w: int, shift: int, rho: float
) -> tuple[NDArray[np.int64], int]:
    """Planar w×w windows (adjacent 4-neighbors share ρK) plus wrap-edge repair.

    Used only when a circular field of length side·shift would be no larger than
    the window, so a naive torus would collapse to full overlap.
    """
    k = w * w
    target = int(round(rho * k))
    H = (side - 1) * shift + w
    W = (side - 1) * shift + w
    # planar indices
    sets: list[set[int]] = []
    for r in range(side):
        for c in range(side):
            s: set[int] = set()
            for di in range(w):
                for dj in range(w):
                    s.add((r * shift + di) * W + (c * shift + dj))
            sets.append(s)
    # Wrap-around pairs: horizontal (r, 0)-(r, side-1) and vertical (0, c)-(side-1, c)
    wrap_pairs = []
    for r in range(side):
        wrap_pairs.append((r * side + 0, r * side + (side - 1)))
    for c in range(side):
        wrap_pairs.append((0 * side + c, (side - 1) * side + c))

    next_id = H * W

    def surplus_drop(i: int, protected: set[int], partners: list[int], target_ov: int) -> int | None:
        # Drop a sample that is not protected and that keeps every adjacent overlap at the target.
        r, c = divmod(i, side)
        adj = []
        if c > 0:
            adj.append(i - 1)
        if c < side - 1:
            adj.append(i + 1)
        if r > 0:
            adj.append(i - side)
        if r < side - 1:
            adj.append(i + side)
        # drop a sample that does not drop any adjacent overlap below target
        candidates = []
        for s in sets[i]:
            if s in protected:
                continue
            safe = True
            for m in adj:
                if s in sets[m] and len(sets[i] & sets[m]) - 1 < target_ov:
                    safe = False
                    break
            if safe:
                # prefer unused by adj
                used = sum(1 for m in adj if s in sets[m])
                candidates.append((used, s))
        if not candidates:
            return None
        candidates.sort()
        return candidates[0][1]

    for i, j in wrap_pairs:
        need = target - len(sets[i] & sets[j])
        for _ in range(max(0, need)):
            prot_i = set(sets[i] & sets[j])
            prot_j = set(prot_i)
            drop_i = surplus_drop(i, prot_i, [], target)
            drop_j = surplus_drop(j, prot_j, [], target)
            if drop_i is None or drop_j is None:
                break
            new_id = next_id
            next_id += 1
            sets[i].remove(drop_i)
            sets[j].remove(drop_j)
            sets[i].add(new_id)
            sets[j].add(new_id)

    n = side * side
    idx = np.zeros((n, k), dtype=np.int64)
    for i in range(n):
        s = sorted(sets[i])
        if len(s) != k:
            # pad or trim (should be exact)
            if len(s) < k:
                while len(s) < k:
                    s.append(next_id)
                    next_id += 1
            s = s[:k]
        idx[i] = np.array(s, dtype=np.int64)
    return idx, next_id

def _arc_overlap(s1: int, s2: int, k: int, L: int) -> int:
    if k >= L:
        return k
    delta = (s2 - s1) % L
    ov = max(0, k - int(delta)) + max(0, k - int(L - delta))
    return int(min(k, ov))

def assign_windows_expander(
    A: FloatArr, k: int, rho: float, rng: Generator
) -> tuple[NDArray[np.int64], int, dict]:
    """Index sets of size K for an arbitrary graph, aiming for |S_n ∩ S_m| ≈ ρK on every edge.

    An expander has no geometric stream, so this is not a sliding window. Each tile
    is given a random K-subset of a universe of size M ≈ K/ρ (so the expected
    overlap is already ρK); under-lapped edges are then repaired by swapping in
    a shared sample. This keeps every edge connected with overlap close to the
    target. The returned dict reports the achieved min/mean overlap.
    """
    n = A.shape[0]
    target = int(round(rho * k))
    if rho >= 1.0 - 1e-12 or target >= k:
        idx = np.broadcast_to(np.arange(k, dtype=np.int64), (n, k)).copy()
        return idx, k, dict(target=k, min_overlap=k, mean_overlap=float(k), ok=True, method="full")

    M = max(k, int(round(k / max(rho, 1e-9))))
    edges = [(int(i), int(j)) for i, j in np.argwhere(np.triu(A, 1) > 0)]
    sets: list[set[int]] = [
        set(int(x) for x in rng.choice(M, size=k, replace=False)) for _ in range(n)
    ]

    def ov(i, j):
        return len(sets[i] & sets[j])

    def can_drop(i: int, s: int) -> bool:
        if s not in sets[i]:
            return False
        for m in np.flatnonzero(A[i] > 0):
            m = int(m)
            if s in sets[m] and len(sets[i] & sets[m]) - 1 < target:
                return False
        return True

    max_passes = 24
    for _pass in range(max_passes):
        bad = [(i, j) for i, j in edges if ov(i, j) < target]
        if not bad:
            break
        progressed = False
        rng.shuffle(bad)
        for i, j in bad:
            need = target - ov(i, j)
            only_j = list(sets[j] - sets[i])
            rng.shuffle(only_j)
            for cand in only_j:
                if need <= 0:
                    break
                drop = next((s for s in list(sets[i] - sets[j]) if can_drop(i, s)), None)
                if drop is None:
                    break
                sets[i].remove(drop)
                sets[i].add(cand)
                need -= 1
                progressed = True
        if progressed:
            continue
        # Fresh shared IDs if swaps stall
        for i, j in bad:
            need = target - ov(i, j)
            for _ in range(max(0, need)):
                drop_i = next((s for s in list(sets[i] - sets[j]) if can_drop(i, s)), None)
                drop_j = next((s for s in list(sets[j] - sets[i]) if can_drop(j, s)), None)
                if drop_i is None or drop_j is None:
                    break
                new_id = M
                M += 1
                sets[i].remove(drop_i)
                sets[i].add(new_id)
                sets[j].remove(drop_j)
                sets[j].add(new_id)
                progressed = True
        if not progressed:
            break

    overlaps = [ov(i, j) for i, j in edges]
    min_ov = int(min(overlaps)) if overlaps else 0
    mean_ov = float(np.mean(overlaps)) if overlaps else 0.0
    ok = min_ov >= max(1, int(0.8 * target))
    idx = np.zeros((n, k), dtype=np.int64)
    for i in range(n):
        s = sorted(sets[i])
        if len(s) != k:
            # should not happen; pad or trim
            if len(s) < k:
                while len(s) < k:
                    s.append(M)
                    M += 1
            s = s[:k]
        idx[i] = np.array(s, dtype=np.int64)
    return idx, M, dict(
        target=target,
        min_overlap=min_ov,
        mean_overlap=mean_ov,
        ok=ok,
        method="random-subset-repair",
        n_edges=len(edges),
        universe=M,
    )

def assign_windows(kind: str, n: int, k: int, rho: float, A: FloatArr, rng: Generator):
    """Place per-tile sample windows for a ring, torus, or expander."""
    if kind == "ring":
        idx, L = sliding_windows_ring(n, k, rho)
        # exact overlap on ring edges
        target = k if rho >= 1 else k - int(round((1.0 - rho) * k))
        info = dict(target=target, min_overlap=target, mean_overlap=float(target), ok=True, method="1d-sliding")
        return idx, L, info
    if kind == "torus":
        # Exact 2-D sliding windows need side*(1-ρ) > 1; otherwise every window
        # covers the whole field and the construction collapses to ρ=1.
        side = int(round(np.sqrt(n)))
        w = int(round(np.sqrt(k)))
        shift = 0 if rho >= 1 else int(round((1.0 - rho) * w))
        degenerate = (rho < 1) and (shift >= 1) and (side * shift <= w)
        if degenerate:
            idx, L, info = assign_windows_expander(A, k, rho, rng)
            info = dict(info)
            info["method"] = "greedy-identify (degenerate 2-D torus)"
            return idx, L, info
        idx, L = sliding_windows_torus(n, k, rho)
        target = k if rho >= 1 else int(round(rho * k))
        info = dict(target=target, min_overlap=target, mean_overlap=float(target), ok=True, method="2d-sliding")
        return idx, L, info
    if kind == "expander":
        return assign_windows_expander(A, k, rho, rng)
    raise ValueError(f"unknown kind {kind}")

def generate_windowed_observations(
    tau: ComplexArr,
    idx: NDArray[np.int64],
    stream_len: int,
    snr_db: float,
    rng: Generator,
    psi_stream: ComplexArr | None = None,
) -> tuple[ComplexArr, ComplexArr]:
    """Y[n, t] = τ_n * ψ[idx[n,t]] + w_{n,t}."""
    n, k = idx.shape
    if psi_stream is None:
        psi_stream = complex_normal(rng, stream_len)
    # guard: indices must lie in [0, stream_len)
    if int(idx.max()) >= stream_len:
        raise ValueError(f"index {int(idx.max())} >= stream_len {stream_len}")
    sigma = noise_std_from_snr(snr_db, signal_power=1.0)
    W = sigma * complex_normal(rng, (n, k))
    psi_obs = psi_stream[idx]
    Y = tau[:, None] * psi_obs + W
    return Y.astype(np.complex128), psi_stream.astype(np.complex128)

def edge_ratios_from_windows(
    Y: ComplexArr, idx: NDArray[np.int64], A: FloatArr
) -> tuple[ComplexArr, NDArray[np.int32]]:
    """r_nm = ⟨y_n,y_m⟩_S / ⟨y_m,y_m⟩_S on S = S_n ∩ S_m, for edges of A only."""
    n, k = Y.shape
    r = np.zeros((n, n), dtype=np.complex128)
    counts = np.zeros((n, n), dtype=np.int32)
    # hash maps per tile: sample-id -> (value)
    maps: list[dict[int, complex]] = []
    for i in range(n):
        maps.append({int(s): complex(z) for s, z in zip(idx[i], Y[i])})
    ii, jj = np.where(np.triu(A, 1) > 0)
    dropped = 0
    for i, j in zip(ii, jj):
        i = int(i)
        j = int(j)
        acc = 0.0 + 0.0j
        e_i = 0.0
        e_j = 0.0
        cnt = 0
        # iterate the smaller map
        mj = maps[j]
        for s, zi in maps[i].items():
            zj = mj.get(s)
            if zj is None:
                continue
            acc += zi * np.conj(zj)
            e_i += (zi.real * zi.real + zi.imag * zi.imag)
            e_j += (zj.real * zj.real + zj.imag * zj.imag)
            cnt += 1
        if cnt <= 0 or e_i <= 1e-30 or e_j <= 1e-30:
            dropped += 1
            continue
        counts[i, j] = counts[j, i] = cnt
        r[i, j] = acc / e_j
        r[j, i] = np.conj(acc) / e_i
    return r, counts

def used_adjacency(A: FloatArr, counts: NDArray[np.int32], min_count: int = 1) -> FloatArr:
    """Drop edges that had no (or too few) shared samples."""
    A_use = A.copy()
    A_use[counts < min_count] = 0.0
    return A_use

def pairwise_inner_products(Y: ComplexArr) -> ComplexArr:
    """Gram matrix of tile observations, G = Y Y^H."""
    return Y @ Y.conj().T

def relative_veil_matrix(Y: ComplexArr, A: FloatArr) -> ComplexArr:
    """r_nm = ⟨y_n, y_m⟩ / ⟨y_m, y_m⟩ ≈ τ_n / τ_m on edges; 0 off edges."""
    G = pairwise_inner_products(Y)
    denom = np.real(np.diag(G)).copy()
    denom = np.maximum(denom, 1e-15)
    r = G / denom[None, :]
    # 0/1 wirings match scaling by A. Weighted wirings keep the ratio
    # and let the Laplacian carry the weight.
    r = r * (A > 1e-15)
    return r.astype(np.complex128)

def _symmetrized_log_amp(r: ComplexArr, A: FloatArr) -> FloatArr:
    mask = A > 1e-15
    z = np.zeros(r.shape, dtype=np.float64)
    abs_r = np.abs(r)
    log_abs = np.zeros_like(abs_r)
    log_abs[mask] = np.log(np.maximum(abs_r[mask], 1e-15))
    z[mask] = 0.5 * (log_abs[mask] - log_abs.T[mask])
    return z

def _connection_matrix(r: ComplexArr, A: FloatArr, unit_modulus: bool = True) -> ComplexArr:
    mask = A > 1e-15
    H = np.zeros_like(r, dtype=np.complex128)
    if unit_modulus:
        abs_r = np.abs(r)
        H[mask] = A[mask] * r[mask] / np.maximum(abs_r[mask], 1e-15)
    else:
        H[mask] = r[mask]
    H = 0.5 * (H + H.conj().T)
    return H

def amplitude_ls(r: ComplexArr, A: FloatArr) -> FloatArr:
    """Least-squares log-amplitude: L η = b, mean-zero. Sparse solve for all N."""
    z = _symmetrized_log_amp(r, A)
    b = (A * z).sum(axis=1)
    n = A.shape[0]
    if n == 1:
        return np.zeros(1, dtype=np.float64)
    L = laplacian_sparse(A).tocsc()
    eta = np.zeros(n, dtype=np.float64)
    Lred = L[: n - 1, : n - 1]
    try:
        eta[:-1] = spsolve(Lred, b[:-1])
        if not np.all(np.isfinite(eta)):
            raise ValueError("non-finite sparse solve")
    except Exception:
        Ld = laplacian(A)
        try:
            eta[:-1] = np.linalg.solve(Ld[:-1, :-1], b[:-1])
        except np.linalg.LinAlgError:
            eta = np.linalg.lstsq(Ld, b, rcond=None)[0]
    eta -= eta.mean()
    return eta

def amplitude_jacobi(r: ComplexArr, A: FloatArr, t_iter: int, eta0: FloatArr | None = None) -> FloatArr:
    """Local Jacobi iteration for the log-amplitude least-squares problem."""
    z = _symmetrized_log_amp(r, A)
    b = (A * z).sum(axis=1)
    deg = np.maximum(A.sum(axis=1), 1.0)
    n = A.shape[0]
    eta = np.zeros(n, dtype=np.float64) if eta0 is None else eta0.copy()
    for _ in range(t_iter):
        eta = (A @ eta + b) / deg
        eta -= eta.mean()
    return eta

def phase_power_iteration(
    r: ComplexArr,
    A: FloatArr,
    t_iter: int,
    u0: ComplexArr | None = None,
) -> ComplexArr:
    """Local power iteration for the relative phase of the veil."""
    H = _connection_matrix(r, A, unit_modulus=True)
    n = A.shape[0]
    u = np.ones(n, dtype=np.complex128) if u0 is None else u0.copy()
    u = u / np.maximum(np.abs(u), 1e-15)
    for _ in range(t_iter):
        v = H @ u
        u = v / np.maximum(np.abs(v), 1e-15)
    return u

def phase_leading_eigenvector(r: ComplexArr, A: FloatArr) -> ComplexArr:
    """Leading eigenvector of the Hermitian connection matrix H."""
    H = _connection_matrix(r, A, unit_modulus=True)
    n = A.shape[0]
    if n <= 192:
        evals, evecs = np.linalg.eigh(H)
        u = evecs[:, -1]
        return u / np.maximum(np.abs(u), 1e-15)
    # sparse path
    H_s = sparse.csr_matrix(H)
    try:
        _evals, evecs = eigsh(H_s, k=1, which="LA", maxiter=max(400, 8 * n), tol=1e-8)
        u = evecs[:, 0]
        return u / np.maximum(np.abs(u), 1e-15)
    except Exception:
        evals, evecs = np.linalg.eigh(H)
        u = evecs[:, -1]
        return u / np.maximum(np.abs(u), 1e-15)

def estimate_decentralized(
    r: ComplexArr,
    A: FloatArr,
    mode: str = "converged",
    t_iter: int = 500,
    eta0: FloatArr | None = None,
    u0: ComplexArr | None = None,
    chebyshev: bool = False,
    lam2: float | None = None,
    lam_max: float | None = None,
) -> ComplexArr:
    """Estimate each tile veil up to one global complex factor, from neighbour ratios only."""
    if mode == "converged":
        eta = amplitude_ls(r, A)
        phase = phase_leading_eigenvector(r, A)
    elif mode == "iterative":
        if chebyshev and lam2 is not None and lam_max is not None:
            eta = amplitude_chebyshev(r, A, t_iter, lam2, lam_max, eta0=eta0)
        else:
            eta = amplitude_jacobi(r, A, t_iter, eta0=eta0)
        phase = phase_power_iteration(r, A, t_iter, u0=u0)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return np.exp(eta) * phase

def gauge_align(tau_hat: ComplexArr, tau_true: ComplexArr) -> tuple[ComplexArr, complex]:
    """Remove the unobservable global complex factor by least squares against a reference."""
    denom = np.vdot(tau_hat, tau_hat)
    if denom == 0:
        return tau_hat.copy(), 1.0 + 0.0j
    alpha = np.vdot(tau_hat, tau_true) / denom
    return alpha * tau_hat, complex(alpha)

def residual_veils(tau_hat: ComplexArr, tau_true: ComplexArr) -> ComplexArr:
    """Per-tile leftover veil after the best global complex factor is removed."""
    aligned, _ = gauge_align(tau_hat, tau_true)
    aligned_safe = np.where(np.abs(aligned) >= 1e-15, aligned, np.complex128(1e-15))
    return tau_true / aligned_safe

def kappa_errors(tau_hat: ComplexArr, tau_true: ComplexArr) -> dict[str, FloatArr | float]:
    """Gauge-fixed amplitude, phase, and combined veil error of an estimate."""
    residual = residual_veils(tau_hat, tau_true)
    aligned, _ = gauge_align(tau_hat, tau_true)
    err_amp = np.log(np.abs(tau_true)) - np.log(np.abs(aligned))
    err_phase = np.angle(residual)
    rms_amp = float(np.sqrt(np.mean(err_amp**2)))
    rms_phase = float(np.sqrt(np.mean(err_phase**2)))
    rms_kappa = float(np.sqrt(np.mean(np.abs(np.log(residual)) ** 2)))
    return dict(
        rms_amp=rms_amp,
        rms_phase=rms_phase,
        rms_kappa=rms_kappa,
        mse_amp=float(np.mean(err_amp**2)),
        mse_phase=float(np.mean(err_phase**2)),
        mse_kappa=float(np.mean(np.abs(np.log(residual)) ** 2)),
        err_amp=err_amp,
        err_phase=err_phase,
        residual=residual,
    )

def xi_from_consistency(r: ComplexArr, A: FloatArr, tau_hat: ComplexArr) -> float:
    """Coherence score from neighbour-ratio residuals. No reference veil is required."""
    mask = A > 1e-15
    if not np.any(mask):
        return 0.0
    s = r * tau_hat[None, :] / tau_hat[:, None]
    vals = s[mask]
    mean_abs2 = np.mean(np.abs(vals) ** 2)
    if mean_abs2 <= 0:
        return 0.0
    return float(np.abs(np.mean(vals)) ** 2 / mean_abs2)

def precompute_edge_shares(idx: NDArray[np.int64], A: FloatArr):
    """Shared-sample positions on every edge, padded to a rectangular array."""
    n, k = idx.shape
    maps = [{int(s): p for p, s in enumerate(idx[t])} for t in range(n)]
    ii, jj = np.where(np.triu(A, 1) > 0)
    shares: list[tuple[int, int, list]] = []
    for i, j in zip(ii, jj):
        i, j = int(i), int(j)
        common = []
        mj = maps[j]
        for s, pi in maps[i].items():
            pj = mj.get(s)
            if pj is not None:
                common.append((pi, pj))
        shares.append((i, j, common))
    o_max = max((len(c) for *_, c in shares), default=0)
    e = len(shares)
    pos_i = np.zeros((e, max(o_max, 1)), dtype=np.int64)
    pos_j = np.zeros((e, max(o_max, 1)), dtype=np.int64)
    mask = np.zeros((e, max(o_max, 1)), dtype=bool)
    i_arr = np.zeros(e, dtype=np.int64)
    j_arr = np.zeros(e, dtype=np.int64)
    for e_i, (i, j, common) in enumerate(shares):
        i_arr[e_i] = i
        j_arr[e_i] = j
        for o, (pi, pj) in enumerate(common):
            pos_i[e_i, o] = pi
            pos_j[e_i, o] = pj
            mask[e_i, o] = True
    return dict(i=i_arr, j=j_arr, pos_i=pos_i, pos_j=pos_j, mask=mask, n=n)

def edge_ratios_batch(Y: ComplexArr, shares: dict) -> ComplexArr:
    """Y (W,N,K) -> r (W,N,N) using only shared samples on each edge."""
    w, n, _k = Y.shape
    i, j = shares["i"], shares["j"]
    pos_i, pos_j, mask = shares["pos_i"], shares["pos_j"], shares["mask"]
    if i.size == 0:
        return np.zeros((w, n, n), dtype=np.complex128)
    # Yi[w, e, o] = Y[w, i[e], pos_i[e, o]]
    Yi = Y[:, i[:, None], pos_i]
    Yj = Y[:, j[:, None], pos_j]
    m = mask[None, :, :]
    acc = (Yi * np.conj(Yj) * m).sum(axis=2)
    e_j = ((Yj.real**2 + Yj.imag**2) * m).sum(axis=2)
    e_i = ((Yi.real**2 + Yi.imag**2) * m).sum(axis=2)
    r = np.zeros((w, n, n), dtype=np.complex128)
    r[:, i, j] = acc / np.maximum(e_j, 1e-30)
    r[:, j, i] = np.conj(acc) / np.maximum(e_i, 1e-30)
    return r

def generate_windowed_observations_batch(
    tau: ComplexArr,
    idx: NDArray[np.int64],
    stream_len: int,
    snr_db: float,
    rng: Generator,
) -> ComplexArr:
    """tau (W,N) -> Y (W,N,K)."""
    w, n = tau.shape
    k = idx.shape[1]
    psi = complex_normal(rng, (w, stream_len))
    sigma = noise_std_from_snr(snr_db, signal_power=1.0)
    noise = sigma * complex_normal(rng, (w, n, k))
    psi_obs = psi[:, idx]  # (W, N, K)
    return (tau[:, :, None] * psi_obs + noise).astype(np.complex128)

def unveil_batch(r: ComplexArr, A: FloatArr) -> ComplexArr:
    """Batched converged estimator. r is (W,N,N) or (N,N). Returns (W,N)."""
    if r.ndim == 2:
        r = r[None, ...]
    w, n, _ = r.shape
    mask = A > 1e-15
    abs_r = np.abs(r)
    log_abs = np.zeros(r.shape, dtype=np.float64)
    log_abs[:, mask] = np.log(np.maximum(abs_r[:, mask], 1e-15))
    z = 0.5 * (log_abs - np.transpose(log_abs, (0, 2, 1)))
    b = (A[None, :, :] * z).sum(axis=2)
    L = laplacian(A)
    eta = np.zeros((w, n), dtype=np.float64)
    if n > 1:
        try:
            eta[:, :-1] = np.linalg.solve(L[:-1, :-1], b[:, :-1].T).T
        except np.linalg.LinAlgError:
            for wi in range(w):
                eta[wi] = np.linalg.lstsq(L, b[wi], rcond=None)[0]
    eta -= eta.mean(axis=1, keepdims=True)
    H = np.zeros_like(r)
    H[:, mask] = r[:, mask] / np.maximum(abs_r[:, mask], 1e-15)
    H = 0.5 * (H + np.conjugate(np.transpose(H, (0, 2, 1))))
    _evals, evecs = np.linalg.eigh(H)
    u = evecs[:, :, -1]
    u = u / np.maximum(np.abs(u), 1e-15)
    return (np.exp(eta) * u).astype(np.complex128)

def unveil_iterative_batch(r: ComplexArr, A: FloatArr, n_iter: int = 80) -> ComplexArr:
    """Batched Jacobi + phase power iteration, for large N."""
    if r.ndim == 2:
        r = r[None, ...]
    w, n, _ = r.shape
    mask = A > 1e-15
    abs_r = np.abs(r)
    log_abs = np.zeros(r.shape, dtype=np.float64)
    log_abs[:, mask] = np.log(np.maximum(abs_r[:, mask], 1e-15))
    z = 0.5 * (log_abs - np.transpose(log_abs, (0, 2, 1)))
    b = (A[None] * z).sum(axis=2)
    deg = np.maximum(A.sum(axis=1), 1.0)
    eta = np.zeros((w, n), dtype=np.float64)
    for _ in range(n_iter):
        eta = (eta @ A.T + b) / deg  # (W,N) @ (N,N) + (W,N)
        eta -= eta.mean(axis=1, keepdims=True)
    H = np.zeros_like(r)
    H[:, mask] = r[:, mask] / np.maximum(abs_r[:, mask], 1e-15)
    H = 0.5 * (H + np.conjugate(np.transpose(H, (0, 2, 1))))
    u = np.ones((w, n), dtype=np.complex128)
    for _ in range(n_iter):
        v = np.einsum("wij,wj->wi", H, u)
        u = v / np.maximum(np.abs(v), 1e-15)
    return (np.exp(eta) * u).astype(np.complex128)

def gauge_align_batch(tau_hat: ComplexArr, tau_ref: ComplexArr) -> tuple[ComplexArr, ComplexArr]:
    """Align each walk by a single complex factor. Shapes (W,N)."""
    denom = np.sum(np.conj(tau_hat) * tau_hat, axis=1, keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    alpha = np.sum(np.conj(tau_hat) * tau_ref, axis=1, keepdims=True) / denom
    return alpha * tau_hat, alpha

def gauge_align_subset(tau_hat: ComplexArr, tau_ref: ComplexArr, idx: np.ndarray) -> tuple[ComplexArr, ComplexArr]:
    """Post-hoc pin: estimate the global factor from a subset of tiles."""
    hat = tau_hat[:, idx]
    ref = tau_ref[:, idx] if tau_ref.ndim == 2 else tau_ref[idx][None]
    denom = np.sum(np.conj(hat) * hat, axis=1, keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    alpha = np.sum(np.conj(hat) * ref, axis=1, keepdims=True) / denom
    return alpha * tau_hat, alpha

def kappa_errors_batch(tau_hat: ComplexArr, tau_true: ComplexArr, align: bool = True) -> dict:
    """If align=True, gauge-fixed (relative) error; else absolute error."""
    if tau_hat.ndim == 1:
        tau_hat = tau_hat[None]
        tau_true = tau_true[None]
    if align:
        aligned, _ = gauge_align_batch(tau_hat, tau_true)
    else:
        aligned = tau_hat
    aligned_safe = np.where(np.abs(aligned) >= 1e-15, aligned, np.complex128(1e-15))
    residual = tau_true / aligned_safe
    err_amp = np.log(np.abs(tau_true)) - np.log(np.abs(aligned_safe))
    err_phase = np.angle(residual)
    mse_amp = np.mean(err_amp**2, axis=1)
    mse_phase = np.mean(err_phase**2, axis=1)
    logres = np.log(residual)
    mse_kappa = np.mean(np.abs(logres) ** 2, axis=1)
    rms_kappa = np.sqrt(mse_kappa)
    per_tile = np.abs(logres) ** 2  # (W, N)
    return dict(
        rms_kappa=rms_kappa,
        rms_amp=np.sqrt(mse_amp),
        rms_phase=np.sqrt(mse_phase),
        mse_kappa=mse_kappa,
        mse_amp=mse_amp,
        mse_phase=mse_phase,
        err_amp=err_amp,
        err_phase=err_phase,
        per_tile_mse=per_tile,
        residual=residual,
    )

def xi_from_consistency_batch(r: ComplexArr, A: FloatArr, tau_hat: ComplexArr) -> FloatArr:
    """Neighbour-consistency score for each walk in a batch."""
    mask = A > 1e-15
    if not np.any(mask):
        return np.zeros(r.shape[0])
    s = r * tau_hat[:, None, :] / tau_hat[:, :, None]
    vals = s[:, mask]
    mean_abs2 = np.mean(np.abs(vals) ** 2, axis=1)
    mean = np.mean(vals, axis=1)
    out = np.zeros(r.shape[0])
    ok = mean_abs2 > 0
    out[ok] = (np.abs(mean[ok]) ** 2) / mean_abs2[ok]
    return out

def amplitude_chebyshev(
    r: ComplexArr,
    A: FloatArr,
    t_iter: int,
    lam2: float,
    lam_max: float,
    eta0: FloatArr | None = None,
) -> FloatArr:
    """Chebyshev-accelerated Richardson for the Jacobi Laplacian system L η = b.

    On a (nearly) regular graph of degree d, I-P = D^{-1} L has spectrum in
    [λ₂/d, λ_max/d] on the mean-zero subspace. The 3-term recurrence is the
    standard Chebyshev semi-iteration (Barrett et al., Templates, Fig. 2.13).
    """
    z = _symmetrized_log_amp(r, A)
    bvec = (A * z).sum(axis=1)
    deg = np.maximum(A.sum(axis=1), 1.0)
    dmean = float(np.mean(deg))
    a = max(lam2 / dmean, 1e-12)
    bb = min(max(lam_max / dmean, a + 1e-12), 2.0 - 1e-12)
    f = bvec / deg
    n = A.shape[0]
    y_prev = np.zeros(n, dtype=np.float64) if eta0 is None else eta0.copy()
    y_prev -= y_prev.mean()

    def apply_IminusP(v):
        return v - (A @ v) / deg

    theta = 2.0 / (a + bb)
    sigma = (bb - a) / (a + bb)
    resid0 = f - apply_IminusP(y_prev)
    resid0 -= resid0.mean()
    y = y_prev + theta * resid0
    y -= y.mean()
    if t_iter <= 1:
        return y
    omega = 1.0
    for k in range(1, t_iter):
        if k == 1:
            omega = 2.0 / (2.0 - sigma * sigma)
        else:
            omega = 1.0 / (1.0 - (sigma * sigma * 0.25) * omega)
        resid = f - apply_IminusP(y)
        resid -= resid.mean()
        y_new = omega * (y + theta * resid - y_prev) + y_prev
        y_new -= y_new.mean()
        y_prev, y = y, y_new
    return y

def estimate_centralized_svd(Y: ComplexArr) -> ComplexArr:
    """Rank-1 estimate of the veil from the full observation matrix."""
    u, s, _vh = np.linalg.svd(Y, full_matrices=False)
    tau_hat = u[:, 0] * np.sqrt(s[0]) if s.size else u[:, 0]
    return tau_hat.astype(np.complex128)

def xi_empirical(tau: ComplexArr) -> float:
    """Coherence score |mean τ|^2 / mean|τ|^2 on the supplied tiles."""
    mean_abs2 = np.mean(np.abs(tau) ** 2)
    if mean_abs2 <= 0:
        return 0.0
    return float(np.abs(np.mean(tau)) ** 2 / mean_abs2)

def independent_edge_ratios(
    tau: ComplexArr, A: FloatArr, sigma: float, rng: Generator
) -> ComplexArr:
    """r_nm = (τ_n / τ_m) exp(ν_nm), with ν_nm = −ν_mn i.i.d. complex Gaussian
    on undirected edges. This is the Barooah–Hespanha measurement model:
    independent edge noise, not derived from a shared observation matrix Y.
    """
    n = tau.shape[0]
    ii, jj = np.where(np.triu(A, 1) > 0)
    noise_a = rng.normal(0.0, sigma, size=ii.size)
    noise_p = rng.normal(0.0, sigma, size=ii.size)
    noise = np.zeros((n, n), dtype=np.complex128)
    noise[ii, jj] = noise_a + 1j * noise_p
    noise[jj, ii] = -(noise_a + 1j * noise_p)
    mask = A > 1e-15
    ratio = tau[:, None] / tau[None, :]
    r = np.zeros((n, n), dtype=np.complex128)
    r[mask] = ratio[mask] * np.exp(noise[mask])
    return r

def mse_kappa(tau_hat: ComplexArr, tau_true: ComplexArr) -> float:
    """Mean squared gauge-fixed log-veil error."""
    aligned, _ = gauge_align(tau_hat, tau_true)
    aligned_safe = np.where(np.abs(aligned) >= 1e-15, aligned, np.complex128(1e-15))
    residual = tau_true / aligned_safe
    return float(np.mean(np.abs(np.log(residual)) ** 2))
