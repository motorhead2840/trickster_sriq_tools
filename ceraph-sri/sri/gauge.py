"""Pinning the unobservable global veil factor.

Neighbour comparisons recover only relative veils. Holding M reference tiles
at a known veil removes that factor. With refreshed, correct references the
leftover error scales as about 1/M. A stale reference or a single faulty
reference does not.

The median-of-logs pin is available as an experiment, not as the default.
The default pin is least squares on the reference tiles.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sri.graphs import is_perfect_square, laplacian

ComplexArr = NDArray[np.complex128]
FloatArr = NDArray[np.float64]

def graph_distance_ring(n: int, src: int) -> np.ndarray:
    """Hop distance from one tile along a ring."""
    i = np.arange(n)
    d = np.abs(i - src)
    return np.minimum(d, n - d)

def graph_distance_torus(n: int, src: int) -> np.ndarray:
    """Hop distance from one tile on a torus."""
    side = int(round(np.sqrt(n)))
    sr, sc = divmod(src, side)
    r, c = np.divmod(np.arange(n), side)
    dr = np.minimum(np.abs(r - sr), side - np.abs(r - sr))
    dc = np.minimum(np.abs(c - sc), side - np.abs(c - sc))
    return dr + dc

def nearest_distance(n: int, refs: np.ndarray, kind: str) -> np.ndarray:
    """Graph distance from each tile to the nearest reference tile."""
    if kind == "ring":
        dists = np.stack([graph_distance_ring(n, int(s)) for s in refs], axis=0)
    else:
        # torus distances are a fine hop metric on a 4-regular expander too
        if kind == "torus":
            dists = np.stack([graph_distance_torus(n, int(s)) for s in refs], axis=0)
        else:
            # BFS on the actual graph is done by the caller when needed
            dists = np.stack([graph_distance_ring(n, int(s)) for s in refs], axis=0)
    return dists.min(axis=0)

def bfs_distance(A: FloatArr, sources) -> np.ndarray:
    """Breadth-first distance from a set of reference tiles."""
    n = A.shape[0]
    dist = np.full(n, n + 1, dtype=np.int64)
    from collections import deque
    q = deque()
    for s in sources:
        dist[int(s)] = 0
        q.append(int(s))
    nbrs = [np.flatnonzero(A[i] > 0).astype(int) for i in range(n)]
    while q:
        u = q.popleft()
        for v in nbrs[u]:
            if dist[v] > dist[u] + 1:
                dist[v] = dist[u] + 1
                q.append(int(v))
    return dist

def resistance_to_set(A: FloatArr, refs) -> np.ndarray:
    """Effective resistance from each node to the shorted set of references."""
    n = A.shape[0]
    S = sorted({int(s) for s in refs})
    R = np.zeros(n, dtype=np.float64)
    if not S:
        return np.full(n, np.nan)
    free = [i for i in range(n) if i not in set(S)]
    if not free:
        return R
    L = laplacian(A)
    Lff = L[np.ix_(free, free)]
    try:
        X = np.linalg.inv(Lff)
        for a, i in enumerate(free):
            R[i] = float(np.real(X[a, a]))
    except np.linalg.LinAlgError:
        I = np.eye(len(free))
        X = np.linalg.lstsq(Lff, I, rcond=None)[0]
        for a, i in enumerate(free):
            R[i] = float(np.real(X[a, a]))
    return R

def spread_refs(n: int, m: int, kind: str) -> np.ndarray:
    """Well-separated reference tiles."""
    m = min(m, n)
    if kind == "torus" and is_perfect_square(n) and is_perfect_square(m):
        side = int(round(np.sqrt(n)))
        ms = int(round(np.sqrt(m)))
        step = side // ms
        refs = [((r * step) % side) * side + ((c * step) % side) for r in range(ms) for c in range(ms)]
        return np.array(refs[:m], dtype=np.int64)
    # even spacing on a ring (also a decent spread on an expander by index)
    return np.array([(i * n) // m for i in range(m)], dtype=np.int64)

def estimate_dirichlet_batch(r: ComplexArr, A: FloatArr, refs: np.ndarray, tau_ref: ComplexArr) -> ComplexArr:
    """Anchored estimator: Dirichlet amplitude + connection-Laplacian phase."""
    if r.ndim == 2:
        r = r[None]
        tau_ref = tau_ref[None]
    w, n, _ = r.shape
    refs = np.asarray(refs, dtype=np.int64)
    S = set(int(s) for s in refs)
    free = np.array([i for i in range(n) if i not in S], dtype=np.int64)
    mask = A > 0
    abs_r = np.abs(r)
    log_abs = np.zeros(r.shape, dtype=np.float64)
    log_abs[:, mask] = np.log(np.maximum(abs_r[:, mask], 1e-15))
    z = 0.5 * (log_abs - np.transpose(log_abs, (0, 2, 1)))
    b = (A[None] * z).sum(axis=2)
    L = laplacian(A)
    eta = np.zeros((w, n), dtype=np.float64)
    eta[:, refs] = np.log(np.maximum(np.abs(tau_ref), 1e-15))
    if free.size:
        Lff = L[np.ix_(free, free)]
        Lfs = L[np.ix_(free, refs)]
        rhs = b[:, free].T - Lfs @ eta[:, refs].T  # (F, W)
        try:
            eta[:, free] = np.linalg.solve(Lff, rhs).T
        except np.linalg.LinAlgError:
            eta[:, free] = np.linalg.lstsq(Lff, rhs, rcond=None)[0].T
    H = np.zeros_like(r)
    H[:, mask] = r[:, mask] / np.maximum(abs_r[:, mask], 1e-15)
    H = 0.5 * (H + np.conjugate(np.transpose(H, (0, 2, 1))))
    deg = A.sum(axis=1)
    # connection Laplacian L = D - H
    Lconn = np.zeros_like(H)
    Lconn[:, np.arange(n), np.arange(n)] = deg
    Lconn -= H
    phi = np.ones((w, n), dtype=np.complex128)
    phi[:, refs] = tau_ref / np.maximum(np.abs(tau_ref), 1e-15)
    if free.size:
        Lff = Lconn[:, np.ix_(free, free)] if False else Lconn[:, free][:, :, free]
        # Lconn is (W,N,N); Lff (W,F,F), Lfs (W,F,M)
        Lfs = Lconn[:, free][:, :, refs]
        rhs = -np.einsum("wfm,wm->wf", Lfs, phi[:, refs])
        try:
            phi[:, free] = np.linalg.solve(Lff, rhs[..., None]).squeeze(-1)
        except np.linalg.LinAlgError:
            for wi in range(w):
                phi[wi, free] = np.linalg.lstsq(Lff[wi], rhs[wi], rcond=None)[0]
    phi = phi / np.maximum(np.abs(phi), 1e-15)
    return (np.exp(eta) * phi).astype(np.complex128)

def _pin_posthoc(hat, refs, tau_ref):
    """hat (N,), refs (M,), tau_ref (M,) -> pinned (N,), alpha, gauge residual."""
    h = hat[refs]
    denom = np.vdot(h, h)
    if denom == 0:
        alpha = 1.0 + 0.0j
    else:
        alpha = np.vdot(h, tau_ref) / denom
    return alpha * hat, complex(alpha)

def _pin_posthoc_robust(hat, refs, tau_ref):
    """Median log-offset across references (resistant to one faulty ref)."""
    h = hat[refs]
    offset = np.log(tau_ref / np.where(np.abs(h) < 1e-15, 1e-15, h))
    # complex median ≈ median real, median imag
    c = np.median(offset.real) + 1j * np.median(offset.imag)
    return hat * np.exp(c), complex(np.exp(c))

def _pin_mean_prior(hat, strength=1.0):
    kap = np.log(hat)
    kap = kap - strength * kap.mean()
    return np.exp(kap)

def predict_gauge_variance(r_ref, r_rel, m):
    """Var(gauge) ≈ (r_ref + r_rel)/M for well-separated, roughly independent refs."""
    return (r_ref + r_rel) / max(m, 1)

def fit_mse_vs_reff(points, layout, m):
    """Linear fit of absolute veil error against resistance to the reference set."""
    pts = [p for p in points if p["layout"] == layout and p["m"] == m and not p["is_ref"]]
    if len(pts) < 4:
        return dict(slope=np.nan, intercept=np.nan, r=np.nan, n=len(pts))
    x = np.array([p["R_eff"] for p in pts])
    y = np.array([p["mse"] for p in pts])
    if np.allclose(x, x[0]):
        return dict(slope=0.0, intercept=float(y.mean()), r=np.nan, n=len(pts))
    A = np.vstack([x, np.ones(len(x))]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = slope * x + intercept
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return dict(slope=float(slope), intercept=float(intercept), r2=float(r2), n=len(pts),
                xmin=float(x.min()), xmax=float(x.max()), ymin=float(y.min()), ymax=float(y.max()))
