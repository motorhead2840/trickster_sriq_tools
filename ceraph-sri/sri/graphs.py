"""Tile wiring graphs.

NumPy arrays are the only backend. Ring, torus, and expander constructors
match the published unveiling runs. Quasiperiodic wirings live in
``sri.wiring_quasi``.
"""
from __future__ import annotations

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray
from scipy import sparse

ComplexArr = NDArray[np.complex128]
FloatArr = NDArray[np.float64]

# Degree of the expander used as the comparison wiring. It matches the torus.
EXPANDER_DEGREE = 4

def is_perfect_square(n: int) -> bool:
    """Return True when n is a perfect square, so a torus wiring exists."""
    side = int(round(np.sqrt(n)))
    return side * side == n

def is_connected(A: FloatArr) -> bool:
    """Return True when every tile can reach every other tile on the wiring."""
    n = A.shape[0]
    if n == 0:
        return True
    seen = {0}
    stack = [0]
    while stack:
        u = stack.pop()
        for v in np.flatnonzero(A[u] > 1e-15):
            v = int(v)
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == n

def _symmetrize(A: FloatArr) -> FloatArr:
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    A[A < 0] = 0.0
    return A

def ring_adjacency(n: int, hops: tuple[int, ...] = (1,)) -> FloatArr:
    """Cycle wiring. hops selects neighbour offsets; the default is the 2-regular ring."""
    A = np.zeros((n, n), dtype=np.float64)
    idx = np.arange(n)
    for h in hops:
        h = int(h) % n
        if h == 0:
            continue
        A[idx, (idx + h) % n] = 1.0
        A[idx, (idx - h) % n] = 1.0
    np.fill_diagonal(A, 0.0)
    return A

def torus_adjacency(n: int) -> FloatArr:
    """4-regular grid on a torus. n must be a perfect square."""
    side = int(round(np.sqrt(n)))
    if side * side != n:
        raise ValueError(f"torus requires a square N, got {n}")
    A = np.zeros((n, n), dtype=np.float64)
    for r in range(side):
        for c in range(side):
            i = r * side + c
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                j = ((r + dr) % side) * side + (c + dc) % side
                A[i, j] = 1.0
    return A

def _stubs_regular_graph(n: int, d: int, rng: Generator, attempts: int = 80) -> FloatArr | None:
    if n <= d or (n * d) % 2 != 0 or d < 1:
        return None
    for _ in range(attempts):
        stubs = np.repeat(np.arange(n, dtype=np.int64), d)
        rng.shuffle(stubs)
        edges = set()
        ok = True
        for i in range(0, stubs.size, 2):
            a = int(stubs[i])
            b = int(stubs[i + 1])
            if a == b:
                ok = False
                break
            edge = (a, b) if a < b else (b, a)
            if edge in edges:
                ok = False
                break
            edges.add(edge)
        if not ok or len(edges) != n * d // 2:
            continue
        A = np.zeros((n, n), dtype=np.float64)
        for a, b in edges:
            A[a, b] = A[b, a] = 1.0
        if np.allclose(A.sum(axis=1), d) and is_connected(A):
            return A
    return None

def circulant_regular_fallback(n: int, d: int) -> FloatArr:
    """Deterministic even-degree circulant used when a random regular draw fails."""
    if d % 2 != 0:
        d = d + 1
    A = np.zeros((n, n), dtype=np.float64)
    n_steps = d // 2
    steps = list(range(1, n_steps))
    jump = max(n_steps, n // 3)
    while jump in steps or jump % n == 0:
        jump += 1
    steps.append(jump % n if jump % n != 0 else 1)
    steps = steps[:n_steps]
    idx = np.arange(n)
    for s in steps:
        A[idx, (idx + s) % n] = 1.0
        A[idx, (idx - s) % n] = 1.0
    np.fill_diagonal(A, 0.0)
    return A

def random_regular_adjacency(n: int, d: int, rng: Generator) -> FloatArr:
    """Connected d-regular wiring.

    The configuration model is tried first. If it does not return a simple
    connected graph, NetworkX's generator is the fallback used by the published
    runs, and a deterministic circulant is the last resort.
    """
    A = _stubs_regular_graph(n, d, rng)
    if A is not None:
        return A
    try:
        import networkx as nx

        graph = nx.random_regular_graph(d, n, seed=int(rng.integers(0, 2**31 - 1)))
        if nx.is_connected(graph):
            return nx.to_numpy_array(graph, dtype=np.float64)
    except Exception:
        pass
    return circulant_regular_fallback(n, d)

def watts_strogatz_adjacency(n: int, degree: int, p: float, rng: Generator) -> FloatArr:
    """Ring of even degree, then rewire each right-hand edge with probability p."""
    if degree % 2 != 0:
        degree += 1
    hops = tuple(range(1, degree // 2 + 1))
    A = ring_adjacency(n, hops)
    # rewire
    for i in range(n):
        for h in hops:
            j = (i + h) % n
            if A[i, j] <= 0:
                continue
            if rng.random() > p:
                continue
            # try a new target
            for _ in range(20):
                k = int(rng.integers(0, n))
                if k != i and A[i, k] == 0:
                    A[i, j] = A[j, i] = 0.0
                    A[i, k] = A[k, i] = 1.0
                    break
    if not is_connected(A):
        # restore a ring backbone
        A = np.maximum(A, ring_adjacency(n, (1,)))
    return A

def laplacian(A: FloatArr) -> FloatArr:
    """Combinatorial Laplacian L = D − A of a tile wiring, weighted or unweighted."""
    deg = A.sum(axis=1)
    return np.diag(deg) - A

def laplacian_sparse(A: FloatArr):
    """Sparse combinatorial Laplacian of a tile wiring."""
    A_s = sparse.csr_matrix(A)
    deg = np.asarray(A_s.sum(axis=1)).ravel()
    return sparse.diags(deg) - A_s

def make_graph(kind: str, n: int, rng: Generator, degree: int = EXPANDER_DEGREE) -> FloatArr:
    """Build a ring, torus, expander, or complete wiring on n tiles."""
    if kind == "ring":
        return ring_adjacency(n)
    if kind == "torus":
        return torus_adjacency(n)
    if kind == "expander":
        return random_regular_adjacency(n, degree, rng)
    if kind == "complete":
        A = np.ones((n, n), dtype=np.float64)
        np.fill_diagonal(A, 0.0)
        return A
    raise ValueError(f"unknown graph kind {kind!r}")

def ring_mean_resistance(n: int) -> float:
    """Exact: eigenvalues 4 sin²(πk/N),  k=1..N-1;  sum 1/λ = (N²-1)/12."""
    if n <= 1:
        return 0.0
    return (n**2 - 1) / (12.0 * n)

def torus_mean_resistance(n: int) -> float:
    """Exact average gauge-fixed node variance from the 2-D Laplacian spectrum."""
    side = int(round(np.sqrt(n)))
    if side * side != n:
        raise ValueError(f"torus requires square N, got {n}")
    if n <= 1:
        return 0.0
    acc = 0.0
    for p in range(side):
        for q in range(side):
            if p == 0 and q == 0:
                continue
            lam = 4.0 * (np.sin(np.pi * p / side) ** 2 + np.sin(np.pi * q / side) ** 2)
            acc += 1.0 / lam
    return acc / n

def mean_resistance(A: FloatArr, kind: str | None = None) -> float:
    """R̄(G) = trace(L⁺)/N = (1/N) Σ_{i: λ_i>0} 1/λ_i.

    For the ring and the torus the spectrum is used in closed form.
    Otherwise a dense (or, for large N, a shift-invert) eigendecomposition.
    """
    n = A.shape[0]
    if n <= 1:
        return 0.0
    if kind == "ring":
        return ring_mean_resistance(n)
    if kind == "torus":
        return torus_mean_resistance(n)
    L = laplacian(A)
    evals = np.linalg.eigvalsh(L)
    pos = evals[evals > 1e-11]
    if pos.size == 0:
        return float("inf")
    return float(np.sum(1.0 / pos) / n)

def laplacian_eigbounds(A: FloatArr) -> dict[str, float]:
    """Laplacian gap of a tile wiring.

    jacobi_gap is λ2 / mean degree, and steps_1_over_gap is the reciprocal,
    which is the predicted Jacobi step count. connected, n_edges, and
    weight_sum are included so weighted wirings can be scored the same way.
    """
    evals = np.sort(np.real(np.linalg.eigvalsh(laplacian(A))))
    lam2 = float(evals[1]) if evals.size > 1 else 0.0
    lam_max = float(evals[-1]) if evals.size else 0.0
    deg = float(np.mean(A.sum(axis=1)))
    gap = lam2 / deg if deg > 0 else 0.0
    n_edges = float(np.sum(A > 1e-15) / 2.0)
    weight_sum = float(np.sum(A) / 2.0)
    return dict(
        lam2=lam2,
        lam_max=lam_max,
        deg=deg,
        jacobi_gap=gap,
        steps_1_over_gap=float(1.0 / gap) if gap > 0 else float("inf"),
        n_edges=n_edges,
        weight_sum=weight_sum,
        connected=bool(is_connected(A) and lam2 > 1e-11),
    )
