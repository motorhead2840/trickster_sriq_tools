"""Quasiperiodic tile wirings.

Fibonacci and Aubry–André hop patterns are included because they were tested
as alternatives to a grid or an expander. On the unveiling error law they
raise the mean resistance, so they are a negative result, not a recommended
wiring. The on-site Aubry–André inverse participation ratio is a diagnostic
of that chain; it is not a score for the wiring.
"""
from __future__ import annotations

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from sri.graphs import (
    random_regular_adjacency,
    ring_adjacency,
    torus_adjacency,
    watts_strogatz_adjacency,
)

ComplexArr = NDArray[np.complex128]
FloatArr = NDArray[np.float64]

# Inverse golden ratio. It sets the Aubry–André hop phase.
GOLDEN_CONJ = 0.5 * (np.sqrt(5.0) - 1.0)

def _fibonacci_list(limit: int) -> list[int]:
    out = []
    a, b = 1, 2
    while a < limit:
        out.append(a)
        a, b = b, a + b
    return out

def fibonacci_word(n: int) -> np.ndarray:
    """Infinite Fibonacci word 0100101... as a 0/1 array of length n."""
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    a, b = "0", "01"
    while len(b) < n:
        a, b = b, b + a
    return np.fromiter((int(ch) for ch in b[:n]), dtype=np.int64, count=n)

def fibonacci_weighted_ring(n: int, hops: tuple[int, ...], weak: float, strong: float = 1.0) -> FloatArr:
    """Ring (or degree-4 ring) with hoppings strong/weak along a Fibonacci word."""
    word = fibonacci_word(n)
    A = np.zeros((n, n), dtype=np.float64)
    for h in hops:
        for i in range(n):
            j = (i + h) % n
            w = strong if word[i] == 0 else weak
            # the opposite direction uses the word at j if we only set i->i+h;
            # make it a single undirected weight from the lower-index side of the hop.
            A[i, j] = max(A[i, j], w)
            A[j, i] = A[i, j]
    np.fill_diagonal(A, 0.0)
    return A

def aa_weighted_ring(n: int, lam: float, hops: tuple[int, ...] = (1,), phi: float = 0.0) -> FloatArr:
    """Ring with hoppings w_j = 1 + λ cos(2π β j + φ), β the golden conjugate.

    The standard Aubry–André *on-site* transition is at λ=1 (V=2λ, t=1) or
    λ=2 depending on normalisation; it does not itself change who is wired to
    whom. This weighted-hopping ring is the natural analogue for unveiling:
    as λ → 1^-, some hoppings approach 0 and the chain almost disconnects.
    We never let a hopping go non-positive: hoppings are clip{1+λ cos, ε}.
    """
    eps = 1e-3
    A = np.zeros((n, n), dtype=np.float64)
    for h in hops:
        for i in range(n):
            j = (i + h) % n
            w = 1.0 + lam * np.cos(2.0 * np.pi * GOLDEN_CONJ * i + phi)
            w = float(max(w, eps))
            A[i, j] = A[j, i] = w
    np.fill_diagonal(A, 0.0)
    return A

def aa_onsite_ipr(n: int, lam: float, t: float = 1.0) -> dict[str, float]:
    """IPR diagnostic of the *standard* on-site AA Hamiltonian (not the wiring).

    H_{j,j}=2λ cos(2π β j), H_{j,j+1}=H_{j+1,j}=t, transition at |λ|=t (=1).
    Mean inverse participation ratio near 1/N is extended; near 1 is localized.
    """
    idx = np.arange(n)
    onsite = 2.0 * lam * np.cos(2.0 * np.pi * GOLDEN_CONJ * idx)
    H = np.diag(onsite)
    H += t * ring_adjacency(n, (1,))
    evals, evecs = np.linalg.eigh(H)
    # IPR_α = Σ_j |ψ_{αj}|^4
    ipr = np.sum(np.abs(evecs) ** 4, axis=0)
    return dict(
        ipr_mean=float(np.mean(ipr)),
        ipr_median=float(np.median(ipr)),
        ipr_max=float(np.max(ipr)),
        ipr_min=float(np.min(ipr)),
        participation_mean=float(np.mean(1.0 / np.maximum(ipr, 1e-18))),
        n=n,
        lam=lam,
        localized_hint=bool(np.median(ipr) > 4.0 / n),
    )

def circulant_from_steps(n: int, steps: tuple[int, ...]) -> FloatArr:
    """Circulant wiring with the given step set."""
    A = np.zeros((n, n), dtype=np.float64)
    idx = np.arange(n)
    used = []
    for s in steps:
        s = int(s) % n
        if s == 0 or n - s in used or s in used:
            continue
        A[idx, (idx + s) % n] = 1.0
        A[idx, (idx - s) % n] = 1.0
        used.append(s)
    np.fill_diagonal(A, 0.0)
    return A

def fibonacci_circulant(n: int, degree: int = 4) -> FloatArr:
    """Unweighted circulant with a ring backbone plus Fibonacci long-range jumps.

    The first hop is always 1 (so the graph stays connected). Further hops are
    Fibonacci lengths near sqrt(N), not 2 — otherwise degree 4 would collapse
    to the ordinary ±1,±2 ring.
    """
    n_steps = max(degree // 2, 1)
    fibs = [f for f in _fibonacci_list(max(n // 2, 3)) if f >= 3]
    steps = [1]
    target = int(round(np.sqrt(n)))
    # pick distinct Fibs closest to sqrt(N), 2 sqrt(N), ...
    for i in range(1, n_steps):
        tgt = i * target
        leftover = [f for f in fibs if f not in steps and f % n != 0]
        if not leftover:
            break
        steps.append(min(leftover, key=lambda f: abs(f - tgt)))
    s = 3
    while len(steps) < n_steps:
        if s not in steps and s < n // 2:
            steps.append(s)
        s += 1
    return circulant_from_steps(n, tuple(steps[:n_steps]))

def mod13_circulant(n: int, degree: int = 4, include_backbone: bool = True) -> FloatArr:
    """pid%13-style modulo routing as a circulant: steps 1 (optional) and 13, 13², ..."""
    n_steps = max(degree // 2, 1)
    steps = []
    if include_backbone:
        steps.append(1)
    jump = 13
    while len(steps) < n_steps:
        s = jump % n
        if s not in steps and s != 0 and (n - s) not in steps:
            steps.append(s)
        if jump > 13 ** 6:
            break
        jump *= 13
    s = 2
    while len(steps) < n_steps:
        if s not in steps and s % n != 0:
            steps.append(s)
        s += 1
    return circulant_from_steps(n, tuple(steps[:n_steps]))

def mod13_naive(n: int) -> FloatArr:
    """Naive routing with only step 13. Disconnected when gcd(13, N) > 1."""
    return circulant_from_steps(n, (13,))

def layout_catalogue(cfg) -> list[dict]:
    """Named layouts. Unweighted family aims at degree 4; weighted family uses the same hop set."""
    return [
        dict(key="ring4", title="ring (deg 4)", family="unweighted", kind="ring4"),
        dict(key="torus", title="2D torus", family="unweighted", kind="torus"),
        dict(key="expander", title="random 4-regular", family="unweighted", kind="expander"),
        dict(key="smallworld", title=f"Watts–Strogatz p={cfg.ws_p:g}", family="unweighted", kind="smallworld"),
        dict(key="fib_circ", title="Fibonacci circulant", family="unweighted", kind="fib_circ"),
        dict(key="mod13", title="mod-13 circulant (+ backbone)", family="unweighted", kind="mod13"),
        dict(key="mod13_naive", title="mod-13 only (no backbone)", family="unweighted", kind="mod13_naive"),
        dict(key="fib_w", title=f"Fibonacci-weighted ring (weak={cfg.fib_weak:g})", family="weighted", kind="fib_w"),
        dict(key="aa_below", title=f"AA-weighted ring λ={cfg.aa_lambda_below:g}", family="weighted", kind="aa_below"),
        dict(key="aa_above", title=f"AA-weighted ring λ={cfg.aa_lambda_above:g} (near-vanishing hops)", family="weighted", kind="aa_above"),
    ]

def make_layout(kind: str, n: int, cfg, rng: Generator) -> FloatArr:
    """Build one named wiring. cfg supplies degree and the quasiperiodic parameters."""
    d = cfg.degree
    hops4 = (1, 2)
    if kind == "ring4":
        return ring_adjacency(n, hops4)
    if kind == "torus":
        return torus_adjacency(n)
    if kind == "expander":
        return random_regular_adjacency(n, d, rng)
    if kind == "smallworld":
        return watts_strogatz_adjacency(n, d, cfg.ws_p, rng)
    if kind == "fib_circ":
        return fibonacci_circulant(n, d)
    if kind == "mod13":
        return mod13_circulant(n, d, include_backbone=True)
    if kind == "mod13_naive":
        return mod13_naive(n)
    if kind == "fib_w":
        return fibonacci_weighted_ring(n, hops4, weak=cfg.fib_weak, strong=1.0)
    if kind == "aa_below":
        return aa_weighted_ring(n, cfg.aa_lambda_below, hops4)
    if kind == "aa_above":
        return aa_weighted_ring(n, cfg.aa_lambda_above, hops4)
    raise ValueError(kind)
