"""Triton kernel for per-shot homodyne estimates."""

import triton
import triton.language as tl


@triton.jit
def kernel(Z, OUT, N: tl.constexpr, SQRT_VS: tl.constexpr, SQRT_VN: tl.constexpr,
           SQRT_ETA: tl.constexpr, LOSS: tl.constexpr, K: tl.constexpr,
           RA: tl.constexpr, RB: tl.constexpr, A: tl.constexpr, B: tl.constexpr,
           CONDITION: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    s = SQRT_VS * tl.load(Z + i, mask, 0)
    n = SQRT_VN * tl.load(Z + N + i, mask, 0)
    ya = SQRT_ETA * (s + n + RA * tl.load(Z + 2 * N + i, mask, 0))
    ya += LOSS * tl.load(Z + 4 * N + i, mask, 0)
    yb = SQRT_ETA * (K * n + RB * tl.load(Z + 3 * N + i, mask, 0))
    yb += LOSS * tl.load(Z + 5 * N + i, mask, 0)
    tl.store(OUT + CONDITION * N + i, A * ya + B * yb, mask)
