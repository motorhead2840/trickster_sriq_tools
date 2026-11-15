"""Triton kernel for Abhaya damping.

The kernel computes Θ · exp(-λ · Ξ) in one launch. Each program loads a tile
of Θ and Ξ, applies the exponential, and stores A. The tile lives in SRAM for
that program; the loads and the store still use device memory. This module
imports Triton only when a caller asks to build or launch the kernel.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from sri.omlang.abhaya import abhaya_damping, sphota_schedule


def triton_available() -> bool:
    try:
        import triton  # noqa: F401
        import triton.language  # noqa: F401
    except ImportError:
        return False
    return True


def cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def build_kernel() -> Any:
    """Return the ``@triton.jit`` kernel. Raises if Triton is not installed."""
    import triton
    import triton.language as tl

    @triton.jit
    def abhaya_damping_kernel(
        theta_ptr,
        xi_ptr,
        out_ptr,
        n_elements,
        lam,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        theta = tl.load(theta_ptr + offsets, mask=mask)
        xi = tl.load(xi_ptr + offsets, mask=mask)
        damped = theta * tl.exp(-lam * xi)
        tl.store(out_ptr + offsets, damped, mask=mask)

    return abhaya_damping_kernel


def launch(theta: Any, xi: Any, lam: float, block: int | None = None) -> Any:
    """Run the kernel on the current CUDA device. Returns a CUDA tensor.

    ``block`` is the width chosen by Julia ``@sphota`` or by ``sphota_schedule``.
    When it is omitted, the Python ladder supplies it.
    """
    import math

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to launch the Abhaya Triton kernel")
    device = torch.device("cuda", torch.cuda.current_device())
    state = torch.as_tensor(theta, dtype=torch.float32, device=device).reshape(-1).contiguous()
    instability = torch.as_tensor(xi, dtype=torch.float32, device=device).reshape(-1).contiguous()
    if state.shape != instability.shape:
        raise ValueError("theta and xi length mismatch")
    if block is None:
        block = sphota_schedule(int(state.numel()), instability.detach().cpu().numpy())["block"]
    if block < 1:
        raise ValueError("block must be positive")
    grid = math.ceil(int(state.numel()) / int(block))
    out = torch.empty_like(state)
    kernel = build_kernel()
    kernel[(grid,)](
        state,
        instability,
        out,
        state.numel(),
        float(lam),
        BLOCK=int(block),
    )
    return out


def launch_numpy(theta: Any, xi: Any, lam: float, block: int | None = None) -> np.ndarray:
    """Launch on CUDA and copy the result back. Falls back nowhere: CUDA is required."""
    return launch(theta, xi, lam, block=block).detach().cpu().numpy()


def reference_matches_formula(theta: Any, xi: Any, lam: float) -> np.ndarray:
    """CPU twin used when the kernel cannot be launched."""
    return abhaya_damping(theta, xi, lam)
