"""One session that runs NumPy, Julia, JAX, and Triton on the same arrays.

The stack is real only when all three optional tiers import, each result
matches the NumPy map, and Triton was launched with the block size Julia
returned. A missing tier is reported. It is not treated as a match.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from sri.omlang.abhaya import abhaya_damping, sigma_density, sphota_schedule

MATCH_ATOL = 1e-3


def _max_abs(got: Any, ref: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(got, dtype=np.float64) - np.asarray(ref, dtype=np.float64))))


def _close(got: Any, ref: np.ndarray) -> bool:
    return _max_abs(got, ref) <= MATCH_ATOL


def run_stack(n: int = 16384, lam: float = 1.0, seed: int = 11) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=n).astype(np.float32)
    xi = rng.normal(size=n).astype(np.float32)
    reference = abhaya_damping(theta, xi, lam).astype(np.float32)
    density = sigma_density(xi)
    python_schedule = sphota_schedule(n, xi)
    report: dict[str, Any] = {
        "length": n,
        "lam": lam,
        "seed": seed,
        "numpy": {"status": "RAN", "schedule": python_schedule, "density": density},
        "julia": {"status": "MISSING"},
        "jax": {"status": "MISSING"},
        "triton": {"status": "MISSING"},
    }

    block = int(python_schedule["block"])
    try:
        from sri.omlang.abhaya_julia import (
            abhaya_damping_julia,
            julia_available,
            sphota_schedule_julia,
        )

        if julia_available():
            schedule = sphota_schedule_julia(n, density)
            produced = abhaya_damping_julia(theta, xi, lam)
            block = int(schedule["block"])
            report["julia"] = {
                "status": "RAN",
                "schedule": schedule,
                "schedule_matches_python": schedule["block"] == python_schedule["block"],
                "max_abs_error": _max_abs(produced, reference),
                "matches_numpy": _close(produced, reference),
            }
    except Exception as exc:  # noqa: BLE001
        report["julia"] = {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}

    try:
        from sri.omlang import abhaya_jax

        if abhaya_jax.available():
            produced = abhaya_jax.apply_numpy(theta, xi, lam)
            report["jax"] = {
                "status": "RAN",
                "max_abs_error": _max_abs(produced, reference),
                "matches_numpy": _close(produced, reference),
            }
    except Exception as exc:  # noqa: BLE001
        report["jax"] = {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}

    try:
        from sri.omlang import abhaya_triton

        if abhaya_triton.triton_available() and abhaya_triton.cuda_available():
            import torch

            devices = []
            for index in range(torch.cuda.device_count()):
                torch.cuda.set_device(index)
                started = time.perf_counter()
                produced = abhaya_triton.launch_numpy(theta, xi, lam, block=block)
                elapsed_ms = (time.perf_counter() - started) * 1e3
                devices.append(
                    {
                        "device": index,
                        "name": torch.cuda.get_device_name(index),
                        "block": block,
                        "elapsed_ms": elapsed_ms,
                        "max_abs_error": _max_abs(produced, reference),
                        "matches_numpy": _close(produced, reference),
                    }
                )
            report["triton"] = {"status": "RAN", "devices": devices}
        elif not abhaya_triton.cuda_available():
            report["triton"] = {"status": "NO_CUDA"}
    except Exception as exc:  # noqa: BLE001
        report["triton"] = {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}

    julia_ok = report["julia"].get("matches_numpy") is True and report["julia"].get("schedule_matches_python") is True
    jax_ok = report["jax"].get("matches_numpy") is True
    triton_rows = report["triton"].get("devices") or []
    triton_ok = bool(triton_rows) and all(row["matches_numpy"] for row in triton_rows)
    report["block_used"] = block
    report["stack_real"] = bool(julia_ok and jax_ok and triton_ok)
    report["sentence"] = (
        "Julia chose the block, JAX and Triton matched NumPy, and Triton launched on every visible GPU."
        if report["stack_real"]
        else "The stack is not real on this machine until Julia, JAX, and Triton have all matched NumPy."
    )
    return report
