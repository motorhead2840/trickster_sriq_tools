"""Julia subprocess bridge for Abhaya damping and the @sphota schedule.

The tested path is little-endian Float32 files, the same pattern as
``julia_bridge.py``. A PythonCall import is not required. When Julia is
absent the callers raise, and the Python reference remains the one the
tests execute.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_JULIA_SOURCE = Path(__file__).resolve().parent / "julia" / "abhaya_bridge.jl"


def julia_available(executable: str | None = None) -> bool:
    return shutil.which(executable or "julia") is not None


def sphota_schedule_julia(
    n: int,
    density: float,
    *,
    executable: str = "julia",
) -> dict[str, int]:
    if not julia_available(executable):
        raise RuntimeError("julia executable not found on PATH")
    done = subprocess.run(
        [
            executable,
            "--startup-file=no",
            str(_JULIA_SOURCE),
            "--schedule",
            str(int(n)),
            str(float(density)),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    block_text, grid_text = done.stdout.split()
    return {"block": int(block_text), "grid": int(grid_text)}


def abhaya_damping_julia(
    theta: NDArray[np.floating],
    xi: NDArray[np.floating],
    lam: float,
    *,
    executable: str = "julia",
) -> NDArray[np.float64]:
    if not julia_available(executable):
        raise RuntimeError("julia executable not found on PATH")
    state = np.asarray(theta, dtype=np.float32).reshape(-1)
    instability = np.asarray(xi, dtype=np.float32).reshape(-1)
    if state.shape != instability.shape:
        raise ValueError("theta and xi length mismatch")
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        inp = folder / "in.bin"
        out = folder / "out.bin"
        np.concatenate([state, instability]).astype("<f4").tofile(inp)
        subprocess.run(
            [
                executable,
                "--startup-file=no",
                str(_JULIA_SOURCE),
                "damp",
                str(inp),
                str(out),
                str(state.size),
                str(float(lam)),
            ],
            check=True,
            timeout=120,
        )
        return np.fromfile(out, dtype="<f4").astype(np.float64)
