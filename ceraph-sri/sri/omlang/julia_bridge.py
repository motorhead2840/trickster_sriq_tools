"""Optional Julia subprocess bridge for Sphota conjugation (binary Float32 I/O)."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_JULIA_SOURCE = Path(__file__).resolve().parent / "julia" / "yoga_buddhi_conjugation.jl"


def julia_available(executable: str | None = None) -> bool:
    return shutil.which(executable or "julia") is not None


def _run_julia(
    mode: str,
    payload: NDArray[np.float32],
    n: int,
    *,
    executable: str = "julia",
) -> NDArray[np.float32]:
    if not julia_available(executable):
        raise RuntimeError("julia executable not found on PATH")
    with tempfile.TemporaryDirectory() as directory:
        inp = Path(directory) / "in.bin"
        out = Path(directory) / "out.bin"
        payload.astype("<f4", copy=False).tofile(inp)
        subprocess.run(
            [
                executable,
                "--startup-file=no",
                str(_JULIA_SOURCE),
                mode,
                str(inp),
                str(out),
                str(n),
            ],
            check=True,
            timeout=120,
        )
        return np.fromfile(out, dtype="<f4")


def oracle_samya_julia(
    value: NDArray[np.floating],
    noise: NDArray[np.floating],
    *,
    executable: str = "julia",
) -> dict[str, NDArray[np.float64]]:
    v = np.asarray(value, dtype=np.float32).reshape(-1)
    xi = np.asarray(noise, dtype=np.float32).reshape(-1)
    if v.size != xi.size:
        raise ValueError("value and noise length mismatch")
    raw = _run_julia("oracle", np.concatenate([v, xi]), v.size, executable=executable)
    n = v.size
    if raw.size != 3 * n:
        raise RuntimeError("julia oracle returned unexpected size")
    return {
        "thesis": raw[:n].astype(np.float64),
        "antithesis": raw[n : 2 * n].astype(np.float64),
        "samya": raw[2 * n :].astype(np.float64),
    }


def observable_upsilon_julia(
    y_a: NDArray[np.floating],
    y_b: NDArray[np.floating],
    weights: NDArray[np.floating],
    *,
    executable: str = "julia",
) -> dict[str, NDArray[np.float64]]:
    ya = np.asarray(y_a, dtype=np.float32).reshape(-1)
    yb = np.asarray(y_b, dtype=np.float32).reshape(-1)
    w = np.asarray(weights, dtype=np.float32).reshape(-1)
    if ya.size != yb.size or w.size != 2:
        raise ValueError("bad shapes for observable julia path")
    payload = np.concatenate([ya, yb, w])
    raw = _run_julia("obs", payload, ya.size, executable=executable)
    n = ya.size
    if raw.size != 3 * n:
        raise RuntimeError("julia obs returned unexpected size")
    return {
        "antithesis": raw[:n].astype(np.float64),
        "projected": raw[n : 2 * n].astype(np.float64),
        "mmse": raw[2 * n :].astype(np.float64),
    }
