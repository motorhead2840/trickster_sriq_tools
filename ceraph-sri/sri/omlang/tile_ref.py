"""Tile-shaped NumPy references for Yoga–Buddhi ops (cuTile translation baseline).

These mirror the *structure* of tile kernels NVIDIA ports with the
Python→Rust agent skill: explicit shapes, element-wise / row tiles, no
hidden JIT specialization. They are the ``analysis.json`` ground truth
until a cuTile Python reference exists.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArr = NDArray[np.floating[Any]]


def oracle_samya_tile(thesis: FloatArr, antithesis: FloatArr) -> FloatArr:
    """Tile op: Samya = ½ (thesis ⊕ antithesis). Latent / oracle chart."""
    t = np.asarray(thesis, dtype=np.float64)
    a = np.asarray(antithesis, dtype=np.float64)
    if t.shape != a.shape:
        raise ValueError("thesis/antithesis shape mismatch")
    return 0.5 * (t + a)


def observable_upsilon_tile(
    y_a: FloatArr,
    y_b: FloatArr,
    w_a: float,
    w_b: float,
) -> tuple[FloatArr, FloatArr]:
    """Tile op: antithesis + Υ_obs for Lemma Z1 (explicit weights in signature)."""
    ya = np.asarray(y_a, dtype=np.float64)
    yb = np.asarray(y_b, dtype=np.float64)
    if ya.shape != yb.shape:
        raise ValueError("y_a/y_b shape mismatch")
    antithesis = (2.0 * float(w_a) - 1.0) * ya + (2.0 * float(w_b)) * yb
    projected = 0.5 * (ya + antithesis)
    return antithesis, projected


def mirror_abs2_tile(real: FloatArr, imag: FloatArr) -> FloatArr:
    """Tile op: Mirror M(C)=|C|² on split real/imag channels."""
    r = np.asarray(real, dtype=np.float64)
    i = np.asarray(imag, dtype=np.float64)
    if r.shape != i.shape:
        raise ValueError("real/imag shape mismatch")
    return r * r + i * i


def analysis_spec() -> dict[str, Any]:
    """Machine-readable analyzer artifact (TileGym-style analysis.json core)."""
    return {
        "source": "sri.omlang.tile_ref",
        "methodology": "nvidia-cutile-python-to-rust-agentic",
        "operators": [
            {
                "name": "oracle_samya_tile",
                "dtype": ["f32", "f64"],
                "kind": "elementwise_binary",
                "tolerance_abs": 1e-6,
            },
            {
                "name": "observable_upsilon_tile",
                "dtype": ["f32", "f64"],
                "kind": "elementwise_binary_weighted",
                "tolerance_rel": 1e-10,
                "forbids_inputs": ["clean_s", "oracle_n"],
            },
            {
                "name": "mirror_abs2_tile",
                "dtype": ["f32", "f64"],
                "kind": "elementwise_binary",
                "tolerance_abs": 1e-12,
            },
        ],
        "perf_gate": {
            "metric": "cupti_device_time_geomean",
            "threshold_vs_reference": 0.95,
            "status": "not_run",
        },
    }
