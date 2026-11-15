"""Numerics for the SRI tile architecture.

Tiles are integrated-circuit computing or sensing units. A veil is the unknown
per-tile complex gain and phase. A reference tile is a tile whose veil is
known and can pin the global factor that neighbour comparisons cannot see.

NumPy is the only array backend in this build. Closed-form modules bind their
arithmetic to ``xp``, which is NumPy, so a later optional GPU backend can
rebind it. Nothing in the public package imports ``sri.legacy``.
"""
from __future__ import annotations

from sri import (
    cv,
    dynamics,
    errorlaw,
    gauge,
    graphs,
    metrics,
    registry,
    selection,
    tracking,
    unveiling,
    wiring_quasi,
)

__all__ = [
    "cv",
    "dynamics",
    "errorlaw",
    "gauge",
    "graphs",
    "metrics",
    "registry",
    "selection",
    "tracking",
    "unveiling",
    "wiring_quasi",
]
