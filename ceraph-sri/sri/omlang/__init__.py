"""Sphota-hybrid Ω-lang: Python control + optional JAX tensors + optional Julia conjugation.

Sphota (स्फोट) = the burst that crystallizes meaning into executable form.
This package translates dissertation Yoga–Buddhi algebra into runnable backends.
"""

from sri.omlang.abhaya import abhaya_damping, omega_dit, sphota_schedule
from sri.omlang.sphota import SphotaCompiler, SphotaReport
from sri.omlang.tile_ref import (
    analysis_spec,
    mirror_abs2_tile,
    observable_upsilon_tile,
    oracle_samya_tile,
)
from sri.omlang.yoga_buddhi import (
    mirror_validation,
    oracle_samya,
    observable_upsilon,
)

__all__ = [
    "SphotaCompiler",
    "SphotaReport",
    "abhaya_damping",
    "omega_dit",
    "sphota_schedule",
    "analysis_spec",
    "mirror_abs2_tile",
    "mirror_validation",
    "observable_upsilon",
    "observable_upsilon_tile",
    "oracle_samya",
    "oracle_samya_tile",
]
