"""V3 damping dynamics and CPTP attenuators."""
from __future__ import annotations

from sri.dynamics.v3_damping import (
    Branch,
    DampingParameters,
    DampingResult,
    apply_logit_damping,
    cptp_attenuate_covariance,
    v3_damping,
)

__all__ = [
    "Branch",
    "DampingParameters",
    "DampingResult",
    "apply_logit_damping",
    "cptp_attenuate_covariance",
    "v3_damping",
]
