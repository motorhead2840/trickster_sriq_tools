"""Execution backends: local classical Gaussian and Braket adapter."""
from __future__ import annotations

from sri.backends.braket_interface import (
    DM1_ARN,
    SV1_ARN,
    BraketJobPlan,
    build_ising_circuit,
    build_smoke_plan,
    credentials_available,
    dry_run_validate,
    emulate_quantum_task,
    estimate_cost_usd,
    resolve_device_arn,
    submit_quantum_task,
)
from sri.backends.classical_gaussian import ClassicalGaussianBackend

__all__ = [
    "DM1_ARN",
    "SV1_ARN",
    "BraketJobPlan",
    "ClassicalGaussianBackend",
    "build_ising_circuit",
    "build_smoke_plan",
    "credentials_available",
    "dry_run_validate",
    "emulate_quantum_task",
    "estimate_cost_usd",
    "resolve_device_arn",
    "submit_quantum_task",
]
