"""Shim. The CPU reference model lives in ``sri.legacy.simulator``."""
from sri.legacy.simulator import (
    BOLTZMANN_CONSTANT,
    DampingParameters,
    DampingResult,
    OmegaDit,
    Projection,
    SimulationConfig,
    damp_logits,
    integrate_omega,
    landauer_minimum_heat,
    main,
    mean_variance,
    simulate,
    stability_coefficient,
    symplectic_project,
    v3_damping,
    write_telemetry,
)

__all__ = [
    "BOLTZMANN_CONSTANT",
    "DampingParameters",
    "DampingResult",
    "OmegaDit",
    "Projection",
    "SimulationConfig",
    "damp_logits",
    "integrate_omega",
    "landauer_minimum_heat",
    "main",
    "mean_variance",
    "simulate",
    "stability_coefficient",
    "symplectic_project",
    "v3_damping",
    "write_telemetry",
]


if __name__ == "__main__":
    raise SystemExit(main())
