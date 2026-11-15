"""Phase 2a / S4: live SV1 (or DM1) smoke (or local emulate) + classical CV shadow."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np

from sri.backends.braket_interface import (
    build_smoke_plan,
    credentials_available,
    dry_run_validate,
    emulate_quantum_task,
    estimate_cost_usd,
    resolve_device_arn,
    submit_quantum_task,
)
from sri.backends.classical_gaussian import ClassicalGaussianBackend
from sri.cv.estimator import SensingConfig


def run(
    output_dir: Path,
    *,
    live: bool = False,
    emulate: bool = True,
    shots: int = 1000,
    cost_cap_usd: float = 5.0,
    device: str = "sv1",
    n_qubits: int = 4,
) -> dict:
    device_arn = resolve_device_arn(device=device)
    plan = build_smoke_plan(
        task="qubit_toy_ising",
        device=device,
        device_arn=device_arn,
        shots=shots,
        cost_cap_usd=cost_cap_usd,
        submit=live,
        s3_bucket=os.environ.get("BRAKET_S3_BUCKET"),
        n_qubits=n_qubits,
    )
    validation = dry_run_validate(plan)
    classical = ClassicalGaussianBackend().run(
        SensingConfig(reference_r=0.7),
        shots=min(shots, 5000),
        rng=np.random.default_rng(108),
    )

    quantum: dict
    if live:
        if not credentials_available():
            raise RuntimeError(
                "AWS credentials not available. Complete docs/aws-braket-setup.md "
                "or pass --emulate for local plumbing."
            )
        os.environ["BRAKET_SUBMIT"] = "1"
        plan.submit = True
        quantum = submit_quantum_task(plan, poll=True)
    elif emulate:
        quantum = emulate_quantum_task(plan)
    else:
        quantum = {"status": "skipped", "mode": "dry_run_only", "plan": validation}

    artifact_name = "dm1_smoke.json" if "dm1" in device_arn.lower() else "sv1_smoke.json"
    summary = {
        "experiment": "braket_live_smoke",
        "phase": "S4",
        "device": device,
        "device_arn": device_arn,
        "live": live,
        "validation": validation,
        "estimated_cost_usd": estimate_cost_usd(plan),
        "quantum_task": quantum,
        "classical_cv_shadow": classical,
        "credentials_available": credentials_available(),
        "provenance": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "disclaimer": (
            "Qubit Ising smoke is hardware/simulator contact only. "
            "Classical CV shadow is Experiment Set 1 sensing math — not a QPU result. "
            "DM1 noise simulations remain qubit plumbing, not CV MMSE."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / artifact_name).write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--live", action="store_true", help="Submit to Braket SV1/DM1")
    parser.add_argument(
        "--emulate",
        action="store_true",
        default=True,
        help="Local emulate when not --live (default)",
    )
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--cost-cap-usd", type=float, default=5.0)
    parser.add_argument(
        "--device",
        choices=["sv1", "dm1"],
        default="sv1",
        help="Simulator alias (default sv1). DM1 for density-matrix/noise path.",
    )
    parser.add_argument("--n-qubits", type=int, default=4)
    args = parser.parse_args(argv)
    if args.live:
        args.emulate = False
    summary = run(
        args.output_dir,
        live=args.live,
        emulate=args.emulate,
        shots=args.shots,
        cost_cap_usd=args.cost_cap_usd,
        device=args.device,
        n_qubits=args.n_qubits,
    )
    print(json.dumps(summary, indent=2))
    status = summary["quantum_task"].get("status")
    if status not in {"COMPLETED", "skipped"}:
        raise SystemExit(f"quantum task status={status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
