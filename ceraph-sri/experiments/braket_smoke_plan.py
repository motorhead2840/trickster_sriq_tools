"""Build and validate the December Braket first-job payload (local dry-run)."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

from sri.backends.braket_interface import build_smoke_plan, dry_run_validate
from sri.backends.classical_gaussian import ClassicalGaussianBackend
from sri.cv.estimator import SensingConfig
import numpy as np


def run(output_dir: Path, task: str = "cv_sensing_analogue", shots: int = 1000) -> dict:
    plan = build_smoke_plan(task=task, shots=shots)
    validation = dry_run_validate(plan)
    # Local classical shadow of the planned CV task for offline confidence.
    backend = ClassicalGaussianBackend()
    local = backend.run(
        SensingConfig(reference_r=0.7),
        shots=min(shots, 5000),
        rng=np.random.default_rng(108),
    )
    summary = {
        "experiment": "braket_smoke_plan",
        "validation": validation,
        "local_classical_shadow": local,
        "provenance": {
            "python": platform.python_version(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "braket_smoke_plan.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--task",
        choices=["cv_sensing_analogue", "qubit_toy_ising"],
        default="cv_sensing_analogue",
    )
    parser.add_argument("--shots", type=int, default=1000)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.output_dir, task=args.task, shots=args.shots), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
