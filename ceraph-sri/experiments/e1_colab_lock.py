"""Phase 1: compress Experiment Set 1 (E1–E4) into one lock run.

Runs observable-only + foundation sweep, freezes Circuit-B Branch W as the
research default, merges conjecture statuses, and records device provenance.
On Colab A100, pass --require-gpu. CPU hosts still produce a valid lock with
device metadata (not a substitute for the A100 confirmation artifact).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.foundation_axioms_sweep import run as run_foundation
from experiments.merge_conjecture_status import merge
from experiments.observable_only_benchmark import run as run_observable
from sri.dynamics.v3_damping import Branch


def _detect_devices() -> dict:
    info = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "jax_default_backend": None,
        "gpu_devices": [],
    }
    try:
        import jax

        info["jax_default_backend"] = jax.default_backend()
        info["gpu_devices"] = [
            f"{d.platform}:{d.device_kind}"
            for d in jax.devices()
            if d.platform in ("gpu", "cuda", "rocm")
        ]
    except Exception as error:  # noqa: BLE001
        info["jax_error"] = str(error)
    return info


def run(
    output_root: Path,
    *,
    shots: int = 20000,
    seeds: list[int] | None = None,
    dimension: int = 4096,
    require_gpu: bool = False,
) -> dict:
    if seeds is None:
        seeds = [108, 109, 110]
    devices = _detect_devices()
    if require_gpu and not devices["gpu_devices"]:
        raise RuntimeError(
            f"--require-gpu set but no JAX GPU found (backend={devices.get('jax_default_backend')})"
        )

    obs_dir = output_root / "cv-observable-a100"
    found_dir = output_root / "foundation-a100"
    obs = run_observable(obs_dir, seeds=seeds, shots=shots)
    found = run_foundation(found_dir, seeds=seeds + [111, 112], dimension=dimension)

    # Equal-resource note for E1 (MMSE uses y_B; filters see only y_A).
    obs["equal_resource_note"] = (
        "MMSE uses two optical modes (signal+reference). Moving-average/median "
        "baselines use y_A only. Prefer single_mode_mse for same-mode comparisons; "
        "two-mode MMSE vs coherent two-mode analytic is the physics-aligned contest."
    )
    (obs_dir / "observable_only_summary.json").write_text(
        json.dumps(obs, indent=2) + "\n", encoding="utf-8"
    )

    merge_path = output_root / "conjecture_status.json"
    merged = merge(
        [
            obs_dir / "observable_only_summary.json",
            found_dir / "foundation_axioms_summary.json",
        ],
        merge_path,
    )

    # Entropy ledger extract for E4 (compact).
    records_path = found_dir / "foundation_axioms_records.json"
    ledger_rows = []
    if records_path.is_file():
        records = json.loads(records_path.read_text(encoding="utf-8"))
        for row in records:
            if row.get("branch") == "W" and row.get("seed") == seeds[0]:
                ledger_rows.append(
                    {
                        "cycle": row["cycle"],
                        "noise_std": row["noise_std"],
                        "chi": row["chi"],
                        "entropy_before": row["entropy_before"],
                        "entropy_after": row["entropy_after"],
                        "xi_snr": row["xi_snr"],
                    }
                )
    ledger_path = found_dir / "entropy_ledger_branch_w.json"
    ledger_path.write_text(json.dumps(ledger_rows, indent=2) + "\n", encoding="utf-8")

    branch_default = Branch.WORKED.value
    lock = {
        "experiment": "e1_colab_lock",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "circuit_b_research_default": branch_default,
        "devices": devices,
        "configuration": {
            "shots": shots,
            "seeds": seeds,
            "dimension": dimension,
            "require_gpu": require_gpu,
        },
        "conjectures": merged["conjectures"],
        "aggregates": {
            "observable": obs.get("aggregate"),
            "foundation": found.get("aggregate"),
        },
        "paths": {
            "observable": str(obs_dir),
            "foundation": str(found_dir),
            "conjecture_status": str(merge_path),
            "entropy_ledger": str(ledger_path),
        },
        "colab_exit_criteria": {
            "c1_c4_confirmed": all(
                c["status"] == "confirmed"
                for c in merged["conjectures"]
                if c["conjecture_id"] in {"C1", "C2", "C3", "C4"}
            ),
            "xi_snr_in_unit_interval": (
                found["aggregate"]["xi_snr_min"] > 0
                and found["aggregate"]["xi_snr_max"] <= 1.0 + 1e-12
            ),
            "branch_w_preferred": branch_default == "W",
            "entropy_nonincreasing": found["aggregate"]["entropy_nonincreasing_under_cptp"],
        },
        "provenance": {
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "e1_colab_lock.json").write_text(
        json.dumps(lock, indent=2) + "\n", encoding="utf-8"
    )

    # Convenience copies into docs/benchmark-results layout when writing there.
    docs_root = Path("docs/benchmark-results")
    if output_root.resolve() == docs_root.resolve() or "benchmark-results" in str(output_root):
        target_status = docs_root / "conjecture_status.json"
        if merge_path.resolve() != target_status.resolve():
            shutil.copy2(merge_path, target_status)
    return lock


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/benchmark-results"),
    )
    parser.add_argument("--shots", type=int, default=20000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[108, 109, 110])
    parser.add_argument("--dimension", type=int, default=4096)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args(argv)
    lock = run(
        args.output_dir,
        shots=args.shots,
        seeds=list(args.seeds),
        dimension=args.dimension,
        require_gpu=args.require_gpu,
    )
    print(json.dumps(lock, indent=2))
    criteria = lock["colab_exit_criteria"]
    if not all(criteria.values()):
        raise SystemExit(f"Colab exit criteria failed: {criteria}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
