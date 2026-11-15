"""Phase 2b / S5: Hybrid Job-shaped runner for Experiment Set 1 (E1–E4).

Default mode runs locally (same numerics a Braket Hybrid Job container would
execute), optionally bundling S2 advantage + S3 efficiency suites.

With --submit-hybrid and AWS credentials + BRAKET_SUBMIT=1 + BRAKET_S3_BUCKET,
attempts AwsQuantumJob.create under a hard cost gate; otherwise records a
packaged job spec for manual launch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
from pathlib import Path

import numpy as np

from experiments.e1_colab_lock import run as run_e1_lock
from sri.backends.braket_interface import (
    SV1_ARN,
    build_smoke_plan,
    credentials_available,
    estimate_cost_usd,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_locks(reference: dict, hybrid: dict, tol_mse: float = 0.02, tol_xi: float = 1e-6) -> dict:
    """Compare observable MSE and Ξ_SNR ranges between A100/CPU lock and hybrid."""
    ref_obs = reference["aggregates"]["observable"]
    hyb_obs = hybrid["aggregates"]["observable"]
    ref_f = reference["aggregates"]["foundation"]
    hyb_f = hybrid["aggregates"]["foundation"]
    mse_delta = abs(
        ref_obs["mean_mmse_sample_mse"] - hyb_obs["mean_mmse_sample_mse"]
    )
    checks = {
        "mse_delta": mse_delta,
        "mse_within_tol": mse_delta <= tol_mse,
        "xi_snr_min_delta": abs(ref_f["xi_snr_min"] - hyb_f["xi_snr_min"]),
        "xi_snr_max_delta": abs(ref_f["xi_snr_max"] - hyb_f["xi_snr_max"]),
        "xi_within_tol": (
            abs(ref_f["xi_snr_min"] - hyb_f["xi_snr_min"]) <= tol_xi
            and abs(ref_f["xi_snr_max"] - hyb_f["xi_snr_max"]) <= tol_xi
        ),
        "entropy_ok_both": (
            ref_f["entropy_nonincreasing_under_cptp"]
            and hyb_f["entropy_nonincreasing_under_cptp"]
        ),
    }
    checks["pass"] = (
        checks["mse_within_tol"] and checks["xi_within_tol"] and checks["entropy_ok_both"]
    )
    return checks


def _run_campaign_bundle(hybrid_root: Path, *, shots: int, seeds: list[int], dimension: int) -> dict:
    """Embed S2 advantage + S3 efficiency into the Hybrid artifact tree."""
    from experiments.braket_advantage_suite import run as run_advantage
    from experiments.braket_efficiency_suite import run as run_efficiency

    advantage = run_advantage(
        hybrid_root / "campaign" / "advantage",
        shots=shots,
        seeds=seeds,
        campaign_shots=0,
        confirmation_subset=True,
    )
    # Keep Hybrid wall-clock modest: efficiency uses the lock dimension neighborhood.
    efficiency = run_efficiency(
        hybrid_root / "campaign" / "efficiency",
        dimensions=[max(256, dimension // 4), dimension, min(dimension * 4, 65536)],
        shots=min(shots, 5000),
        seeds=seeds[:1],
    )
    return {
        "advantage_pass": bool(advantage.get("pass")),
        "efficiency_pass": bool(efficiency.get("pass")),
        "advantage_path": str(hybrid_root / "campaign" / "advantage" / "advantage_summary.json"),
        "efficiency_path": str(hybrid_root / "campaign" / "efficiency" / "efficiency_summary.json"),
    }


def run_local_hybrid(
    output_dir: Path,
    *,
    shots: int = 20000,
    dimension: int = 4096,
    seeds: list[int] | None = None,
    reference_lock: Path | None = None,
    include_campaign_bundle: bool = True,
) -> dict:
    if seeds is None:
        seeds = [108, 109, 110]
    hybrid_root = output_dir / "e1_e4_hybrid"
    lock = run_e1_lock(
        hybrid_root,
        shots=shots,
        seeds=seeds,
        dimension=dimension,
        require_gpu=False,
    )
    # Move lock file to expected name for Braket artifact layout.
    shutil.copy2(hybrid_root / "e1_colab_lock.json", hybrid_root / "hybrid_lock.json")

    comparison = None
    if reference_lock and reference_lock.is_file():
        comparison = compare_locks(_load_json(reference_lock), lock)

    campaign_bundle = None
    if include_campaign_bundle:
        campaign_bundle = _run_campaign_bundle(
            hybrid_root, shots=shots, seeds=seeds, dimension=dimension
        )

    plan = build_smoke_plan(task="hybrid_e1_e4", shots=1, cost_cap_usd=25.0, submit=False)
    summary = {
        "experiment": "braket_hybrid_e1_e4",
        "phase": "S5",
        "mode": "local_hybrid_container_equivalent",
        "plan": {
            "task": plan.task,
            "cost_cap_usd": plan.cost_cap_usd,
            "estimated_cost_usd": estimate_cost_usd(plan),
            "parameters": plan.parameters,
        },
        "lock": lock,
        "comparison_to_reference": comparison,
        "campaign_bundle": campaign_bundle,
        "credentials_available": credentials_available(),
        "note": (
            "Numerics are identical to the Colab/CPU E1 lock. A live AwsQuantumJob "
            "submit is attempted when --submit-hybrid is used with BRAKET_SUBMIT=1, "
            "credentials, and BRAKET_S3_BUCKET."
        ),
        "provenance": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hybrid_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def package_hybrid_job_spec(output_dir: Path) -> dict:
    """Write a launch spec the user can submit from Colab with braket AwsQuantumJob."""
    spec = {
        "job_name": "sri-e1-e4-hybrid",
        "entry_point": "experiments/braket_hybrid_e1_e4.py",
        "args": [
            "--output-dir",
            "/opt/ml/model",
            "--local-hybrid",
            "--include-campaign-bundle",
            "--shots",
            "20000",
            "--dimension",
            "4096",
        ],
        "cost_cap_usd": 25.0,
        "instance": "ml.m5.large",
        "source_module": "experiments",
        "campaign_bundle": [
            "experiments/braket_advantage_suite.py",
            "experiments/braket_efficiency_suite.py",
        ],
        "environment": {
            "BRAKET_SUBMIT": "0",
        },
        "instructions": (
            "From an environment with AWS credentials: "
            "BRAKET_SUBMIT=1 BRAKET_S3_BUCKET=... python experiments/braket_hybrid_e1_e4.py "
            "--output-dir ... --submit-hybrid. Automated create uses AwsQuantumJob.create "
            "when the SDK is available; otherwise use this packaged spec manually. "
            "Do not exceed cost_cap_usd=$25 for this job."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hybrid_job_spec.json").write_text(
        json.dumps(spec, indent=2) + "\n", encoding="utf-8"
    )
    return spec


def try_submit_hybrid(output_dir: Path, *, wait: bool = False) -> dict:
    """Attempt AwsQuantumJob.create under live gates; fall back to packaging the spec."""
    if os.environ.get("BRAKET_SUBMIT", "0") != "1":
        return {
            "status": "not_submitted",
            "reason": "BRAKET_SUBMIT!=1",
            "spec": package_hybrid_job_spec(output_dir),
        }
    s3_bucket = os.environ.get("BRAKET_S3_BUCKET")
    if not s3_bucket:
        return {
            "status": "not_submitted",
            "reason": "BRAKET_S3_BUCKET unset",
            "spec": package_hybrid_job_spec(output_dir),
        }
    if not credentials_available():
        return {
            "status": "not_submitted",
            "reason": "no_aws_credentials",
            "spec": package_hybrid_job_spec(output_dir),
        }

    plan = build_smoke_plan(
        task="hybrid_e1_e4",
        shots=1,
        cost_cap_usd=25.0,
        submit=True,
        s3_bucket=s3_bucket,
    )
    estimated = estimate_cost_usd(plan)
    if estimated > plan.cost_cap_usd:
        return {
            "status": "not_submitted",
            "reason": f"estimated ${estimated:.4f} exceeds cap ${plan.cost_cap_usd:.4f}",
            "spec": package_hybrid_job_spec(output_dir),
        }

    spec = package_hybrid_job_spec(output_dir)
    repo_root = Path(__file__).resolve().parents[1]
    try:
        from braket.aws import AwsQuantumJob
    except Exception as error:  # noqa: BLE001
        return {
            "status": "not_submitted",
            "reason": f"braket_sdk_unavailable: {error}",
            "spec": spec,
        }

    create_kwargs = {
        # Classical E1–E4 numerics run in the Hybrid container; SV1 is the
        # declared device ARN for job routing (no CV MMSE on the simulator).
        "device": SV1_ARN,
        "source_module": str(repo_root / "experiments"),
        "entry_point": "braket_hybrid_e1_e4.py:main",
        "job_name": spec["job_name"],
        "hyperparameters": {
            "shots": "20000",
            "dimension": "4096",
            "include_campaign_bundle": "1",
        },
        "wait_until_complete": wait,
    }
    try:
        from braket.jobs.config import InstanceConfig

        create_kwargs["instance_config"] = InstanceConfig(instanceType=spec["instance"])
    except Exception:  # noqa: BLE001 - older SDKs may lack InstanceConfig
        pass

    try:
        create_kwargs["output_data_config"] = {
            "s3Path": f"s3://{s3_bucket}/sri-braket/hybrid-e1-e4"
        }
    except Exception:  # noqa: BLE001
        pass

    try:
        job = AwsQuantumJob.create(**create_kwargs)
        job_arn = getattr(job, "arn", None) or getattr(job, "id", None) or str(job)
        return {
            "status": "submitted",
            "job_arn": job_arn,
            "estimated_cost_usd": estimated,
            "cost_cap_usd": plan.cost_cap_usd,
            "device_arn": SV1_ARN,
            "spec": spec,
            "note": (
                "Hybrid Job submitted. Container runs classical CV E1–E4 + campaign "
                "bundle; SV1 ARN is routing only."
            ),
        }
    except TypeError as error:
        # Retry with a minimal kwargs set for SDK variance.
        try:
            job = AwsQuantumJob.create(
                device=SV1_ARN,
                source_module=str(repo_root / "experiments"),
                entry_point="braket_hybrid_e1_e4.py:main",
                wait_until_complete=wait,
            )
            job_arn = getattr(job, "arn", None) or getattr(job, "id", None) or str(job)
            return {
                "status": "submitted",
                "job_arn": job_arn,
                "estimated_cost_usd": estimated,
                "cost_cap_usd": plan.cost_cap_usd,
                "spec": spec,
                "note": f"Submitted with minimal kwargs after TypeError: {error}",
            }
        except Exception as retry_error:  # noqa: BLE001
            return {
                "status": "spec_ready_manual_launch",
                "reason": (
                    f"AwsQuantumJob.create failed ({retry_error}); packaged "
                    "hybrid_job_spec.json for manual launch. No surprise spend."
                ),
                "estimated_cost_usd": estimated,
                "spec": spec,
            }
    except Exception as error:  # noqa: BLE001
        # Keep spend-safe: if create fails (permissions, API shape), ship the spec.
        return {
            "status": "spec_ready_manual_launch",
            "reason": (
                f"AwsQuantumJob.create failed ({error}); packaged hybrid_job_spec.json "
                "for manual launch from Colab/CLI. No surprise spend beyond this attempt."
            ),
            "estimated_cost_usd": estimated,
            "spec": spec,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--local-hybrid", action="store_true", default=True)
    parser.add_argument("--submit-hybrid", action="store_true")
    parser.add_argument("--wait", action="store_true", help="Wait for Hybrid Job completion")
    parser.add_argument("--shots", type=int, default=20000)
    parser.add_argument("--dimension", type=int, default=4096)
    parser.add_argument(
        "--include-campaign-bundle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run S2 advantage + S3 efficiency inside the hybrid artifact tree",
    )
    parser.add_argument(
        "--reference-lock",
        type=Path,
        default=Path("docs/benchmark-results/e1_colab_lock.json"),
    )
    args = parser.parse_args(argv)

    summary = run_local_hybrid(
        args.output_dir,
        shots=args.shots,
        dimension=args.dimension,
        reference_lock=args.reference_lock if args.reference_lock.is_file() else None,
        include_campaign_bundle=args.include_campaign_bundle,
    )
    if args.submit_hybrid:
        summary["hybrid_submit"] = try_submit_hybrid(args.output_dir, wait=args.wait)
    else:
        summary["hybrid_submit"] = {
            "status": "spec_only",
            "spec": package_hybrid_job_spec(args.output_dir),
        }
    (args.output_dir / "hybrid_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps({k: summary[k] for k in summary if k != "lock"}, indent=2))
    comparison = summary.get("comparison_to_reference")
    if comparison is not None and not comparison.get("pass", False):
        raise SystemExit(f"hybrid vs reference comparison failed: {comparison}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
