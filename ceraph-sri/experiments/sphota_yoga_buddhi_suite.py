"""Sphota-hybrid Yoga–Buddhi validation suite (Braket Hybrid–shaped).

Runs locally by default (NumPy + optional JAX/Julia). With
``--submit-hybrid`` + BRAKET_SUBMIT=1 + BRAKET_S3_BUCKET, packages a Hybrid
Job spec under a cost gate (same pattern as ``braket_hybrid_e1_e4``).

This validates Lemma Z1 / dissertation Υ_YB translation — classical math in a
Hybrid container — not photonic CV on a QPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

from sri.backends.braket_interface import (
    SV1_ARN,
    BraketJobPlan,
    credentials_available,
    dry_run_validate,
    estimate_cost_usd,
)
from sri.omlang.sphota import SphotaCompiler
from sri.registry.conjectures import get, list_conjectures, record_result, reset_registry


def run(
    output_dir: Path,
    *,
    seed: int = 108,
    n_samples: int = 4096,
    prefer_jax: bool = True,
    prefer_julia: bool = True,
    promote_c5: bool = True,
    submit_hybrid: bool = False,
    cost_cap_usd: float = 25.0,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reset_registry()
    # Re-seed C1–C4 as confirmed context for the lock file; C5 starts open.
    for cid in ("C1", "C2", "C3", "C4"):
        record_result(cid, "confirmed", notes="prior foundation lock")

    compiler = SphotaCompiler(
        seed=seed,
        n_samples=n_samples,
        prefer_jax=prefer_jax,
        prefer_julia=prefer_julia,
    )
    report = compiler.crystallize()
    report_dict = report.to_dict()

    c5_notes = (
        f"Sphota suite: oracle_err={report.oracle_samya_max_abs_err:.3e}, "
        f"obs_err={report.observable_equals_mmse_max_abs:.3e}, "
        f"mse_within_5sigma={report.mse_within_5sigma}, "
        f"backends={{{', '.join(k for k, v in report.backends.items() if v.get('ok'))}}}"
    )
    if promote_c5 and report.pass_lemma_z1:
        record_result(
            "C5",
            "confirmed",
            artifact_paths=[str(output_dir / "sphota_yoga_buddhi_summary.json")],
            notes=c5_notes,
        )
    else:
        record_result(
            "C5",
            "open",
            artifact_paths=[str(output_dir / "sphota_yoga_buddhi_summary.json")],
            notes=(
                c5_notes
                if report.pass_lemma_z1
                else "Lemma Z1 checks failed; C5 remains open"
            ),
        )

    plan = BraketJobPlan(
        job_name="sphota-yoga-buddhi",
        device_arn=SV1_ARN,
        shots=max(100, n_samples // 4),
        task="sphota_yoga_buddhi",
        parameters={
            "seed": seed,
            "n_samples": n_samples,
            "encoding": "classical_hybrid_container",
            "note": (
                "Yoga–Buddhi / Lemma Z1 validation in Hybrid-shaped compute. "
                "Not a photonic QPU MMSE claim."
            ),
        },
        cost_cap_usd=cost_cap_usd,
        submit=submit_hybrid,
        s3_bucket=os.environ.get("BRAKET_S3_BUCKET"),
        s3_prefix="sri-braket/sphota-yoga-buddhi",
        region=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )
    validation = dry_run_validate(plan)
    hybrid_spec = {
        "plan": {
            "job_name": plan.job_name,
            "device_arn": plan.device_arn,
            "shots": plan.shots,
            "task": plan.task,
            "parameters": plan.parameters,
            "cost_cap_usd": plan.cost_cap_usd,
            "s3_bucket": plan.s3_bucket,
            "s3_prefix": plan.s3_prefix,
            "region": plan.region,
            "submit": plan.submit,
        },
        "estimate_usd": estimate_cost_usd(plan),
        "dry_run": validation,
        "credentials_available": credentials_available(),
        "submit_requested": submit_hybrid,
        "submit_executed": False,
        "note": (
            "Package local Sphota numerics as Hybrid Job payload. "
            "Live AwsQuantumJob.create requires BRAKET_SUBMIT=1 and credentials."
        ),
    }
    if (
        submit_hybrid
        and validation.get("status") == "dry_run_ok"
        and validation.get("within_cost_cap")
        and credentials_available()
    ):
        if os.environ.get("BRAKET_SUBMIT") == "1":
            hybrid_spec["submit_executed"] = False
            hybrid_spec["note"] = (
                "Live Hybrid submit gated: run this suite inside a Braket notebook / "
                "Hybrid container entrypoint; agent VM may lack job-role wiring. "
                "Local pass_lemma_z1 is the math gate."
            )
        else:
            hybrid_spec["note"] = "BRAKET_SUBMIT!=1; dry-run only"

    summary = {
        "experiment": "sphota_yoga_buddhi",
        "recorded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "platform": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "configuration": {
            "seed": seed,
            "n_samples": n_samples,
            "prefer_jax": prefer_jax,
            "prefer_julia": prefer_julia,
        },
        "sphota": report_dict,
        "conjectures": [
            {
                "conjecture_id": c.conjecture_id,
                "status": c.status,
                "notes": c.notes,
                "statement": c.statement,
                "pass_criterion": c.pass_criterion,
                "artifact_paths": c.artifact_paths,
            }
            for c in list_conjectures()
        ],
        "hybrid": hybrid_spec,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "verdict": {
            "pass_lemma_z1": report.pass_lemma_z1,
            "c5_status": get("C5").status,
        },
    }
    (output_dir / "sphota_yoga_buddhi_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "hybrid_job_spec.json").write_text(
        json.dumps(hybrid_spec, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/benchmark-results/sphota-yoga-buddhi"),
    )
    parser.add_argument("--seed", type=int, default=108)
    parser.add_argument("--n-samples", type=int, default=4096)
    parser.add_argument("--no-jax", action="store_true")
    parser.add_argument("--no-julia", action="store_true")
    parser.add_argument("--no-promote-c5", action="store_true")
    parser.add_argument("--submit-hybrid", action="store_true")
    parser.add_argument("--cost-cap-usd", type=float, default=25.0)
    args = parser.parse_args(argv)
    summary = run(
        args.output_dir,
        seed=args.seed,
        n_samples=args.n_samples,
        prefer_jax=not args.no_jax,
        prefer_julia=not args.no_julia,
        promote_c5=not args.no_promote_c5,
        submit_hybrid=args.submit_hybrid,
        cost_cap_usd=args.cost_cap_usd,
    )
    print(json.dumps(summary["verdict"], indent=2))
    print("pass_lemma_z1:", summary["sphota"]["pass_lemma_z1"])
    print("wrote", args.output_dir / "sphota_yoga_buddhi_summary.json")
    return 0 if summary["verdict"]["pass_lemma_z1"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
