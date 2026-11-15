"""Elaborate Braket Hybrid validation: Colab oracle fixtures + Lemma Z1 + Sphota.

Designed for a Braket notebook / Hybrid container after the Colab evidence audit:

* Replays **oracle Samya** (Untitled10 known-baseline cancellation) as regression.
* Stress-tests **Lemma Z1 / C5** (observable antithesis; never feeds s or n).
* Cross-checks Sphota backends (NumPy, tile_ref, optional JAX/Julia).
* Emits theory-hook metrics (latent↔observable antithesis distance, softwalls)
  so A0–A5 / Theorem Zero can be extended from measured artifacts — not vibes.

Default: local Hybrid-shaped run (no AWS spend). Live packaging gated by
``BRAKET_SUBMIT=1`` + ``BRAKET_S3_BUCKET`` + ``--submit-hybrid``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from sri.backends.braket_interface import (
    SV1_ARN,
    BraketJobPlan,
    credentials_available,
    dry_run_validate,
    estimate_cost_usd,
)
from sri.cv.estimator import (
    SensingConfig,
    analytic_mse,
    estimate_signal,
    mmse_coefficients,
    observable_antithesis,
    observable_yoga_buddhi,
    oracle_paired_projection,
    simulate_measurements,
)
from sri.omlang import jax_backend, julia_bridge, tile_ref
from sri.omlang.sphota import SphotaCompiler
from sri.omlang.yoga_buddhi import (
    latent_involution_holds,
    mirror_validation,
    observable_upsilon,
    oracle_samya,
)
from sri.registry.conjectures import get, list_conjectures, record_result, reset_registry
from sri.sree.moe_te_config import mentor_te_moe_plan


@dataclass(frozen=True)
class ConfigPoint:
    name: str
    config: SensingConfig


DEFAULT_CONFIG_GRID: tuple[ConfigPoint, ...] = (
    ConfigPoint("default_r07", SensingConfig(reference_r=0.7)),
    ConfigPoint("coherent", SensingConfig(reference_r=0.0, signal_r=0.0)),
    ConfigPoint("sq_signal", SensingConfig(signal_r=0.7, reference_r=0.0)),
    ConfigPoint("low_eta", SensingConfig(efficiency=0.5, reference_r=0.7)),
    ConfigPoint("strong_noise", SensingConfig(noise_variance=2.0, reference_r=0.7)),
    ConfigPoint("weak_signal", SensingConfig(signal_variance=0.25, reference_r=0.7)),
)


def _devices() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "jax_available": jax_backend.available(),
        "jax_default_backend": None,
        "jax_devices": [],
        "julia_available": julia_bridge.julia_available(),
        "braket_sdk": None,
        "rustc": None,
        "cutile_stub_path": str(Path("embodiment/cutile_rs_stub").resolve()),
    }
    try:
        import jax

        info["jax_default_backend"] = jax.default_backend()
        info["jax_devices"] = [str(d) for d in jax.devices()]
    except Exception as exc:  # noqa: BLE001
        info["jax_error"] = str(exc)
    try:
        import braket  # type: ignore

        info["braket_sdk"] = getattr(braket, "__version__", "present")
    except Exception as exc:  # noqa: BLE001
        info["braket_error"] = str(exc)
    try:
        proc = subprocess.run(
            ["rustc", "--version"], capture_output=True, text=True, check=False
        )
        if proc.returncode == 0:
            info["rustc"] = proc.stdout.strip()
    except Exception as exc:  # noqa: BLE001
        info["rustc_error"] = str(exc)
    return info


def check_colab_oracle_fixture(
    *,
    seeds: list[int],
    dimensions: list[int],
    var_tol: float = 1e-24,
) -> dict[str, Any]:
    """Untitled10 pattern: known baseline → conjugate → Samya residual ~0."""
    rows: list[dict[str, Any]] = []
    all_ok = True
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for n in dimensions:
            baseline = rng.normal(0.0, 1.0, size=n)
            flux = rng.normal(0.0, 0.35, size=n)
            active = baseline + flux
            # Colab Sphota cell: conjugate = baseline - (active - baseline)
            conjugate = baseline - (active - baseline)
            final = 0.5 * (active + conjugate)
            residual = final - baseline
            var = float(np.var(residual))
            max_abs = float(np.max(np.abs(residual)))
            # Also via package oracle_samya / paired projection
            pack = oracle_samya(baseline, flux)
            paired = oracle_paired_projection(baseline, flux)
            pack_ok = bool(np.allclose(pack["samya"], baseline, atol=1e-12))
            paired_ok = bool(np.allclose(paired, baseline, atol=1e-12))
            involution = latent_involution_holds(baseline, flux)
            ok = var <= var_tol and max_abs < 1e-12 and pack_ok and paired_ok and involution
            all_ok = all_ok and ok
            rows.append(
                {
                    "seed": seed,
                    "n": n,
                    "residual_var": var,
                    "residual_max_abs": max_abs,
                    "package_oracle_ok": pack_ok,
                    "paired_projection_ok": paired_ok,
                    "latent_involution": involution,
                    "pass": ok,
                }
            )
    return {
        "name": "colab_oracle_samya_fixture",
        "source": "Untitled10 known-baseline / dissertation Samya",
        "pass": all_ok,
        "rows": rows,
        "note": (
            "Proves oracle chart only. Does not prove blind antithesis or hardware O(0)."
        ),
    }


def check_lemma_z1_grid(
    *,
    seeds: list[int],
    n_samples: int,
    config_grid: tuple[ConfigPoint, ...] = DEFAULT_CONFIG_GRID,
) -> dict[str, Any]:
    """Observable Υ ≡ ŝ ≡ estimate_signal; MSE within 5σ; never uses s/n in estimator."""
    rows: list[dict[str, Any]] = []
    all_ok = True
    for point in config_grid:
        analytic = analytic_mse(point.config)
        se = float(np.sqrt(max(analytic, 1e-30) * 2.0 / n_samples))
        for seed in seeds:
            rng = np.random.default_rng(seed)
            signal, noise, y_a, y_b = simulate_measurements(point.config, n_samples, rng)
            # Guard: estimator APIs must not need signal/noise — we only use them for scoring.
            yb_result = observable_yoga_buddhi(y_a, y_b, point.config)
            est = estimate_signal(y_a, y_b, point.config)
            weights = mmse_coefficients(point.config)
            ups = observable_upsilon(y_a, y_b, weights)
            anti = observable_antithesis(y_a, y_b, point.config)
            tile_anti, tile_proj = tile_ref.observable_upsilon_tile(
                y_a, y_b, float(weights[0]), float(weights[1])
            )

            alg_err = float(
                max(
                    np.max(np.abs(yb_result.projected - est.estimate)),
                    np.max(np.abs(ups["projected"] - est.estimate)),
                    np.max(np.abs(tile_proj - est.estimate)),
                    np.max(np.abs(anti - yb_result.antithesis)),
                    np.max(np.abs(tile_anti - anti)),
                )
            )
            sample_mse = float(np.mean((yb_result.projected - signal) ** 2))
            within = abs(sample_mse - analytic) <= 5.0 * se

            # Latent antithesis for diagnostic distance (oracle n available only here).
            # Dual-mode thesis chart is y_A; latent conjugate uses known n only for metric.
            eta = point.config.efficiency
            # Approximate latent noise imprint on mode A: √η n (ignores q_A) — diagnostic only.
            latent_noise_proxy = np.sqrt(eta) * noise
            latent_antithesis_proxy = signal - noise  # pure latent chart (v-ξ), not y-space
            # Distance between observable antithesis and -y_A (non-involution witness on raw Y)
            not_neg_ya = float(np.mean((anti + y_a) ** 2))
            raw_y_is_involution = not_neg_ya < 1e-8

            # Oracle Samya residual on latent chart for same draw
            oracle_err = float(
                np.max(np.abs(oracle_paired_projection(signal, noise) - signal))
            )

            ok = alg_err < 1e-10 and within and oracle_err < 1e-12 and not raw_y_is_involution
            all_ok = all_ok and ok
            rows.append(
                {
                    "config": point.name,
                    "seed": seed,
                    "analytic_mse": analytic,
                    "sample_mse": sample_mse,
                    "se_5sigma": 5.0 * se,
                    "mse_within_5sigma": within,
                    "algebra_max_abs_err": alg_err,
                    "oracle_samya_max_abs_err": oracle_err,
                    "mean_sq_obs_anti_plus_ya": not_neg_ya,
                    "raw_y_falsely_involution": raw_y_is_involution,
                    "weights": [float(weights[0]), float(weights[1])],
                    "latent_proxy_note": (
                        "latent_antithesis_proxy is v-ξ chart; not used by estimator"
                    ),
                    "latent_antithesis_proxy_norm": float(np.linalg.norm(latent_antithesis_proxy)),
                    "observable_antithesis_norm": float(np.linalg.norm(anti)),
                    "latent_noise_proxy_norm": float(np.linalg.norm(latent_noise_proxy)),
                    "pass": ok,
                }
            )
    return {
        "name": "lemma_z1_config_seed_grid",
        "pass": all_ok,
        "n_samples": n_samples,
        "n_configs": len(config_grid),
        "n_seeds": len(seeds),
        "rows": rows,
    }


def check_mirror_and_softwalls() -> dict[str, Any]:
    z = np.array([1 + 2j, 0.5 - 0.5j, -3j, 2.0], dtype=np.complex128)
    m = mirror_validation(z)
    tile_m = tile_ref.mirror_abs2_tile(z.real, z.imag)
    mirror_ok = bool(np.all(m >= -1e-15) and np.allclose(m, np.abs(z) ** 2) and np.allclose(m, tile_m))
    softwalls = {
        "claims_landauer_circumvention": False,
        "claims_hardware_O0_msb_flip": False,
        "claims_helmholtz_equals_u_min": False,
        "claims_nvidia_deepseek_tflops_as_ours": False,
        "claims_cutile_geomean_as_ours": False,
        "hybrid_is_photonic_qpu": False,
    }
    return {
        "name": "mirror_and_softwalls",
        "pass": mirror_ok and all(v is False for v in softwalls.values()),
        "mirror_nonneg": mirror_ok,
        "softwalls": softwalls,
        "note": "Mirror is A5 channel; softwalls must stay false in this artifact.",
    }


def check_sphota_backends(
    *,
    seed: int,
    n_samples: int,
    prefer_jax: bool,
    prefer_julia: bool,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    report = SphotaCompiler(
        seed=seed,
        n_samples=n_samples,
        prefer_jax=prefer_jax,
        prefer_julia=prefer_julia,
    ).crystallize()
    elapsed = time.perf_counter() - t0
    payload = report.to_dict()
    payload["wall_seconds"] = elapsed
    payload["name"] = "sphota_crystallize"
    payload["pass"] = bool(report.pass_lemma_z1)
    return payload


def check_cutile_stub() -> dict[str, Any]:
    stub = Path("embodiment/cutile_rs_stub")
    if not stub.is_dir():
        return {"name": "cutile_rs_stub", "pass": False, "status": "missing"}
    try:
        test = subprocess.run(
            ["cargo", "test", "--quiet"],
            cwd=stub,
            capture_output=True,
            text=True,
            check=False,
        )
        run = subprocess.run(
            ["cargo", "run", "--quiet", "--bin", "sri_cutile_rs_stub_check"],
            cwd=stub,
            capture_output=True,
            text=True,
            check=False,
        )
        ok = test.returncode == 0 and run.returncode == 0 and "VERDICT: KERNEL_OK" in run.stdout
        return {
            "name": "cutile_rs_stub",
            "pass": ok,
            "status": "KERNEL_OK" if ok else "KERNEL_FAIL",
            "test_returncode": test.returncode,
            "run_returncode": run.returncode,
            "stdout_tail": run.stdout[-500:],
        }
    except FileNotFoundError:
        return {
            "name": "cutile_rs_stub",
            "pass": True,
            "status": "PERF_SKIP",
            "reason": "cargo_unavailable",
            "note": "Optional on Braket images without Rust; tile_ref remains ground truth.",
        }


def check_sree_te_plan() -> dict[str, Any]:
    plan = mentor_te_moe_plan(mentor_profile="smoke")
    ok = (
        plan["maxtext_flags"]["te_moe_block"] is True
        and plan["honesty"]["do_not_claim_nvidia_deepseek_tflops"] is True
        and plan["witness_attach"]["must_run_before"] == "softmax"
    )
    return {"name": "sree_te_moe_plan", "pass": ok, "plan": plan, "status": "CONFIG_OK" if ok else "CONFIG_FAIL"}


def compare_e1_lock(reference_lock: Path | None, lemma_grid: dict[str, Any]) -> dict[str, Any] | None:
    if reference_lock is None or not reference_lock.is_file():
        return None
    ref = json.loads(reference_lock.read_text(encoding="utf-8"))
    ref_mse = float(ref["aggregates"]["observable"]["mean_mmse_sample_mse"])
    default_rows = [
        r for r in lemma_grid["rows"] if r["config"] == "default_r07"
    ]
    if not default_rows:
        return {"pass": False, "reason": "no_default_r07_rows"}
    mean_sample = float(np.mean([r["sample_mse"] for r in default_rows]))
    delta = abs(mean_sample - ref_mse)
    # Different n_samples / shots ⇒ loose band; still useful provenance check.
    ok = delta <= 0.15
    return {
        "name": "compare_e1_colab_lock",
        "pass": ok,
        "reference": str(reference_lock),
        "reference_mean_mmse_sample_mse": ref_mse,
        "elaborate_mean_sample_mse_default_r07": mean_sample,
        "delta": delta,
        "tol": 0.15,
        "note": "Loose continuity check vs prior E1 lock; not a C1 re-proof.",
    }


def theory_hooks(checks: dict[str, Any]) -> dict[str, Any]:
    """Machine hints for next math/framework updates after this run lands."""
    z1 = checks.get("lemma_z1_config_seed_grid", {})
    rows = z1.get("rows", [])
    mean_gap = float(
        np.mean([r["mean_sq_obs_anti_plus_ya"] for r in rows]) if rows else float("nan")
    )
    return {
        "lemma_z1_status": "supported" if z1.get("pass") else "failed",
        "colab_oracle_regression": (
            "supported" if checks.get("colab_oracle_samya_fixture", {}).get("pass") else "failed"
        ),
        "proposed_next_lemmas": [
            {
                "id": "Z2",
                "question": "unveil→Z1 vs select→Z1 composition order",
                "needs_from_this_run": "stable Z1 grid + antithesis distance diagnostics",
            },
            {
                "id": "Z1b_distance",
                "question": (
                    "Formalize E[‖Ĉ_S(Y)+y_A‖²] as a non-involution witness on raw Y "
                    f"(empirical mean_sq≈{mean_gap:.6g})"
                ),
                "status": "metric_emitted",
            },
        ],
        "do_not_promote": [
            "Landauer circumvention",
            "hardware MSB XOR O(0)",
            "NVIDIA DeepSeek TFLOPS as SREE",
            "cuTile CUPTI geomean without Blackwell run",
        ],
        "framework_files_to_update_after_pasteback": [
            "docs/architecture-axioms.md",
            "docs/yoga-buddhi-projector.md",
            "docs/theoretical-framework.md",
            "docs/conjecture-registry.md",
        ],
    }


def run(
    output_dir: Path,
    *,
    seeds: list[int] | None = None,
    n_samples: int = 8192,
    oracle_dimensions: list[int] | None = None,
    prefer_jax: bool = True,
    prefer_julia: bool = True,
    promote_c5: bool = True,
    reference_lock: Path | None = None,
    run_cutile_stub: bool = True,
    submit_hybrid: bool = False,
    cost_cap_usd: float = 25.0,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if seeds is None:
        seeds = [108, 109, 110, 111]
    if oracle_dimensions is None:
        oracle_dimensions = [256, 1024, 4096, 16384]

    reset_registry()
    for cid in ("C1", "C2", "C3", "C4"):
        record_result(cid, "confirmed", notes="prior foundation lock (context)")

    t0 = time.perf_counter()
    devices = _devices()

    checks: dict[str, Any] = {}
    checks["colab_oracle_samya_fixture"] = check_colab_oracle_fixture(
        seeds=seeds, dimensions=oracle_dimensions
    )
    checks["lemma_z1_config_seed_grid"] = check_lemma_z1_grid(
        seeds=seeds, n_samples=n_samples
    )
    checks["mirror_and_softwalls"] = check_mirror_and_softwalls()
    checks["sphota_crystallize"] = check_sphota_backends(
        seed=seeds[0],
        n_samples=n_samples,
        prefer_jax=prefer_jax,
        prefer_julia=prefer_julia,
    )
    checks["sree_te_moe_plan"] = check_sree_te_plan()
    if run_cutile_stub:
        checks["cutile_rs_stub"] = check_cutile_stub()
    else:
        checks["cutile_rs_stub"] = {
            "name": "cutile_rs_stub",
            "pass": True,
            "status": "SKIPPED",
        }

    comparison = compare_e1_lock(reference_lock, checks["lemma_z1_config_seed_grid"])
    if comparison is not None:
        checks["compare_e1_colab_lock"] = comparison

    required = [
        "colab_oracle_samya_fixture",
        "lemma_z1_config_seed_grid",
        "mirror_and_softwalls",
        "sphota_crystallize",
        "sree_te_moe_plan",
    ]
    pass_core = all(checks[k].get("pass") for k in required)
    # cutile stub optional skip still ok; fail only on explicit KERNEL_FAIL
    cutile = checks["cutile_rs_stub"]
    cutile_ok = cutile.get("pass", True) or cutile.get("status") in {"SKIPPED", "PERF_SKIP"}
    compare_ok = comparison is None or comparison.get("pass", False)
    pass_all = pass_core and cutile_ok and compare_ok

    c5_notes = (
        f"elaborate Braket suite: z1={checks['lemma_z1_config_seed_grid']['pass']}, "
        f"oracle_fixture={checks['colab_oracle_samya_fixture']['pass']}, "
        f"sphota={checks['sphota_crystallize']['pass']}"
    )
    if promote_c5 and pass_core:
        record_result(
            "C5",
            "confirmed",
            artifact_paths=[str(output_dir / "elaborate_summary.json")],
            notes=c5_notes,
        )
    else:
        record_result(
            "C5",
            "open",
            artifact_paths=[str(output_dir / "elaborate_summary.json")],
            notes=c5_notes if pass_core else "core checks failed; C5 remains open",
        )

    plan = BraketJobPlan(
        job_name="sphota-yb-elaborate",
        device_arn=SV1_ARN,
        shots=max(100, n_samples // 4),
        task="sphota_yoga_buddhi",
        parameters={
            "suite": "braket_sphota_yoga_buddhi_elaborate",
            "seeds": seeds,
            "n_samples": n_samples,
            "oracle_dimensions": oracle_dimensions,
            "encoding": "classical_hybrid_container",
            "note": (
                "Elaborate Yoga–Buddhi / Lemma Z1 + Colab oracle regression. "
                "Not a photonic QPU MMSE claim."
            ),
        },
        cost_cap_usd=cost_cap_usd,
        submit=submit_hybrid,
        s3_bucket=os.environ.get("BRAKET_S3_BUCKET"),
        s3_prefix="sri-braket/sphota-yoga-buddhi-elaborate",
        region=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )
    validation = dry_run_validate(plan)
    hybrid_spec = {
        "plan": asdict(plan),
        "estimate_usd": estimate_cost_usd(plan),
        "dry_run": validation,
        "credentials_available": credentials_available(),
        "submit_requested": submit_hybrid,
        "submit_executed": False,
        "notebook_hint": (
            "On Braket notebook: clone/sync repo, PYTHONPATH=., run this module, "
            "upload elaborate_summary.json to S3, paste verdict + theory_hooks."
        ),
    }
    if (
        submit_hybrid
        and validation.get("status") == "dry_run_ok"
        and validation.get("within_cost_cap")
        and credentials_available()
        and os.environ.get("BRAKET_SUBMIT") == "1"
    ):
        hybrid_spec["note"] = (
            "Live Hybrid create still gated to notebook/Hybrid entrypoint wiring; "
            "local elaborate pass is the math gate. Set package entrypoint to this script."
        )
    else:
        hybrid_spec["note"] = "Dry-run / local elaborate only unless BRAKET_SUBMIT=1"

    elapsed = time.perf_counter() - t0
    summary: dict[str, Any] = {
        "experiment": "braket_sphota_yoga_buddhi_elaborate",
        "recorded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "wall_seconds": elapsed,
        "devices": devices,
        "configuration": {
            "seeds": seeds,
            "n_samples": n_samples,
            "oracle_dimensions": oracle_dimensions,
            "prefer_jax": prefer_jax,
            "prefer_julia": prefer_julia,
            "promote_c5": promote_c5,
            "reference_lock": str(reference_lock) if reference_lock else None,
            "config_grid": [p.name for p in DEFAULT_CONFIG_GRID],
        },
        "checks": checks,
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
        "theory_hooks": theory_hooks(checks),
        "tile_analysis": tile_ref.analysis_spec(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "verdict": {
            "pass_all": pass_all,
            "pass_core": pass_core,
            "pass_lemma_z1": bool(checks["lemma_z1_config_seed_grid"]["pass"]),
            "pass_colab_oracle_fixture": bool(checks["colab_oracle_samya_fixture"]["pass"]),
            "c5_status": get("C5").status,
            "failed_checks": [k for k, v in checks.items() if not v.get("pass", False)],
        },
    }

    summary_path = output_dir / "elaborate_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "hybrid_job_spec.json").write_text(
        json.dumps(hybrid_spec, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "theory_hooks.json").write_text(
        json.dumps(summary["theory_hooks"], indent=2) + "\n", encoding="utf-8"
    )
    # Compact paste-back for chat / notebook
    paste = {
        "verdict": summary["verdict"],
        "devices": {
            "jax_default_backend": devices.get("jax_default_backend"),
            "jax_devices": devices.get("jax_devices"),
            "julia_available": devices.get("julia_available"),
            "braket_sdk": devices.get("braket_sdk"),
        },
        "theory_hooks": summary["theory_hooks"],
        "c5_status": summary["verdict"]["c5_status"],
        "wall_seconds": elapsed,
    }
    (output_dir / "pasteback.json").write_text(
        json.dumps(paste, indent=2) + "\n", encoding="utf-8"
    )

    # Convenience copy under docs/benchmark-results when writing there
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/benchmark-results/braket-live/sphota-yoga-buddhi-elaborate"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[108, 109, 110, 111])
    parser.add_argument("--n-samples", type=int, default=8192)
    parser.add_argument(
        "--oracle-dimensions",
        type=int,
        nargs="+",
        default=[256, 1024, 4096, 16384],
    )
    parser.add_argument("--no-jax", action="store_true")
    parser.add_argument("--no-julia", action="store_true")
    parser.add_argument("--no-promote-c5", action="store_true")
    parser.add_argument("--no-cutile-stub", action="store_true")
    parser.add_argument(
        "--reference-lock",
        type=Path,
        default=Path("docs/benchmark-results/e1_colab_lock.json"),
    )
    parser.add_argument("--submit-hybrid", action="store_true")
    parser.add_argument("--cost-cap-usd", type=float, default=25.0)
    args = parser.parse_args(argv)

    ref = args.reference_lock if args.reference_lock.is_file() else None
    summary = run(
        args.output_dir,
        seeds=list(args.seeds),
        n_samples=args.n_samples,
        oracle_dimensions=list(args.oracle_dimensions),
        prefer_jax=not args.no_jax,
        prefer_julia=not args.no_julia,
        promote_c5=not args.no_promote_c5,
        reference_lock=ref,
        run_cutile_stub=not args.no_cutile_stub,
        submit_hybrid=args.submit_hybrid,
        cost_cap_usd=args.cost_cap_usd,
    )
    print(json.dumps(summary["verdict"], indent=2))
    print("wrote", args.output_dir / "elaborate_summary.json")
    print("pasteback", args.output_dir / "pasteback.json")
    return 0 if summary["verdict"]["pass_all"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
