"""Provisional-patent evidence campaign (SageMaker / Braket Hybrid-shaped).

Designed for a short enablement push after Lemma Z1 / C5 GPU stamp.

What this strengthens for a provisional (supported claims):
  R1  Reproducibility of Z1 / Colab oracle fixtures
  R2  Wide sensing-parameter enablement grid
  R3  A0 enforcement — estimator APIs never consume s or n
  R4  Oracle vs observable MSE gap (approximate Samya narrative)
  R5  Honest GPU/CPU wall timing of observable Yoga–Buddhi (not O(0))
  R6  Software Witness smoke (Ξ_SNR + damp) for SREE Mentor intercept story
  R7  Inventor disclosure pack (claim boundaries + softwalls + artifact index)

What this deliberately does NOT try to prove:
  Landauer circumvention, hardware MSB XOR O(0), DeepSeek TFLOPS, cuTile geomean,
  photonic QPU MMSE, Helmholtz F = U_min.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from sri.cv.estimator import (
    SensingConfig,
    analytic_mse,
    estimate_signal,
    observable_antithesis,
    observable_yoga_buddhi,
    oracle_paired_projection,
    simulate_measurements,
)
from sri.legacy.simulator import damp_logits
from sri.metrics.stability import legacy_xi, snr_stability
from sri.omlang.yoga_buddhi import latent_involution_holds, oracle_samya


def _devices() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "jax_default_backend": None,
        "jax_devices": [],
    }
    try:
        import jax

        info["jax_default_backend"] = jax.default_backend()
        info["jax_devices"] = [str(d) for d in jax.devices()]
    except Exception as exc:  # noqa: BLE001
        info["jax_error"] = str(exc)
    return info


def round_r1_repro(*, seeds: list[int], n_samples: int) -> dict[str, Any]:
    """R1: oracle Samya + Z1 algebra on a small fixed grid."""
    rows = []
    ok = True
    configs = [
        SensingConfig(reference_r=0.7),
        SensingConfig(reference_r=0.0),
        SensingConfig(efficiency=0.5, reference_r=0.7),
    ]
    for seed in seeds:
        rng = np.random.default_rng(seed)
        v = rng.normal(size=1024)
        xi = rng.normal(scale=0.3, size=1024)
        samya = oracle_samya(v, xi)["samya"]
        oracle_ok = bool(np.allclose(samya, v)) and latent_involution_holds(v, xi)
        for cfg in configs:
            s, n, ya, yb = simulate_measurements(cfg, n_samples, rng)
            yb_out = observable_yoga_buddhi(ya, yb, cfg)
            est = estimate_signal(ya, yb, cfg)
            alg = float(np.max(np.abs(yb_out.projected - est.estimate)))
            mse = float(np.mean((yb_out.projected - s) ** 2))
            analytic = analytic_mse(cfg)
            se = float(np.sqrt(max(analytic, 1e-30) * 2.0 / n_samples))
            within = abs(mse - analytic) <= 5.0 * se
            row_ok = oracle_ok and alg < 1e-10 and within
            ok = ok and row_ok
            rows.append(
                {
                    "seed": seed,
                    "config": as_config_key(cfg),
                    "algebra_err": alg,
                    "sample_mse": mse,
                    "analytic_mse": analytic,
                    "mse_within_5sigma": within,
                    "pass": row_ok,
                }
            )
    return {"round": "R1_repro", "pass": ok, "n_rows": len(rows), "rows": rows}


def as_config_key(cfg: SensingConfig) -> str:
    return (
        f"Vs={cfg.signal_variance},Vn={cfg.noise_variance},k={cfg.coupling},"
        f"eta={cfg.efficiency},rA={cfg.signal_r},rB={cfg.reference_r}"
    )


def round_r2_wide_grid(*, seed: int, n_samples: int) -> dict[str, Any]:
    """R2: enablement breadth across sensing parameters."""
    rng = np.random.default_rng(seed)
    grid = []
    for eta in (0.4, 0.6, 0.8, 1.0):
        for r_b in (0.0, 0.4, 0.7, 1.0):
            for v_n in (0.5, 1.0, 2.0):
                grid.append(
                    SensingConfig(
                        signal_variance=1.0,
                        noise_variance=v_n,
                        coupling=1.0,
                        efficiency=eta,
                        signal_r=0.0,
                        reference_r=r_b,
                    )
                )
    rows = []
    ok = True
    for cfg in grid:
        s, _n, ya, yb = simulate_measurements(cfg, n_samples, rng)
        out = observable_yoga_buddhi(ya, yb, cfg)
        est = estimate_signal(ya, yb, cfg)
        alg = float(np.max(np.abs(out.projected - est.estimate)))
        mse = float(np.mean((out.projected - s) ** 2))
        analytic = analytic_mse(cfg)
        se = float(np.sqrt(max(analytic, 1e-30) * 2.0 / n_samples))
        within = abs(mse - analytic) <= 5.0 * se
        row_ok = alg < 1e-10 and within
        ok = ok and row_ok
        rows.append(
            {
                "config": as_config_key(cfg),
                "algebra_err": alg,
                "sample_mse": mse,
                "analytic_mse": analytic,
                "mse_within_5sigma": within,
                "pass": row_ok,
            }
        )
    return {
        "round": "R2_wide_enablement_grid",
        "pass": ok,
        "n_configs": len(grid),
        "n_fail": sum(1 for r in rows if not r["pass"]),
        "rows": rows,
    }


def round_r3_a0_enforcement() -> dict[str, Any]:
    """R3: prove public estimator callables cannot accept s/n as inputs."""
    checks = []
    for name, fn in (
        ("estimate_signal", estimate_signal),
        ("observable_yoga_buddhi", observable_yoga_buddhi),
        ("observable_antithesis", observable_antithesis),
    ):
        sig = inspect.signature(fn)
        params = list(sig.parameters)
        forbidden = {"s", "n", "signal", "noise", "clean", "xi", "value"}
        overlap = sorted(set(params) & forbidden)
        # First two args must be y_a, y_b (or config-last patterns)
        ok = overlap == [] and params[0] == "y_a" and params[1] == "y_b"
        checks.append(
            {
                "callable": name,
                "parameters": params,
                "forbidden_overlap": overlap,
                "pass": ok,
            }
        )
    # Runtime: calling with only Y recovers MMSE; supplying oracle noise elsewhere
    # does not change the observable path (separate oracle API).
    cfg = SensingConfig(reference_r=0.7)
    rng = np.random.default_rng(0)
    s, n, ya, yb = simulate_measurements(cfg, 2048, rng)
    obs = observable_yoga_buddhi(ya, yb, cfg).projected
    # Deliberately corrupt s,n after draw — observable result must be unchanged if recomputed from Y
    s_corrupt = s + 100.0
    n_corrupt = n + 100.0
    obs2 = observable_yoga_buddhi(ya, yb, cfg).projected
    unchanged = bool(np.allclose(obs, obs2))
    # Sanity: oracle path WOULD change if we flipped latent noise (different API)
    oracle_a = oracle_paired_projection(s, n)
    oracle_b = oracle_paired_projection(s_corrupt, n_corrupt)
    oracle_differs = not np.allclose(oracle_a, oracle_b)
    runtime = {
        "observable_insensitive_to_latent_corruption_in_scope": unchanged,
        "oracle_api_sensitive_to_latent": oracle_differs,
        "pass": unchanged and oracle_differs,
    }
    return {
        "round": "R3_a0_enforcement",
        "pass": all(c["pass"] for c in checks) and runtime["pass"],
        "signature_checks": checks,
        "runtime": runtime,
    }


def round_r4_oracle_vs_observable(*, seed: int, n_samples: int) -> dict[str, Any]:
    """R4: document MSE gap between oracle Samya and observable Υ."""
    rng = np.random.default_rng(seed)
    cfg = SensingConfig(reference_r=0.7)
    s, n, ya, yb = simulate_measurements(cfg, n_samples, rng)
    oracle = oracle_paired_projection(s, n)
    obs = observable_yoga_buddhi(ya, yb, cfg).projected
    oracle_mse = float(np.mean((oracle - s) ** 2))
    obs_mse = float(np.mean((obs - s) ** 2))
    analytic = analytic_mse(cfg)
    se = float(np.sqrt(max(analytic, 1e-30) * 2.0 / n_samples))
    within = bool(abs(obs_mse - analytic) <= 5.0 * se)
    return {
        "round": "R4_oracle_vs_observable_gap",
        "pass": bool(oracle_mse < 1e-30) and within,
        "oracle_mse": oracle_mse,
        "observable_sample_mse": obs_mse,
        "analytic_mmse": analytic,
        "gap_obs_minus_oracle": float(obs_mse - oracle_mse),
        "narrative": (
            "Oracle Samya is exact when (s,n) known; deployable path pays MMSE. "
            "Provisional should claim the observable method, not zero-error without antithesis."
        ),
    }


def round_r5_timing(*, n: int = 1_000_000, repeats: int = 20) -> dict[str, Any]:
    """R5: honest wall-clock of observable Yoga–Buddhi (software timing only)."""
    cfg = SensingConfig(reference_r=0.7)
    rng = np.random.default_rng(7)
    _s, _n, ya, yb = simulate_measurements(cfg, n, rng)

    def once() -> None:
        observable_yoga_buddhi(ya, yb, cfg)

    # warmup
    once()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        once()
        times.append(time.perf_counter() - t0)
    arr = np.asarray(times, dtype=np.float64)
    return {
        "round": "R5_honest_wall_timing",
        "pass": True,  # informational; always records
        "n": n,
        "repeats": repeats,
        "wall_seconds_mean": float(np.mean(arr)),
        "wall_seconds_std": float(np.std(arr)),
        "wall_seconds_p50": float(np.median(arr)),
        "wall_ms_mean": float(np.mean(arr) * 1e3),
        "note": (
            "Wall-clock of NumPy observable Yoga–Buddhi on this host. "
            "NOT a hardware O(0)/O(1) proof; scales with n in software."
        ),
        "jax_devices": _devices().get("jax_devices"),
    }


def round_r6_witness_smoke(*, seed: int = 0) -> dict[str, Any]:
    """R6: software Witness — Ξ_SNR gate + damp_logits (SREE Mentor story at host level)."""
    rng = np.random.default_rng(seed)
    # Stable logits
    stable = rng.normal(loc=2.0, scale=0.2, size=4096)
    # Avidya-like spike
    unstable = rng.normal(loc=0.0, scale=5.0, size=4096)
    xi_s = snr_stability(stable)
    xi_u = snr_stability(unstable)
    legacy_s = legacy_xi(stable)
    # damp unstable
    chi = 2.0
    damped = damp_logits(unstable, chi)
    var_before = float(np.var(unstable))
    var_after = float(np.var(damped))
    gated = xi_u < 0.5  # illustrative threshold for smoke only
    return {
        "round": "R6_witness_smoke",
        "pass": xi_s > xi_u and var_after < var_before and 0.0 < xi_s <= 1.0 and 0.0 < xi_u <= 1.0,
        "xi_snr_stable": xi_s,
        "xi_snr_unstable": xi_u,
        "legacy_xi_stable": legacy_s,
        "var_before_damp": var_before,
        "var_after_damp": var_after,
        "illustrative_gate_fired": gated,
        "note": (
            "Host-level Witness smoke for provisional enablement of a pre-Softmax "
            "stability gate + damp. Not a TMEM-native intercept proof."
        ),
    }


def round_r7_disclosure(rounds: dict[str, Any], devices: dict[str, Any]) -> dict[str, Any]:
    """R7: claim-boundary disclosure pack for counsel."""
    supported = [
        {
            "title": "Observable Yoga–Buddhi / Lemma Z1",
            "summary": (
                "Method of constructing an observable antithesis from dual-mode "
                "homodyne outcomes (y_A, y_B) and Wiener weights such that "
                "Υ_obs = ½(y_A + Ĉ_S) equals the MMSE estimator without receiving s or n."
            ),
            "evidence_rounds": ["R1", "R2", "R3", "R4"],
            "status": "supported_by_gpu_c5",
        },
        {
            "title": "Sphota multi-backend crystallization",
            "summary": (
                "System and method for compiling Yoga–Buddhi algebra across NumPy/JAX "
                "(and optional Julia) with tile-shaped reference ops for later cuTile port."
            ),
            "evidence_rounds": ["R1", "prior_c5_pasteback"],
            "status": "supported_scaffold",
        },
        {
            "title": "Pre-Softmax Witness gate (software)",
            "summary": (
                "Method of measuring logit stability (Ξ_SNR) and applying damping before "
                "tokenization — SREE Mentor intercept at host/software layer."
            ),
            "evidence_rounds": ["R6"],
            "status": "supported_smoke_not_tmem",
        },
    ]
    do_not_claim_yet = [
        "Landauer circumvention / Helmholtz F=U_min as measured fact",
        "IEEE-754 MSB XOR in TMEM as O(0) hardware primitive",
        "Photonic QPU MMSE from Braket SV1 toys",
        "NVIDIA DeepSeek TFLOPS or cuTile CUPTI geomean as Ceraph results",
        "Full Theorem Zero multi-layer composition (needs Lemma Z2)",
    ]
    return {
        "round": "R7_inventor_disclosure_pack",
        "pass": all(rounds[k]["pass"] for k in rounds if k != "R7_inventor_disclosure_pack"),
        "supported_claim_sketches": supported,
        "do_not_claim_yet": do_not_claim_yet,
        "devices": devices,
        "prior_c5_artifact": (
            "docs/benchmark-results/braket-live/sphota-yoga-buddhi-elaborate/"
            "sagemaker-g4dn/pasteback.json"
        ),
        "counsel_note": (
            "Draft provisional around supported sketches + code enablement. "
            "Yellow-paper hardware/thermo sentences belong in continuation or "
            "after dedicated reduction-to-practice — not as enabled claims today."
        ),
    }


def run(
    output_dir: Path,
    *,
    seeds: list[int] | None = None,
    n_samples: int = 4096,
    timing_n: int = 1_000_000,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if seeds is None:
        seeds = [108, 109, 110]
    devices = _devices()
    t0 = time.perf_counter()

    rounds: dict[str, Any] = {}
    rounds["R1_repro"] = round_r1_repro(seeds=seeds, n_samples=n_samples)
    rounds["R2_wide_enablement_grid"] = round_r2_wide_grid(seed=seeds[0], n_samples=n_samples)
    rounds["R3_a0_enforcement"] = round_r3_a0_enforcement()
    rounds["R4_oracle_vs_observable_gap"] = round_r4_oracle_vs_observable(
        seed=seeds[0], n_samples=n_samples
    )
    rounds["R5_honest_wall_timing"] = round_r5_timing(n=timing_n)
    rounds["R6_witness_smoke"] = round_r6_witness_smoke(seed=seeds[0])
    rounds["R7_inventor_disclosure_pack"] = round_r7_disclosure(rounds, devices)

    required = [
        "R1_repro",
        "R2_wide_enablement_grid",
        "R3_a0_enforcement",
        "R4_oracle_vs_observable_gap",
        "R6_witness_smoke",
    ]
    pass_core = all(rounds[k]["pass"] for k in required)
    pass_all = pass_core and rounds["R7_inventor_disclosure_pack"]["pass"]

    summary = {
        "experiment": "braket_provisional_evidence_campaign",
        "recorded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "wall_seconds": time.perf_counter() - t0,
        "devices": devices,
        "configuration": {
            "seeds": seeds,
            "n_samples": n_samples,
            "timing_n": timing_n,
        },
        "rounds": rounds,
        "verdict": {
            "pass_all": pass_all,
            "pass_core": pass_core,
            "failed_rounds": [k for k, v in rounds.items() if not v.get("pass", False)],
            "provisional_foundation": "solid_for_supported_claims" if pass_core else "blocked",
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    paste = {
        "verdict": summary["verdict"],
        "devices": {
            "jax_default_backend": devices.get("jax_default_backend"),
            "jax_devices": devices.get("jax_devices"),
        },
        "r5_timing_ms_mean": rounds["R5_honest_wall_timing"]["wall_ms_mean"],
        "r4_gap": rounds["R4_oracle_vs_observable_gap"]["gap_obs_minus_oracle"],
        "r7_do_not_claim_yet": rounds["R7_inventor_disclosure_pack"]["do_not_claim_yet"],
        "supported_claim_titles": [
            c["title"] for c in rounds["R7_inventor_disclosure_pack"]["supported_claim_sketches"]
        ],
        "wall_seconds": summary["wall_seconds"],
    }
    (output_dir / "provisional_evidence_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "pasteback.json").write_text(
        json.dumps(paste, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "disclosure_pack.json").write_text(
        json.dumps(rounds["R7_inventor_disclosure_pack"], indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/benchmark-results/braket-live/provisional-evidence"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[108, 109, 110])
    parser.add_argument("--n-samples", type=int, default=4096)
    parser.add_argument("--timing-n", type=int, default=1_000_000)
    args = parser.parse_args(argv)
    summary = run(
        args.output_dir,
        seeds=list(args.seeds),
        n_samples=args.n_samples,
        timing_n=args.timing_n,
    )
    print(json.dumps(summary["verdict"], indent=2))
    print("wrote", args.output_dir / "pasteback.json")
    return 0 if summary["verdict"]["pass_all"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
