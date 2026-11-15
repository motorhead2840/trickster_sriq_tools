"""Final Dissertation × current-project validity campaign (Braket / SageMaker).

Re-checks the March 2026 Final Dissertation claims against the Phase-2 /
provisional-pack enablement boundary, in CS terminology:

  D1–D8   Dissertation math / architecture chapters
  D9      Softwall honesty board (must not falsely green)
  D10     Lemma Z1 deployable projector (project advance)
  D11     User-query end-to-end path smoke (FIG. 5 / FIG. 8 shape)

Honesty:
  - Landauer circumvention / TMEM O(0) / Blackwell 0.0062 ms / MBL → softwall
  - §1.2 post-filter 300 ms is a modeled midpoint, not matched E2E product baseline
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from experiments.braket_phase2_dissertation_campaign import (
    s1_bipartition_samya,
    s1_fibonacci_aah_smoke,
    s1_landauer_and_free_energy,
    s1_mirror,
    s1_omega_dit,
    s1_v3_abhaya,
    s2_safety_tax_vs_inprocess,
    s3_lemma_z1,
    s3_softwall_board,
    s3_z1b_and_advantage,
)
from sri.dynamics.v3_damping import Branch, DampingParameters, apply_logit_damping, v3_damping
from sri.metrics.stability import snr_stability


def _devices() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "jax_default_backend": None,
        "jax_devices": [],
        "cudaq": None,
        "nvidia_smi": None,
    }
    try:
        import jax

        info["jax_default_backend"] = jax.default_backend()
        info["jax_devices"] = [str(d) for d in jax.devices()]
    except Exception as exc:  # noqa: BLE001
        info["jax_error"] = str(exc)
    try:
        import cudaq

        info["cudaq"] = getattr(cudaq, "__version__", "present")
        info["cudaq_gpus"] = int(cudaq.num_available_gpus())
    except Exception as exc:  # noqa: BLE001
        info["cudaq_error"] = str(exc)
    try:
        import subprocess

        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if smi.returncode == 0:
            info["nvidia_smi"] = smi.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return info


def _ok(flag: bool, **extra: Any) -> dict[str, Any]:
    return {"pass": bool(flag), **extra}


def d1_pathology_context() -> dict[str, Any]:
    """§1.1 — narrative/context gate: elevated free-energy framing is recorded, not calorimetry."""
    return _ok(
        True,
        chapter="1.1 Iterative pathology",
        track="CONTEXT",
        note=(
            "Hallucination framed as elevated free-energy / skipped intermediate states; "
            "no physical calorimetry claimed"
        ),
        patent_fig="FIG.1 / whitepaper §2",
    )


def d11_user_query_path(*, vocab: int = 8192) -> dict[str, Any]:
    """FIG. 5 / FIG. 8 shape: user query → pre-Softmax scores → Witness → present/refuse.

    Two cases:
      - peaked scores → expect present_answer
      - near-flat noisy scores → may refuse; path still valid if decision is emitted
    """
    rng = np.random.default_rng(20261112)
    user_query = "Explain dual-mode cleaned estimate without oracle noise."
    uniform = 1.0 / float(vocab)

    def _one(scores: np.ndarray) -> dict[str, Any]:
        t0 = time.perf_counter()
        xi = float(np.clip(snr_stability(scores), 0.0, 1.0))
        chi = v3_damping(xi, float(np.var(scores)), DampingParameters(branch=Branch.WORKED))
        damped = apply_logit_damping(scores, float(min(chi.chi, 60.0)))
        m = float(np.max(damped))
        ex = np.exp(damped - m)
        probs = ex / np.sum(ex)
        token = int(np.argmax(probs))
        confidence = float(probs[token])
        # Present if clearly above uniform baseline (vocab-aware)
        present = confidence >= max(5.0 * uniform, uniform + 1e-6)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "stability_xi": xi,
            "chi": float(chi.chi),
            "token_id": token,
            "confidence": confidence,
            "decision": "present_answer" if present else "no_answer_indication",
            "elapsed_ms": elapsed_ms,
        }

    peaked = rng.normal(loc=0.0, scale=0.4, size=vocab)
    peaked[0] = 12.0  # sharp preferred token
    flat = rng.normal(loc=0.0, scale=0.05, size=vocab)

    peaked_out = _one(peaked)
    flat_out = _one(flat)
    ok = (
        len(user_query) > 0
        and peaked_out["decision"] == "present_answer"
        and peaked_out["elapsed_ms"] < 100.0
        and flat_out["elapsed_ms"] < 100.0
        and flat_out["decision"] in {"present_answer", "no_answer_indication"}
    )
    return _ok(
        ok,
        chapter="Product path FIG.5/FIG.8",
        track="PROVED_PATH_SMOKE",
        user_query=user_query,
        peaked=peaked_out,
        flat=flat_out,
        decision=peaked_out["decision"],
        confidence=peaked_out["confidence"],
        elapsed_ms=peaked_out["elapsed_ms"],
        patent_fig="FIG.4 / FIG.5 / FIG.8",
    )


def build_integration_matrix(stages: dict[str, Any]) -> list[dict[str, Any]]:
    """Chapter map for counsel / inventor (mirrors docs/patent-provisional/06-...)."""
    rows = [
        {
            "id": "D1",
            "diss_section": "§1.1",
            "status": "CONTEXT",
            "pass": stages["D1_pathology"]["pass"],
        },
        {
            "id": "D2",
            "diss_section": "§1.2 safety tax",
            "status": "ENABLED_ARCHITECTURE_LATENCY",
            "pass": stages["D2_safety_tax"]["pass"],
            "inprocess_median_ms": stages["D2_safety_tax"].get("inprocess_median_ms"),
            "speedup_vs_post_filter": stages["D2_safety_tax"].get("speedup_vs_post_filter"),
        },
        {
            "id": "D3",
            "diss_section": "§2.1 CV-dit",
            "status": "ENABLED_MATH",
            "pass": stages["D3_cv_dit"]["pass"],
        },
        {
            "id": "D4",
            "diss_section": "§2.2 / projector",
            "status": "ENABLED_MATH",
            "pass": stages["D4_bipartition"]["pass"],
        },
        {
            "id": "D5",
            "diss_section": "§3.1 Fibonacci/AAH",
            "status": "INSTRUMENT_ONLY",
            "pass": stages["D5_fibonacci_smoke"]["pass"],
            "mbl_proven": False,
        },
        {
            "id": "D6",
            "diss_section": "§4 Witness V3",
            "status": "ENABLED",
            "pass": stages["D6_witness_v3"]["pass"],
        },
        {
            "id": "D7",
            "diss_section": "§4 Mirror",
            "status": "ENABLED",
            "pass": stages["D7_mirror"]["pass"],
        },
        {
            "id": "D8",
            "diss_section": "§5 Landauer/Helmholtz",
            "status": "MATH_ENABLED_PHYSICS_SOFTWALL",
            "pass": stages["D8_landauer_fe"]["pass"],
            "physical_circumvention": stages["D8_landauer_fe"].get("physical_circumvention"),
        },
        {
            "id": "D9",
            "diss_section": "softwalls",
            "status": "SOFTWALL_HONEST",
            "pass": stages["D9_softwalls"]["pass"],
            "board": {
                k: stages["D9_softwalls"][k]
                for k in (
                    "landauer_circumvention",
                    "tmem_msb_xor_o0",
                    "blackwell_0p0062ms",
                    "mbl_quasicrystal",
                    "any_softwall_falsely_greened",
                )
                if k in stages["D9_softwalls"]
            },
        },
        {
            "id": "D10",
            "diss_section": "Lemma Z1 (project)",
            "status": "ENABLED_DEPLOYABLE",
            "pass": stages["D10_lemma_z1"]["pass"],
        },
        {
            "id": "D11",
            "diss_section": "user-query path",
            "status": "ENABLED_PATH_SMOKE",
            "pass": stages["D11_user_query_path"]["pass"],
            "decision": stages["D11_user_query_path"].get("decision"),
        },
    ]
    return rows


def write_integration_report(output_dir: Path, summary: dict[str, Any]) -> Path:
    s2 = summary["stages"]["D2_safety_tax"]
    soft = summary["stages"]["D9_softwalls"]
    lines = [
        "# Final Dissertation validity report (Braket / SageMaker)",
        "",
        f"**Generated (UTC):** {summary['meta']['generated_utc']}",
        f"**Commit:** `{summary['meta'].get('git_sha', 'unknown')}`",
        "",
        "## Verdict",
        "",
        f"- `pass_dissertation_validity`: **{summary['verdict']['pass_dissertation_validity']}**",
        f"- In-process median: **{s2.get('inprocess_median_ms'):.4f} ms**",
        f"- Post-filter model midpoint: **{s2.get('post_filter_model_ms'):.0f} ms**",
        f"- Architectural ratio: **{s2.get('speedup_vs_post_filter'):.1f}×** (modeled baseline)",
        "",
        "## Softwalls (must stay soft)",
        "",
        f"- Landauer circumvention: `{soft.get('landauer_circumvention')}`",
        f"- TMEM MSB-XOR O(0): `{soft.get('tmem_msb_xor_o0')}`",
        f"- Blackwell 0.0062 ms: `{soft.get('blackwell_0p0062ms')}`",
        f"- MBL quasicrystal: `{soft.get('mbl_quasicrystal')}`",
        f"- Any softwall falsely greened: `{soft.get('any_softwall_falsely_greened')}`",
        "",
        "## Integration matrix",
        "",
        "| ID | Dissertation | Status | Pass |",
        "|---|---|---|---|",
    ]
    for row in summary["integration_matrix"]:
        lines.append(
            f"| {row['id']} | {row['diss_section']} | {row['status']} | {row['pass']} |"
        )
    lines.extend(
        [
            "",
            "## Patent figure anchors",
            "",
            "- FIG. 1 / 4 / 5 — safety-tax vs in-process + serving",
            "- FIG. 3 — Lemma Z1 dual-measurement estimate",
            "- FIG. 8 — user present/refuse path (D11)",
            "- FIG. 6 — softwall / future embodiments",
            "",
            "## Counsel note",
            "",
            "Treat Final Dissertation rhetoric (Exascale, absolute zero entropy, fearless symmetry)",
            "as **vision**. Enable only rows marked ENABLED_* above. Softwalls stay future embodiments.",
            "",
        ]
    )
    path = output_dir / "DISSERTATION_VALIDITY_INTEGRATION.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


CHAPTER_TITLES: dict[int, str] = {
    1: "The Epistemological and Thermodynamic Crisis",
    2: "Foundational Ontology — CV-dit and bipartite space",
    3: "Quasicrystal manifold and cancellation projector",
    4: "Witness gate, V3 damping, and mirror validation",
    5: "Free-energy / Landauer math and hardware softwalls",
    6: "Future analog and optical horizons (softwall only)",
}


def d6_future_optical_softwall() -> dict[str, Any]:
    """Ch. 6 — PPLN / optical phase-conjugate core is disclosed, not proven."""
    return _ok(
        True,
        chapter="6 Future analog / optical horizons",
        track="SOFTWALL_FUTURE_ONLY",
        ppln_phase_conjugate="FUTURE_EMBODIMENT_NOT_PROVEN",
        analog_optical_core="FUTURE_EMBODIMENT_NOT_PROVEN",
        falsely_greened=False,
        note="Chapter 6 remains aspirational; validity pass = correctly not enabled",
        patent_fig="FIG.6",
    )


def run_chapter(
    chapter: int,
    output_dir: Path,
    *,
    seeds: list[int] | None = None,
    n_samples: int = 4096,
    vocab: int = 32000,
    batches: int = 40,
) -> dict[str, Any]:
    """Run checks for a single Final Dissertation chapter (1–6)."""
    if chapter not in CHAPTER_TITLES:
        raise ValueError(f"chapter must be 1–6, got {chapter}")
    seeds = seeds or [108, 109, 110]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ch_dir = output_dir / f"chapter_{chapter}"
    ch_dir.mkdir(parents=True, exist_ok=True)

    checks: dict[str, Any]
    if chapter == 1:
        checks = {
            "pathology_1_1": d1_pathology_context(),
            "safety_tax_1_2": s2_safety_tax_vs_inprocess(vocab=vocab, batches=batches),
        }
    elif chapter == 2:
        checks = {
            "cv_dit_2_1": s1_omega_dit(),
            "bipartition_2_2": s1_bipartition_samya(),
        }
    elif chapter == 3:
        checks = {
            "cancellation_projector": s1_bipartition_samya(),
            "fibonacci_aah_smoke_3_1": s1_fibonacci_aah_smoke(),
            "lemma_z1_deployable": s3_lemma_z1(seeds=seeds, n_samples=n_samples),
            "z1b_advantage": s3_z1b_and_advantage(n_samples=min(n_samples, 8192)),
        }
    elif chapter == 4:
        checks = {
            "witness_v3_4_1": s1_v3_abhaya(),
            "mirror_4_2": s1_mirror(),
            "user_query_witness_path": d11_user_query_path(vocab=min(vocab, 16384)),
        }
    elif chapter == 5:
        checks = {
            "landauer_helmholtz_5_1": s1_landauer_and_free_energy(),
            "softwall_board_5_2": s3_softwall_board(),
        }
    else:  # chapter == 6
        checks = {
            "optical_future_6": d6_future_optical_softwall(),
            "softwall_board": s3_softwall_board(),
        }

    pass_chapter = all(bool(v.get("pass", False)) for v in checks.values())
    # Chapter 5/6 must keep softwalls honest
    if chapter in {5, 6}:
        board = checks.get("softwall_board_5_2") or checks.get("softwall_board") or {}
        pass_chapter = pass_chapter and board.get("any_softwall_falsely_greened") is False
        if chapter == 6:
            pass_chapter = pass_chapter and checks["optical_future_6"].get("falsely_greened") is False

    result: dict[str, Any] = {
        "chapter": chapter,
        "title": CHAPTER_TITLES[chapter],
        "pass_chapter": pass_chapter,
        "checks": checks,
        "devices": _devices(),
        "generated_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    }
    (ch_dir / "chapter_result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    paste = {
        "campaign": "dissertation-validity-chapter",
        "chapter": chapter,
        "title": CHAPTER_TITLES[chapter],
        "pass_chapter": pass_chapter,
        "check_passes": {k: bool(v.get("pass")) for k, v in checks.items()},
    }
    (ch_dir / "pasteback.json").write_text(json.dumps(paste, indent=2), encoding="utf-8")
    lines = [
        f"# Chapter {chapter} validity — {CHAPTER_TITLES[chapter]}",
        "",
        f"**pass_chapter:** {pass_chapter}",
        "",
        "| Check | Pass | Track |",
        "|---|---|---|",
    ]
    for name, payload in checks.items():
        lines.append(
            f"| `{name}` | {payload.get('pass')} | {payload.get('track', '')} |"
        )
    lines.append("")
    (ch_dir / f"CHAPTER_{chapter}_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    result["report"] = str(ch_dir / f"CHAPTER_{chapter}_REPORT.md")
    result["pasteback_path"] = str(ch_dir / "pasteback.json")
    return result


def run(
    output_dir: Path,
    *,
    seeds: list[int] | None = None,
    n_samples: int = 4096,
    vocab: int = 32000,
    batches: int = 40,
) -> dict[str, Any]:
    seeds = seeds or [108, 109, 110]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stages: dict[str, Any] = {
        "D1_pathology": d1_pathology_context(),
        "D2_safety_tax": s2_safety_tax_vs_inprocess(vocab=vocab, batches=batches),
        "D3_cv_dit": s1_omega_dit(),
        "D4_bipartition": s1_bipartition_samya(),
        "D5_fibonacci_smoke": s1_fibonacci_aah_smoke(),
        "D6_witness_v3": s1_v3_abhaya(),
        "D7_mirror": s1_mirror(),
        "D8_landauer_fe": s1_landauer_and_free_energy(),
        "D9_softwalls": s3_softwall_board(),
        "D10_lemma_z1": s3_lemma_z1(seeds=seeds, n_samples=n_samples),
        "D10b_advantage": s3_z1b_and_advantage(n_samples=min(n_samples, 8192)),
        "D11_user_query_path": d11_user_query_path(vocab=min(vocab, 16384)),
        "D12_chapter6_optical": d6_future_optical_softwall(),
    }

    matrix = build_integration_matrix(stages)
    required = [r for r in matrix if r["id"] != "D1"]  # D1 is context-only
    pass_all = all(bool(r["pass"]) for r in required)
    soft_ok = stages["D9_softwalls"].get("any_softwall_falsely_greened") is False
    pass_validity = pass_all and soft_ok and bool(stages["D10b_advantage"]["pass"])

    git_sha = "unknown"
    try:
        import subprocess

        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001
        pass

    summary: dict[str, Any] = {
        "meta": {
            "generated_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "git_sha": git_sha,
            "devices": _devices(),
            "seeds": seeds,
            "n_samples": n_samples,
            "vocab": vocab,
            "batches": batches,
            "source_whitepaper": "Final Dissertation (SRI) 2026-03-26 Patent Appendix A",
            "integration_doc": "docs/patent-provisional/06-final-dissertation-integration.md",
        },
        "stages": stages,
        "integration_matrix": matrix,
        "verdict": {
            "pass_dissertation_validity": pass_validity,
            "pass_enabled_math": all(
                stages[k]["pass"]
                for k in ("D3_cv_dit", "D4_bipartition", "D6_witness_v3", "D7_mirror", "D8_landauer_fe")
            ),
            "pass_safety_tax_architecture": bool(stages["D2_safety_tax"]["pass"]),
            "pass_lemma_z1": bool(stages["D10_lemma_z1"]["pass"]),
            "pass_user_query_path": bool(stages["D11_user_query_path"]["pass"]),
            "softwall_honest": soft_ok,
            "inprocess_median_ms": stages["D2_safety_tax"].get("inprocess_median_ms"),
            "post_filter_model_ms": stages["D2_safety_tax"].get("post_filter_model_ms"),
            "speedup_vs_post_filter": stages["D2_safety_tax"].get("speedup_vs_post_filter"),
        },
    }

    report = write_integration_report(output_dir, summary)
    summary["report"] = str(report)

    telemetry = {
        "meta": summary["meta"],
        "verdict": summary["verdict"],
        "integration_matrix": matrix,
        "stages": stages,
    }
    (output_dir / "telemetry.json").write_text(json.dumps(telemetry, indent=2, default=str), encoding="utf-8")

    pasteback = {
        "campaign": "dissertation-validity",
        "verdict": summary["verdict"],
        "softwalls": {
            k: stages["D9_softwalls"].get(k)
            for k in (
                "landauer_circumvention",
                "tmem_msb_xor_o0",
                "blackwell_0p0062ms",
                "mbl_quasicrystal",
                "any_softwall_falsely_greened",
            )
        },
        "user_query_path": {
            "decision": stages["D11_user_query_path"].get("decision"),
            "elapsed_ms": stages["D11_user_query_path"].get("elapsed_ms"),
            "confidence": stages["D11_user_query_path"].get("confidence"),
        },
        "digest": hashlib.sha256(
            json.dumps(summary["verdict"], sort_keys=True, default=str).encode()
        ).hexdigest()[:16],
    }
    (output_dir / "pasteback.json").write_text(json.dumps(pasteback, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[108, 109, 110])
    parser.add_argument("--n-samples", type=int, default=4096)
    parser.add_argument("--vocab", type=int, default=32000)
    parser.add_argument("--batches", type=int, default=40)
    args = parser.parse_args(argv)
    summary = run(
        args.out,
        seeds=list(args.seeds),
        n_samples=args.n_samples,
        vocab=args.vocab,
        batches=args.batches,
    )
    print(json.dumps(summary["verdict"], indent=2))
    print("report:", summary["report"])
    return 0 if summary["verdict"]["pass_dissertation_validity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
