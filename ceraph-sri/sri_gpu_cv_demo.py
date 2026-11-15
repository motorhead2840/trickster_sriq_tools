"""CV-aware SRI cycle demo with observable-only MMSE and axiomatic Ξ.

Uses NumPy by default so CI and CPU hosts can reproduce provenance. When JAX
reports a GPU and --require-gpu is set, measurements are noted as GPU-capable
host telemetry (the CV algebra remains the Gaussian closed form / NumPy path
in this foundation sprint).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sri.cv.estimator import SensingConfig, estimate_signal, simulate_measurements
from sri.dynamics.v3_damping import Branch, DampingParameters, apply_logit_damping, v3_damping
from sri.metrics.stability import stability_report


@dataclass(frozen=True)
class DemoConfig:
    seed: int = 108
    shots_per_cycle: int = 4096
    cycles: int = 25
    spike_start: int = 10
    spike_end: int = 20
    baseline_noise_variance: float = 0.25
    spike_noise_variance: float = 225.0
    efficiency: float = 0.8
    coupling: float = 1.0
    reference_r: float = 0.7
    branch: str = "W"
    repeats: int = 5


def _detect_gpu() -> list[str]:
    try:
        import jax

        return [
            f"{d.platform}:{d.device_kind}"
            for d in jax.devices()
            if d.platform in ("gpu", "cuda", "rocm")
        ]
    except Exception:
        return []


def run_demo(config: DemoConfig, output_dir: Path, require_gpu: bool = False) -> dict:
    gpus = _detect_gpu()
    if require_gpu and not gpus:
        raise RuntimeError("No JAX GPU backend active; refuse CPU fallback under --require-gpu")

    branch = Branch.WORKED if config.branch.upper() == "W" else Branch.PRINTED
    rng = np.random.default_rng(config.seed)
    rows: list[dict] = []

    def once() -> list[dict]:
        local_rows: list[dict] = []
        for cycle in range(config.cycles):
            vn = (
                config.spike_noise_variance
                if config.spike_start <= cycle <= config.spike_end
                else config.baseline_noise_variance
            )
            sensing = SensingConfig(
                signal_variance=1.0,
                noise_variance=vn,
                coupling=config.coupling,
                efficiency=config.efficiency,
                signal_r=0.0,
                reference_r=config.reference_r,
            )
            signal, _noise, y_a, y_b = simulate_measurements(
                sensing, config.shots_per_cycle, rng
            )
            mmse = estimate_signal(y_a, y_b, sensing)
            residuals = y_a - mmse.estimate
            report = stability_report(residuals)
            grad_var = float(np.var(np.diff(residuals)))
            damping = v3_damping(
                report.xi_snr,
                max(grad_var, 0.0),
                DampingParameters(branch=branch),
            )
            damped = apply_logit_damping(residuals, damping.chi)
            mse = float(np.mean((mmse.estimate - signal) ** 2))
            local_rows.append(
                {
                    "cycle": cycle,
                    "noise_variance": vn,
                    "mmse_mse": mse,
                    "analytic_mse": mmse.analytic_mse,
                    "legacy_xi": report.legacy_xi,
                    "xi_snr": report.xi_snr,
                    "xi_fisher": report.xi_fisher,
                    "circuit_a": damping.circuit_a,
                    "circuit_b": damping.circuit_b,
                    "chi_v3": damping.chi,
                    "damped_residual_variance": float(np.var(damped)),
                    "branch": branch.value,
                }
            )
        return local_rows

    # Warm-up
    rows = once()
    timings_ms: list[float] = []
    for _ in range(config.repeats):
        start = time.perf_counter()
        rows = once()
        timings_ms.append((time.perf_counter() - start) * 1000.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "sri_gpu_cv_telemetry.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "source_equations": {
            "mmse": "w = Cov(s,y) Cov(y,y)^{-1}; s_hat = w·y",
            "xi_snr": "E(logits)^2 / (E(logits)^2 + Var(logits) + eps)",
            "legacy_xi": "1 - Var(logits)/(E(logits)+eps)",
            "circuit_b_branch_W": (
                "lambda_max * sigmoid(alpha * (Var(grad)/sigma_sat^2 - theta_crit))"
            ),
            "circuit_b_branch_P": (
                "lambda_max * sigmoid(alpha * (Var(grad)/(sigma_sat^2*eps) - theta_crit))"
            ),
            "chi_v3": "circuit_a + circuit_b",
        },
        "configuration": asdict(config),
        "runtime": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "gpu_devices": gpus,
            "backend": "numpy_cv_gaussian",
        },
        "measurement": {
            "timing_scope": "full multi-cycle NumPy CV demo including measurement draws",
            "repeat_count": config.repeats,
            "execution_ms_each_repeat": timings_ms,
            "execution_ms_median": statistics.median(timings_ms),
            "max_mmse_mse": max(row["mmse_mse"] for row in rows),
            "min_xi_snr": min(row["xi_snr"] for row in rows),
            "max_xi_snr": max(row["xi_snr"] for row in rows),
            "note": (
                "Observable-only MMSE; axiomatic xi_snr used for V3. "
                "Classical simulation of CV optics, not quantum hardware."
            ),
        },
        "provenance": {
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "telemetry_csv": csv_path.name,
        },
    }
    report_path = output_dir / "sri_gpu_cv_run.json"
    report_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata["measurement"], indent=2))
    print(f"telemetry={csv_path}")
    print(f"run metadata={report_path}")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("sri_gpu_cv_results"))
    parser.add_argument("--seed", type=int, default=108)
    parser.add_argument("--shots-per-cycle", type=int, default=4096)
    parser.add_argument("--cycles", type=int, default=25)
    parser.add_argument("--branch", choices=["P", "W", "p", "w"], default="W")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args(argv)
    config = DemoConfig(
        seed=args.seed,
        shots_per_cycle=args.shots_per_cycle,
        cycles=args.cycles,
        branch=args.branch.upper(),
        repeats=args.repeats,
    )
    run_demo(config, args.output_dir, require_gpu=args.require_gpu)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
