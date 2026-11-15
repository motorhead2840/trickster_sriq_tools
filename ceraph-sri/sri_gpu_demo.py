"""GPU-first JAX demonstration of the dissertation's SRI equations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


SOURCE_DOCUMENT = "sri-provisional-patent-dissertation-5.pdf"
EPSILON = 1e-7


@dataclass(frozen=True)
class GPUConfig:
    seed: int = 108
    dimension: int = 16_384
    cycles: int = 25
    baseline_noise_std: float = 0.5
    spike_noise_std: float = 15.0
    spike_start: int = 10
    spike_end: int = 20
    lambda_0: float = 1.0
    kappa: float = 4.0
    lambda_max: float = 50.0
    alpha: float = 10.0
    theta_crit: float = 5.0
    sigma_sat: float = 1.0
    epsilon: float = EPSILON
    repeats: int = 10

    def validate(self) -> None:
        if self.dimension <= 0 or self.cycles <= 0 or self.repeats <= 0:
            raise ValueError("dimension, cycles, and repeats must be positive")
        if not 0 <= self.spike_start <= self.spike_end < self.cycles:
            raise ValueError("noise spike interval must satisfy 0 <= start <= end < cycles")
        numeric_values = (
            self.baseline_noise_std,
            self.spike_noise_std,
            self.lambda_0,
            self.kappa,
            self.lambda_max,
            self.alpha,
            self.theta_crit,
            self.sigma_sat,
            self.epsilon,
        )
        if any(not math.isfinite(value) for value in numeric_values):
            raise ValueError("simulation parameters must be finite")
        if min(self.baseline_noise_std, self.spike_noise_std) < 0:
            raise ValueError("noise standard deviations must be non-negative")
        if min(self.lambda_0, self.kappa, self.lambda_max, self.alpha, self.theta_crit) < 0:
            raise ValueError("V3 damping gains and threshold must be non-negative")
        if self.sigma_sat < 0 or self.epsilon <= 0:
            raise ValueError("sigma_sat must be non-negative and epsilon must be positive")


def _build_kernel(jax, jnp, config: GPUConfig):
    """Build the batched Equation 11/12/13 and projector computation."""

    @jax.jit
    def kernel(clean, noise):
        thesis = clean[jnp.newaxis, :] + noise
        antithesis = clean[jnp.newaxis, :] - noise
        projected = 0.5 * (thesis + antithesis)

        logits_mean = jnp.mean(thesis, axis=1)
        logits_variance = jnp.var(thesis, axis=1)
        gradient_variance = jnp.var(noise, axis=1)
        xi = 1.0 - logits_variance / (logits_mean + config.epsilon)

        circuit_a = config.lambda_0 * jnp.exp(-config.kappa * (1.0 - xi))
        circuit_b_input = config.alpha * (
            gradient_variance / (config.sigma_sat**2 * config.epsilon)
            - config.theta_crit
        )
        circuit_b = config.lambda_max * jax.nn.sigmoid(circuit_b_input)
        chi = circuit_a + circuit_b
        damped_logits = thesis * jnp.exp(-chi[:, jnp.newaxis])

        return (
            thesis,
            antithesis,
            projected,
            logits_mean,
            logits_variance,
            gradient_variance,
            xi,
            circuit_a,
            circuit_b,
            chi,
            jnp.var(damped_logits, axis=1),
            jnp.mean(jnp.square(projected - clean[jnp.newaxis, :]), axis=1),
        )

    return kernel


def _load_jax():
    try:
        import jax
        import jax.numpy as jnp
        import jaxlib
    except ImportError as error:
        raise RuntimeError(
            "JAX is required. In Colab, select a GPU runtime and install a CUDA-enabled "
            "JAX build only if the runtime does not already provide one."
        ) from error
    return jax, jnp, jaxlib


def run_gpu_demo(config: GPUConfig, output_dir: Path, require_a100: bool = False) -> dict:
    config.validate()
    jax, jnp, jaxlib = _load_jax()
    devices = jax.devices()
    gpu_devices = [device for device in devices if device.platform in ("gpu", "cuda", "rocm")]
    if not gpu_devices:
        raise RuntimeError(
            f"No GPU backend is active (JAX default backend: {jax.default_backend()!r}). "
            "Select a GPU runtime in Colab; this demo intentionally does not fall back to CPU."
        )
    device_names = [device.device_kind for device in gpu_devices]
    if require_a100 and not any("A100" in name.upper() for name in device_names):
        raise RuntimeError(f"--require-a100 was set, but detected GPU(s): {device_names}")

    key = jax.random.PRNGKey(config.seed)
    key, clean_key, noise_key = jax.random.split(key, 3)
    clean = 2.0 + 0.25 * jax.random.normal(clean_key, (config.dimension,), dtype=jnp.float32)
    standard_noise = jax.random.normal(
        noise_key, (config.cycles, config.dimension), dtype=jnp.float32
    )
    cycle_ids = jnp.arange(config.cycles)
    noise_scales = jnp.where(
        (cycle_ids >= config.spike_start) & (cycle_ids <= config.spike_end),
        config.spike_noise_std,
        config.baseline_noise_std,
    ).astype(jnp.float32)
    noise = standard_noise * noise_scales[:, jnp.newaxis]

    kernel = _build_kernel(jax, jnp, config)
    outputs = kernel(clean, noise)
    jax.block_until_ready(outputs)
    timings_ms = []
    for _ in range(config.repeats):
        start = time.perf_counter()
        outputs = kernel(clean, noise)
        jax.block_until_ready(outputs)
        timings_ms.append((time.perf_counter() - start) * 1000.0)

    (
        _thesis,
        _antithesis,
        _projected,
        logits_mean,
        logits_variance,
        gradient_variance,
        xi,
        circuit_a,
        circuit_b,
        chi,
        damped_variance,
        projection_mse,
    ) = jax.device_get(outputs)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "sri_gpu_telemetry.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = (
            "cycle",
            "noise_std",
            "logits_mean",
            "logits_variance",
            "gradient_variance",
            "xi_stability",
            "circuit_a",
            "circuit_b",
            "chi_v3",
            "damped_variance",
            "known_noise_projection_mse",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(config.cycles):
            writer.writerow(
                {
                    "cycle": index,
                    "noise_std": (
                        config.spike_noise_std
                        if config.spike_start <= index <= config.spike_end
                        else config.baseline_noise_std
                    ),
                    "logits_mean": float(logits_mean[index]),
                    "logits_variance": float(logits_variance[index]),
                    "gradient_variance": float(gradient_variance[index]),
                    "xi_stability": float(xi[index]),
                    "circuit_a": float(circuit_a[index]),
                    "circuit_b": float(circuit_b[index]),
                    "chi_v3": float(chi[index]),
                    "damped_variance": float(damped_variance[index]),
                    "known_noise_projection_mse": float(projection_mse[index]),
                }
            )

    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metadata = {
        "source_document": SOURCE_DOCUMENT,
        "source_equations": {
            "xi": "1 - Var(logits) / (E(logits) + epsilon)",
            "circuit_a": "lambda_0 * exp(-kappa * (1 - Xi))",
            "circuit_b": (
                "lambda_max * sigmoid(alpha * "
                "(Var(gradient) / (sigma_sat^2 * epsilon) - theta_crit))"
            ),
            "chi_v3": "circuit_a + circuit_b",
            "logit_rescaling": "logits_raw * exp(-chi_v3)",
            "projection": "0.5 * ((v + xi_noise) + (v - xi_noise))",
        },
        "configuration": asdict(config),
        "runtime": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "jax_version": jax.__version__,
            "jaxlib_version": jaxlib.__version__,
            "default_backend": jax.default_backend(),
            "devices": [
                {
                    "platform": device.platform,
                    "device_kind": device.device_kind,
                    "device_id": device.id,
                }
                for device in devices
            ],
            "gpu_devices_used": device_names,
            "array_dtype": "float32",
        },
        "measurement": {
            "timing_scope": "JIT compilation excluded; full batched kernel synchronized to host",
            "repeat_count": config.repeats,
            "execution_ms_each_repeat": timings_ms,
            "execution_ms_median": statistics.median(timings_ms),
            "known_noise_projection_mse_max": max(float(value) for value in projection_mse),
            "note": (
                "Synthetic known-signal experiment. Exact injected noise is available to the "
                "paired-state projector. Not blind denoising, factuality validation, or a "
                "hardware-independent performance claim."
            ),
        },
        "provenance": {
            "script_sha256": script_hash,
            "generated_result": True,
            "source_equations_only": True,
        },
        "telemetry_csv": csv_path.name,
    }
    report_path = output_dir / "sri_gpu_run.json"
    report_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print("SRI GPU-first reference run")
    print(f"GPU(s): {', '.join(device_names)}")
    print(f"shape: cycles={config.cycles}, dimension={config.dimension}, dtype=float32")
    print(f"median synchronized kernel time: {statistics.median(timings_ms):.6g} ms")
    print(
        "maximum known-noise projection MSE: "
        f"{metadata['measurement']['known_noise_projection_mse_max']:.6g}"
    )
    print(f"telemetry: {csv_path}")
    print(f"run metadata: {report_path}")
    print("Synthetic known-noise result; not evidence of blind denoising or hallucination prevention.")
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the dissertation-equation SRI simulation on an active GPU."
    )
    parser.add_argument("--seed", type=int, default=108)
    parser.add_argument("--dimension", type=int, default=16_384)
    parser.add_argument("--cycles", type=int, default=25)
    parser.add_argument("--spike-start", type=int, default=10)
    parser.add_argument("--spike-end", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("sri_gpu_results"))
    parser.add_argument("--require-a100", action="store_true")
    args = parser.parse_args(argv)
    config = GPUConfig(
        seed=args.seed,
        dimension=args.dimension,
        cycles=args.cycles,
        spike_start=args.spike_start,
        spike_end=args.spike_end,
        repeats=args.repeats,
    )
    try:
        run_gpu_demo(config, args.output_dir, require_a100=args.require_a100)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
