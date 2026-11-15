"""Cross-backend validation of the ideal Gaussian photonic sensing model."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

if __package__:
    from .cv_photonics_benchmark import posterior
else:
    from cv_photonics_benchmark import posterior

CONDITIONS = (
    ("coherent_signal_only", 0.0, None),
    ("coherent_signal_and_reference", 0.0, 0.0),
    ("squeezed_signal_only", "r", None),
    ("squeezed_signal_and_reference", "r", 0.0),
    ("squeezed_reference", 0.0, "r"),
)


def configuration(shots, vs, vn, eta, coupling, squeezing):
    values = (vs, vn, eta, coupling, squeezing)
    if (shots < 2 or not all(math.isfinite(x) for x in values) or vs <= 0 or vn < 0
            or not 0 < eta <= 1 or coupling < 0 or not 0 <= squeezing <= 3):
        raise ValueError("Invalid parameters: require shots >= 2, Vs > 0, Vn >= 0, "
                         "0 < eta <= 1, k >= 0, and 0 <= r <= 3")


def coefficients(vs, vn, eta, coupling, squeezing):
    entries = []
    for name, ra, rb in CONDITIONS:
        ra = squeezing if ra == "r" else ra
        rb = squeezing if rb == "r" else rb
        coef, theory = posterior(vs, vn, eta, coupling, ra, rb)
        entries.append((name, ra, rb, coef[0], coef[1] if rb is not None else 0.0, theory))
    return entries


def calculate(xp, z, entries, vs, vn, eta, coupling):
    signal = math.sqrt(vs) * z[0]
    noise = math.sqrt(vn) * z[1]
    estimates = []
    for _, ra, rb, a, b, _ in entries:
        ya = math.sqrt(eta) * (signal + noise + math.exp(-ra) * z[2] / math.sqrt(2))
        ya = ya + math.sqrt((1 - eta) / 2) * z[4]
        if rb is None:
            estimates.append(a * ya)
        else:
            yb = math.sqrt(eta) * (coupling * noise + math.exp(-rb) * z[3] / math.sqrt(2))
            yb = yb + math.sqrt((1 - eta) / 2) * z[5]
            estimates.append(a * ya + b * yb)
    return xp.stack(estimates)


def julia_estimates(z, executable, source, args):
    if sys.byteorder != "little":
        raise RuntimeError("The Julia binary interchange requires a little-endian host")
    with tempfile.TemporaryDirectory() as directory:
        input_path = Path(directory) / "input.bin"
        output_path = Path(directory) / "output.bin"
        z.astype("<f4", copy=False).tofile(input_path)
        subprocess.run([executable, "--startup-file=no", str(source), str(input_path),
                        str(output_path), str(z.shape[1]), *(str(value) for value in args)],
                       check=True, timeout=300)
        result = np.fromfile(output_path, dtype="<f4")
        if result.size != 5 * z.shape[1]:
            raise RuntimeError("Julia returned an incorrect number of estimates")
        return result.reshape(5, z.shape[1])


def gpu_estimates(z, entries, vs, vn, eta, coupling):
    import torch
    import jax
    import jax.numpy as jnp
    import triton
    if __package__:
        from .cv_a100_kernels import kernel
    else:
        from cv_a100_kernels import kernel

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA. Select a Colab GPU runtime.")
    if "A100" not in torch.cuda.get_device_name(0).upper():
        raise RuntimeError(f"A100 required; got {torch.cuda.get_device_name(0)}")
    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("JAX has no GPU device; install a compatible CUDA-enabled JAX and restart.")
    device = next(device for device in jax.devices() if device.platform == "gpu")
    tz = torch.from_numpy(z).cuda()
    torch_result = calculate(torch, tz, entries, vs, vn, eta, coupling)
    torch.cuda.synchronize()
    jz = jax.device_put(z, device)
    jax_result = jax.jit(lambda x: calculate(jnp, x, entries, vs, vn, eta, coupling))(jz)
    jax_result.block_until_ready()

    output = torch.empty((len(entries), z.shape[1]), device="cuda", dtype=torch.float32)
    for j, (_, ra, rb, a, b, _) in enumerate(entries):
        kernel[(triton.cdiv(z.shape[1], 256),)](
            tz, output, z.shape[1], math.sqrt(vs), math.sqrt(vn), math.sqrt(eta),
            math.sqrt((1 - eta) / 2), coupling, math.exp(-ra) / math.sqrt(2),
            0.0 if rb is None else math.exp(-rb) / math.sqrt(2), a, b, j, 256)
    torch.cuda.synchronize()
    info = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "jax": jax.__version__,
            "jax_device": str(device), "triton": triton.__version__}
    return {"pytorch": torch_result.cpu().numpy(), "jax": np.asarray(jax_result),
            "triton": output.cpu().numpy()}, info


def run(output, shots=20000, seed=108, vs=1.0, vn=1.0, eta=0.8, coupling=1.0,
        squeezing=0.7, julia="julia", gpu=True):
    configuration(shots, vs, vn, eta, coupling, squeezing)
    executable = shutil.which(julia)
    if not executable:
        raise RuntimeError("Julia executable not found; install Julia or supply --julia PATH")
    source = Path(__file__).with_name("cv_photonics_julia.jl")
    entries = coefficients(vs, vn, eta, coupling, squeezing)
    z = np.random.default_rng(seed).standard_normal((6, shots), dtype=np.float32)
    reference = calculate(np, z, entries, vs, vn, eta, coupling)
    outputs = {"julia_cpu": julia_estimates(z, executable, source, (vs, vn, eta, coupling, squeezing))}
    environment = {"python": platform.python_version(), "numpy": np.__version__,
                   "julia": subprocess.run([executable, "--version"], check=True, capture_output=True,
                                           text=True, timeout=30).stdout.strip()}
    if gpu:
        gpu_outputs, info = gpu_estimates(z, entries, vs, vn, eta, coupling)
        outputs.update(gpu_outputs)
        environment.update(info)
        environment["jaxlib"] = importlib.metadata.version("jaxlib")
        environment["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=30).stdout.strip()
    checks = {}
    for name, values in outputs.items():
        if values.shape != reference.shape or not np.isfinite(values).all():
            raise RuntimeError(f"{name} returned invalid estimates")
        difference = np.abs(values - reference)
        checks[name] = {"max_abs_difference": float(difference.max()),
                        "tolerance_atol": 2e-6, "tolerance_rtol": 3e-4}
        if not np.allclose(values, reference, rtol=3e-4, atol=2e-6):
            raise RuntimeError(f"{name} differs from NumPy: max error {difference.max()}")
    signal = math.sqrt(vs) * z[0].astype(np.float64)
    errors = (reference.astype(np.float64) - signal[None, :]) ** 2
    results = {}
    for j, (name, _, _, _, _, theory) in enumerate(entries):
        mse = float(errors[j].mean())
        se = float(errors[j].std(ddof=1) / math.sqrt(shots))
        if abs(mse - theory) > 6 * se + 2e-6:
            raise RuntimeError(f"{name}: sampled MSE disagrees with analytic prediction")
        results[name] = {"sample_mse": mse, "sample_standard_error": se, "analytic_mse": theory}
    comparisons = {}
    for j in (1, 2, 3):
        delta = errors[j] - errors[4]
        mean = float(delta.mean())
        se = float(delta.std(ddof=1) / math.sqrt(shots))
        comparisons[entries[j][0]] = {"mse_reduction_vs_squeezed_reference": mean,
                                      "normal_approx_95_interval": [mean - 1.96 * se, mean + 1.96 * se]}
    report = {
        "status": "completed", "experiment": "classical_simulation_of_gaussian_photonic_sensing",
        "configuration": {"shots": shots, "seed": seed, "signal_variance": vs, "noise_variance": vn,
                          "efficiency": eta, "reference_noise_coupling": coupling, "squeezing_r": squeezing,
                          "gpu_validation_enabled": gpu},
        "environment": environment, "backend_checks": checks, "results": results, "comparisons": comparisons,
        "resources": {"squeezing_photons_per_squeezed_mode": math.sinh(squeezing) ** 2,
                      "signal_displacement_photons": vs / 2, "signal_noise_displacement_photons": vn / 2,
                      "reference_noise_displacement_photons": coupling ** 2 * vn / 2,
                      "mode_counts": [1, 2, 1, 2, 2],
                      "unmeasured": "Preparation, local oscillator, detector, control, calibration, energy and error correction."},
        "limitations": ["Classical Gaussian Monte Carlo, not a quantum processor or physical experiment.",
                        "Known distribution parameters and optimal Gaussian estimators; no quantum state is recovered.",
                        "Shared random numbers enable numerical comparison, not a physical paired measurement.",
                        "Intervals quantify Monte Carlo error only; no hardware uncertainty or proof of quantum advantage."],
        "input_sha256": hashlib.sha256(z.tobytes()).hexdigest(),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__), source, Path(__file__).with_name("cv_a100_kernels.py"),
                                    Path(__file__).with_name("cv_photonics_benchmark.py"))},
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "cv_a100_validation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--shots", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=108)
    parser.add_argument("--signal-variance", type=float, default=1.0)
    parser.add_argument("--noise-variance", type=float, default=1.0)
    parser.add_argument("--efficiency", type=float, default=0.8)
    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--squeezing", type=float, default=0.7)
    parser.add_argument("--julia", default="julia")
    parser.add_argument("--cpu-only", action="store_true", help="Validate NumPy and Julia only; no GPU claim")
    args = parser.parse_args()
    result = run(args.output_dir, shots=args.shots, seed=args.seed, vs=args.signal_variance,
                 vn=args.noise_variance, eta=args.efficiency, coupling=args.coupling,
                 squeezing=args.squeezing, julia=args.julia, gpu=not args.cpu_only)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
