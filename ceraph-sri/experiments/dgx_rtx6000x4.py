"""Four-GPU RTX 6000 session for the Witness claims.

Self-contained: numpy, and torch plus matplotlib when a GPU is present.
CPU claims always run. GPU claims run on every visible device and, when at
least four devices are present, also as a four-way shard.

Outputs under the chosen directory:
  session_report.json
  tech_spec_notes.md
  patent_spec_notes.md
  figures/*.png
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

SEED = 11
LENGTH = 8192
LAM = 1.0
WINDOW = 9
HALL_RATIO = 0.95
FUSION_RATIO = 0.90
SLOPE_MAX = 0.15
BUDGET_MS = 0.2379
LENGTHS = (16384, 65536, 262144, 1048576)
PHI = (1.0 + math.sqrt(5.0)) / 2.0


def window_reduce(values: np.ndarray, width: int, moment: str) -> np.ndarray:
    radius = width // 2
    series = np.asarray(values, dtype=np.float64)
    out = np.empty(series.shape, dtype=np.float64)
    for index in range(series.size):
        chunk = series[max(0, index - radius) : min(series.size, index + radius + 1)]
        out[index] = float(np.mean(chunk) if moment == "mean" else np.var(chunk))
    return out


def make_series(offset: float = 0.0, n: int = LENGTH, seed: int = SEED):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n) + offset
    observed = latent + rng.normal(scale=0.5, size=n)
    spikes = rng.random(n) < 0.10
    observed = observed.copy()
    observed[spikes] += rng.normal(scale=8.0, size=int(np.sum(spikes)))
    return latent, observed, spikes


def mse(estimate, latent) -> float:
    return float(np.mean((np.asarray(estimate) - np.asarray(latent)) ** 2))


def quality_block(offset: float) -> dict:
    latent, observed, _spikes = make_series(offset)
    center = window_reduce(observed, WINDOW, "mean")
    xi = window_reduce(observed, WINDOW, "var")
    raw = observed * np.exp(-LAM * xi)
    flux = center + (observed - center) * np.exp(-LAM * xi)
    average = window_reduce(observed, WINDOW, "mean")
    identity = mse(observed, latent)
    moving = mse(average, latent)
    raw_mse = mse(raw, latent)
    flux_mse = mse(flux, latent)
    bar = HALL_RATIO * min(identity, moving)
    remaining = float(np.var(flux) / np.var(observed))
    sigma = float(np.mean(np.abs(observed)))
    beta = 1.0
    upsilon = float(np.mean(center))
    flux_level = float(np.mean(np.abs(observed - center)))
    omega = sigma * beta * (upsilon + flux_level) * (1.0 / observed.size)
    return {
        "offset": offset,
        "identity_mse": identity,
        "moving_average_mse": moving,
        "raw_mse": raw_mse,
        "flux_mse": flux_mse,
        "raw": "SUPPORTED" if raw_mse <= bar else "NOT_MET",
        "flux": "SUPPORTED" if flux_mse <= bar else "NOT_MET",
        "flux_variance_remaining": remaining,
        "flux_variance": "SUPPORTED" if remaining <= 0.50 else "NOT_MET",
        "omega_dit": {
            "sigma": sigma,
            "beta": beta,
            "upsilon": upsilon,
            "xi": flux_level,
            "sample": omega,
        },
    }


def abstain_sweep() -> dict:
    _latent, observed, spikes = make_series(0.0)
    xi = window_reduce(observed, WINDOW, "var")
    rows = []
    for threshold in (0.5, 1, 2, 4, 8, 16, 32, 64):
        silent = xi > threshold
        tp = int(np.sum(silent & spikes))
        fp = int(np.sum(silent & ~spikes))
        fn = int(np.sum(~silent & spikes))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        rows.append({"threshold": threshold, "precision": precision, "recall": recall, "f1": f1})
    best = max(rows, key=lambda row: row["f1"])
    return {
        "spikes": int(np.sum(spikes)),
        "sweep": rows,
        "best": best,
        "abstain": "SUPPORTED" if best["f1"] >= 0.5 else "NOT_MET",
    }


def paired_algebra() -> dict:
    truth = np.linspace(-1.0, 1.0, 10000)
    noise = np.random.default_rng(5).normal(size=truth.size)
    projected = 0.5 * ((truth + noise) + (truth - noise))
    residual = float(np.max(np.abs(projected - truth)))
    bits = noise.astype(np.float32).view(np.uint32)
    sign_flipped = np.bitwise_xor(bits, np.uint32(0x80000000)).view(np.float32)
    xor_err = float(np.max(np.abs(sign_flipped + noise.astype(np.float32))))
    weights = np.array([PHI ** (-k) for k in range(13)], dtype=np.float64)
    return {
        "paired_residual": residual,
        "paired_cancellation": "SUPPORTED" if residual < 1e-5 else "NOT_MET",
        "sign_xor_max_abs_error": xor_err,
        "sign_xor": "SUPPORTED" if xor_err < 1e-5 else "NOT_MET",
        "fibonacci_window": 13,
        "fibonacci_weight_sum": float(weights.sum()),
        "helmholtz_algebra": "SUPPORTED" if residual < 1e-8 else "NOT_MET",
        "reading": "Paired tensors were both inputs. This is algebra, not measured heat.",
    }


def query_gpus() -> list[dict]:
    rows = []
    try:
        done = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,power.draw,power.limit,temperature.gpu,clocks.sm,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
        )
    except FileNotFoundError:
        return rows
    if done.returncode != 0:
        return rows
    for line in done.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 8:
            continue
        def num(value: str):
            try:
                return float(value)
            except ValueError:
                return None
        rows.append(
            {
                "index": int(float(parts[0])),
                "name": parts[1],
                "power_w": num(parts[2]),
                "limit_w": num(parts[3]),
                "temperature_c": num(parts[4]),
                "sm_clock_mhz": num(parts[5]),
                "utilization_pct": num(parts[6]),
                "memory_used_mib": num(parts[7]),
            }
        )
    return rows


def _median_ms(fn, repeats: int = 12) -> float:
    import torch

    fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    samples = []
    for _ in range(repeats):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    samples.sort()
    return samples[len(samples) // 2]


def gpu_scaling(device: int) -> dict:
    import torch

    torch.cuda.set_device(device)
    rows = []
    for n in LENGTHS:
        y = torch.randn(n, device=device)
        rows.append({"length": n, "median_ms": _median_ms(lambda y=y: y * torch.exp(-(y * y)))})
    log_n = np.log([row["length"] for row in rows])
    log_t = np.log([row["median_ms"] for row in rows])
    slope = float(np.polyfit(log_n, log_t, 1)[0])
    ten_k = torch.randn(10000, device=device)
    ten_k_ms = _median_ms(lambda: ten_k * torch.exp(-(ten_k * ten_k)), repeats=20)
    return {
        "device": device,
        "rows": rows,
        "log_log_slope": slope,
        "sublinear": "SUPPORTED" if slope < 1.0 else "NOT_MET",
        "near_flat": "SUPPORTED" if slope < SLOPE_MAX else "NOT_MET",
        "n10000_median_ms": ten_k_ms,
        "under_internal_budget": "SUPPORTED" if ten_k_ms < BUDGET_MS else "NOT_MET",
    }


def gpu_dtype(device: int) -> dict:
    import torch

    torch.cuda.set_device(device)
    rows = {}
    for dtype, name in ((torch.float32, "float32"), (torch.float16, "float16"), (torch.bfloat16, "bfloat16")):
        y = torch.randn(1048576, device=device, dtype=dtype)
        rows[name] = _median_ms(lambda y=y: y * torch.exp(-(y * y)))
    fp8 = "unavailable"
    if hasattr(torch, "float8_e4m3fn"):
        try:
            _ = torch.randn(1024, device=device).to(torch.float8_e4m3fn)
            fp8 = "allocated"
        except Exception as exc:  # noqa: BLE001
            fp8 = f"unavailable: {type(exc).__name__}"
    return {"device": device, "median_ms": rows, "float8": fp8}


def gpu_shard(devices: list[int], n: int = 4_194_304) -> dict:
    import torch

    shard = n // len(devices)
    tensors = [torch.randn(shard, device=device) for device in devices]

    def once():
        for tensor in tensors:
            _ = tensor * torch.exp(-(tensor * tensor))
        for device in devices:
            torch.cuda.synchronize(device)

    single = torch.randn(n, device=devices[0])

    def once_one():
        _ = single * torch.exp(-(single * single))
        torch.cuda.synchronize(devices[0])

    four_ms = _median_ms(once, repeats=8)
    one_ms = _median_ms(once_one, repeats=8)
    return {
        "devices": devices,
        "elements": n,
        "one_gpu_ms": one_ms,
        "sharded_ms": four_ms,
        "shards": len(devices),
        "sharded_sooner": "SUPPORTED" if four_ms < one_ms else "NOT_MET",
    }


def gpu_power(devices: list[int], pulls: int = 25) -> dict:
    import torch

    tensors = [torch.randn(1_048_576, device=device) for device in devices]
    samples = []
    for _ in range(pulls):
        for tensor in tensors:
            _ = tensor * torch.exp(-(tensor * tensor))
        for device in devices:
            torch.cuda.synchronize(device)
        samples.append(query_gpus())
    per = {device: {"power_w": [], "temperature_c": []} for device in devices}
    for sample in samples:
        for row in sample:
            if row["index"] in per and row["power_w"] is not None:
                per[row["index"]]["power_w"].append(row["power_w"])
                per[row["index"]]["temperature_c"].append(row["temperature_c"])
    summary = []
    for device, bucket in per.items():
        if not bucket["power_w"]:
            continue
        summary.append(
            {
                "device": device,
                "mean_power_w": float(np.mean(bucket["power_w"])),
                "max_temperature_c": float(np.max(bucket["temperature_c"])),
                "samples": bucket,
            }
        )
    return {"pulls": pulls, "gpus": summary}


def save_figures(out: Path, report: dict) -> dict[str, str]:
    import matplotlib.pyplot as plt

    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    paths = {}

    quality = report["cpu"]["quality"]
    labels = ["identity", "moving average", "raw Abhaya", "flux Abhaya"]
    fig = plt.figure(figsize=(10, 6), facecolor="#10141c")
    ax = fig.add_subplot(111, projection="3d")
    for column, block in enumerate(quality):
        heights = [block["identity_mse"], block["moving_average_mse"], block["raw_mse"], block["flux_mse"]]
        colors = ["#8aa0b8", "#e09a55", "#7d8cff", "#3ecf8e"]
        ax.bar3d(np.arange(4) + column * 0.0, np.full(4, column), np.zeros(4), 0.45, 0.45, heights, color=colors, shade=True)
    ax.set_xticks(np.arange(4) + 0.2)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["offset 0", "offset 5"])
    ax.set_zlabel("MSE")
    fig.suptitle("Abhaya quality, latent value withheld", color="#f4f7fb")
    path = figures / "quality_mse_3d.png"
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    paths["quality"] = str(path)

    sweep = report["cpu"]["abstain"]["sweep"]
    fig = plt.figure(figsize=(10, 6), facecolor="#10141c")
    ax = fig.add_subplot(111, projection="3d")
    xs = np.arange(len(sweep))
    for shift, key, color in ((0, "precision", "#7d8cff"), (1, "recall", "#3ecf8e"), (2, "f1", "#ffb703")):
        ax.bar3d(xs + shift * 0.25, np.full(len(xs), shift), np.zeros(len(xs)), 0.22, 0.5, [row[key] for row in sweep], color=color, shade=True)
    ax.set_xticks(xs + 0.3)
    ax.set_xticklabels([str(row["threshold"]) for row in sweep])
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["precision", "recall", "F1"])
    ax.set_zlabel("Score")
    fig.suptitle("Abstain sweep on window variance", color="#f4f7fb")
    path = figures / "abstain_3d.png"
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    paths["abstain"] = str(path)

    scaling = report["gpu"].get("scaling") or []
    if scaling:
        fig = plt.figure(figsize=(10, 6), facecolor="#10141c")
        ax = fig.add_subplot(111, projection="3d")
        for row in scaling:
            xs = np.arange(len(row["rows"]))
            ax.bar3d(xs, np.full(len(xs), row["device"]), np.zeros(len(xs)), 0.4, 0.4, [item["median_ms"] for item in row["rows"]], shade=True)
        ax.set_xticks(np.arange(len(LENGTHS)))
        ax.set_xticklabels([str(n) for n in LENGTHS], rotation=15)
        ax.set_ylabel("GPU")
        ax.set_zlabel("Median ms")
        fig.suptitle("Latency by length and GPU", color="#f4f7fb")
        path = figures / "latency_by_gpu_3d.png"
        fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
        paths["latency"] = str(path)

    gpus = (report["gpu"].get("power") or {}).get("gpus") or []
    if gpus:
        fig = plt.figure(figsize=(10, 6), facecolor="#10141c")
        ax = fig.add_subplot(111, projection="3d")
        for gpu in gpus:
            samples = gpu["samples"]["power_w"]
            xs = np.arange(len(samples))
            ax.bar3d(xs, np.full(len(xs), gpu["device"]), np.zeros(len(xs)), 0.5, 0.45, samples, shade=True)
        ax.set_ylabel("GPU")
        ax.set_zlabel("Power W")
        fig.suptitle("Board power under the sustained map", color="#f4f7fb")
        path = figures / "power_3d.png"
        fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
        paths["power"] = str(path)
        fig = plt.figure(figsize=(10, 6), facecolor="#10141c")
        ax = fig.add_subplot(111, projection="3d")
        for gpu in gpus:
            samples = gpu["samples"]["temperature_c"]
            xs = np.arange(len(samples))
            ax.bar3d(xs, np.full(len(xs), gpu["device"]), np.zeros(len(xs)), 0.5, 0.45, samples, shade=True)
        ax.set_ylabel("GPU")
        ax.set_zlabel("Temperature C")
        fig.suptitle("Board temperature under the sustained map", color="#f4f7fb")
        path = figures / "temperature_3d.png"
        fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
        paths["temperature"] = str(path)
    return paths


def write_notes(out: Path, report: dict) -> None:
    quality = report["cpu"]["quality"]
    zero, offset = quality[0], quality[1]
    tech = [
        "# Technical specification notes — RTX 6000 session",
        "",
        f"Host: {report['host']}",
        f"GPUs visible: {report['gpu_count']}",
        "",
        "## Quality, latent value withheld",
        "",
        f"- Offset 0 flux MSE {zero['flux_mse']:.4f}, moving average {zero['moving_average_mse']:.4f}, status {zero['flux']}.",
        f"- Offset 5 flux MSE {offset['flux_mse']:.4f}, raw MSE {offset['raw_mse']:.4f}, raw status {offset['raw']}.",
        f"- Flux variance remaining {zero['flux_variance_remaining']:.4f} ({zero['flux_variance']}).",
        "",
        "## Algebra",
        "",
        f"- Paired residual {report['cpu']['algebra']['paired_residual']}.",
        f"- Sign-bit XOR error {report['cpu']['algebra']['sign_xor_max_abs_error']}.",
        "",
        "## Device",
        "",
        json.dumps({key: report["gpu"].get(key) for key in ("scaling", "dtype", "shard", "power")}, indent=2),
        "",
        "Figures:",
        "",
    ]
    for name, path in report.get("figures", {}).items():
        tech.append(f"- {name}: {path}")
    (out / "tech_spec_notes.md").write_text("\n".join(tech) + "\n", encoding="utf-8")
    patent = [
        "# Patent specification notes — this session only",
        "",
        "These notes record measurements from this run. They are source material for a specification. They are not a grant, and they do not describe a calorimeter result, a lithium-niobate crystal, or a 10,000-qubit device.",
        "",
        f"Flux Abhaya status at offset 0: {zero['flux']}. Flux Abhaya status at offset 5: {offset['flux']}. Raw Abhaya at offset 5: {offset['raw']}.",
        f"Paired cancellation: {report['cpu']['algebra']['paired_cancellation']}, with both tensors supplied.",
        f"Abstain best F1: {report['cpu']['abstain']['best']['f1']:.4f} ({report['cpu']['abstain']['abstain']}).",
        "",
        "Attach the PNG files in figures/ as the drawing set for latency, power, temperature, quality, and the abstain sweep.",
        "",
    ]
    (out / "patent_spec_notes.md").write_text("\n".join(patent), encoding="utf-8")


def run(out: Path, allow_fewer: bool) -> dict:
    try:
        import torch
    except ImportError:
        torch = None

    out.mkdir(parents=True, exist_ok=True)
    gpu_count = torch.cuda.device_count() if torch is not None and torch.cuda.is_available() else 0
    names = [torch.cuda.get_device_name(i) for i in range(gpu_count)] if torch is not None else []
    torch_version = torch.__version__ if torch is not None else None
    report = {
        "host": platform.platform(),
        "torch": torch_version,
        "gpu_count": gpu_count,
        "gpu_names": names,
        "idle": query_gpus(),
        "cpu": {
            "quality": [quality_block(0.0), quality_block(5.0)],
            "abstain": abstain_sweep(),
            "algebra": paired_algebra(),
        },
        "gpu": {},
        "four_gpu": gpu_count >= 4,
    }
    if gpu_count == 0:
        report["gpu"]["status"] = "NO_CUDA"
    elif gpu_count < 4 and not allow_fewer:
        report["gpu"]["status"] = "NEED_4_GPUS"
    else:
        devices = list(range(min(gpu_count, 4)))
        report["gpu"]["scaling"] = [gpu_scaling(device) for device in devices]
        report["gpu"]["dtype"] = gpu_dtype(devices[0])
        report["gpu"]["shard"] = gpu_shard(devices, n=1_048_576 * len(devices))
        report["gpu"]["power"] = gpu_power(devices)
        report["gpu"]["status"] = "RAN"
    report["figures"] = save_figures(out, report)
    (out / "session_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_notes(out, report)
    summary = {
        "output": str(out),
        "gpu_count": gpu_count,
        "gpu_names": names,
        "flux_offset0": report["cpu"]["quality"][0]["flux"],
        "raw_offset5": report["cpu"]["quality"][1]["raw"],
        "abstain": report["cpu"]["abstain"]["abstain"],
        "paired": report["cpu"]["algebra"]["paired_cancellation"],
        "sign_xor": report["cpu"]["algebra"]["sign_xor"],
        "gpu_status": report["gpu"].get("status"),
        "figures": report["figures"],
    }
    print(json.dumps(summary, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="RTX 6000 x4 Witness session")
    parser.add_argument("--output", type=Path, default=Path("spec_session_out/rtx6000x4"))
    parser.add_argument("--allow-fewer-gpus", action="store_true")
    args = parser.parse_args()
    run(args.output, args.allow_fewer_gpus)


if __name__ == "__main__":
    main()
