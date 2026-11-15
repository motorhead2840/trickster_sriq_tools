"""One-session measurement kit for a rented NVIDIA GPU.

Records device identity, optional board power and temperature, CPU scaling,
a synthetic denoising check, and the existing GPU demo when JAX can see a GPU.

This does not estimate a data-center bill. Energy is reported only when
nvidia-smi returns numeric power samples for the same interval.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from experiments.signal_benchmark import run as run_signal


def _median(samples: list[float]) -> float:
    ordered = sorted(samples)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def probe_nvidia_smi() -> dict:
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return {"available": False, "reason": "nvidia-smi not on PATH"}
    query = [
        binary,
        "--query-gpu=name,uuid,driver_version,memory.total,power.draw,power.limit,temperature.gpu,clocks.sm",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(query, check=False, text=True, capture_output=True)
    return {
        "available": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


class PowerLog:
    def __init__(self, path: Path):
        self.path = path
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[dict] = []

    def start(self) -> None:
        if shutil.which("nvidia-smi") is None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["t_unix", "power_w", "temperature_c", "sm_clock_mhz", "raw"],
            )
            writer.writeheader()
            while not self._stop.is_set():
                completed = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=power.draw,temperature.gpu,clocks.sm",
                        "--format=csv,noheader,nounits",
                    ],
                    check=False,
                    text=True,
                    capture_output=True,
                )
                raw = completed.stdout.strip().splitlines()[:1]
                row = {"t_unix": time.time(), "power_w": "", "temperature_c": "", "sm_clock_mhz": "", "raw": raw[0] if raw else completed.stderr.strip()}
                if raw:
                    parts = [part.strip() for part in raw[0].split(",")]
                    if len(parts) >= 3:
                        row["power_w"], row["temperature_c"], row["sm_clock_mhz"] = parts[:3]
                writer.writerow(row)
                handle.flush()
                self.samples.append(row)
                self._stop.wait(1.0)

    def summary(self) -> dict:
        powers = []
        temps = []
        for row in self.samples:
            try:
                powers.append(float(row["power_w"]))
                temps.append(float(row["temperature_c"]))
            except (TypeError, ValueError):
                continue
        if not powers:
            return {"samples": len(self.samples), "power_measured": False}
        return {
            "samples": len(self.samples),
            "power_measured": True,
            "mean_power_w": sum(powers) / len(powers),
            "max_power_w": max(powers),
            "min_temperature_c": min(temps) if temps else None,
            "max_temperature_c": max(temps) if temps else None,
        }


def time_map_reduce(length: int, repeats: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    values = rng.standard_normal(length).astype(np.float32)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        transformed = np.sin(values) * np.cos(0.5 * values) + np.tanh(values)
        float(np.sum(transformed * transformed))
        samples.append((time.perf_counter() - start) * 1e3)
    return {"length": length, "median_ms": _median(samples), "min_ms": min(samples), "max_ms": max(samples)}


def maybe_gpu_demo(output_dir: Path, dimension: int, repeats: int) -> dict:
    try:
        from sri_gpu_demo import GPUConfig, run_gpu_demo
    except Exception as error:  # noqa: BLE001
        return {"ran": False, "reason": f"import failed: {error}"}
    try:
        report = run_gpu_demo(
            GPUConfig(dimension=dimension, repeats=repeats, cycles=25, spike_start=10, spike_end=20),
            output_dir / f"gpu_demo_{dimension}",
        )
    except (RuntimeError, ValueError) as error:
        return {"ran": False, "reason": str(error)}
    measurement = report.get("measurement", {})
    return {
        "ran": True,
        "dimension": dimension,
        "median_ms": measurement.get("execution_ms_median"),
        "devices": report.get("runtime", {}).get("gpu_devices_used"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("spec_session_out"))
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096, 16384, 65536, 262144])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--gpu-dimensions", type=int, nargs="*", default=[4096, 16384])
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    power = PowerLog(args.output_dir / "nvidia_smi.csv")
    power.start()
    scaling = [time_map_reduce(length, args.repeats, 108) for length in args.lengths]
    gpu_runs = [maybe_gpu_demo(args.output_dir, dimension, args.repeats) for dimension in args.gpu_dimensions]
    signal = run_signal(args.output_dir / "signal", seed=108, train_count=40, test_count=80, length=128, noise_std=0.5)
    time.sleep(1.2)
    power.stop()
    elapsed_s = time.time() - started
    power_summary = power.summary()
    energy_j = None
    if power_summary.get("power_measured"):
        energy_j = power_summary["mean_power_w"] * elapsed_s
    report = {
        "experiment": "spec_session_measurement",
        "host": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
        },
        "nvidia_smi_probe": probe_nvidia_smi(),
        "elapsed_s": elapsed_s,
        "power": power_summary,
        "energy_joules_if_power_sampled": energy_j,
        "energy_note": "Mean board watts times wall seconds for this process interval. Not a facility bill, and absent when power.draw is blank.",
        "cpu_map_reduce": scaling,
        "gpu_demo": gpu_runs,
        "signal_mean_mse": signal["mean_mse"],
    }
    (args.output_dir / "spec_session.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("elapsed_s", "power", "energy_joules_if_power_sampled", "cpu_map_reduce", "gpu_demo", "signal_mean_mse")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
