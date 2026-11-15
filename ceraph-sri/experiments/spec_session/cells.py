"""One command per cell. Each cell writes a JSON summary and one histogram.

Paste the printed JSON back. Keep the PNG beside it. Run from the repository
root on the single GPU you rented.

    python experiments/spec_session/cells.py 0
    python experiments/spec_session/cells.py 1
    python experiments/spec_session/cells.py 2
    python experiments/spec_session/cells.py 3
    python experiments/spec_session/cells.py 4
    python experiments/spec_session/cells.py 5
"""
from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path("spec_session_out")


def _save(cell: int, payload: dict, figure) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"cell_{cell}_histogram.png"
    figure.savefig(png, dpi=140)
    import matplotlib.pyplot as plt
    plt.close(figure)
    payload["histogram"] = str(png)
    path = OUT / f"cell_{cell}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


def _figure(title: str):
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.set_title(title)
    return figure, axis


def _nvidia_query(fields: str) -> list[str]:
    if shutil.which("nvidia-smi") is None:
        return []
    completed = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        return []
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def cell_0() -> None:
    rows = _nvidia_query("name,driver_version,memory.total,power.draw,power.limit,temperature.gpu")
    figure, axis = _figure("Cell 0 — device probe")
    axis.axis("off")
    axis.text(0.0, 0.5, "\n".join(rows) or "nvidia-smi not available", family="monospace", va="center")
    _save(0, {
        "cell": 0,
        "name": "probe",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "nvidia_smi": rows or None,
        "stop_if_power_blank": True,
    }, figure)


def cell_1() -> None:
    powers, temps = [], []
    deadline = time.time() + 20
    while time.time() < deadline:
        rows = _nvidia_query("power.draw,temperature.gpu")
        if rows:
            parts = [part.strip() for part in rows[0].split(",")]
            try:
                powers.append(float(parts[0]))
                temps.append(float(parts[1]))
            except (IndexError, ValueError):
                pass
        time.sleep(1)
    figure, axis = _figure("Cell 1 — idle power samples")
    if powers:
        axis.hist(powers, bins=8, color="#7ee0c6")
        axis.set_xlabel("Power (W)")
        axis.set_ylabel("Samples")
    else:
        axis.text(0.5, 0.5, "No numeric power samples", ha="center")
    _save(1, {
        "cell": 1,
        "name": "idle_20s",
        "samples": len(powers),
        "mean_power_w": (sum(powers) / len(powers)) if powers else None,
        "min_temperature_c": min(temps) if temps else None,
        "max_temperature_c": max(temps) if temps else None,
    }, figure)


def _time_cpu(length: int, repeats: int) -> list[float]:
    values = np.random.default_rng(108).standard_normal(length).astype(np.float32)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        transformed = np.sin(values) * np.cos(0.5 * values) + np.tanh(values)
        float(np.sum(transformed * transformed))
        samples.append((time.perf_counter() - start) * 1e3)
    return samples


def cell_2() -> None:
    lengths = [4096, 16384, 65536, 262144, 1048576]
    rows = []
    for length in lengths:
        samples = _time_cpu(length, 5)
        rows.append({"length": length, "median_ms": _median(samples)})
    slope = float(np.polyfit(np.log([r["length"] for r in rows]), np.log([r["median_ms"] for r in rows]), 1)[0])
    figure, axis = _figure("Cell 2 — CPU time versus length")
    axis.bar([str(r["length"]) for r in rows], [r["median_ms"] for r in rows], color="#e0a36b")
    axis.set_ylabel("Median ms")
    axis.tick_params(axis="x", rotation=30)
    _save(2, {"cell": 2, "name": "cpu_scaling", "log_log_slope": slope, "rows": rows}, figure)


def cell_3() -> None:
    try:
        import torch
    except ImportError:
        torch = None
    lengths = [4096, 65536, 1048576]
    rows = []
    if torch is not None and torch.cuda.is_available():
        for length in lengths:
            values = torch.randn(length, device="cuda", dtype=torch.float32)
            starter, ender = torch.cuda.Event(True), torch.cuda.Event(True)
            samples = []
            for _ in range(8):
                starter.record()
                transformed = torch.sin(values) * torch.cos(0.5 * values) + torch.tanh(values)
                torch.sum(transformed * transformed)
                ender.record()
                torch.cuda.synchronize()
                samples.append(float(starter.elapsed_time(ender)))
            rows.append({"length": length, "median_ms": _median(samples), "samples_ms": samples})
    figure, axis = _figure("Cell 3 — GPU map-reduce latency")
    if rows:
        axis.hist(np.concatenate([r["samples_ms"] for r in rows]), bins=12, color="#7ee0c6")
        axis.set_xlabel("Latency (ms)")
    else:
        axis.text(0.5, 0.5, "No CUDA torch device", ha="center")
    public = [{"length": r["length"], "median_ms": r["median_ms"]} for r in rows]
    _save(3, {"cell": 3, "name": "gpu_map_reduce", "rows": public, "ran": bool(rows)}, figure)


def cell_4() -> None:
    from sri_gpu_demo import GPUConfig, run_gpu_demo

    rows = []
    for dimension in (4096, 16384):
        try:
            report = run_gpu_demo(
                GPUConfig(dimension=dimension, repeats=8, cycles=25, spike_start=10, spike_end=20),
                OUT / f"gpu_demo_{dimension}",
            )
            rows.append({
                "dimension": dimension,
                "median_ms": report["measurement"]["execution_ms_median"],
                "devices": report["runtime"].get("gpu_devices_used"),
            })
        except (RuntimeError, ValueError) as error:
            rows.append({"dimension": dimension, "ran": False, "reason": str(error)})
    figure, axis = _title_bars("Cell 4 — existing GPU demo", rows, "dimension", "median_ms")
    _save(4, {"cell": 4, "name": "gpu_demo", "rows": rows}, figure)


def _title_bars(title: str, rows: list[dict], label_key: str, value_key: str):
    figure, axis = _figure(title)
    labels, values = [], []
    for row in rows:
        if value_key in row and row[value_key] is not None:
            labels.append(str(row[label_key]))
            values.append(row[value_key])
    if values:
        axis.bar(labels, values, color="#8fb8e8")
        axis.set_ylabel(value_key)
    else:
        axis.text(0.5, 0.5, "No numeric results", ha="center")
    return figure, axis


def cell_5() -> None:
    from experiments.signal_benchmark import run as run_signal

    summary = run_signal(OUT / "signal", seed=108, train_count=40, test_count=80, length=128, noise_std=0.5)
    means = summary["mean_mse"]
    figure, axis = _figure("Cell 5 — synthetic error by method")
    names = ["moving_average_mse", "median_mse", "paired_mse"]
    axis.bar(names, [means[name] for name in names], color="#c7b0e0")
    axis.tick_params(axis="x", rotation=20)
    axis.set_ylabel("Mean squared error")
    _save(5, {"cell": 5, "name": "signal", "mean_mse": means}, figure)


def main(argv: list[str] | None = None) -> int:
    choice = (argv or sys.argv[1:])[:1]
    cells = {"0": cell_0, "1": cell_1, "2": cell_2, "3": cell_3, "4": cell_4, "5": cell_5}
    if not choice or choice[0] not in cells:
        print("usage: python experiments/spec_session/cells.py <0|1|2|3|4|5>")
        return 2
    cells[choice[0]]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
