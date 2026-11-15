# DGX round — Abhaya-gate and hybrid speed

**When:** the next NVIDIA DGX session. This file is the run card.  
**Scorer:** `experiments/dgx_abhaya_round.py`  
**CPU preview already run:** `docs/benchmark-results/dgx-abhaya/cpu_preview.json`

The Colab archive gives us a set of claims worth taking onto the machine.
Two of them already clear their bars on CPU. The rest are scheduled because
the saved cells point at them: sub-millisecond Triton times, variance driven
down, a cool board, and a hybrid Julia / JAX / Triton stack. Paste the cells
below in order. Each one prints a dictionary. Bring those dictionaries back.

A printed number is not a competitor. The internal budget **0.2379 ms** is
re-timed because the dissertation cell beat it with a measured **0.0082 ms**.
Beating that budget again supports the replication. The budget itself is not evidence of a timed opponent, and it does not replace the named-operator comparison.

## Locked bars

| Bar | Value |
|---|---|
| Seed, length, λ, window | 11, 8192, 1.0, 9 |
| Hallucination pass | Abhaya MSE ≤ **0.95** × the better of identity and moving average |
| Flux variance | Remaining variance ≤ **0.50** of the observation variance |
| Fusion | Triton median ≤ **0.90** × eager PyTorch of the same map |
| Near-flat scaling | log-log slope **< 0.15** |
| Sublinear scaling | log-log slope **< 1** |
| Thermal | Fewer joules, temperature within **5** °C, and under half the board limit |
| Sphota | Scheduled block within 10% of the best measured block |
| Paired residual | Max absolute error **< 1e-5** when the paired tensors are inputs |

## What the CPU preview already shows

Zero-mean spikes, latent value withheld from the estimator:

| Estimator | MSE |
|---|---:|
| Observation | 6.910 |
| Moving average | 1.675 |
| Raw Abhaya | **0.774** |
| Flux Abhaya | **1.472** |

Flux variance remaining: **0.117**. The same flux numbers hold when the carrier
is shifted by 5. The raw map on that shift rises to **18.17** and does not pass.
Both readings are reported.

## Cells

Run them on the DGX notebook. CUDA is required from cell 1 onward. Cell 0 is the
CPU confirmation and can run anywhere.

### Cell 0 — confirm the CPU preview

```python
import numpy as np

SEED, N, LAM, WINDOW = 11, 8192, 1.0, 9

def window_reduce(values, width, moment):
    radius = width // 2
    series = np.asarray(values, dtype=np.float64)
    out = np.empty_like(series)
    for i in range(series.size):
        chunk = series[max(0, i - radius):min(series.size, i + radius + 1)]
        out[i] = float(np.mean(chunk) if moment == "mean" else np.var(chunk))
    return out

def series(offset):
    rng = np.random.default_rng(SEED)
    latent = rng.normal(size=N) + offset
    observed = latent + rng.normal(scale=0.5, size=N)
    spikes = rng.random(N) < 0.10
    observed = observed.copy()
    observed[spikes] += rng.normal(scale=8.0, size=int(spikes.sum()))
    return latent, observed

def score(offset):
    latent, observed = series(offset)
    center = window_reduce(observed, WINDOW, "mean")
    xi = window_reduce(observed, WINDOW, "var")
    raw = observed * np.exp(-LAM * xi)
    flux = center + (observed - center) * np.exp(-LAM * xi)
    avg = window_reduce(observed, WINDOW, "mean")
    def mse(est):
        return float(np.mean((est - latent) ** 2))
    identity, ma, raw_mse, flux_mse = mse(observed), mse(avg), mse(raw), mse(flux)
    bar = 0.95 * min(identity, ma)
    return {
        "offset": offset,
        "identity_mse": identity,
        "moving_average_mse": ma,
        "raw_mse": raw_mse,
        "flux_mse": flux_mse,
        "raw": "SUPPORTED" if raw_mse <= bar else "NOT_MET",
        "flux": "SUPPORTED" if flux_mse <= bar else "NOT_MET",
        "flux_variance_remaining": float(np.var(flux) / np.var(observed)),
    }

print({"cell": 0, "zero": score(0.0), "offset": score(5.0)})
```

### Cell 1 — name the board

```python
import subprocess, torch
line = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,power.draw,power.limit,temperature.gpu", "--format=csv,noheader,nounits"],
    text=True, capture_output=True,
).stdout.strip().splitlines()[0]
name, power, limit, temp = [part.strip() for part in line.split(",")]
print({
    "cell": 1,
    "gpu": name,
    "torch": torch.__version__,
    "cuda": torch.cuda.is_available(),
    "idle_power_w": float(power),
    "board_limit_w": float(limit),
    "temperature_c": float(temp),
})
```

### Cell 2 — flux and raw maps on the GPU, timed against a moving average

```python
import time, subprocess
import numpy as np
import torch

SEED, N, LAM, WIDTH, REPEATS = 11, 1048576, 1.0, 9, 20

def score_cpu_match():
    rng = np.random.default_rng(SEED)
    latent = rng.normal(size=8192)
    observed = latent + rng.normal(scale=0.5, size=8192)
    return latent, observed

def gpu_maps(observed):
    y = observed
    center = torch.nn.functional.avg_pool1d(y.view(1, 1, -1), WIDTH, stride=1, padding=WIDTH // 2).view(-1)
    # Window variance from E[y^2] - (E[y])^2 on the same pool.
    y2 = torch.nn.functional.avg_pool1d((y * y).view(1, 1, -1), WIDTH, stride=1, padding=WIDTH // 2).view(-1)
    xi = (y2 - center * center).clamp_min(0)
    raw = y * torch.exp(-LAM * xi)
    flux = center + (y - center) * torch.exp(-LAM * xi)
    return raw, flux, center

def median_ms(fn):
    fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    samples = []
    for _ in range(REPEATS):
        start.record(); fn(); end.record(); torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    samples.sort()
    return samples[len(samples) // 2]

y = torch.randn(N, device="cuda")
raw_ms = median_ms(lambda: gpu_maps(y))
avg_ms = median_ms(lambda: torch.nn.functional.avg_pool1d(y.view(1, 1, -1), WIDTH, stride=1, padding=WIDTH // 2))
print({
    "cell": 2,
    "length": N,
    "flux_and_raw_median_ms": raw_ms,
    "avg_pool_median_ms": avg_ms,
    "fusion_versus_pool": "SUPPORTED" if raw_ms < avg_ms else "NOT_MET",
    "note": "Quality for this cell is decided by cell 0. Speed counts as a product win only together with that quality.",
})
```

### Cell 3 — Triton launch against eager PyTorch, and the block ladder

```python
import torch
import triton
import triton.language as tl

@triton.jit
def abhaya_kernel(theta_ptr, xi_ptr, out_ptr, n, lam, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    theta = tl.load(theta_ptr + offsets, mask=mask)
    xi = tl.load(xi_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, theta * tl.exp(-lam * xi), mask=mask)

def median_ms(fn, repeats=20):
    fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    samples = []
    for _ in range(repeats):
        start.record(); fn(); end.record(); torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    samples.sort()
    return samples[len(samples) // 2]

n = 1048576
theta = torch.randn(n, device="cuda")
xi = torch.rand(n, device="cuda")
out = torch.empty_like(theta)

def eager():
    return theta * torch.exp(-1.0 * xi)

def triton_at(block):
    grid = (triton.cdiv(n, block),)
    abhaya_kernel[grid](theta, xi, out, n, 1.0, BLOCK=block)

blocks = [128, 256, 512, 1024]
times = {block: median_ms(lambda block=block: triton_at(block)) for block in blocks}
best = min(times, key=times.get)
# Density ladder from sri.omlang.abhaya.sphota_schedule: mean(|sigma|) < 0.5 -> 128.
scheduled = 128
ref = median_ms(eager)
launched = triton_at(scheduled)
torch.cuda.synchronize()
err = float(torch.max(torch.abs(out - eager())).item())
print({
    "cell": 3,
    "triton_ms_by_block": times,
    "scheduled_block": scheduled,
    "best_block": best,
    "sphota": "SUPPORTED" if times[scheduled] <= 1.10 * times[best] else "NOT_MET",
    "eager_ms": ref,
    "fusion": "SUPPORTED" if times[scheduled] <= 0.90 * ref else "NOT_MET",
    "max_abs_error_versus_eager": err,
    "under_internal_budget_0_2379_ms": times[scheduled] < 0.2379,
})
```

### Cell 4 — scaling

```python
import math
import torch

def median_ms(fn, repeats=12):
    fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    samples = []
    for _ in range(repeats):
        start.record(); fn(); end.record(); torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    samples.sort()
    return samples[len(samples) // 2]

lengths = [16384, 65536, 262144, 1048576]
rows = []
for n in lengths:
    y = torch.randn(n, device="cuda")
    def once(y=y):
        return y * torch.exp(-y * y)
    rows.append({"length": n, "median_ms": median_ms(once)})

log_n = [math.log(row["length"]) for row in rows]
log_t = [math.log(row["median_ms"]) for row in rows]
mean_n, mean_t = sum(log_n) / len(log_n), sum(log_t) / len(log_t)
var_n = sum((x - mean_n) ** 2 for x in log_n)
slope = sum((x - mean_n) * (t - mean_t) for x, t in zip(log_n, log_t)) / var_n
print({
    "cell": 4,
    "rows": rows,
    "log_log_slope": slope,
    "sublinear": "SUPPORTED" if slope < 1.0 else "NOT_MET",
    "near_flat": "SUPPORTED" if slope < 0.15 else "NOT_MET",
})
```

### Cell 5 — paired cancellation, the cell that is allowed to see both tensors

```python
import torch
truth = torch.linspace(-1.0, 1.0, 10000, device="cuda")
noise = torch.randn_like(truth)
thesis = truth + noise
antithesis = truth - noise
projected = 0.5 * (thesis + antithesis)
residual = float(torch.max(torch.abs(projected - truth)).item())
print({
    "cell": 5,
    "length": 10000,
    "max_residual": residual,
    "paired_cancellation": "SUPPORTED" if residual < 1e-5 else "NOT_MET",
    "inputs": "truth and noise were both supplied",
})
```

### Cell 6 — joules and temperature under a sustained flux loop

```python
import subprocess
import torch

def sample():
    line = subprocess.run(
        ["nvidia-smi", "--query-gpu=power.draw,temperature.gpu,power.limit", "--format=csv,noheader,nounits"],
        text=True, capture_output=True,
    ).stdout.strip().splitlines()[0]
    power, temp, limit = [float(part.strip()) for part in line.split(",")]
    return power, temp, limit

y = torch.randn(1048576, device="cuda")
watts, temps = [], []
for _ in range(40):
    center = torch.nn.functional.avg_pool1d(y.view(1, 1, -1), 9, stride=1, padding=4)
    y2 = torch.nn.functional.avg_pool1d((y * y).view(1, 1, -1), 9, stride=1, padding=4)
    xi = (y2 - center * center).clamp_min(0)
    _ = center + (y.view(1, 1, -1) - center) * torch.exp(-xi)
    torch.cuda.synchronize()
    power, temp, limit = sample()
    watts.append(power)
    temps.append(temp)
mean_w = sum(watts) / len(watts)
print({
    "cell": 6,
    "mean_power_w": mean_w,
    "max_temperature_c": max(temps),
    "board_limit_w": limit,
    "headroom": "SUPPORTED" if max(temps) < 0.5 * 100 and mean_w < 0.5 * limit else "NOT_MET",
    "note": "Headroom uses a 100 C scale for the temperature half and half the board limit for power. Joules versus a second operator need that operator's own timed loop beside this one.",
})
```

### Cell 7 — labeled answers, only if a local model is already loaded

```python
print({
    "cell": 7,
    "labeled_generation": "OPEN",
    "why": "Run this only when the notebook already holds a model and a tokenizer. Score exact agreement on a list of questions whose answers you wrote down before generating. Accuracy per second has to meet or beat the ungated model. Untitled4's 56 suppressed tokens are the motive, not the score.",
})
```

## How a claim becomes one we say out loud

Cell 0's flux line and the matching GPU rerun, together, support the sentence
"on this spike series, damping the oscillation beat a moving average without
seeing the latent value." Cell 3's fusion line supports "the Triton launch of
this map was faster than eager PyTorch." Cell 4's sublinear line supports
"time grew slower than the length." Cell 5 supports the paired identity.
Cell 6 supports thermal headroom on that board.

The product sentence "faster and cooler than current compute" waits for cell 2
or cell 3 to beat a named operator while cell 0's quality bar still holds, and
for a second timed loop in cell 6. That is the cell pair to watch.

Cells 8–11, for the dissertation and notebook claims that this card did not
yet cover, are in [future-tech-register.md](future-tech-register.md). Paste
them after cell 7.
