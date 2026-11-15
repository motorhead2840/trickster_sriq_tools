# RTX 6000 × 4 session

Run this on the DGX machine that has four RTX PRO 6000 GPUs. The program is
`experiments/dgx_rtx6000x4.py`. It does not need the rest of the repository
beyond that file, numpy, torch, and matplotlib.

Stop the instance when the folder `spec_session_out/rtx6000x4` has been downloaded.

## 1. Choose the machine

Pick the instance with **4× NVIDIA RTX PRO 6000**. Four L4s were an earlier,
smaller test. This session uses the four boards you have now. One GPU still
runs the quality and algebra checks. Four GPUs add a latency curve per board,
a sharded-versus-single timing, and a power and temperature trace per board.

## 2. Open a terminal on that machine

A notebook terminal or a plain shell is enough. Confirm four boards:

```bash
nvidia-smi --query-gpu=index,name,memory.total,power.limit --format=csv
```

You want four rows, each named RTX PRO 6000 (or RTX 6000), before you continue.

## 3. Install the Python packages into this environment

Blackwell boards (compute capability 12.0) need a CUDA 13 PyTorch build.
Ada boards accept the default CUDA 12 build. Try the CUDA 13 line first.

```bash
python3 -m pip install --upgrade pip
python3 -m pip install numpy matplotlib
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu130
python3 -c "import torch; print(torch.__version__, torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
```

If that torch wheel refuses to start, reinstall the CUDA 12.6 build and print
the same line again:

```bash
python3 -m pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126
```

`device_count()` must be 4 before the session.

## 4. Put the runner on the machine

If this repository is already cloned:

```bash
python3 experiments/dgx_rtx6000x4.py --output spec_session_out/rtx6000x4
```

If it is not cloned, copy `experiments/dgx_rtx6000x4.py` from the Phase 4
branch onto the machine as `rtx6000x4.py` and run:

```bash
python3 rtx6000x4.py --output spec_session_out/rtx6000x4
```

Do not pass `--allow-fewer-gpus` on this session. With fewer than four boards
the runner records `NEED_4_GPUS` and skips the device section.

## 5. What it measures

CPU, same on every machine:

- Raw and flux Abhaya against a moving average, offset 0 and offset 5.
- Flux variance remaining.
- Abstain sweep and best F1.
- Paired cancellation, sign-bit XOR, Fibonacci window of 13, omega-dit sample.

On each of the four GPUs:

- Median time at lengths 16,384, 65,536, 262,144, and 1,048,576.
- Log-log slope, scored sublinear if the slope is under 1 and near-flat if under 0.15.
- A 10,000-element launch scored against the internal 0.2379 ms budget.

On GPU 0:

- fp32, fp16, and bf16 medians at length 1,048,576, plus an fp8 allocation check.

Across the four GPUs:

- One board holding the whole vector versus the same element count split in four.
- Twenty-five power and temperature samples while all four boards run the map.

## 6. What to download

The folder `spec_session_out/rtx6000x4` contains:

- `session_report.json` — every number and every status word
- `tech_spec_notes.md` — tables for the technical specification
- `patent_spec_notes.md` — the measured sentences for the patent specification
- `figures/quality_mse_3d.png`
- `figures/abstain_3d.png`
- `figures/latency_by_gpu_3d.png`
- `figures/power_3d.png`
- `figures/temperature_3d.png`

Paste the printed JSON summary back here, and keep the folder. The figures are
the drawings for both specification documents.

## 7. Make the hybrid stack real

The cells above measure PyTorch and NumPy. The Witness stack is real only when Julia, JAX, and Triton agree on one array. After the four-GPU command, run:

```bash
# Julia, if the machine does not already have it
command -v julia || (curl -fsSL https://install.julialang.org | sh -s -- --yes)
export PATH="$HOME/.juliaup/bin:$PATH"
python3 -m pip install "jax[cuda13]" triton
python3 -c "from sri.omlang.hybrid_stack import run_stack; import json; print(json.dumps(run_stack(), indent=2))"
```

`stack_real` is true only when Julia's `@sphota` block matches the Python ladder, Julia's damping matches NumPy, JAX matches NumPy, and Triton matches NumPy on every visible GPU using that block. Paste that dictionary back with the four-GPU summary.

## 8. Stop the instance

Download first. Then stop or pause the four-GPU instance so it does not keep billing.
