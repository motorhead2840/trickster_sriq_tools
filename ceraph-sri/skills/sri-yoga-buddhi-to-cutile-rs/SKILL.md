# Skill: sri-yoga-buddhi-to-cutile-rs

Translate SRI Yoga–Buddhi / Lemma Z1 **tile reference ops** into a cuTile Rust backend using NVIDIA’s verdict-driven agent pattern (TileGym `tilegym-converting-python-to-rust`).

**Upstream method:** https://developer.nvidia.com/blog/translating-cuda-tile-operations-from-python-to-rust-using-agentic-ai/

**Activation:** user asks to port / convert / cutile-rs Yoga–Buddhi, Samya, observable antithesis, or Sphota GPU substrate.

## Orchestrator contract (lean)

You are a **pure orchestrator**. Do not author kernels in the top-level turn. Spawn / simulate stages; advance only on machine verdicts.

Stages (bounded retries):

| Stage | Owner | Required artifacts | Verdict keys |
|---|---|---|---|
| 0 preflight | scripts | toolchain note | `PREFLIGHT_OK` / `ENV_FAIL` |
| 1 analyze | analyzer | `analysis.json`, reference dump path | `ANALYSIS_OK` |
| 2 kernel | kernel writer | `kernel.rs` sketch or cutile module + in-Rust test log | `KERNEL_OK` / `KERNEL_FAIL` |
| 3 host/FFI | host builder | C-ABI sketch + Python wrapper + functional log | `HOST_ALL_PASS` / `HOST_FAIL` |
| 4 perf | perf validator | CUPTI or honest skip | `PERF_OK` / `PERF_SKIP` / `PERF_FAIL` |

Hard caps: analysis×1, kernel×2, host×2, perf×2. On fail, write diagnosis under `embodiment/cutile_rs_stub/diagnosis/` — do not thrash.

## Step 0 — preflight

Check and record:

- CUDA ≥ 13.1 and `tileiras` (if missing → `PERF_SKIP` / `ENV_FAIL` for GPU; CPU stub still allowed)
- Rust ≥ 1.89 preferred (repo may build CPU stub on older rustc)
- Blackwell available only for perf gate
- Reference Python: `sri.omlang.tile_ref` + `sri.omlang.yoga_buddhi`

## Step 1 — analyze

From `sri.omlang.tile_ref`, emit `analysis.json` with:

- operators: `oracle_samya_tile`, `observable_upsilon_tile`, `mirror_abs2_tile`
- dtypes: at least `f32` (extend later)
- shapes / tile sizes
- tolerances (match C5 / Lemma Z1)
- baseline backend: NumPy tile_ref (until a cuTile Python reference exists)

## Step 2 — kernel writer rules (distilled)

1. Prefer idiomatic safe tile surface; only use unchecked APIs if needed to match reference IR.  
2. Spell specializations in the signature (const tile size, dtype) — no Python-style implicit JIT.  
3. Oracle Samya must remain the latent involution identity when antithesis is supplied.  
4. Observable path must **not** take clean `s` or oracle `n`.  
5. Prove functionally **in-Rust / CPU stub first** before FFI.  
6. If CUDA Tile IR dumps exist, IR-diff before celebrating.

## Step 3 — host / FFI

- Zero-copy descriptor style (`ptr, ndim, shape, strides`) when bridging to Python/PyTorch/JAX.  
- SphotaCompiler should gain `prefer_cutile_rs=True` only after `HOST_ALL_PASS`.

## Step 4 — perf

- CUPTI device-time geomean ≥ 0.95 vs reference on Blackwell.  
- If no Blackwell/tileiras: emit `PERF_SKIP` with reason; do not fake numbers.

## Return format (every stage)

End with:

```text
<VALIDATOR_OUTPUT>
VERDICT: <NAME>
</VALIDATOR_OUTPUT>
```

Orchestrator routes only on that line + exit codes — never on chat vibes.
