# CUDA Tile → Rust (agentic) embodiment path for Sphota / Yoga–Buddhi

**Source:** [Translating CUDA Tile Operations from Python to Rust Using Agentic AI](https://developer.nvidia.com/blog/translating-cuda-tile-operations-from-python-to-rust-using-agentic-ai/) (NVIDIA, 2026-09-16; Liu / Elibol / Manur).

**SRI integration intent:** reuse NVIDIA’s *verdict-driven, IR-checkable* Python→Rust tile translation methodology for Yoga–Buddhi / Lemma Z1 kernels on Blackwell — without claiming we have already matched TileGym’s 0.995 geomean.

Related: [yoga-buddhi-projector.md](yoga-buddhi-projector.md), [sphota-yoga-buddhi-braket-runbook.md](sphota-yoga-buddhi-braket-runbook.md), [sree-mentor-jax-te-moe.md](sree-mentor-jax-te-moe.md).

---

## What NVIDIA showed (facts we adopt)

| Idea | NVIDIA result | Why it fits SRI |
|---|---|---|
| Shared **CUDA Tile IR** (`cuda_tile`) | cuTile Python, Triton-TileIR, cuTile Rust → same `tileiras` → cubin | Porting is re-expression, not re-optimization; **IR diff** catches structural lies |
| **Agent skill** with stage subagents | Analyzer → kernel writer → host/FFI → perf; machine `VERDICT:` lines | Same anti-sycophancy discipline as Darśana / C5: no stage advances on prose |
| Explicit specialization | Rust const-generics vs Python JIT implicits | Matches our Lemma Z1 lock: latent involution explicit; deployable maps typed |
| C-ABI zero-copy host | `TensorDesc` into PyTorch/Python | Matches Sphota host (Python/Julia) calling a GPU substrate |
| Perf bar | CUPTI geomean ≥ 0.95 vs cuTile Python; overall 0.995 on B200 | Future embodiment gate for Yoga–Buddhi tile ops — **not claimed yet** |

Pipeline (adopted):

```text
preflight → analyze (dump reference Tile IR + analysis.json)
         → write kernel.rs (in-Rust functional test + IR self-check)
         → host/FFI + Python wrapper (TileGym-style ALL_PASS)
         → CUPTI perf (≥0.95 geomean) with bounded IR-diff / residual-perf retries
```

Skill location upstream: `skills/tilegym-converting-python-to-rust/` in [NVIDIA/TileGym](https://github.com/NVIDIA/TileGym).

Requirements they state: CUDA 13.1+, Blackwell for perf check, Rust 1.89+, `tileiras`.

---

## Honesty boundary for this repo

- We **integrate the method and scaffolds**.
- We do **not** claim Yoga–Buddhi is already a cuTile Rust operator at TileGym parity.
- Colab Yoga–Buddhi successes remain **oracle / known-baseline** unless Lemma Z1 path is used.
- Softwall: no Landauer / hardware \(\mathcal O(0)\) claim from a future tile port alone.

---

## SRI mapping

| SRI object | Python/JAX/Julia today | cuTile Rust target (future) |
|---|---|---|
| Oracle Samya \(\hat\Upsilon_{YB}\) | `sri.omlang.yoga_buddhi.oracle_samya` | Element-wise tile: `samya = 0.5*(thesis+antithesis)` with known \(\xi\) |
| Observable \(\hat\Upsilon_{\mathrm{obs}}\) | Lemma Z1 / `observable_upsilon` | Fused tile: Wiener weights + antithesis + average (explicit dtypes/shapes) |
| Mirror \(\mathcal{M}\) | `mirror_validation` | Element-wise `|z|²` tile |
| Sphota host | `SphotaCompiler` + optional Julia | Python/Julia host → C-ABI → `cutile-rs` backend |

Agent skill in this repo: [`skills/sri-yoga-buddhi-to-cutile-rs/SKILL.md`](../skills/sri-yoga-buddhi-to-cutile-rs/SKILL.md).

Reference tile ops (NumPy, always on): `sri.omlang.tile_ref`.

Rust CPU mirror crate (compiles without CUDA Tile): `embodiment/cutile_rs_stub/`.

---

## Pass criteria when a real cuTile port lands

1. **Structural:** Tile IR diff vs Python/cuTile reference for the same op inventory (loads/stores/reduces as applicable).  
2. **Functional:** oracle Samya max abs err \< 1e-6 (f32) / Lemma Z1 \(\hat\Upsilon\equiv\hat s\) within agreed rtol.  
3. **Perf (Blackwell):** CUPTI geomean ≥ 0.95 vs the chosen Python/cuTile reference (NVIDIA’s bar).  
4. **Honesty:** report device-time vs wall-clock separately; no free-energy claim from variance alone.

Until those four are green in CI, status stays **scaffold / open**.
