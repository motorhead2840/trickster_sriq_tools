# SREE Mentor × JAX Transformer Engine (dropless MoE)

**Source:** [Accelerating Dropless MoE Training in JAX with NVIDIA Transformer Engine](https://developer.nvidia.com/blog/accelerating-dropless-moe-training-in-jax-with-nvidia-transformer-engine/) (NVIDIA, 2026-09-14; Lee / Berchtold / Nguyen / Do / Shah).

**SRI product target:** **SREE.edu / SREE AI Mentor** — logit-intercept mentoring platform described in the yellow paper (SRI-Witness™ + \(\Xi\) gate + Yoga–Buddhi before Softmax). This note adopts NVIDIA’s *training-stack* pattern so a Mentor backbone can scale without capacity-drop quality loss — without claiming DeepSeek-V3’s 10.4× / 1,068 TFLOPS/GPU numbers as ours.

Related: [cuda-tile-rust-embodiment.md](cuda-tile-rust-embodiment.md), [yoga-buddhi-projector.md](yoga-buddhi-projector.md), [technical-foundation.md](technical-foundation.md).

---

## What NVIDIA showed (facts we adopt)

| Idea | NVIDIA result | Why it fits SREE Mentor |
|---|---|---|
| **Dropless MoE** | Every token hits its Top-K experts; no capacity trim / pad | Mentor quality must not silently drop hard tokens (uncertain / out-of-distribution student queries) |
| **Grouped GEMM / `ragged_dot`** | One kernel, per-expert actual lengths; MXFP8 on Blackwell | Ragged expert loads match uneven educational traffic better than padded capacity MoE |
| **NCCL EP Dispatch+Combine** | Fused EP path + token dedup | Multi-GPU Mentor training without communication-dominated stalls |
| **JAX host offload + XLA multi-stream collectives** | Activation remat/offload; overlap NVLink + IB | Memory headroom for long-context tutoring + Witness intercept buffers |
| **Stack impact** | 103 → 1,068 TFLOPS/GPU (~10.4×) on GB200/GB300; 97% efficiency @ 1,024 GPUs | **Reference bar only** — reproduce later in MaxText/TE; do not paste into SREE marketing |

Container path: NGC / `ghcr.io/nvidia/jax:maxtext-2026-09-10`+ with Transformer Engine; MaxText flags `te_moe_block`, `te_gmm_quantization: te_mxfp8`, `sparse_matmul`, etc.

---

## Honesty boundary

- Yellow-paper SREE numbers (0.0082 ms annihilate, 94% throughput, \(\Xi<0.98\)) remain **claims under audit** — see Colab/whitepaper evidence notes. This doc does **not** re-assert them.
- TE MoE gains are **training throughput**, not proof of zero-hallucination Mentor inference.
- Softwall: no Landauer / hardware \(\mathcal O(0)\) claim from enabling `te_moe_block`.

---

## Architecture mapping (SRI)

```text
Student turn
  → Mentor MoE backbone (train with TE dropless JAX; serve dense or sparse)
  → raw logits ℓ in TMEM / device buffer
  → SRI-Witness: Ξ(ℓ) stability / avidya gate
  → if unstable: Yoga–Buddhi / Lemma Z1 observable projector (or abstain)
  → Softmax / sample only after gate
```

| Layer | Role | Code / doc today | TE / MaxText tomorrow |
|---|---|---|---|
| MoE train | Dropless experts, EP, MXFP8 GMM | scaffold config below | MaxText + TE MoEBlock |
| Witness gate | \(\Xi\), damp / abstain | `sri` logit helpers / technical-foundation | keep host-side or fused custom |
| Yoga–Buddhi | Samya / \(\hat\Upsilon_{\mathrm{obs}}\) | `sri.omlang` + Lemma Z1 | optional cuTile Rust tile ops |
| Sphota host | multi-backend crystallize | `SphotaCompiler` | C-ABI / JAX TE dual path |

**Dropless ↔ Avidya.** Capacity MoE *drops* overflow tokens — analogous to silently discarding hard student tokens. Dropless keeps every routed token; the Mentor’s honesty still depends on the **Witness gate**, not on MoE alone.

---

## Scaffold in this repo

| Artifact | Purpose |
|---|---|
| [`skills/sri-sree-mentor-te-moe/SKILL.md`](../skills/sri-sree-mentor-te-moe/SKILL.md) | Agent skill: TE/MaxText enablement with verdict gates |
| [`sri/sree/moe_te_config.py`](../sri/sree/moe_te_config.py) | Machine-readable MaxText/TE flag bundle + honesty metadata |
| [`experiments/sree_mentor_te_moe_plan.py`](../experiments/sree_mentor_te_moe_plan.py) | Emit plan JSON (no GPU TE required in CI) |

Pass criteria when a live TE run lands:

1. **Correctness:** small MoE smoke (loss finite; dropless = no capacity drop counters).  
2. **Witness attach:** Ξ / Yoga–Buddhi intercept still callable on logits (even if not fused).  
3. **Perf (optional):** step-time / TFLOPS/GPU vs non-TE baseline on the *same* Mentor config — report deltas; do not import NVIDIA DeepSeek figures.  
4. **Scale note:** EP/FSDP topology documented; 97% @ 1k GPUs is NVIDIA’s result, not ours until measured.

Until then status = **plan / scaffold**.
