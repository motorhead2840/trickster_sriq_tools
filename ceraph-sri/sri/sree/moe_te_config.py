"""MaxText / Transformer Engine dropless MoE flag bundles for SREE Mentor.

Derived from NVIDIA's TE+JAX MoE guidance (2026-09). These are *plans* —
enabling them requires the NGC MaxText+TE container and GPUs. CI must not
import DeepSeek-V3 TFLOPS figures as SREE results.
"""
from __future__ import annotations

from typing import Any


def te_moe_maxtext_flags(*, mentor_profile: str = "smoke") -> dict[str, Any]:
    """Return MaxText-oriented TE MoE flags.

    ``smoke``: small local enablement checklist.
    ``scale_ref``: NVIDIA DeepSeek-V3 reproduction *reference* (not SREE metrics).
    """
    base = {
        "te_moe_block": True,
        "te_gmm_quantization": "te_mxfp8",
        "ragged_buffer_factor": 2.0,
        "te_ep_overflow_check_every_n_steps": 20,
        "sparse_matmul": True,
        "prefuse_moe_weights": True,
        "capacity_factor": 1.0,
        "megablox": False,
        "attention": "cudnn_flash_te",
    }
    if mentor_profile == "smoke":
        return {
            **base,
            "model_name": "sree-mentor-moe-smoke",
            "max_target_length": 2048,
            "per_device_batch_size": 1,
            "steps": 5,
            "notes": "Local / single-node smoke; replace model_name with real Mentor backbone.",
        }
    if mentor_profile == "scale_ref":
        # NVIDIA blog reproduction flags — labeled reference-only.
        return {
            **base,
            "model_name": "deepseek3-671b",
            "max_target_length": 4096,
            "per_device_batch_size": 6,
            "quantization": "te_fp8_currentscaling",
            "reference_only": True,
            "reference_source": (
                "https://developer.nvidia.com/blog/"
                "accelerating-dropless-moe-training-in-jax-with-nvidia-transformer-engine/"
            ),
            "notes": "Do not report these TFLOPS as SREE Mentor results.",
        }
    raise ValueError(f"unknown mentor_profile: {mentor_profile}")


def mentor_te_moe_plan(*, mentor_profile: str = "smoke") -> dict[str, Any]:
    """Full SREE Mentor TE MoE plan artifact for experiments / agents."""
    return {
        "product": "SREE.ai Mentor / SREE.edu",
        "methodology": "nvidia-transformer-engine-jax-dropless-moe",
        "container_hint": "ghcr.io/nvidia/jax:maxtext-2026-09-10 or newer",
        "maxtext_flags": te_moe_maxtext_flags(mentor_profile=mentor_profile),
        "witness_attach": {
            "gate": "stability_coefficient / Ξ on pre-softmax logits",
            "projector": "Yoga–Buddhi observable (Lemma Z1) or abstain",
            "must_run_before": "softmax",
            "status": "scaffold",
        },
        "pairing": {
            "cutile_yoga_buddhi": "docs/cuda-tile-rust-embodiment.md",
            "skill": "skills/sri-sree-mentor-te-moe/SKILL.md",
        },
        "honesty": {
            "do_not_claim_nvidia_deepseek_tflops": True,
            "dropless_does_not_imply_zero_hallucination": True,
            "perf_status": "not_run",
        },
        "pass_criteria": [
            "smoke_loss_finite",
            "no_capacity_token_drop_counters",
            "witness_hook_callable",
            "perf_delta_vs_own_baseline_only",
        ],
    }
