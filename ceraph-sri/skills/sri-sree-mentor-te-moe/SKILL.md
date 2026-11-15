# Skill: sri-sree-mentor-te-moe

Enable or plan **dropless MoE training** for SREE AI Mentor using NVIDIA Transformer Engine + JAX (MaxText), per:

https://developer.nvidia.com/blog/accelerating-dropless-moe-training-in-jax-with-nvidia-transformer-engine/

**Activation:** user asks about SREE Mentor training, dropless MoE, Transformer Engine JAX, MaxText TE MoEBlock, MXFP8 grouped GEMM, or EP dispatch/combine.

## Orchestrator contract

Pure orchestrator. Advance only on machine verdicts.

| Stage | Required | Verdict |
|---|---|---|
| 0 inventory | map Mentor backbone size / EP need / GPU gen | `INVENTORY_OK` |
| 1 config | emit TE MaxText flag bundle from `sri.sree.moe_te_config` | `CONFIG_OK` |
| 2 smoke | small MoE forward/train step (or honest skip) | `SMOKE_OK` / `SMOKE_SKIP` |
| 3 witness | Ξ / Yoga–Buddhi hook still attached post-logits | `WITNESS_OK` |
| 4 scale-notes | document topology; no fake TFLOPS paste | `SCALE_DOC_OK` |

Hard caps: smoke×2. On missing TE/MaxText container → `SMOKE_SKIP` with reason; still ship config + Witness attach plan.

## Rules

1. Prefer **dropless** (`capacity_factor` / no token drop) for Mentor quality.  
2. Never copy NVIDIA DeepSeek-V3 10.4× / 1,068 TFLOPS as SREE results.  
3. Keep Witness + Yoga–Buddhi **outside** Softmax; TE accelerates MoE train, not the theorem.  
4. Pair with cuTile skill only when porting Yoga–Buddhi *tiles*, not the whole MoE stack.  
5. End every stage with:

```text
<VALIDATOR_OUTPUT>
VERDICT: <NAME>
</VALIDATOR_OUTPUT>
```
