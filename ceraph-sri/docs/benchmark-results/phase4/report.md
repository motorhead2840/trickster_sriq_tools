# Phase 4 report — head-to-head scoreboard

**Narrow claim:** SUPPORTED  
**Broad claim (faster and cooler than a named current method on the same GPU):** NOT_MET

The deployable dual-mode path remains ahead of single-mode Wiener, and the archived kernels have already been timed on an RTX PRO 6000 Blackwell, including sub-millisecond medians at the small sizes. The same card's average-pool comparison is the open race: the current operator finished first, so the broad speed and thermal claim stays unmet until a later same-GPU row reverses it.

## Confirmation

Mean relative reduction versus single-mode Wiener on the short CPU check: **0.1538**.

| Noise variance | SRI MSE | Single-mode MSE | R |
|---:|---:|---:|---:|
| 1.0 | 0.6122 | 0.6901 | 0.1129 |
| 2.0 | 0.6035 | 0.7495 | 0.1947 |

## Claims

- `dual_mode_beats_single_mode` — **SUPPORTED**. Phase 3 locked mean R=0.1205 against threshold 0.08. This run's confirmation mean R=0.1538.
- `blackwell_execution` — **SUPPORTED**. 5 measured RTX PRO 6000 kernel records are in the catalog.
- `submillisecond_on_recorded_sizes` — **SUPPORTED**. Smallest recorded kernel time is 0.0062 ms.
- `faster_than_same_host_cpu_at_large_length` — **SUPPORTED**. Peak CPU/GPU ratio on the 14 November Blackwell kernel is 148.8.
- `same_gpu_faster_than_named_baseline` — **NOT_MET**. torch.nn.functional.avg_pool1d kernel 9 median 0.257 ms, Ceraph kernel median 1.178 ms, temperatures 30.0 °C and 31.0 °C.
- `thermal_savings_against_a_named_accelerator` — **OPEN**. Board readings exist for one L40S and one RTX PRO 6000. No second accelerator ran the same job.
- `printed_latency_budget` — **NOT_EVIDENCE**. 0.2379 ms is stored and excluded. It is not a timed competitor.

## Next row that would move the broad claim

Same GPU, same tensor, a named current operator, and this kernel or the dual-mode estimator. Record median time, board power, and temperature for both. The broad claim becomes SUPPORTED when the Ceraph side matches quality and finishes sooner or uses fewer joules.
