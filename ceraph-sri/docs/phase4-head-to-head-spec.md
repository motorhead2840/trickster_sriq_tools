# Phase 4 — same-device head-to-head

**Status:** OPEN from 14 November 2026  
**Runner:** `experiments/phase4_head_to_head.py`  
**Catalog:** `docs/benchmark-results/phase4/colab_device_catalog.json`

Phase 4 is a different path from the Colab cells that print a latency budget and then time only one kernel. The optimistic reading of the work so far is specific, and it is already measured.

## What already holds

1. **Quality, deployable path.** Phase 3 passed. On the locked dual-mode model the observable path, which matches Wiener MMSE, has mean relative MSE reduction **0.1205** against single-mode Wiener. The threshold was 0.08.
2. **The target card runs the code.** Saved Colab outputs and `sri_gpu_run.json` name an NVIDIA RTX PRO 6000 Blackwell Server Edition. Triton medians in those cells are **0.0082 ms** at length 8,192 and **0.0062 ms** at 10,000 nodes. The JAX run median is **0.08884 ms** at length 16,384.
3. **Same-host speed at large length.** On 14 November the Blackwell kernel was **7.1× to 149×** sooner than the CPU in that machine, and the board averaged **105 W** at **30 °C**. On the L40S, a synthetic map-and-reduce at length 1,048,576 was about **41×** sooner than that machine’s CPU, near **85 W** and **36 °C**.

Those three points are the startup claim Phase 4 is allowed to carry: a dual-mode estimator that beats single-mode sensing on the declared model, with kernels that have already been timed on current NVIDIA boards.

## What Phase 4 adds

A scoreboard with four outcomes.

| Status | Meaning |
|---|---|
| `SUPPORTED` | A named measurement met the bar |
| `NOT_MET` | A named measurement missed the bar |
| `OPEN` | The measurement that would settle it has not been taken |
| `NOT_EVIDENCE` | The number is in the archive and must not be scored |

Pass for the **narrow claim** requires the Phase 3 quality result to stay intact and at least one Blackwell kernel time to remain in the catalog. The runner re-checks a short CPU confirmation so a regression in the dual-mode path fails Phase 4 even if the old report is still on disk.

Pass for the **broad claim** (faster, lower energy, and lower thermal cost than current compute) requires a future same-GPU row where the Ceraph method matches a named current method on the task metric and finishes sooner or draws fewer joules, with temperature recorded the same way. The 14 November average-pool row is the first such row. The average pool was faster, and the temperatures were 31 °C and 30 °C. That row stays `NOT_MET`. It is how the next attempt will be judged.

## Rules for the Colab numbers

- `0.0082`, `0.0062`, `0.08884`, the patent-cell timings, and the 14 November tables are measurements. They stay.
- `0.2379 ms` is a printed budget in `DISSERTATION.ipynb`. The runner refuses to treat it as a measured baseline. The same literal appears in other notebooks. Repeating it does not create a competitor.
- A residual of zero in the 10,000-node cell is the output minus a truth vector the kernel received.
- “Baseline variance” in `Patent.ipynb` is the variance of the input tensor before the same function runs.
- The four Colab zip archives from 5, 10, and 12 November are one notebook set. The notebook whose name refers to token theft is not part of this catalog.

## Out of scope

Constant-time hardware, physical Landauer circumvention, a data-center bill, and a win over every current library. Phase 4 records those as `OPEN` until a same-device row supports them.
