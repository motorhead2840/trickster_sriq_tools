# Workstation and concurrent jobs, 14 November 2026

Same RTX PRO 6000 Blackwell Server Edition boards as the earlier four-GPU session. PyTorch 2.14.1+cu130. These two runs ask a different question from splitting one vector into four pieces.

## One board, a single workstation

Ordinary load. Flux error **1.472** versus a moving average at **1.675** and the raw series at **6.910**, at offset 0 and offset 5. Latency of a one-line map stayed about **0.021 ms** from length 256 through 1,048,576. The flux map at one million elements took **0.063 ms**. An average pool took **0.016 ms**. Idle **30.3 W** at **29 °C**. During the short loop, **47.6 W** and **31 °C**. Limit **600 W**.

Heavy load. Length 16,777,216, forty damping passes. Median before the burn **17.54 ms**, after **17.43 ms**. The result stayed finite. Drift against the first pass was **0**. Mean power **195 W**, utilization about **53%**, peak temperature **36 °C**.

## Four boards, four separate jobs

Ordinary load, one million elements each. Four jobs queued on one board took **0.062 ms**. The same four jobs launched together, one per board, took **0.065 ms**. At this size the jobs are too small for a second board to help. Power during that light concurrent loop was about **85–86 W** per board, temperature **29–30 °C**.

Heavy load, same length and forty passes as the one-board burn. Wall time for all four jobs together was **17.39 ms** before the burn and **17.37 ms** after. One board had needed **17.5 ms** to finish one of those jobs. Four boards therefore finished four heavy jobs in the same wall time as one board finished one. Every board stayed finite, drift was **0**, and the timing held. Mean power was **138–139 W** per board, utilization about **26–28%**, peak temperature **33–35 °C**.

The light split and the light concurrent test do not show a multi-board speedup. The heavy concurrent test does: same wall time, four times the jobs. Total electricity is four boards, not one. Per job, the person waiting sees one board’s time.
