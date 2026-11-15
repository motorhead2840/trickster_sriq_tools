# Darśana validation notebook, v2

This directory archives the uploaded notebook and a normalized tally of its
saved per-test outputs. The notebook's experiments are exploratory benchmark
results, not part of the automated `pytest` suite.

The per-test output cells report **23 PASS, 8 FAIL, and 1 NOT RUN** across
T1–T32. The failed tests are T6, T13, T20, T21, T24, T26, T29, and T31.
Post-hoc diagnostics are preserved in the notebook but do not change those
pre-registered outcomes.

The notebook's final v2 verdict says 24 of 32 tests passed, which conflicts
with its saved per-test outputs: T7 has no execution count or output. The
pre-registration cell (cell 2) also has a saved `SyntaxError`. Accordingly,
`results.json` follows the individual test output cells, records T7 as not
run, and does not treat the notebook as a clean, reproducible rerun. Results
were not re-executed during integration.
