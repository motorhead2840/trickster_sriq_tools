# 6pm spec session

Budget cap: $150–$200. Planned spend: $40–$70. Reserve the rest.

Use one RTX PRO 6000 Blackwell, or the single GPU your DGX Cloud screen actually offers. Do not launch an 8-GPU node. If `nvidia-smi` prints a blank `power.draw`, stop the rental.

## Order

1. Confirm the device and that power and temperature are numeric.
2. Run cells 0 through 5. Paste each printed JSON back, and keep the matching PNG.

```bash
python experiments/spec_session/cells.py 0
python experiments/spec_session/cells.py 1
python experiments/spec_session/cells.py 2
python experiments/spec_session/cells.py 3
python experiments/spec_session/cells.py 4
python experiments/spec_session/cells.py 5
```

3. Stop the GPU. The proof note is written from those six JSON files and histograms. A result can be called faster or cheaper only where those numbers say so.

A second, different GPU is a separate session. It is how a comparison is earned. It is not required to finish the math checks.

## What tonight can and cannot change

Math checks move from “implemented” to “re-run on this host.” Power and temperature move from “unmeasured” to “measured on this board” only if `nvidia-smi` returns numbers. No equation on the softwall list becomes a silicon result because a GPU hour finished.
