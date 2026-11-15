"""Reproduce the v5 selection round with the sri package.

Default size is ``small``, NumPy, seed 11.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from suite import SimConfig, run_suite  # noqa: E402


def main(size: str = "small", backend: str = "numpy"):
    out = Path(__file__).resolve().parent / "out"
    return run_suite(SimConfig.from_size(size), out, backend=backend, do_agreement=False, do_timing=False)


if __name__ == "__main__":
    size = sys.argv[1] if len(sys.argv) > 1 else "small"
    backend = sys.argv[2] if len(sys.argv) > 2 else "numpy"
    main(size, backend)
