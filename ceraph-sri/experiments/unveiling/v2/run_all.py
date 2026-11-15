"""Reproduce the v2 error-law round with the sri package.

Default size is ``small`` and the seed is the published seed (7).
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from suite import SimConfig, run_suite  # noqa: E402


def main(size: str = "small"):
    out = Path(__file__).resolve().parent / "out"
    return run_suite(SimConfig.from_size(size), out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "small")
