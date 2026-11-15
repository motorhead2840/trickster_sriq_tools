"""Shim. The CPU entry point lives in ``sri.legacy.core_engine``."""
from sri.legacy.core_engine import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
