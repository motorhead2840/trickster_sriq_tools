"""Backwards-compatible command-line entry point for the CPU SRI simulation."""

from sri.legacy.simulator import main


if __name__ == "__main__":
    raise SystemExit(main())
