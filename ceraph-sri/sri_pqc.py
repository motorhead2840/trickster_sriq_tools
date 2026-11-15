"""Shim. The known-noise projection demo lives in ``sri.legacy.pqc``."""
from sri.legacy.pqc import main, verify_known_noise_projection

__all__ = ["main", "verify_known_noise_projection"]


if __name__ == "__main__":
    raise SystemExit(main())
