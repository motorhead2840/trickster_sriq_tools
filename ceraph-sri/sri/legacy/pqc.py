"""Toy known-noise cancellation demo; this module does not implement PQC."""

from statistics import fmean

from sri.legacy.simulator import symplectic_project


def verify_known_noise_projection(dimension: int = 8192) -> float:
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    clean = tuple(5.0 * ((index % 17) - 8) for index in range(dimension))
    noise = tuple(25.0 * (((index * 13) % 29) - 14) for index in range(dimension))
    projection = symplectic_project(clean, noise)
    return fmean((actual - expected) ** 2 for actual, expected in zip(projection.projected, clean))


def main() -> int:
    mse = verify_known_noise_projection()
    print("Known-noise algebra demonstration (not LWE, ML-KEM, or post-quantum cryptography)")
    print(f"dimension=8192 paired-signal projection MSE={mse:.8e}")
    print("The exact injected noise is supplied to the projection; this is not decryption.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
