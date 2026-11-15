"""Optional JAX tensor path for Sphota Yoga–Buddhi (CPU or GPU)."""
from __future__ import annotations

from typing import Any


def available() -> bool:
    try:
        import jax  # noqa: F401
        import jax.numpy as jnp  # noqa: F401

        return True
    except ImportError:
        return False


def _ensure_x64() -> None:
    """Prefer f64 so Lemma Z1 checks match NumPy; no-op if JAX already configured."""
    import jax

    if not jax.config.read("jax_enable_x64"):
        jax.config.update("jax_enable_x64", True)


def observable_upsilon_jax(
    y_a: Any,
    y_b: Any,
    weights: Any,
) -> dict[str, Any]:
    """JAX twin of ``yoga_buddhi.observable_upsilon`` (JIT-friendly pure arrays)."""
    _ensure_x64()
    import jax.numpy as jnp

    ya = jnp.asarray(y_a, dtype=jnp.float64)
    yb = jnp.asarray(y_b, dtype=jnp.float64)
    w = jnp.asarray(weights, dtype=jnp.float64)
    antithesis = (2.0 * w[0] - 1.0) * ya + (2.0 * w[1]) * yb
    projected = 0.5 * (ya + antithesis)
    mmse = w[0] * ya + w[1] * yb
    return {
        "thesis": ya,
        "antithesis": antithesis,
        "projected": projected,
        "mmse": mmse,
        "max_abs_proj_minus_mmse": jnp.max(jnp.abs(projected - mmse)),
    }


def oracle_samya_jax(value: Any, noise: Any) -> dict[str, Any]:
    _ensure_x64()
    import jax.numpy as jnp

    v = jnp.asarray(value, dtype=jnp.float64)
    xi = jnp.asarray(noise, dtype=jnp.float64)
    thesis = v + xi
    antithesis = v - xi
    samya = 0.5 * (thesis + antithesis)
    return {
        "thesis": thesis,
        "antithesis": antithesis,
        "samya": samya,
        "max_abs_samya_minus_v": jnp.max(jnp.abs(samya - v)),
    }
