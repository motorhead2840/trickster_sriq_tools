"""JAX primitive for Abhaya damping.

When JAX is installed, ``apply`` binds a primitive whose JVP is

    dA = exp(-λ Ξ) dΘ - λ Θ exp(-λ Ξ) dΞ

which is the derivative of Θ · exp(-λ · Ξ). The impl is the same elementwise
map. Triton is not required for the primitive; ``apply_triton`` uses the
kernel only when CUDA and Triton import.
"""
from __future__ import annotations

from typing import Any, Callable

_APPLY: Callable[..., Any] | None = None


def available() -> bool:
    try:
        import jax  # noqa: F401
        import jax.numpy  # noqa: F401
    except ImportError:
        return False
    return True


def _build() -> Callable[..., Any]:
    import jax
    import jax.numpy as jnp
    from jax.interpreters import ad

    primitive = jax.core.Primitive("sri_abhaya_damping")

    def _impl(theta: Any, xi: Any, *, lam: float) -> Any:
        return jnp.asarray(theta) * jnp.exp(-lam * jnp.asarray(xi))

    def _jvp(primals: tuple[Any, Any], tangents: tuple[Any, Any], *, lam: float) -> tuple[Any, Any]:
        theta, xi = primals
        dtheta, dxi = tangents
        scale = jnp.exp(-lam * xi)
        out = theta * scale
        dout = dtheta * scale + dxi * (out * (-lam))
        return out, dout

    primitive.def_impl(_impl)
    ad.defjvp(primitive, _jvp)

    def apply(theta: Any, xi: Any, lam: float) -> Any:
        return primitive.bind(jnp.asarray(theta), jnp.asarray(xi), lam=float(lam))

    return apply


def apply(theta: Any, xi: Any, lam: float) -> Any:
    """Bind the primitive. Raises if JAX is not installed."""
    global _APPLY
    if not available():
        raise RuntimeError("jax is not installed")
    if _APPLY is None:
        _APPLY = _build()
    return _APPLY(theta, xi, lam)


def apply_numpy(theta: Any, xi: Any, lam: float) -> Any:
    """Run the primitive and return a NumPy array."""
    import numpy as np

    return np.asarray(apply(theta, xi, lam))
