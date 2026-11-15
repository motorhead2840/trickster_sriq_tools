"""SphotaCompiler — crystallize Yoga–Buddhi math across Python / JAX / Julia."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from sri.cv.estimator import SensingConfig, mmse_coefficients, simulate_measurements
from sri.omlang import jax_backend, julia_bridge, tile_ref
from sri.omlang.yoga_buddhi import (
    latent_involution_holds,
    mirror_validation,
    observable_upsilon,
    oracle_samya,
)


@dataclass
class SphotaReport:
    """Pass/fail report for a Sphota Yoga–Buddhi crystallization."""

    oracle_samya_max_abs_err: float
    latent_involution: bool
    observable_equals_mmse_max_abs: float
    sample_mse: float
    analytic_mse: float
    mse_within_5sigma: bool
    mirror_nonneg: bool
    backends: dict[str, Any] = field(default_factory=dict)
    pass_lemma_z1: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SphotaCompiler:
    """Compile and execute Yoga–Buddhi checks on available backends."""

    def __init__(
        self,
        *,
        config: SensingConfig | None = None,
        prefer_jax: bool = True,
        prefer_julia: bool = True,
        julia_executable: str = "julia",
        seed: int = 108,
        n_samples: int = 4096,
    ) -> None:
        self.config = config or SensingConfig(reference_r=0.7)
        self.prefer_jax = prefer_jax
        self.prefer_julia = prefer_julia
        self.julia_executable = julia_executable
        self.seed = seed
        self.n_samples = n_samples

    def crystallize(self) -> SphotaReport:
        rng = np.random.default_rng(self.seed)
        signal, noise, y_a, y_b = simulate_measurements(self.config, self.n_samples, rng)
        weights = mmse_coefficients(self.config)

        # --- Python / NumPy reference (always on) ---
        oracle = oracle_samya(signal[:64], noise[:64])
        oracle_err = float(np.max(np.abs(oracle["samya"] - signal[:64])))
        involution = latent_involution_holds(signal[:64], noise[:64])
        obs = observable_upsilon(y_a, y_b, weights)
        obs_err = float(np.max(np.abs(obs["projected"] - obs["mmse"])))
        sample_mse = float(np.mean((obs["projected"] - signal) ** 2))
        from sri.cv.estimator import analytic_mse

        analytic = float(analytic_mse(self.config))
        # Monte Carlo SE under Gaussian: rough 5σ gate used by C1/C5.
        se = float(np.sqrt(max(analytic, 1e-30) * 2.0 / self.n_samples))
        within = abs(sample_mse - analytic) <= 5.0 * se

        cplx = (y_a[:128] + 1j * y_b[:128]).astype(np.complex128)
        mirror = mirror_validation(cplx)
        mirror_ok = bool(np.all(mirror >= -1e-15))

        backends: dict[str, Any] = {
            "numpy": {"ok": True, "oracle_err": oracle_err, "obs_err": obs_err}
        }

        # --- JAX optional ---
        if self.prefer_jax and jax_backend.available():
            j_oracle = jax_backend.oracle_samya_jax(signal[:64], noise[:64])
            j_obs = jax_backend.observable_upsilon_jax(y_a, y_b, weights)
            backends["jax"] = {
                "ok": True,
                "devices": [str(d) for d in __import__("jax").devices()],
                "oracle_err": float(j_oracle["max_abs_samya_minus_v"]),
                "obs_err": float(j_obs["max_abs_proj_minus_mmse"]),
                "matches_numpy_obs": bool(
                    np.allclose(
                        np.asarray(j_obs["projected"]),
                        obs["projected"],
                        rtol=1e-10,
                        atol=1e-12,
                    )
                ),
            }
        else:
            backends["jax"] = {"ok": False, "reason": "jax_unavailable_or_disabled"}

        # --- Julia optional ---
        if self.prefer_julia and julia_bridge.julia_available(self.julia_executable):
            try:
                j_or = julia_bridge.oracle_samya_julia(
                    signal[:64], noise[:64], executable=self.julia_executable
                )
                j_ob = julia_bridge.observable_upsilon_julia(
                    y_a, y_b, weights, executable=self.julia_executable
                )
                backends["julia"] = {
                    "ok": True,
                    "oracle_err": float(np.max(np.abs(j_or["samya"] - signal[:64]))),
                    "obs_err": float(np.max(np.abs(j_ob["projected"] - obs["mmse"]))),
                    "matches_numpy_obs": bool(
                        np.allclose(j_ob["projected"], obs["projected"], rtol=1e-5, atol=1e-5)
                    ),
                }
            except Exception as exc:  # noqa: BLE001 — report bridge failure honestly
                backends["julia"] = {"ok": False, "reason": str(exc)}
        else:
            backends["julia"] = {"ok": False, "reason": "julia_unavailable_or_disabled"}

        # --- Tile-shaped NumPy baseline (cuTile translation ground truth) ---
        thesis = signal[:64] + noise[:64]
        antithesis = signal[:64] - noise[:64]
        tile_samya = tile_ref.oracle_samya_tile(thesis, antithesis)
        _tile_anti, tile_proj = tile_ref.observable_upsilon_tile(
            y_a, y_b, float(weights[0]), float(weights[1])
        )
        tile_mirror = tile_ref.mirror_abs2_tile(y_a[:128], y_b[:128])
        backends["tile_ref"] = {
            "ok": True,
            "analysis": tile_ref.analysis_spec(),
            "oracle_err": float(np.max(np.abs(tile_samya - signal[:64]))),
            "obs_err": float(np.max(np.abs(tile_proj - obs["mmse"]))),
            "mirror_nonneg": bool(np.all(tile_mirror >= -1e-15)),
            "cutile_rs_status": "stub_only",
            "embodiment": "embodiment/cutile_rs_stub",
        }

        # Real cuTile Rust is optional; report availability without failing Z1.
        backends["cutile_rs"] = {
            "ok": False,
            "reason": "cuda_tile_not_bound; see embodiment/cutile_rs_stub + docs/cuda-tile-rust-embodiment.md",
            "perf_status": "not_run",
        }

        pass_z1 = (
            oracle_err < 1e-12
            and involution
            and obs_err < 1e-10
            and within
            and mirror_ok
            and backends["tile_ref"]["oracle_err"] < 1e-12
            and backends["tile_ref"]["obs_err"] < 1e-10
        )
        return SphotaReport(
            oracle_samya_max_abs_err=oracle_err,
            latent_involution=involution,
            observable_equals_mmse_max_abs=obs_err,
            sample_mse=sample_mse,
            analytic_mse=analytic,
            mse_within_5sigma=within,
            mirror_nonneg=mirror_ok,
            backends=backends,
            pass_lemma_z1=pass_z1,
            notes=(
                "Sphota hybrid: NumPy+tile_ref required; JAX/Julia optional; "
                "cuTile Rust scaffold only. "
                "pass_lemma_z1 is algebra+MSE only — C5 promotion via suite runner."
            ),
        )
