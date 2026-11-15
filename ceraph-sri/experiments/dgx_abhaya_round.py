"""Preregistered DGX round for Abhaya-gate and same-device speed.

CPU pieces run anywhere. CUDA pieces are specified in docs/dgx-abhaya-round.md
and are not executed by this module. A printed latency budget is not an input
to any decision.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Locked before the next DGX session. Do not retune after seeing that session.
SEED = 11
LENGTH = 8192
SPIKE_FRACTION = 0.10
SPIKE_STD = 8.0
BACKGROUND_STD = 0.5
WINDOW = 9
LAM = 1.0
HALL_RATIO = 0.95
FUSION_RATIO = 0.90
SLOPE_MAX = 0.15
SUBLINEAR_MAX = 1.0
TEMP_MARGIN_C = 5.0
VAR_REMAINING_MAX = 0.50
INTERNAL_BUDGET_MS = 0.2379
LENGTHS = (16384, 65536, 262144, 1048576)


def window_reduce(values: np.ndarray, width: int, moment: str) -> np.ndarray:
    radius = width // 2
    out = np.empty(values.shape, dtype=np.float64)
    series = np.asarray(values, dtype=np.float64)
    for index in range(series.size):
        lo = max(0, index - radius)
        hi = min(series.size, index + radius + 1)
        chunk = series[lo:hi]
        if moment == "mean":
            out[index] = float(np.mean(chunk))
        elif moment == "var":
            out[index] = float(np.var(chunk))
        else:
            raise ValueError(moment)
    return out


def make_series(
    n: int = LENGTH,
    seed: int = SEED,
    spike_fraction: float = SPIKE_FRACTION,
    spike_std: float = SPIKE_STD,
) -> tuple[np.ndarray, np.ndarray]:
    """Latent values and observations. Estimators may receive only the observations."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n)
    observed = latent + rng.normal(scale=BACKGROUND_STD, size=n)
    spikes = rng.random(n) < spike_fraction
    observed = observed.copy()
    observed[spikes] += rng.normal(scale=spike_std, size=int(np.sum(spikes)))
    return latent, observed


def abhaya_from_observations(observed: np.ndarray, lam: float = LAM, width: int = WINDOW) -> np.ndarray:
    """A(y) = y · exp(-λ · Ξ), with Ξ the window variance of y. y is not the latent value."""
    instability = window_reduce(observed, width, "var")
    return np.asarray(observed, dtype=np.float64) * np.exp(-float(lam) * instability)


def abhaya_flux(observed: np.ndarray, lam: float = LAM, width: int = WINDOW) -> np.ndarray:
    """Apply A to the oscillation, which is the Colab reading of anxiety-damping.

    The local mean is kept. The deviation is Θ in A(Θ) = Θ · exp(-λ · Ξ), and Ξ
    is the window variance. The latent value is not an input.
    """
    center = window_reduce(observed, width, "mean")
    instability = window_reduce(observed, width, "var")
    deviation = np.asarray(observed, dtype=np.float64) - center
    return center + deviation * np.exp(-float(lam) * instability)


def moving_average(observed: np.ndarray, width: int = WINDOW) -> np.ndarray:
    return window_reduce(observed, width, "mean")


def abstain_mask(observed: np.ndarray, threshold: float = 1.0, width: int = WINDOW) -> np.ndarray:
    """Courage of non-action: high window variance means do not emit that sample."""
    return window_reduce(observed, width, "var") > threshold


def mse(estimate: np.ndarray, latent: np.ndarray) -> float:
    return float(np.mean((np.asarray(estimate) - np.asarray(latent)) ** 2))


def _scores(latent: np.ndarray, observed: np.ndarray) -> dict:
    raw = abhaya_from_observations(observed)
    flux = abhaya_flux(observed)
    average = moving_average(observed)
    identity_mse = mse(observed, latent)
    average_mse = mse(average, latent)
    raw_mse = mse(raw, latent)
    flux_mse = mse(flux, latent)
    pre = float(np.var(np.asarray(observed, dtype=np.float64)))
    post = float(np.var(flux))
    remaining = post / pre if pre > 0 else 1.0
    return {
        "identity_mse": identity_mse,
        "moving_average_mse": average_mse,
        "abhaya_mse": raw_mse,
        "flux_mse": flux_mse,
        "flux_variance_remaining": remaining,
        "hallucination": decide_hallucination(raw_mse, identity_mse, average_mse),
        "flux_hallucination": decide_hallucination(flux_mse, identity_mse, average_mse),
        "flux_variance": "SUPPORTED" if remaining <= VAR_REMAINING_MAX else "NOT_MET",
    }


def cpu_preview() -> dict:
    latent, observed = make_series()
    return {
        "seed": SEED,
        "length": LENGTH,
        "lam": LAM,
        "window": WINDOW,
        "xi_definition": "sliding window variance of the observation; latent value withheld",
        "pass_series": "zero-mean latent plus sparse spikes",
        **_scores(latent, observed),
    }


def cpu_boundary() -> dict:
    """Same noise draw added to a latent with mean 5. Reported, not the pass bar.

    Multiplying by exp(-λ Ξ) pulls samples toward 0, so a large carrier is damaged
    even when the additive spikes match the pass series.
    """
    latent, observed = make_series()
    return {
        "latent_mean": 5.0,
        "role": "boundary, not the pass bar",
        **_scores(latent + 5.0, observed + 5.0),
    }


def decide_hallucination(abhaya_mse: float, identity_mse: float, average_mse: float) -> str:
    best = min(identity_mse, average_mse)
    if abhaya_mse <= HALL_RATIO * best:
        return "SUPPORTED"
    return "NOT_MET"


def decide_fusion(triton_ms: float, eager_ms: float) -> str:
    if triton_ms <= FUSION_RATIO * eager_ms:
        return "SUPPORTED"
    return "NOT_MET"


def decide_product_speed(ceraph_ms: float, baseline_ms: float, quality: str) -> str:
    if quality == "SUPPORTED" and ceraph_ms < baseline_ms:
        return "SUPPORTED"
    if quality != "SUPPORTED":
        return "NOT_MET"
    return "NOT_MET"


def decide_scaling(lengths: list[int], times_ms: list[float]) -> dict:
    if len(lengths) < 3 or len(lengths) != len(times_ms):
        raise ValueError("need at least three paired lengths and times")
    log_n = np.log(np.asarray(lengths, dtype=np.float64))
    log_t = np.log(np.asarray(times_ms, dtype=np.float64))
    slope = float(np.polyfit(log_n, log_t, 1)[0])
    return {
        "log_log_slope": slope,
        "scaling": "SUPPORTED" if slope < SLOPE_MAX else "NOT_MET",
        "sublinear": "SUPPORTED" if slope < SUBLINEAR_MAX else "NOT_MET",
    }


def decide_thermal(
    ceraph_joules: float,
    baseline_joules: float,
    ceraph_c: float,
    baseline_c: float,
) -> str:
    cooler_or_close = ceraph_c <= baseline_c + TEMP_MARGIN_C
    if ceraph_joules < baseline_joules and cooler_or_close:
        return "SUPPORTED"
    return "NOT_MET"


def write_preview(path: Path) -> dict:
    payload = {"preview": cpu_preview(), "boundary": cpu_boundary()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    destination = Path("docs/benchmark-results/dgx-abhaya/cpu_preview.json")
    print(json.dumps(write_preview(destination), indent=2))
