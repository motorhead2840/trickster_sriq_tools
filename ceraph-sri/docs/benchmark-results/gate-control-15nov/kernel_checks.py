"""CPU checks of the gate kernel in abhaya_gate(1).ipynb.

The kernel is re-expressed in NumPy float32 from the Triton source in cells 0
and 7. Nothing here loads a language model. Run from the repository root:

    python docs/benchmark-results/gate-control-15nov/kernel_checks.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from sri.omlang.abhaya import abhaya_damping  # noqa: E402

F = np.float32


def gate_v1(x, threshold=0.88, scale=5.0, gain=0.5):
    """Cell 0 kernel, one row."""
    x = np.asarray(x, dtype=F)
    v = x.size
    m = x.max()
    mean = F(x.sum(dtype=F) / F(v))
    e = np.exp(x - m)
    s = e.sum(dtype=F)
    sx = (e * x).sum(dtype=F)
    entropy = m + np.log(s) - sx / s
    xi = F(np.clip(1.0 - entropy / F(math.log(v)), 0.0, 1.0))
    instability = max(F(0.0), F(threshold) - xi)
    damping = F(np.exp(-F(scale) * (F(1.0) - xi)))
    applied = max(damping if xi < threshold else F(1.0), F(1e-3))
    gamma = F(gain) * (F(1.0) - F(np.exp(-F(scale) * instability))) if xi < threshold else F(0.0)
    flux = x - mean
    tail = x < (m - F(1.0))
    out = (mean + np.where(tail, flux * (F(1.0) - gamma), flux)) / applied
    return out.astype(F), float(xi), float(entropy), float(applied), float(gamma)


def gate_v4(x, threshold=0.86, scale=3.5, gain=0.50, snr_floor=5.8):
    """Cell 7 kernel, one row. Adds collision and SNR, and passes x through when SNR is low."""
    x = np.asarray(x, dtype=F)
    out, xi, entropy, applied, gamma = gate_v1(x, threshold, scale, gain)
    mean = F(x.sum(dtype=F) / F(x.size))
    std = F(np.sqrt(max(((x - mean) ** 2).sum(dtype=F) / F(x.size), F(1e-6))))
    snr = float((x.max() - mean) / std)
    e = np.exp(x - x.max())
    collision = float((e * e).sum(dtype=F) / (e.sum(dtype=F) ** 2))
    if snr < snr_floor:
        out = x.copy()
    return out, xi, collision, snr


def softmax_entropy(x):
    x = np.asarray(x, dtype=np.float64)
    p = np.exp(x - x.max())
    p /= p.sum()
    return float(-(p * np.log(p + 1e-9)).sum())


def map_a(s):
    c = F(s.mean())
    d = F(np.mean(np.abs(s - c)))
    return c + (s - c) * F(np.exp(-d))


def map_r(s):
    c = F(s.mean())
    d = F(np.mean(np.abs(s - c)))
    return s * F(np.exp(-d))


def repo_damping(s):
    c = F(s.mean())
    d = F(np.mean(np.abs(s - c)))
    return abhaya_damping(s, d, 1.0)


def drift(fn, s, shift=5.0):
    # Same float32 expression as shift_drift in the v8 module.
    a = np.asarray(fn(s), dtype=F)
    b = np.asarray(fn(np.asarray(s, dtype=F) + F(shift)), dtype=F)
    return float(np.max(np.abs(b - (a + F(shift)))))


def build_report(sizes=(17, 1000, 8192, 151936)) -> dict:
    report = {}

    # 1. Reproduce cell 0 on the same kind of input: 151,936 normal draws times 3.5.
    V = 151936
    rows = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        x = (rng.standard_normal(V) * 3.5).astype(F)
        out, xi, ent, applied, gamma = gate_v1(x)
        rows.append(
            {
                "seed": seed,
                "xi": round(xi, 4),
                "entropy_before_nats": round(softmax_entropy(x), 4),
                "entropy_after_nats": round(softmax_entropy(out), 6),
                "applied_damping": round(applied, 5),
                "logit_multiplier": round(1.0 / applied, 2),
                "top1_probability_before": round(float(np.exp(x.max() - x.max()) / np.exp(x.astype(np.float64) - x.max()).sum()), 4),
                "top1_probability_after": round(float(1.0 / np.exp(out.astype(np.float64) - out.max()).sum()), 6),
                "top1_same": bool(out.argmax() == x.argmax()),
            }
        )
    report["cell0_reproduction_on_random_logits"] = {
        "notebook_printed": {"xi": 0.4248, "entropy_before_nats": 6.8623, "entropy_after_nats": -0.0, "top1_same": True},
        "numpy_rows": rows,
    }

    # 2. Does the kernel ever change the largest logit? Random, peaked, flat, shifted, near-tied rows.
    changed = 0
    trials = 0
    rng = np.random.default_rng(2026)
    for kernel in (gate_v1, gate_v4):
        for n in sizes:
            for sigma in (0.01, 0.5, 3.5, 12.0):
                for shift in (-40.0, 0.0, 25.0):
                    for kind in ("normal", "peaked", "two_way_tie"):
                        x = (rng.standard_normal(n) * sigma + shift).astype(F)
                        if kind == "peaked":
                            x[rng.integers(n)] += F(22.0)
                        if kind == "two_way_tie":
                            i, j = rng.choice(n, 2, replace=False)
                            top = x.max() + F(3.0)
                            x[i] = top
                            x[j] = top - F(0.05)
                        out = kernel(x)[0]
                        trials += 1
                        if int(out.argmax()) != int(x.argmax()):
                            changed += 1
    report["largest_logit_changed"] = {"rows_tested": trials, "rows_where_winner_changed": changed}

    # 3. The v8 selection test applied to each map: add 5, compare with output plus 5.
    rng = np.random.default_rng(11)
    y = rng.standard_normal(8192).astype(F) + F(2.0)
    mu = F(y.mean())
    mad = F(np.mean(np.abs(y - mu)))


    gate_out, gate_xi, _, gate_applied, _ = gate_v1(y)
    report["v8_shift_test_seed_11"] = {
        "logged_in_v8_sheets": {"drift_A": 9.5367431640625e-07, "drift_R": 2.757575511932373},
        "map_A_selected_by_v8": drift(map_a, y),
        "map_R_rejected_by_v8": drift(map_r, y),
        "repo_sri_omlang_abhaya_damping": drift(repo_damping, y),
        "notebook_gate_kernel_cell0": drift(lambda s: gate_v1(s)[0], y),
        "notebook_gate_xi_on_this_series": gate_xi,
        "notebook_gate_logit_multiplier_on_this_series": 1.0 / gate_applied,
        "v8_map_A_departure_multiplier": float(np.exp(-mad)),
    }

    # 4. What the stability score says about a confident wrong row.
    V = 151936
    rng = np.random.default_rng(7)
    base = (rng.standard_normal(V) * 2.0).astype(F)
    confident = base.copy()
    confident[123] = base.max() + F(22.0)  # one token far ahead: right or wrong, the score is the same
    split = base.copy()
    split[123] = base.max() + F(12.0)
    split[456] = split[123] - F(0.1)
    report["stability_score_examples"] = {
        "one_token_22_logits_ahead": {"xi": gate_v4(confident)[1], "collision": gate_v4(confident)[2]},
        "two_tokens_nearly_tied": {"xi": gate_v4(split)[1], "collision": gate_v4(split)[2]},
        "note": "Neither number uses the identity of the token. A wrong token 22 logits ahead scores as stable.",
    }

    return report


if __name__ == "__main__":
    result = build_report()
    print(json.dumps(result, indent=2))
    (HERE / "kernel_checks.json").write_text(json.dumps(result, indent=2) + "\n")
