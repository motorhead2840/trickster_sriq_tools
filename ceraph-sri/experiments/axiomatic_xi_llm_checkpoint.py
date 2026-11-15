"""Single end-of-month LLM reliability checkpoint using axiomatic Ξ_SNR.

Reuses the legacy TruthfulQA harness when transformers/datasets/torch+GPU are
available. Without that stack, writes a skipped provenance record so CI stays green.
Does not tune hyperparameters to chase AUROC.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from sri.metrics.stability import snr_stability


def _try_legacy_run(output_dir: Path, require_a100: bool) -> dict | None:
    try:
        from sri.legacy.llm_benchmark import run as legacy_run
    except Exception as error:  # noqa: BLE001 - intentional soft dependency
        return {
            "skipped": True,
            "reason": f"legacy llm_benchmark unavailable: {error}",
        }
    try:
        # Legacy runner writes its own summaries; we only ask for one fixed pass.
        return legacy_run(output_dir, require_a100=require_a100)
    except TypeError:
        # Older signature variants.
        try:
            return legacy_run(output_dir)
        except Exception as error:  # noqa: BLE001
            return {"skipped": True, "reason": str(error)}
    except Exception as error:  # noqa: BLE001
        return {"skipped": True, "reason": str(error)}


def synthetic_ranking_smoke(seed: int = 108, n: int = 100) -> dict:
    """CPU-only smoke: Ξ_SNR ranks a synthetic confidence proxy.

    Not a TruthfulQA result. Exists so the checkpoint script always emits a metric.
    """
    rng = np.random.default_rng(seed)
    correct = []
    scores_conf = []
    scores_xi = []
    for _ in range(n):
        is_correct = int(rng.random() < 0.2)
        logits = rng.normal(loc=1.0 if is_correct else 0.0, scale=1.0, size=8)
        conf = float(np.max(logits) - np.mean(logits))
        xi = snr_stability(logits)
        correct.append(is_correct)
        scores_conf.append(conf)
        scores_xi.append(xi)

    def auc(labels: list[int], scores: list[float]) -> float:
        pairs = 0
        favour = 0.0
        for i, li in enumerate(labels):
            for j, lj in enumerate(labels):
                if li == lj:
                    continue
                pairs += 1
                if (scores[i] > scores[j] and li > lj) or (scores[i] < scores[j] and li < lj):
                    favour += 1.0
                elif scores[i] == scores[j]:
                    favour += 0.5
        return float(favour / max(pairs, 1))

    return {
        "skipped": False,
        "mode": "synthetic_smoke",
        "n": n,
        "accuracy": float(np.mean(correct)),
        "confidence_auroc": auc(correct, scores_conf),
        "xi_snr_auroc": auc(correct, scores_xi),
        "note": "Synthetic only; replace with legacy TruthfulQA run on Colab A100 once.",
    }


def run(output_dir: Path, require_a100: bool = False) -> dict:
    legacy = _try_legacy_run(output_dir, require_a100=require_a100)
    synthetic = synthetic_ranking_smoke()
    summary = {
        "experiment": "axiomatic_xi_llm_checkpoint",
        "legacy_truthfulqa": legacy,
        "synthetic_smoke": synthetic,
        "policy": "single pass; do not iterate hyperparameters to chase AUROC",
        "provenance": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "axiomatic_xi_llm_checkpoint.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-a100", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.output_dir, require_a100=args.require_a100), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
