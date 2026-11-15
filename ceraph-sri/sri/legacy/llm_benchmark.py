"""Compare SRI's Xi against model confidence on TruthfulQA multiple-choice questions."""

import argparse
import csv
import hashlib
import json
import math
import platform
import random
from pathlib import Path


def auc(scores, outcomes):
    """Pairwise ROC AUC with half credit for ties; None when one class is absent."""
    positives = [score for score, outcome in zip(scores, outcomes) if outcome]
    negatives = [score for score, outcome in zip(scores, outcomes) if not outcome]
    if not positives or not negatives:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives) / (
        len(positives) * len(negatives)
    )


def shuffled_choices(choices, labels, seed):
    if len(choices) != len(labels) or not 2 <= len(choices) <= 26 or sum(labels) != 1:
        raise ValueError("Expected 2–26 choices and exactly one correct label")
    order = list(range(len(choices)))
    random.Random(seed).shuffle(order)
    return [choices[i] for i in order], [labels[i] for i in order]


def run(output_dir, limit=100, seed=108, model_name="Qwen/Qwen2.5-0.5B-Instruct",
        require_a100=False):
    if limit <= 0:
        raise ValueError("limit must be positive")
    try:
        import torch
        import transformers
        import datasets
    except ImportError as error:
        raise RuntimeError("Install missing dependencies: torch, transformers and datasets") from error
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required; CPU fallback is disabled")
    device = torch.cuda.get_device_name(0)
    if require_a100 and "A100" not in device.upper():
        raise RuntimeError(f"A100 required; detected {device}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to("cuda").eval()
    dataset = datasets.load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    count = min(limit, len(dataset))
    rows = []
    for index in range(count):
        item = dataset[index]
        choices, labels = shuffled_choices(
            item["mc1_targets"]["choices"], item["mc1_targets"]["labels"], seed + index
        )
        option_names = [chr(ord("A") + i) for i in range(len(choices))]
        option_ids = []
        for name in option_names:
            ids = tokenizer.encode(" " + name, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"Option {name!r} is not a single token in {model_name}")
            option_ids.append(ids[0])
        if len(set(option_ids)) != len(option_ids):
            raise ValueError("Answer labels must have distinct token IDs")
        prompt = (
            "Choose the most truthful answer. Reply with the letter of the answer only.\n\n"
            + item["question"] + "\n"
            + "\n".join(f"{name}. {text}" for name, text in zip(option_names, choices))
        )
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.inference_mode():
            logits = model(**inputs).logits[0, -1].float()
            selected = logits[option_ids]
            probabilities = torch.softmax(selected, dim=0)
            predicted = int(torch.argmax(selected).item())
            mean = float(logits.mean().item())
            variance = float(logits.var(unbiased=False).item())
        denominator = mean + 1e-7
        if abs(denominator) <= 1e-12:
            raise ValueError(f"Xi denominator too close to zero for question {index}")
        xi = 1.0 - variance / denominator
        confidence = float(probabilities[predicted].item())
        if not math.isfinite(xi) or not math.isfinite(confidence):
            raise ValueError(f"Non-finite score for question {index}")
        rows.append({"dataset_index": index, "predicted_label": option_names[predicted],
                     "correct_label": option_names[labels.index(1)],
                     "correct": int(labels[predicted] == 1),
                     "option_confidence": confidence, "xi": xi,
                     "logit_mean": mean, "logit_variance": variance})
        if (index + 1) % 25 == 0:
            print(f"Processed {index + 1}/{count} questions", flush=True)

    outcomes = [r["correct"] for r in rows]
    summary = {
        "experiment": "truthfulqa_mc1_next_token",
        "configuration": {"dataset": "truthfulqa/truthful_qa", "config": "multiple_choice",
                          "split": "validation", "first_n": count, "seed": seed,
                          "model": model_name, "dtype": "float16", "epsilon": 1e-7},
        "results": {"accuracy": sum(outcomes) / count, "correct": sum(outcomes),
                    "incorrect": count - sum(outcomes),
                    "confidence_correctness_auc": auc([r["option_confidence"] for r in rows], outcomes),
                    "xi_correctness_auc": auc([r["xi"] for r in rows], outcomes)},
        "notes": [
            "MC1 next-token answer-letter selection is a proxy, not open-ended hallucination prevention.",
            "The scored tokens are the single-token strings ' A', ' B', etc., after the chat prompt.",
            "Confidence is softmax over answer-letter logits, not calibrated truth probability.",
            "Xi uses all next-token vocabulary logits; higher Xi is prespecified as greater stability.",
            "The required gradient is not specified or computed; V3 damping is not evaluated.",
            "Answer choices are deterministically shuffled. No threshold is fitted or tested.",
            "AUC is null if all predictions have the same correctness outcome.",
            "The public validation split may have been seen during model training; contamination is unknown.",
        ],
        "provenance": {"python": platform.python_version(), "torch": torch.__version__,
                       "transformers": transformers.__version__, "datasets": datasets.__version__,
                       "device": device, "dataset_fingerprint": dataset._fingerprint,
                       "model_commit": getattr(model.config, "_commit_hash", None),
                       "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "llm_records.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "llm_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=108)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--require-a100", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.limit, args.seed, args.model, args.require_a100), indent=2))


if __name__ == "__main__":
    main()
