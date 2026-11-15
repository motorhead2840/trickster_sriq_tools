"""Shim. The Ξ-as-LLM-score script lives in ``sri.legacy.llm_benchmark``."""
from sri.legacy.llm_benchmark import auc, main, run, shuffled_choices

__all__ = ["auc", "main", "run", "shuffled_choices"]


if __name__ == "__main__":
    main()
