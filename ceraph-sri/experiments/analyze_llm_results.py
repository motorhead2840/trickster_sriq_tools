"""Shim. The Ξ-score summary script lives in ``sri.legacy.llm_xi_analysis``."""
from sri.legacy.llm_xi_analysis import analyze, main, percentile, wilson

__all__ = ["analyze", "main", "percentile", "wilson"]


if __name__ == "__main__":
    main()
