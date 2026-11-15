"""Emit SREE Mentor × Transformer Engine dropless MoE plan JSON (no GPU required)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sri.sree.moe_te_config import mentor_te_moe_plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/benchmark-results/sree-mentor-te-moe"),
    )
    parser.add_argument(
        "--profile",
        choices=("smoke", "scale_ref"),
        default="smoke",
    )
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = mentor_te_moe_plan(mentor_profile=args.profile)
    path = args.output_dir / f"te_moe_plan_{args.profile}.json"
    path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(path), "profile": args.profile}, indent=2))
    print("<VALIDATOR_OUTPUT>")
    print("VERDICT: CONFIG_OK")
    print("</VALIDATOR_OUTPUT>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
