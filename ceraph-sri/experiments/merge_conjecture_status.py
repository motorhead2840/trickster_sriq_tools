"""Merge conjecture statuses from foundation-sprint JSON summaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sri.registry.conjectures import list_conjectures, record_result, reset_registry


def _status_rank(status: str) -> int:
    return {"refuted": 2, "confirmed": 1, "open": 0}.get(status, 0)


def merge(paths: list[Path], output: Path) -> dict:
    reset_registry()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload.get("conjectures", []):
            current = next(
                c for c in list_conjectures() if c.conjecture_id == item["conjecture_id"]
            )
            if _status_rank(item["status"]) >= _status_rank(current.status):
                record_result(
                    item["conjecture_id"],
                    item["status"],
                    artifact_paths=item.get("artifact_paths", [str(path)]),
                    notes=item.get("notes", ""),
                )
    merged = {
        "conjectures": [
            {
                "conjecture_id": c.conjecture_id,
                "status": c.status,
                "notes": c.notes,
                "artifact_paths": c.artifact_paths,
                "statement": c.statement,
                "pass_criterion": c.pass_criterion,
            }
            for c in list_conjectures()
        ],
        "sources": [str(path) for path in paths],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    return merged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(merge(list(args.inputs), args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
