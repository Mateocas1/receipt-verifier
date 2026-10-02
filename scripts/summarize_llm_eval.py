"""Aggregate the live harness reports into the committed comparison summary.

    uv run python scripts/summarize_llm_eval.py \
        --probe reports/vision-probe.json \
        --report reports/eval-llm-qwen3.6.json --report reports/eval-cascade.json \
        --out results/llm-eval.json --markdown

`--report` may be repeated. `--markdown` prints the README table rendered from the same
rows, so the table cannot drift from the runs it describes. Reports that do not exist are
reported and skipped, so a partially finished matrix still produces a usable summary.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from receipt_verifier.eval_summary import render_markdown, summarize_report

DEFAULT_OUTPUT: Final = Path("results/llm-eval.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, default=Path("reports/vision-probe.json"))
    parser.add_argument("--report", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", action="store_true", help="print the README table")
    parser.add_argument(
        "--role",
        action="append",
        default=[],
        help="role label per --report, in the same order (e.g. primary, secondary, cascade)",
    )
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows: list[dict[str, Any]] = []
    for index, path in enumerate(args.report):
        if not path.is_file():
            print(f"missing report, skipped: {path}")
            continue
        row = summarize_report(load_json(path))
        if index < len(args.role):
            row["role"] = args.role[index]
        rows.append(row)

    probe: dict[str, Any] | None = None
    if args.probe.is_file():
        probe = load_json(args.probe)
    else:
        print(f"missing probe report: {args.probe}")

    summary = {
        "schema": 1,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "provider": {
            "base_url": (probe or {}).get("base_url", ""),
            "rpm": (probe or {}).get("rpm", 0),
        },
        "vision_probe": probe,
        "runs": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"summary: {args.out} ({len(rows)} run(s))")
    if args.markdown:
        print()
        print(render_markdown(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
