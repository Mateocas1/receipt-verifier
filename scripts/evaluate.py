"""Evaluate an extractor over a dataset and report the harness metrics.

uv run python scripts/evaluate.py --dataset dataset/synthetic/v1 --extractor dummy
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Final

from receipt_verifier.dataset import load_dataset
from receipt_verifier.extractors.dummy import DummyExtractor
from receipt_verifier.harness import default_report_path, render_table, run_evaluation, write_report
from receipt_verifier.schema import AR_TZ

DEFAULT_DATASET: Final = Path("dataset/synthetic/v1")
EXTRACTORS: Final = ("dummy",)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--extractor", choices=EXTRACTORS, default="dummy")
    parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="probability of corrupting each field (dummy extractor only)",
    )
    parser.add_argument("--noise-seed", type=int, default=1717)
    parser.add_argument(
        "--now",
        default=None,
        help="override the evaluation clock (ISO-8601). Defaults to the dataset manifest clock.",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        type=Path,
        default=None,
        help="where to write the JSON report (default: reports/eval-<extractor>.json)",
    )
    parser.add_argument("--no-json", action="store_true", help="print the table only")
    parser.add_argument(
        "--max-false-approvals",
        type=int,
        default=0,
        help="exit non-zero when more than this many receipts are wrongly approved",
    )
    return parser.parse_args(argv)


def build_extractor(args: argparse.Namespace) -> DummyExtractor:
    if args.extractor != "dummy":
        raise SystemExit(f"unknown extractor: {args.extractor}")
    return DummyExtractor.from_dataset(args.dataset, noise=args.noise, seed=args.noise_seed)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset = load_dataset(args.dataset)
    extractor = build_extractor(args)
    now = datetime.fromisoformat(args.now).astimezone(AR_TZ) if args.now else None

    report = run_evaluation(
        dataset,
        extractor,
        now=now,
        noise=extractor.noise,
    )
    print(render_table(report))

    if not args.no_json:
        json_path = args.json_path or default_report_path(extractor.name, extractor.noise)
        write_report(report, json_path)
        print(f"\njson report: {json_path}")

    failures = report.metrics.false_approvals
    if failures > args.max_false_approvals:
        print(f"\nFAIL: {failures} false approvals, allowed {args.max_false_approvals}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
