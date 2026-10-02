"""Evaluate an extractor over a dataset and report the harness metrics.

    uv run python scripts/evaluate.py --dataset dataset/synthetic/v2 --extractor dummy
    uv run python scripts/evaluate.py --dataset dataset/synthetic/v2 --extractor ocr
    uv run python scripts/evaluate.py --dataset dataset/synthetic/v2 --extractor llm --model qwen3.6
    uv run python scripts/evaluate.py --dataset dataset/synthetic/v2 --extractor cascade

``llm`` measures exactly one vision model (``--model``, defaulting to
``VISION_MODEL_PRIMARY``) with no fallback, so the numbers belong to that model alone;
``cascade`` uses the configured stages in order. Both need provider credentials
(``LLM_API_KEY``) and both pace themselves through ``EVAL_RPM`` (default 20 requests per
minute) because the provider window budget is shared.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Final

from receipt_verifier.dataset import load_dataset
from receipt_verifier.extraction import ReceiptExtractor
from receipt_verifier.extractors.dummy import DummyExtractor
from receipt_verifier.extractors.resilient import ResilientExtractor
from receipt_verifier.harness import default_report_path, render_table, run_evaluation, write_report
from receipt_verifier.ratelimit import limiter_from_env
from receipt_verifier.schema import AR_TZ

DEFAULT_DATASET: Final = Path("dataset/synthetic/v2")
LLM_MODEL_ENV: Final = "VISION_MODEL_PRIMARY"
EXTRACTORS: Final = ("dummy", "ocr", "llm", "cascade")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--extractor", choices=EXTRACTORS, default="dummy")
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "vision model id for --extractor llm (default: VISION_MODEL_PRIMARY). "
            "Never used by --extractor cascade, which reads the configured stages."
        ),
    )
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
    parser.add_argument(
        "--strict-errors",
        action="store_true",
        help=(
            "abort the run on the first extractor exception; by default a failed sample "
            "is recorded as an empty reading so a live run survives provider hiccups"
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="report progress every N samples on stderr; 0 disables it",
    )
    args = parser.parse_args(argv)
    args.tolerate_errors = not args.strict_errors
    return args


def build_extractor(args: argparse.Namespace) -> tuple[ReceiptExtractor, float]:
    """Build the requested extractor, or explain what is missing."""
    if args.extractor == "dummy":
        extractor = DummyExtractor.from_dataset(
            args.dataset, noise=args.noise, seed=args.noise_seed
        )
        return extractor, extractor.noise
    if args.noise:
        raise SystemExit("--noise only applies to the dummy extractor")
    if args.model and args.extractor != "llm":
        raise SystemExit("--model only applies to --extractor llm")
    if args.extractor == "ocr":
        from receipt_verifier.extractors.ocr import OcrUnavailable, default_extractor

        try:
            return default_extractor(), 0.0
        except OcrUnavailable as exc:
            raise SystemExit(
                f"local OCR is unavailable: {exc}\n"
                "install the extra (uv sync --extra ocr) and point OCR_TESSDATA at a "
                "tessdata directory with eng/spa traineddata"
            ) from exc
    from receipt_verifier.extractors.resilient import ResilientExtractor
    from receipt_verifier.service.settings import NoExtractorConfigured, Settings

    settings = Settings.from_env()
    limiter = limiter_from_env()
    if args.extractor == "llm":
        model = args.model or settings.llm.primary_model
        if not settings.llm.api_key or not model:
            raise SystemExit(
                "the LLM extractor needs LLM_API_KEY and a model "
                "(--model <id> or VISION_MODEL_PRIMARY); the comparison table in the "
                "README marks this row as pending a key"
            )
        engine = settings.build_vision_model(model, limiter=limiter)
        if args.tolerate_errors:
            return ResilientExtractor(engine), 0.0
        return engine, 0.0
    try:
        return settings.build_extractor(limiter=limiter), 0.0
    except NoExtractorConfigured as exc:
        raise SystemExit(f"no extractor is configured: {exc}") from exc


def progress(args: argparse.Namespace) -> Callable[[int, int], None] | None:
    """A stderr progress reporter for long live runs (``--progress-every 0`` disables)."""
    every = int(args.progress_every)
    if every <= 0:
        return None

    def report(done: int, total: int) -> None:
        if done % every == 0 or done == total:
            print(f"  {done}/{total} samples", file=sys.stderr, flush=True)

    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset = load_dataset(args.dataset)
    extractor, noise = build_extractor(args)
    now = datetime.fromisoformat(args.now).astimezone(AR_TZ) if args.now else None

    report = run_evaluation(dataset, extractor, now=now, noise=noise, on_sample=progress(args))
    print(render_table(report))
    if isinstance(extractor, ResilientExtractor) and extractor.failures:
        print(f"\nextractor errors: {extractor.failures} sample(s) ({extractor.last_error})")

    if not args.no_json:
        json_path = args.json_path or default_report_path(extractor.name, noise)
        write_report(report, json_path)
        print(f"\njson report: {json_path}")

    failures = report.metrics.false_approvals
    if failures > args.max_false_approvals:
        print(f"\nFAIL: {failures} false approvals, allowed {args.max_false_approvals}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
