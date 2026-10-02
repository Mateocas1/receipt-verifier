"""Harness plumbing: the run loop can report progress without changing the result."""

from __future__ import annotations

from pathlib import Path

from receipt_verifier.dataset import load_dataset
from receipt_verifier.extractors.dummy import DummyExtractor
from receipt_verifier.extractors.retry import DestinationRetryExtractor
from receipt_verifier.harness import render_table, run_evaluation

DATASET = Path(__file__).resolve().parents[1] / "dataset" / "synthetic" / "v1"


def test_progress_callback_sees_every_sample() -> None:
    dataset = load_dataset(DATASET)
    extractor = DummyExtractor.from_dataset(DATASET)
    seen: list[tuple[int, int]] = []
    report = run_evaluation(
        dataset, extractor, on_sample=lambda done, total: seen.append((done, total))
    )
    total = len(dataset.labels)
    assert seen[0] == (1, total)
    assert seen[-1] == (total, total)
    assert len(seen) == total
    assert report.metrics.n == total


def test_a_missing_callback_changes_nothing() -> None:
    dataset = load_dataset(DATASET)
    extractor = DummyExtractor.from_dataset(DATASET)
    report = run_evaluation(dataset, extractor)
    assert report.metrics.n == len(dataset.labels)
    assert report.metrics.false_approvals == 0


def test_the_report_records_the_retry_setting() -> None:
    dataset = load_dataset(DATASET)
    extractor = DummyExtractor.from_dataset(DATASET)
    report = run_evaluation(dataset, extractor, destination_retry=True)
    assert report.destination_retry is True
    assert "destination_retry" in render_table(report)


def test_a_perfect_offline_reading_never_retries() -> None:
    """The gate extractors read these fixtures exactly, so the retry must stay idle."""
    dataset = load_dataset(DATASET)
    extractor = DestinationRetryExtractor(
        inner=DummyExtractor.from_dataset(DATASET),
        allowed_destinations=dataset.allowed_destinations,
    )
    report = run_evaluation(dataset, extractor, destination_retry=True)
    assert report.metrics.false_approvals == 0
    assert report.metrics.destination_retries == 0
    assert report.metrics.approve_recall == 1.0
