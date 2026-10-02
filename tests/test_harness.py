"""Harness plumbing: the run loop can report progress without changing the result."""

from __future__ import annotations

from pathlib import Path

from receipt_verifier.dataset import load_dataset
from receipt_verifier.extractors.dummy import DummyExtractor
from receipt_verifier.harness import run_evaluation

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
