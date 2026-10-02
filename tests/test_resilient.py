"""Resilience-wrapper tests: one failed sample must not end an evaluation run."""

from __future__ import annotations

from decimal import Decimal

from receipt_verifier.builders import build_extraction
from receipt_verifier.extraction import ExtractionResult
from receipt_verifier.extractors.resilient import ResilientExtractor

IMAGE = b"\x89PNG\r\n\x1a\n fake"


class WorkingExtractor:
    @property
    def name(self) -> str:
        return "llm-test"

    def extract(self, image: bytes) -> ExtractionResult:
        return build_extraction(
            "llm-test", {"amount": Decimal("10.00")}, prompt_tokens=5, completion_tokens=1
        )


class FailingExtractor:
    @property
    def name(self) -> str:
        return "llm-flaky"

    def extract(self, image: bytes) -> ExtractionResult:
        raise RuntimeError("HTTP 502: bad gateway")


class TestResilientExtractor:
    def test_keeps_the_inner_name(self) -> None:
        assert ResilientExtractor(WorkingExtractor()).name == "llm-test"

    def test_passes_a_good_reading_through(self) -> None:
        extractor = ResilientExtractor(WorkingExtractor())
        result = extractor.extract(IMAGE)
        assert result.amount.value == Decimal("10.00")
        assert result.prompt_tokens == 5
        assert extractor.failures == 0
        assert result.error == ""

    def test_a_failure_becomes_an_empty_reading(self) -> None:
        extractor = ResilientExtractor(FailingExtractor())
        result = extractor.extract(IMAGE)
        assert result.extractor == "llm-flaky"
        assert result.amount.value is None
        assert result.error
        assert "HTTP 502" in result.error
        assert extractor.failures == 1

    def test_failures_are_counted_and_the_last_error_kept(self) -> None:
        extractor = ResilientExtractor(FailingExtractor())
        extractor.extract(IMAGE)
        extractor.extract(IMAGE)
        assert extractor.failures == 2
        assert extractor.last_error.startswith("RuntimeError")
