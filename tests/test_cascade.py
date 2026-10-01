"""Cascade tests: fallback order, breakers, deadlines and partial readings."""

from __future__ import annotations

from decimal import Decimal

import pytest

from receipt_verifier.circuit import CircuitBreaker, CircuitState, GuardedExtractor
from receipt_verifier.confidence import SourceQuality, score_extraction
from receipt_verifier.extraction import ExtractionResult
from receipt_verifier.extractors.cascade import (
    NO_EXTRACTOR,
    CascadeExhausted,
    CascadeExtractor,
)
from receipt_verifier.extractors.llm import LlmTransportError
from receipt_verifier.validate import ValidationPolicy
from tests.helpers import StubExtractor

IMAGE = b"\x89PNG-irrelevant"
CRITICAL = ("amount", "transferred_at", "sender_name", "destination", "operation_id", "issuer")


def reading(
    extractor: str,
    *,
    missing: str | None = None,
    weak: str | None = None,
    confidence: float = 0.95,
) -> ExtractionResult:
    """A full reading, optionally missing one critical field or weak on one of them."""
    values: dict[str, object] = {
        "amount": "2.500,00",
        "amount_detail": "2.500,00",
        "transferred_at": "30/06/2025 18:20",
        "sender_name": "Camila Gómez",
        "sender_bank": "Banco del Río",
        "destination": {"kind": "alias", "value": "camila.gomez.ar", "holder": "Lautaro Ojeda"},
        "operation_id": "MP-6E81DA675F9D",
        "issuer": "mp",
        "memo": "Alquiler julio",
    }
    qualities = dict.fromkeys(values, 0.95)
    if missing == "amount":
        values["amount"] = "no legible"
        qualities["amount"] = 0.0
    elif missing is not None:
        values.pop(missing, None)
        qualities.pop(missing, None)
    if weak is not None:
        qualities[weak] = confidence
    return score_extraction(
        extractor, values, source_quality=SourceQuality(default=0.95, per_field=qualities)
    )


class TestFallback:
    def test_primary_wins_when_it_is_complete(self) -> None:
        primary = StubExtractor("llm")
        fallback = StubExtractor("ocr")
        cascade = CascadeExtractor([primary, fallback])
        result = cascade.extract(IMAGE)
        assert result.extractor == "llm"
        assert fallback.calls == 0
        assert cascade.last_outcome is not None
        assert cascade.last_outcome.usable

    def test_transport_error_falls_back_to_local_ocr(self) -> None:
        """Acceptance: with the provider down, the request is still answered by OCR."""
        primary = StubExtractor("llm", error=LlmTransportError("connection refused"))
        fallback = StubExtractor("ocr")
        cascade = CascadeExtractor([primary, fallback])
        result = cascade.extract(IMAGE)
        assert result.extractor == "ocr"
        assert result.amount.value == Decimal("2500.00")
        outcome = cascade.last_outcome
        assert outcome is not None
        assert outcome.attempts == ("llm", "ocr")
        assert outcome.failures == {"llm": "LlmTransportError"}
        assert cascade.usage_counts() == {"ocr": 1}

    def test_timeout_falls_back_to_local_ocr(self) -> None:
        primary = StubExtractor("llm", delay=0.3)
        fallback = StubExtractor("ocr")
        cascade = CascadeExtractor(
            [
                GuardedExtractor(
                    primary,
                    timeout_seconds=0.01,
                    breaker=CircuitBreaker(name="llm", failure_threshold=1),
                ),
                fallback,
            ]
        )
        assert cascade.extract(IMAGE).extractor == "ocr"

    def test_partial_reading_falls_back(self) -> None:
        primary = StubExtractor("llm", result=reading("llm", missing="amount"))
        fallback = StubExtractor("ocr")
        cascade = CascadeExtractor([primary, fallback])
        assert cascade.extract(IMAGE).extractor == "ocr"

    def test_open_breaker_skips_the_stage(self) -> None:
        breaker = CircuitBreaker(name="llm", failure_threshold=1)
        breaker.record_failure()
        primary = StubExtractor("llm")
        fallback = StubExtractor("ocr")
        cascade = CascadeExtractor([GuardedExtractor(primary, breaker=breaker), fallback])
        assert cascade.extract(IMAGE).extractor == "ocr"
        assert primary.calls == 0
        assert cascade.last_outcome is not None
        assert cascade.last_outcome.failures == {"llm": "CircuitOpenError"}

    def test_only_the_secondary_model_stage_is_tried_when_primary_fails(self) -> None:
        primary = StubExtractor("llm-primary", error=LlmTransportError("down"))
        secondary = StubExtractor("llm-secondary")
        ocr = StubExtractor("ocr")
        cascade = CascadeExtractor([primary, secondary, ocr])
        assert cascade.extract(IMAGE).extractor == "llm-secondary"
        assert ocr.calls == 0


class TestExhaustion:
    def test_all_stages_failing_returns_a_none_reading(self) -> None:
        cascade = CascadeExtractor(
            [
                StubExtractor("llm", error=LlmTransportError("down")),
                StubExtractor("ocr", error=RuntimeError("no engine")),
            ]
        )
        result = cascade.extract(IMAGE)
        assert result.extractor == NO_EXTRACTOR
        assert result.amount.value is None
        outcome = cascade.last_outcome
        assert outcome is not None
        assert not outcome.usable
        assert outcome.failures == {"llm": "LlmTransportError", "ocr": "RuntimeError"}

    def test_extract_or_raise_reports_the_failures(self) -> None:
        cascade = CascadeExtractor([StubExtractor("llm", error=LlmTransportError("down"))])
        with pytest.raises(CascadeExhausted) as excinfo:
            cascade.extract_or_raise(IMAGE)
        assert excinfo.value.failures == {"llm": "LlmTransportError"}

    def test_extract_or_raise_passes_usable_readings_through(self) -> None:
        cascade = CascadeExtractor([StubExtractor("llm")])
        assert cascade.extract_or_raise(IMAGE).amount.value == Decimal("2500.00")

    def test_the_most_complete_partial_reading_is_returned(self) -> None:
        """A partial reading beats nothing: the reviewer sees what was readable."""
        more = StubExtractor("thin", result=reading("thin", missing="operation_id"))
        less = StubExtractor("thinner", result=reading("thinner", missing="amount"))
        cascade = CascadeExtractor([more, less])
        result = cascade.extract(IMAGE)
        assert result.extractor == "thin"
        assert cascade.last_outcome is not None
        assert not cascade.last_outcome.usable
        assert cascade.usage_counts() == {"thin": 1}

    def test_a_cascade_without_stages_is_a_programming_error(self) -> None:
        with pytest.raises(ValueError, match="at least one extractor"):
            CascadeExtractor([])


class TestPolicy:
    def test_min_confidence_comes_from_the_validation_policy(self) -> None:
        borderline = reading("llm", weak="amount", confidence=0.6)
        lenient = CascadeExtractor(
            [StubExtractor("llm", result=borderline), StubExtractor("ocr")],
            policy=ValidationPolicy(min_field_confidence=0.5),
        )
        assert lenient.extract(IMAGE).extractor == "llm"
        strict = CascadeExtractor(
            [StubExtractor("llm", result=borderline), StubExtractor("ocr")],
            policy=ValidationPolicy(min_field_confidence=0.9),
        )
        assert strict.extract(IMAGE).extractor == "ocr"


class TestObservability:
    def test_statuses_expose_every_stage(self) -> None:
        cascade = CascadeExtractor(
            [
                GuardedExtractor(StubExtractor("llm")),
                GuardedExtractor(StubExtractor("ocr")),
            ]
        )
        statuses = cascade.statuses()
        assert [status.name for status in statuses] == ["llm", "ocr"]
        assert all(status.state is CircuitState.CLOSED for status in statuses)

    def test_failure_counts_accumulate(self) -> None:
        cascade = CascadeExtractor(
            [StubExtractor("llm", error=LlmTransportError("down")), StubExtractor("ocr")]
        )
        for _ in range(3):
            cascade.extract(IMAGE)
        assert cascade.failure_counts() == {"llm": 3}
        assert cascade.usage_counts() == {"ocr": 3}

    def test_bare_extractors_are_wrapped_by_the_cascade(self) -> None:
        cascade = CascadeExtractor([StubExtractor("llm")])
        assert isinstance(cascade.stages[0], GuardedExtractor)
        assert cascade.stages[0].name == "llm"
