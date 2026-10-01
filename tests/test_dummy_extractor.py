from decimal import Decimal

import pytest

from receipt_verifier.extraction import FIELD_NAMES
from receipt_verifier.extractors.dummy import DummyExtractor, image_digest
from receipt_verifier.metrics import evaluate_sample, expected_field_value, fields_match, is_covered
from receipt_verifier.schema import ReceiptLabel
from receipt_verifier.validate import ReceiptValidator
from tests.helpers import ALIAS_DESTINATION, NOW, make_label

IMAGE = b"\x89PNG\r\n\x1a\n-fake-bytes"


def build(
    noise: float = 0.0, *, label: ReceiptLabel | None = None, seed: int = 1717
) -> DummyExtractor:
    target = label if label is not None else make_label()
    return DummyExtractor({image_digest(IMAGE): target}, noise=noise, seed=seed)


class TestPerfectReplay:
    def test_all_fields_match_the_label(self) -> None:
        label = make_label()
        result = build().extract(IMAGE)
        assert result.extractor == "dummy"
        fields = result.field_map()
        assert set(fields) == set(FIELD_NAMES)
        for name in FIELD_NAMES:
            assert fields[name].confidence == 1.0
            assert fields_match(name, expected_field_value(label, name), fields[name].value)

    def test_zero_noise_is_repeatable(self) -> None:
        extractor = build()
        assert extractor.extract(IMAGE) == extractor.extract(IMAGE)

    def test_name_and_noise_are_exposed(self) -> None:
        extractor = build(noise=0.25)
        assert extractor.name == "dummy"
        assert extractor.noise == 0.25

    def test_unknown_image_raises(self) -> None:
        with pytest.raises(KeyError, match="no ground truth"):
            build().extract(b"other-image")

    def test_invalid_noise_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="noise"):
            build(noise=1.5)

    def test_dummy_wins_on_coverage_and_validation(self) -> None:
        label = make_label()
        extraction = build().extract(IMAGE)
        validation = ReceiptValidator({ALIAS_DESTINATION.key}).validate(
            extraction, label.expectation, now=NOW
        )
        assert validation.approved
        assert is_covered(extraction, min_confidence=0.5)
        outcome = evaluate_sample(
            label,
            extraction,
            validation,
            latency_ms=1.0,
            cost_usd=Decimal("0"),
            min_confidence=0.5,
        )
        assert all(outcome.field_matches.values())
        assert outcome.copied_reasons == ()


class TestNoise:
    def test_full_noise_degrades_every_field(self) -> None:
        extraction = build(noise=1.0).extract(IMAGE)
        fields = extraction.field_map()
        assert all(field.value is None or field.confidence < 1.0 for field in fields.values())

    def test_full_noise_blocks_approval(self) -> None:
        label = make_label()
        extraction = build(noise=1.0).extract(IMAGE)
        validation = ReceiptValidator({ALIAS_DESTINATION.key}).validate(
            extraction, label.expectation, now=NOW
        )
        assert validation.decision.value == "reject"

    def test_noise_is_seeded(self) -> None:
        first = build(noise=0.5, seed=7).extract(IMAGE)
        second = build(noise=0.5, seed=7).extract(IMAGE)
        assert first == second

    def test_partial_noise_breaks_coverage(self) -> None:
        extraction = build(noise=0.5, seed=3).extract(IMAGE)
        assert not is_covered(extraction, min_confidence=0.5)
