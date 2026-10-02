"""Real-engine integration tests for the OCR extractor.

Skipped unless the ``ocr`` extra is installed *and* Tesseract language data is present
(``OCR_TESSDATA`` or a system ``tessdata`` directory). CI runs them in the dedicated OCR
job; the rest of the suite never needs an OCR engine.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from receipt_verifier.dataset import load_dataset
from receipt_verifier.extractors.ocr import (
    OcrUnavailable,
    TesserocrEngine,
    default_extractor,
    discover_tessdata,
)
from receipt_verifier.replay import receipt_hash
from receipt_verifier.schema import AdversarialKind, Decision
from receipt_verifier.validate import ReceiptValidator, ValidationPolicy

SYNTHETIC_ROOT = Path(__file__).resolve().parents[1] / "dataset" / "synthetic"
COMMITTED_DATASETS = (SYNTHETIC_ROOT / "v1", SYNTHETIC_ROOT / "v2")

tesserocr = pytest.importorskip("tesserocr", reason="install the 'ocr' extra to run OCR tests")
pytestmark = pytest.mark.ocr

if discover_tessdata() is None:  # pragma: no cover - depends on the machine
    pytest.skip(
        "no tesseract language data found; set OCR_TESSDATA or install tesseract-ocr-spa",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def engine() -> TesserocrEngine:
    return TesserocrEngine()


@pytest.fixture(scope="module", params=COMMITTED_DATASETS, ids=["v1", "v2"])
def dataset(request: pytest.FixtureRequest) -> object:
    return load_dataset(request.param)


class TestEngineWiring:
    def test_languages_and_upscale_are_reported(self, engine: TesserocrEngine) -> None:
        assert engine.name == "ocr"
        assert engine.languages == "spa+eng"
        assert engine.upscale == 2.0

    def test_missing_language_data_is_reported_clearly(self) -> None:
        with pytest.raises(OcrUnavailable):
            TesserocrEngine(tessdata=Path("/nonexistent/tessdata"))

    def test_invalid_upscale_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="upscale"):
            TesserocrEngine(upscale=0.5)

    def test_unavailable_engine_is_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "receipt_verifier.extractors.ocr.TesserocrEngine", _raising_engine, raising=True
        )
        with pytest.raises(OcrUnavailable):
            default_extractor()


def _raising_engine(**_: object) -> None:
    raise OcrUnavailable("no engine")


class TestRealReceipts:
    def test_one_layout_per_issuer_is_approved(self, dataset) -> None:
        extractor = default_extractor()
        validator = ReceiptValidator(dataset.allowed_destinations, ValidationPolicy())
        first_per_issuer: dict[object, object] = {}
        for label in dataset.labels:
            if label.adversarial is AdversarialKind.NONE:
                first_per_issuer.setdefault(label.issuer, label)
        assert len(first_per_issuer) == 6
        for label in first_per_issuer.values():
            extraction = extractor.extract(dataset.image_bytes(label))
            result = validator.validate(extraction, label.expectation, now=dataset.evaluation_at)
            assert extraction.amount.value == label.amount, label.sample_id
            assert result.decision is Decision.APPROVE, (label.sample_id, result.reasons)

    def test_adversarial_set_produces_no_false_approval(self, dataset) -> None:
        """Acceptance criterion: the local OCR path never approves a forged receipt."""
        extractor = default_extractor()
        validator = ReceiptValidator(dataset.allowed_destinations, ValidationPolicy())
        seen: dict[str, str] = {}
        decisions: dict[AdversarialKind, set[Decision]] = {}
        for label in dataset.labels:
            image = dataset.image_bytes(label)
            image_hash = receipt_hash(image)
            extraction = extractor.extract(image)
            result = validator.validate(
                extraction,
                label.expectation,
                now=dataset.evaluation_at,
                seen_operation_ids=seen,
                receipt_hash=image_hash,
            )
            if extraction.operation_id.value is not None:
                seen.setdefault(extraction.operation_id.value, image_hash)
            if label.adversarial is not AdversarialKind.NONE:
                decisions.setdefault(label.adversarial, set()).add(result.decision)
        assert decisions, "the adversarial set must not be empty"
        for kind, outcomes in decisions.items():
            assert Decision.APPROVE not in outcomes, kind
        assert decisions[AdversarialKind.DUPLICATE_OPERATION_ID] == {Decision.MANUAL_REVIEW}
        assert decisions[AdversarialKind.EDITED_AMOUNT] == {Decision.REJECT}
        assert decisions[AdversarialKind.WRONG_DESTINATION] == {Decision.REJECT}
        assert decisions[AdversarialKind.INJECTED_INSTRUCTION] == {Decision.REJECT}
        assert decisions[AdversarialKind.STALE_DATE] == {Decision.REJECT}

    def test_injected_instruction_reaches_the_validator(self, dataset) -> None:
        """The injection text must survive OCR, or the check would be vacuous."""
        extractor = default_extractor()
        sample = next(
            label
            for label in dataset.labels
            if label.adversarial is AdversarialKind.INJECTED_INSTRUCTION
        )
        extraction = extractor.extract(dataset.image_bytes(sample))
        assert extraction.memo.value is not None
        assert "instrucciones" in extraction.memo.value.lower() or "instructions" in (
            extraction.memo.value.lower()
        )

    def test_engine_is_deterministic(self, engine: TesserocrEngine, dataset) -> None:
        sample = dataset.labels[0]
        image = dataset.image_bytes(sample)
        first = engine.read(image)
        second = engine.read(image)
        assert [line.text for line in first.lines] == [line.text for line in second.lines]
