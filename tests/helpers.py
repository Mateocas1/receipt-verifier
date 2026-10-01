"""Shared builders for unit tests: valid labels and extractions without rendering."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from time import sleep

from receipt_verifier.builders import build_extraction
from receipt_verifier.confidence import SourceQuality, score_extraction
from receipt_verifier.extraction import FIELD_NAMES, ExtractionResult
from receipt_verifier.schema import (
    AR_TZ,
    AdversarialKind,
    Decision,
    Destination,
    DestinationKind,
    Issuer,
    LedgerEntry,
    ReceiptLabel,
    VerdictReason,
)

NOW = datetime(2025, 7, 1, 9, 0, tzinfo=AR_TZ)
RECEIVED_AT = NOW - timedelta(hours=1)
ALIAS_DESTINATION = Destination(
    kind=DestinationKind.ALIAS, value="camila.gomez.ar", holder="Camila Gómez"
)
CVU_DESTINATION = Destination(
    kind=DestinationKind.CVU,
    value="2852240785731632146398",
    holder="Lautaro Ojeda",
)


def make_label(
    *,
    sample_id: str = "sample-0001",
    issuer: Issuer = Issuer.MP,
    amount: Decimal = Decimal("2500.00"),
    amount_detail: Decimal | None = None,
    transferred_at: datetime = RECEIVED_AT,
    sender_name: str = "Camila Gómez",
    sender_bank: str = "Banco del Río",
    destination: Destination = ALIAS_DESTINATION,
    operation_id: str = "MP-000000000001",
    memo: str = "Alquiler julio",
    expected_decision: Decision = Decision.APPROVE,
    reasons: tuple[VerdictReason, ...] = (),
    adversarial: AdversarialKind = AdversarialKind.NONE,
    expectation: LedgerEntry | None = None,
    expected_amount: Decimal | None = None,
    expected_destination: Destination | None = None,
) -> ReceiptLabel:
    """Build a valid :class:`ReceiptLabel`; overrides make adversarial cases easy."""
    ledger = expectation or LedgerEntry(
        payment_id="pay-00001",
        amount=expected_amount if expected_amount is not None else amount,
        destination=expected_destination if expected_destination is not None else destination,
        requested_at=transferred_at,
    )
    return ReceiptLabel(
        sample_id=sample_id,
        image=f"images/{sample_id}.png",
        issuer=issuer,
        amount=amount,
        amount_detail=amount_detail if amount_detail is not None else amount,
        transferred_at=transferred_at,
        sender_name=sender_name,
        sender_bank=sender_bank,
        destination=destination,
        operation_id=operation_id,
        memo=memo,
        expected_decision=expected_decision,
        reasons=reasons,
        adversarial=adversarial,
        expectation=ledger,
    )


def extraction_from(
    label: ReceiptLabel,
    *,
    extractor: str = "test",
    confidences: dict[str, float] | None = None,
    **overrides: object,
) -> ExtractionResult:
    """Build an extraction that mirrors ``label``, with optional field overrides."""
    values: dict[str, object] = {
        "amount": label.amount,
        "amount_detail": label.amount_detail,
        "transferred_at": label.transferred_at,
        "sender_name": label.sender_name,
        "sender_bank": label.sender_bank,
        "destination": label.destination,
        "operation_id": label.operation_id,
        "issuer": label.issuer,
        "memo": label.memo,
    }
    values.update(overrides)
    confidences = dict(confidences or {})
    for name in FIELD_NAMES:
        if values.get(name) is None:
            confidences.setdefault(name, 0.0)
    return build_extraction(extractor, values, confidences=confidences)


class StubExtractor:
    """Test-only extractor: canned result, optional failure and optional latency."""

    def __init__(
        self,
        name: str = "stub",
        *,
        result: ExtractionResult | None = None,
        error: Exception | None = None,
        delay: float = 0.0,
    ) -> None:
        self._name = name
        self._result = result
        self._error = error
        self._delay = delay
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    def set_error(self, error: Exception | None) -> None:
        """Switch the stub between healthy and failing mode."""
        self._error = error

    def extract(self, image: bytes) -> ExtractionResult:
        self.calls += 1
        if self._delay:
            sleep(self._delay)
        if self._error is not None:
            raise self._error
        if self._result is not None:
            return self._result
        return build_extraction(
            self._name,
            {
                "amount": Decimal("2500.00"),
                "amount_detail": Decimal("2500.00"),
                "transferred_at": RECEIVED_AT,
                "sender_name": "Camila Gómez",
                "sender_bank": "Banco del Río",
                "destination": ALIAS_DESTINATION,
                "operation_id": "MP-000000000001",
                "issuer": Issuer.MP,
                "memo": "Alquiler julio",
            },
        )


def printed_now() -> str:
    """Current Argentine local time in the receipts' printed format."""
    return datetime.now(tz=AR_TZ).strftime("%d/%m/%Y %H:%M")


def reading(
    extractor: str,
    *,
    missing: str | None = None,
    weak: str | None = None,
    confidence: float = 0.95,
    **overrides: object,
) -> ExtractionResult:
    """A complete reading, optionally missing one critical field or weak on one of them."""
    values: dict[str, object] = {
        "amount": "2.500,00",
        "amount_detail": "2.500,00",
        # "Now" by default: a fixture date would go stale against the service clock.
        "transferred_at": printed_now(),
        "sender_name": "Camila Gómez",
        "sender_bank": "Banco del Río",
        "destination": {"kind": "alias", "value": "camila.gomez.ar", "holder": "Lautaro Ojeda"},
        "operation_id": "MP-6E81DA675F9D",
        "issuer": "mp",
        "memo": "Alquiler julio",
    }
    values.update(overrides)
    qualities = dict.fromkeys(values, confidence)
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
