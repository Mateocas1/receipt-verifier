"""Shared builders for unit tests: valid labels and extractions without rendering."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

from receipt_verifier.builders import build_extraction
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
