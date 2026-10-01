"""Request and response models for the public API."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from receipt_verifier.extraction import ExtractedField, ExtractionResult
from receipt_verifier.identifiers import quantize_amount
from receipt_verifier.schema import (
    AR_TZ,
    Decision,
    Destination,
    DestinationKind,
    LedgerEntry,
    VerdictReason,
)

Amount = Annotated[Decimal, Field(gt=Decimal("0"))]


class DestinationInput(BaseModel):
    """Where the payment was supposed to go, from the caller's own records."""

    model_config = ConfigDict(extra="ignore")

    kind: DestinationKind
    value: str


class PaymentInput(BaseModel):
    """Optional ledger evidence: without it the amount cannot be verified."""

    model_config = ConfigDict(extra="ignore")

    payment_id: str = "unverified"
    amount: Amount
    destination: DestinationInput

    def to_ledger_entry(self, *, now: datetime) -> LedgerEntry:
        return LedgerEntry(
            payment_id=self.payment_id,
            amount=quantize_amount(self.amount),
            destination=Destination(
                kind=self.destination.kind, value=self.destination.value, holder=""
            ),
            requested_at=now,
        )


class ReceiptRequest(BaseModel):
    """JSON body form: a base64 image plus optional ledger evidence."""

    model_config = ConfigDict(extra="ignore")

    image_base64: str = Field(min_length=1)
    payment: PaymentInput | None = None


class ReceiptResponse(BaseModel):
    """What the caller needs: fields, confidence, who read it and what code decided."""

    model_config = ConfigDict(frozen=True)

    decision: Decision
    reasons: tuple[VerdictReason, ...] = ()
    media_type: str
    extractor_used: str
    extractors_attempted: tuple[str, ...] = ()
    fields: dict[str, str | None]
    per_field_confidence: dict[str, float]
    checks: dict[str, bool]
    latency_ms: float


def render_field(name: str, field: ExtractedField[object]) -> str | None:
    """Serialize one extracted value for a JSON response (no binary, no raw pixels)."""
    value = field.value
    if value is None:
        return None
    if name == "destination" and isinstance(value, Destination):
        return f"{value.kind.value} {value.value}"
    if isinstance(value, datetime):
        return value.astimezone(AR_TZ).isoformat()
    if isinstance(value, Decimal):
        return f"{quantize_amount(value):.2f}"
    return str(value)


def response_from(
    extraction: ExtractionResult,
    *,
    decision: Decision,
    reasons: tuple[VerdictReason, ...],
    checks: dict[str, bool],
    latency_ms: float,
    media_type: str,
    attempted: tuple[str, ...] = (),
) -> ReceiptResponse:
    fields = extraction.field_map()
    return ReceiptResponse(
        decision=decision,
        reasons=reasons,
        media_type=media_type,
        extractor_used=extraction.extractor,
        extractors_attempted=attempted,
        fields={name: render_field(name, field) for name, field in fields.items()},
        per_field_confidence={name: field.confidence for name, field in fields.items()},
        checks=checks,
        latency_ms=latency_ms,
    )


__all__ = [
    "DestinationInput",
    "PaymentInput",
    "ReceiptRequest",
    "ReceiptResponse",
    "render_field",
    "response_from",
]
