"""Extractor contract: image bytes in, typed fields with per-field confidence out.

Every extractor in the cascade (vision LLM, local OCR, or the dummy used by the
harness) implements :class:`ReceiptExtractor`, so the validator and the evaluation
metrics never need to know which engine produced the fields.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from receipt_verifier.schema import Destination, Issuer

FIELD_NAMES: tuple[str, ...] = (
    "amount",
    "amount_detail",
    "transferred_at",
    "sender_name",
    "sender_bank",
    "destination",
    "operation_id",
    "issuer",
    "memo",
)
"""Canonical field order. Metrics, tables and the dummy extractor all use it."""

CRITICAL_FIELD_NAMES: tuple[str, ...] = (
    "amount",
    "transferred_at",
    "sender_name",
    "destination",
    "operation_id",
    "issuer",
)
"""Fields a receipt must expose before a verdict can be reached."""


class ExtractedField[T](BaseModel):
    """A single extracted value plus the model's confidence in it."""

    model_config = ConfigDict(frozen=True)

    value: T | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def usable(self) -> bool:
        """True when a value exists and carries non-zero confidence."""
        return self.value is not None and self.confidence > 0.0


class ExtractionResult(BaseModel):
    """All fields of one receipt as read by one extractor."""

    model_config = ConfigDict(frozen=True)

    extractor: str
    amount: ExtractedField[Decimal] = ExtractedField()
    amount_detail: ExtractedField[Decimal] = ExtractedField()
    transferred_at: ExtractedField[datetime] = ExtractedField()
    sender_name: ExtractedField[str] = ExtractedField()
    sender_bank: ExtractedField[str] = ExtractedField()
    destination: ExtractedField[Destination] = ExtractedField()
    operation_id: ExtractedField[str] = ExtractedField()
    issuer: ExtractedField[Issuer] = ExtractedField()
    memo: ExtractedField[str] = ExtractedField()
    raw_text: str = ""
    latency_ms: float = 0.0
    cost_usd: Decimal = Decimal("0")
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)

    def field_map(self) -> dict[str, ExtractedField[Any]]:
        """Mapping of canonical field name to its extraction, in ``FIELD_NAMES`` order."""
        return {name: getattr(self, name) for name in FIELD_NAMES}


@runtime_checkable
class ReceiptExtractor(Protocol):
    """Anything that turns image bytes into :class:`ExtractionResult`."""

    @property
    def name(self) -> str:
        """Stable identifier reported in evaluation output."""
        ...

    def extract(self, image: bytes) -> ExtractionResult:
        """Extract fields from one receipt image."""
        ...
