"""Labeled receipt schema: the single source of truth for the dataset and the harness.

A label is ground truth for one rendered receipt image (or, later, one anonymized
real receipt). A ``LedgerEntry`` is what the business side expected to receive, i.e.
the external evidence the deterministic validator compares against. Keeping both in
the same record lets the harness replay adversarial cases without a live payment API.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Self
from zoneinfo import ZoneInfo

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from receipt_verifier.identifiers import (
    is_valid_alias,
    is_valid_cbu_or_cvu,
    quantize_amount,
)

AR_TZ = ZoneInfo("America/Argentina/Buenos_Aires")
AR_UTC_OFFSET = timedelta(hours=-3)

Amount = Annotated[Decimal, Field(gt=Decimal("0"), max_digits=14, decimal_places=2)]


class Issuer(StrEnum):
    """Payment issuer (wallet, bank or state bank) that produced the receipt."""

    MP = "mp"
    UALA = "uala"
    BRUBANK = "brubank"
    GALICIA = "galicia"
    SANTANDER = "santander"
    BNA = "bna"


class Decision(StrEnum):
    """Verifier verdict. ``MANUAL_REVIEW`` means a human must decide: the code could not
    establish a confident approve and found no hard violation either."""

    APPROVE = "approve"
    REJECT = "reject"
    MANUAL_REVIEW = "manual_review"


class VerdictReason(StrEnum):
    """Machine-readable reason attached to a non-approve verdict.

    Reasons are split into two severities by :mod:`receipt_verifier.validate`: hard
    violations that force ``reject``, and uncertainty flags that route to
    ``manual_review``. Also used as the adversarial label taxonomy.
    """

    AMOUNT_MISMATCH = "amount_mismatch"
    DESTINATION_MISMATCH = "destination_mismatch"
    DUPLICATE_OPERATION_ID = "duplicate_operation_id"
    STALE_DATE = "stale_date"
    FUTURE_DATE = "future_date"
    PROMPT_INJECTION = "prompt_injection"
    MISSING_FIELD = "missing_field"
    LOW_CONFIDENCE = "low_confidence"
    UNVERIFIED_PAYMENT = "unverified_payment"
    EXTRACTION_FAILED = "extraction_failed"


class DestinationKind(StrEnum):
    ALIAS = "alias"
    CVU = "cvu"
    CBU = "cbu"


class AdversarialKind(StrEnum):
    """Which adversarial mutation (if any) produced this sample."""

    NONE = "none"
    EDITED_AMOUNT = "edited_amount"
    WRONG_DESTINATION = "wrong_destination"
    DUPLICATE_OPERATION_ID = "duplicate_operation_id"
    INJECTED_INSTRUCTION = "injected_instruction"
    STALE_DATE = "stale_date"


class Destination(BaseModel):
    """Where the money was sent: an alias, a CVU or a CBU plus its holder name."""

    model_config = ConfigDict(frozen=True)

    kind: DestinationKind
    value: str
    holder: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_value(self) -> Self:
        if self.kind is DestinationKind.ALIAS:
            if not is_valid_alias(self.value):
                raise ValueError(f"invalid alias: {self.value!r}")
            return self
        if not is_valid_cbu_or_cvu(self.value):
            raise ValueError(f"invalid {self.kind.value} check digits: {self.value!r}")
        return self

    @property
    def key(self) -> str:
        """Comparison key: aliases are case-insensitive, CBU/CVU are digits."""
        if self.kind is DestinationKind.ALIAS:
            return f"alias:{self.value.lower()}"
        return f"{self.kind.value}:{self.value}"

    def __str__(self) -> str:
        return f"{self.kind.value} {self.value} ({self.holder})"


class LedgerEntry(BaseModel):
    """The payment the verifier expected to receive, from its own records."""

    model_config = ConfigDict(frozen=True)

    payment_id: str = Field(min_length=1)
    amount: Amount
    destination: Destination
    requested_at: datetime


class ReceiptLabel(BaseModel):
    """Ground truth for one receipt, including its expected verdict and reasons."""

    model_config = ConfigDict(frozen=True)

    sample_id: str = Field(min_length=1)
    image: str = Field(description="Image path relative to the dataset root.")
    issuer: Issuer
    amount: Amount = Field(description="Amount displayed as the headline figure.")
    amount_detail: Amount = Field(description="Amount displayed on the receipt detail line.")
    transferred_at: datetime
    sender_name: str = Field(min_length=1)
    sender_bank: str = Field(min_length=1)
    destination: Destination
    operation_id: str = Field(min_length=1)
    memo: str = ""
    expected_decision: Decision
    reasons: tuple[VerdictReason, ...] = ()
    adversarial: AdversarialKind = AdversarialKind.NONE
    expectation: LedgerEntry

    @field_validator("image")
    @classmethod
    def _validate_image_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"image path must be relative and inside the dataset: {value!r}")
        if path.suffix.lower() != ".png":
            raise ValueError(f"image path must point to a .png file: {value!r}")
        return value

    @field_validator("transferred_at")
    @classmethod
    def _validate_transferred_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("transferred_at must be timezone-aware")
        if value.utcoffset() != AR_UTC_OFFSET:
            raise ValueError(
                "transferred_at must use the America/Argentina/Buenos_Aires offset (-03:00)"
            )
        return value

    @field_validator("amount", "amount_detail", mode="before")
    @classmethod
    def _quantize(cls, value: object) -> Decimal:
        if isinstance(value, Decimal):
            return quantize_amount(value)
        if isinstance(value, int | str | float):
            try:
                return quantize_amount(Decimal(str(value)))
            except InvalidOperation as exc:
                raise ValueError(f"invalid amount: {value!r}") from exc
        raise ValueError(f"cannot interpret {type(value).__name__} as an amount")

    @model_validator(mode="after")
    def _validate_verdict(self) -> Self:
        if self.expected_decision is Decision.APPROVE and self.reasons:
            raise ValueError("an approved sample cannot carry reasons")
        if self.expected_decision is not Decision.APPROVE and not self.reasons:
            raise ValueError("a non-approved sample must carry at least one reason")
        if (
            self.adversarial is not AdversarialKind.NONE
            and self.expected_decision is Decision.APPROVE
        ):
            raise ValueError("an adversarial sample cannot be expected to approve")
        return self

    @property
    def critical_fields(self) -> dict[str, object]:
        """Fields the validator requires before it can reach a verdict."""
        return {
            "amount": self.amount,
            "transferred_at": self.transferred_at,
            "sender_name": self.sender_name,
            "destination": self.destination,
            "operation_id": self.operation_id,
            "issuer": self.issuer,
        }


class DatasetManifest(BaseModel):
    """Dataset-level metadata: reproducibility inputs and the evaluation clock."""

    model_config = ConfigDict(frozen=True)

    name: str
    version: str
    seed: int
    generated_at: datetime
    evaluation_at: datetime = Field(
        description=(
            "Reference 'now' for date-window checks. Evaluations default to this clock so "
            "results stay reproducible long after generation."
        )
    )
    max_receipt_age_seconds: int = Field(gt=0)
    sample_count: int = Field(ge=0)
    counts: dict[str, int]
    adversarial_counts: dict[str, int]
