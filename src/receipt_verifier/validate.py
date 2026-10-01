"""Deterministic validator: the only component allowed to approve a receipt.

The validator never trusts the extractor's own confidence as a verdict. It compares
the extracted fields against independent evidence (the expected payment, the
destination allowlist, the operation-id registry) and against the receipt's own
internal consistency, and it refuses to approve when untrusted text carries
instructions. A rejection always carries at least one machine-readable reason.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from datetime import datetime, timedelta
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from receipt_verifier.extraction import CRITICAL_FIELD_NAMES, ExtractionResult
from receipt_verifier.identifiers import quantize_amount
from receipt_verifier.schema import Decision, LedgerEntry, RejectReason

DEFAULT_INJECTION_PATTERNS: tuple[str, ...] = (
    r"ignora las instrucciones",
    r"ignorar las instrucciones",
    r"olvida las instrucciones",
    r"omite las instrucciones",
    r"ignore (all )?(previous|prior|above) instructions",
    r"disregard (the )?(previous|prior|above) instructions",
    r"aprueba (este|el) (pago|comprobante|transferencia)",
    r"aprueba el pago",
    r"aproba (este|el) (pago|comprobante|transferencia)",
    r"marca como aprobado",
    r"no verifiques",
    r"sin verificar",
    r"salta (las )?validaciones",
    r"bypass (the )?(verification|validation)",
    r"you are now",
    r"actua como",
    r"system\s*:",
    r"assistant\s*:",
)
"""Instruction-like phrases that make untrusted receipt text unsafe to act on."""


class ValidationPolicy(BaseModel):
    """Tunable thresholds. Defaults are the ones the synthetic dataset is judged with."""

    model_config = ConfigDict(frozen=True)

    max_receipt_age: timedelta = Field(default=timedelta(days=7))
    max_future_skew: timedelta = Field(default=timedelta(minutes=10))
    min_field_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    injection_patterns: tuple[str, ...] = DEFAULT_INJECTION_PATTERNS


class ValidationResult(BaseModel):
    """Verdict plus the reasons and the individual check outcomes behind it."""

    model_config = ConfigDict(frozen=True)

    decision: Decision
    reasons: tuple[RejectReason, ...] = ()
    checks: dict[str, bool]

    @property
    def approved(self) -> bool:
        return self.decision is Decision.APPROVE


def normalize_untrusted_text(text: str) -> str:
    """Lowercase, strip accents and collapse whitespace before pattern matching."""
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_text = decomposed.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", ascii_text).strip().lower()


class ReceiptValidator:
    """Applies every deterministic check and returns a single approve/reject verdict."""

    def __init__(
        self,
        allowed_destinations: Iterable[str],
        policy: ValidationPolicy | None = None,
    ) -> None:
        self._allowed = frozenset(allowed_destinations)
        self._policy = policy or ValidationPolicy()
        self._patterns = tuple(re.compile(pattern) for pattern in self._policy.injection_patterns)

    @property
    def policy(self) -> ValidationPolicy:
        return self._policy

    @property
    def allowed_destinations(self) -> frozenset[str]:
        return self._allowed

    def validate(
        self,
        extraction: ExtractionResult,
        expectation: LedgerEntry,
        *,
        now: datetime,
        seen_operation_ids: Iterable[str] = (),
    ) -> ValidationResult:
        """Validate one extraction against its expected payment and the running state."""
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")

        seen = frozenset(seen_operation_ids)
        fields = extraction.field_map()
        checks: dict[str, bool] = {}
        reasons: list[RejectReason] = []

        def record(name: str, passed: bool, reason: RejectReason) -> None:
            checks[name] = passed
            if not passed and reason not in reasons:
                reasons.append(reason)

        missing = [
            name
            for name in CRITICAL_FIELD_NAMES
            if fields[name].value is None
            or fields[name].confidence < self._policy.min_field_confidence
        ]
        low_confidence = [
            name
            for name in CRITICAL_FIELD_NAMES
            if fields[name].value is not None
            and fields[name].confidence < self._policy.min_field_confidence
        ]
        record("required_fields", not missing, RejectReason.MISSING_FIELD)
        record("field_confidence", not low_confidence, RejectReason.LOW_CONFIDENCE)

        untrusted = [
            str(fields[name].value)
            for name in ("sender_name", "memo")
            if fields[name].value is not None
        ]
        untrusted.append(extraction.raw_text)
        haystack = normalize_untrusted_text(" \n ".join(untrusted))
        injected = any(pattern.search(haystack) for pattern in self._patterns)
        record("no_prompt_injection", not injected, RejectReason.PROMPT_INJECTION)

        amount = fields["amount"].value
        amount_detail = fields["amount_detail"].value
        amounts_consistent = (
            amount is None
            or amount_detail is None
            or quantize_amount(amount) == quantize_amount(amount_detail)
        )
        record("amount_self_consistent", amounts_consistent, RejectReason.AMOUNT_MISMATCH)

        amount_matches = amount is not None and quantize_amount(amount) == quantize_amount(
            expectation.amount
        )
        record("amount_matches_expectation", amount_matches, RejectReason.AMOUNT_MISMATCH)

        destination = fields["destination"].value
        destination_allowed = destination is not None and destination.key in self._allowed
        record("destination_allowed", destination_allowed, RejectReason.DESTINATION_MISMATCH)
        destination_matches = (
            destination is not None and destination.key == expectation.destination.key
        )
        record(
            "destination_matches_expectation",
            destination_matches,
            RejectReason.DESTINATION_MISMATCH,
        )

        transferred_at = fields["transferred_at"].value
        if transferred_at is None:
            checks["date_window"] = False
        else:
            age = now - transferred_at
            if age > self._policy.max_receipt_age:
                record("date_window", False, RejectReason.STALE_DATE)
            elif transferred_at - now > self._policy.max_future_skew:
                record("date_window", False, RejectReason.FUTURE_DATE)
            else:
                checks["date_window"] = True

        operation_id = fields["operation_id"].value
        unique = operation_id is not None and operation_id not in seen
        record("operation_id_unique", unique, RejectReason.DUPLICATE_OPERATION_ID)

        decision = Decision.REJECT if reasons else Decision.APPROVE
        return ValidationResult(decision=decision, reasons=tuple(reasons), checks=checks)


def is_approved_amount(value: Decimal, expected: Decimal) -> bool:
    """Helper used by tests and reports: money equality with cent precision."""
    return quantize_amount(value) == quantize_amount(expected)
