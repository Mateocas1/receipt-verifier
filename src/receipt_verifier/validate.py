"""Deterministic validator: the only component allowed to approve a receipt.

The validator never trusts the extractor's own confidence as a verdict. It compares
the extracted fields against independent evidence (the expected payment, the
destination allowlist, the operation-id registry) and against the receipt's own
internal consistency, and it refuses to approve when untrusted text carries
instructions.

Every non-approve verdict carries at least one machine-readable reason, and every
reason has a severity:

* ``reject`` — hard evidence that the receipt is invalid or hostile (amount or
destination does not match the ledger, date outside the window, injection text);
* ``review`` — the code could not establish a confident approve (missing or
  low-confidence field, replay of a known operation id, no ledger evidence to compare
  against). A human decides, and the machine never approves on its own.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from receipt_verifier.extraction import CRITICAL_FIELD_NAMES, ExtractionResult
from receipt_verifier.identifiers import quantize_amount
from receipt_verifier.schema import Decision, LedgerEntry, VerdictReason

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


class ReasonSeverity(StrEnum):
    """How a reason maps onto the verdict."""

    REJECT = "reject"
    REVIEW = "review"


HARD_REJECT_REASONS: frozenset[VerdictReason] = frozenset(
    {
        VerdictReason.AMOUNT_MISMATCH,
        VerdictReason.DESTINATION_MISMATCH,
        VerdictReason.IDENTICAL_RECEIPT_REPLAY,
        VerdictReason.STALE_DATE,
        VerdictReason.FUTURE_DATE,
        VerdictReason.PROMPT_INJECTION,
    }
)
"""Reasons that are evidence of an invalid or hostile receipt.

``IDENTICAL_RECEIPT_REPLAY`` is hard because byte-identical arrival of a document already
processed under the same operation id is not a retry with a new image; the byte-identical
original is preserved in the registry, so the identification is exact, not inferred.
"""


def severity_of(reason: VerdictReason) -> ReasonSeverity:
    """Hard violations reject; everything else is routed to a human."""
    return ReasonSeverity.REJECT if reason in HARD_REJECT_REASONS else ReasonSeverity.REVIEW


def decide(reasons: Iterable[VerdictReason]) -> Decision:
    """Verdict from reasons: reject > manual review > approve.

    An empty reason list is the only path to ``approve``; anything the code could not
    settle ends in ``manual_review`` and never in an automatic approval.
    """
    collected = tuple(reasons)
    if any(severity_of(reason) is ReasonSeverity.REJECT for reason in collected):
        return Decision.REJECT
    return Decision.MANUAL_REVIEW if collected else Decision.APPROVE


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
    reasons: tuple[VerdictReason, ...] = ()
    checks: dict[str, bool]

    @property
    def approved(self) -> bool:
        return self.decision is Decision.APPROVE

    @property
    def reject_reasons(self) -> tuple[VerdictReason, ...]:
        return tuple(r for r in self.reasons if severity_of(r) is ReasonSeverity.REJECT)

    @property
    def review_reasons(self) -> tuple[VerdictReason, ...]:
        return tuple(r for r in self.reasons if severity_of(r) is ReasonSeverity.REVIEW)

    @property
    def needs_human(self) -> bool:
        return self.decision is Decision.MANUAL_REVIEW


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
        expectation: LedgerEntry | None,
        *,
        now: datetime,
        seen_operation_ids: Mapping[str, str] | None = None,
        receipt_hash: str | None = None,
    ) -> ValidationResult:
        """Validate one extraction against its expected payment and the running state.

        ``expectation`` may be ``None`` when no ledger evidence is available (for example a
        receipt forwarded without the matching payment). The amount and destination then
        cannot be verified, which is reported as ``unverified_payment`` and routes the
        verdict to ``manual_review``: the service never approves on the receipt's word
        alone.

        ``seen_operation_ids`` maps every operation id the pipeline has already extracted
        to the hash of the first image that carried it, whatever verdict that receipt got.
        Reusing an operation id is never approved: a different image is routed to a human
        (``duplicate_operation_id``) and the identical image is rejected
        (``identical_receipt_replay``) because byte equality is proof, not inference.
        """
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")

        seen = dict(seen_operation_ids or {})
        fields = extraction.field_map()
        checks: dict[str, bool] = {}
        reasons: list[VerdictReason] = []

        def record(name: str, passed: bool, reason: VerdictReason) -> None:
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
        record("required_fields", not missing, VerdictReason.MISSING_FIELD)
        record("field_confidence", not low_confidence, VerdictReason.LOW_CONFIDENCE)

        untrusted = [
            str(fields[name].value)
            for name in ("sender_name", "memo")
            if fields[name].value is not None
        ]
        untrusted.append(extraction.raw_text)
        haystack = normalize_untrusted_text(" \n ".join(untrusted))
        injected = any(pattern.search(haystack) for pattern in self._patterns)
        record("no_prompt_injection", not injected, VerdictReason.PROMPT_INJECTION)

        amount = fields["amount"].value
        amount_detail = fields["amount_detail"].value
        amounts_consistent = (
            amount is None
            or amount_detail is None
            or quantize_amount(amount) == quantize_amount(amount_detail)
        )
        record("amount_self_consistent", amounts_consistent, VerdictReason.AMOUNT_MISMATCH)

        destination = fields["destination"].value
        if destination is None:
            # Same principle as the amount: an unreadable destination is missing, not
            # disallowed, and `missing_field` already sends it to a human.
            checks["destination_allowed"] = False
        else:
            record(
                "destination_allowed",
                destination.key in self._allowed,
                VerdictReason.DESTINATION_MISMATCH,
            )

        if expectation is None:
            checks["ledger_evidence"] = False
            checks["amount_matches_expectation"] = False
            checks["destination_matches_expectation"] = False
            reasons.append(VerdictReason.UNVERIFIED_PAYMENT)
        else:
            checks["ledger_evidence"] = True
            # An unreadable field is not a mismatch: `missing_field` already routes the
            # receipt to a human, and calling it "amount_mismatch" would look like fraud.
            if amount is None:
                checks["amount_matches_expectation"] = False
            else:
                record(
                    "amount_matches_expectation",
                    quantize_amount(amount) == quantize_amount(expectation.amount),
                    VerdictReason.AMOUNT_MISMATCH,
                )
            if destination is None:
                checks["destination_matches_expectation"] = False
            else:
                record(
                    "destination_matches_expectation",
                    destination.key == expectation.destination.key,
                    VerdictReason.DESTINATION_MISMATCH,
                )

        transferred_at = fields["transferred_at"].value
        if transferred_at is None:
            checks["date_window"] = False
        else:
            age = now - transferred_at
            if age > self._policy.max_receipt_age:
                record("date_window", False, VerdictReason.STALE_DATE)
            elif transferred_at - now > self._policy.max_future_skew:
                record("date_window", False, VerdictReason.FUTURE_DATE)
            else:
                checks["date_window"] = True

        operation_id = fields["operation_id"].value
        if operation_id is None:
            # A missing id is reported once, as a missing field: it cannot be a duplicate
            # of anything, and a second reason would only add noise for the reviewer.
            checks["operation_id_unique"] = False
        else:
            first_seen_hash = seen.get(operation_id)
            if first_seen_hash is None:
                checks["operation_id_unique"] = True
            else:
                record("operation_id_unique", False, VerdictReason.DUPLICATE_OPERATION_ID)
                if receipt_hash is not None and receipt_hash == first_seen_hash:
                    record(
                        "receipt_not_a_replay",
                        False,
                        VerdictReason.IDENTICAL_RECEIPT_REPLAY,
                    )

        return ValidationResult(decision=decide(reasons), reasons=tuple(reasons), checks=checks)


def is_approved_amount(value: Decimal, expected: Decimal) -> bool:
    """Helper used by tests and reports: money equality with cent precision."""
    return quantize_amount(value) == quantize_amount(expected)
