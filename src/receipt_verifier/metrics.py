"""Evaluation metrics.

Definitions are deliberately explicit because "false approval rate" can mean several
things and the difference matters for a money-moving system:

* per-field exact match — the extracted value equals ground truth after normalization
  (money compared at cent precision, timestamps at minute precision, aliases
  case-insensitively, free text casefolded);
* approve precision/recall — ``approve`` is the positive class, so recall answers
  "how many of the receipts we should have approved did we approve?" and precision
  answers "how many of our approvals were correct?";
* false approval rate — share of receipts that should have been rejected but were
  approved (``FP / (FP + TN)``), reported both over the whole dataset and over the
  adversarial subset only;
* coverage — share of receipts where every critical field was extracted with
  confidence at or above the policy threshold (no silent guessing);
* latency and cost — per receipt, measured by the harness around ``extract``.
"""

from __future__ import annotations

import unicodedata
from datetime import datetime
from decimal import Decimal
from statistics import mean

from pydantic import BaseModel, ConfigDict

from receipt_verifier.extraction import CRITICAL_FIELD_NAMES, FIELD_NAMES, ExtractionResult
from receipt_verifier.identifiers import quantize_amount
from receipt_verifier.schema import (
    AR_TZ,
    AdversarialKind,
    Decision,
    Destination,
    Issuer,
    ReceiptLabel,
    VerdictReason,
)
from receipt_verifier.validate import ValidationResult

MetricValue = float | None


class FieldMetric(BaseModel):
    """Exact-match accuracy for one field across the dataset."""

    model_config = ConfigDict(frozen=True)

    field: str
    matches: int
    total: int

    @property
    def accuracy(self) -> MetricValue:
        return self.matches / self.total if self.total else None


class SampleOutcome(BaseModel):
    """Per-sample record: truth, prediction and the field-level comparison."""

    model_config = ConfigDict(frozen=True)

    sample_id: str
    adversarial: AdversarialKind
    extractor_used: str
    expected_decision: Decision
    predicted_decision: Decision
    expected_reasons: tuple[VerdictReason, ...]
    predicted_reasons: tuple[VerdictReason, ...]
    field_matches: dict[str, bool]
    covered: bool
    latency_ms: float
    cost_usd: Decimal
    prompt_tokens: int = 0
    completion_tokens: int = 0
    extractor_error: str = ""

    @property
    def copied_reasons(self) -> tuple[VerdictReason, ...]:
        """Predicted reasons that the ground truth also lists (correct explanations)."""
        return tuple(reason for reason in self.predicted_reasons if reason in self.expected_reasons)


class Metrics(BaseModel):
    """Aggregated harness output."""

    model_config = ConfigDict(frozen=True)

    n: int
    per_field: tuple[FieldMetric, ...]
    extractor_usage: dict[str, int] = {}
    approve_true_positives: int
    approve_false_positives: int
    approve_true_negatives: int
    approve_false_negatives: int
    manual_reviews: int
    false_approvals: int
    adversarial_n: int
    adversarial_false_approvals: int
    coverage: MetricValue
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    total_cost_usd: Decimal
    mean_cost_usd: Decimal
    prompt_tokens: int = 0
    completion_tokens: int = 0
    extractor_errors: int = 0
    """Samples where the extractor failed instead of returning a reading."""

    @property
    def approve_precision(self) -> MetricValue:
        predicted = self.approve_true_positives + self.approve_false_positives
        return self.approve_true_positives / predicted if predicted else None

    @property
    def approve_recall(self) -> MetricValue:
        actual = self.approve_true_positives + self.approve_false_negatives
        return self.approve_true_positives / actual if actual else None

    @property
    def approve_f1(self) -> MetricValue:
        precision, recall = self.approve_precision, self.approve_recall
        if precision is None or recall is None or precision + recall == 0:
            return None
        return 2 * precision * recall / (precision + recall)

    @property
    def false_approval_rate(self) -> MetricValue:
        should_reject = self.approve_false_positives + self.approve_true_negatives
        return self.approve_false_positives / should_reject if should_reject else None

    @property
    def adversarial_false_approval_rate(self) -> MetricValue:
        return self.adversarial_false_approvals / self.adversarial_n if self.adversarial_n else None

    @property
    def manual_review_rate(self) -> MetricValue:
        """Share of receipts a human has to look at."""
        return self.manual_reviews / self.n if self.n else None

    @property
    def approval_rate(self) -> MetricValue:
        """Share of receipts approved without human intervention."""
        return self.approve_true_positives / self.n if self.n else None


def _normalize_datetime(value: datetime) -> datetime:
    return value.astimezone(AR_TZ).replace(second=0, microsecond=0)


def normalize_field_value(name: str, value: object) -> object:
    """Normalize a field value so equality is meaningful for that field type."""
    if isinstance(value, Decimal):
        return quantize_amount(value)
    if isinstance(value, datetime):
        return _normalize_datetime(value)
    if isinstance(value, Destination):
        return value.key
    if isinstance(value, Issuer):
        return value.value
    if isinstance(value, str):
        return _fold_text(value)
    return value


def _fold_text(value: str) -> str:
    """Accent- and case-insensitive comparison form for free text.

    "Gómez" and "Gomez" are the same name; OCR and font rendering disagree about the
    accent, the receipt does not.
    """
    decomposed = unicodedata.normalize("NFKD", value)
    ascii_text = decomposed.encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_text.split()).casefold()


def expected_field_value(label: ReceiptLabel, name: str) -> object:
    """Ground-truth value for a canonical field name."""
    mapping: dict[str, object] = {
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
    return mapping[name]


def _is_absent(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def fields_match(name: str, expected: object, extracted: object) -> bool:
    """True when both sides are equivalent — including "both absent".

    A receipt that prints no memo and an extraction that reports no memo agree; that is
    a correct extraction, not a miss.
    """
    if _is_absent(expected) or _is_absent(extracted):
        return _is_absent(expected) and _is_absent(extracted)
    return normalize_field_value(name, expected) == normalize_field_value(name, extracted)


def is_covered(extraction: ExtractionResult, *, min_confidence: float) -> bool:
    """Every critical field present with at least the required confidence."""
    fields = extraction.field_map()
    return all(
        fields[name].value is not None and fields[name].confidence >= min_confidence
        for name in CRITICAL_FIELD_NAMES
    )


def evaluate_sample(
    label: ReceiptLabel,
    extraction: ExtractionResult,
    validation: ValidationResult,
    *,
    latency_ms: float,
    cost_usd: Decimal,
    min_confidence: float,
) -> SampleOutcome:
    """Build the per-sample comparison record."""
    fields = extraction.field_map()
    matches = {
        name: fields_match(name, expected_field_value(label, name), fields[name].value)
        for name in fields
    }
    return SampleOutcome(
        sample_id=label.sample_id,
        adversarial=label.adversarial,
        extractor_used=extraction.extractor,
        expected_decision=label.expected_decision,
        predicted_decision=validation.decision,
        expected_reasons=label.reasons,
        predicted_reasons=validation.reasons,
        field_matches=matches,
        covered=is_covered(extraction, min_confidence=min_confidence),
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        prompt_tokens=extraction.prompt_tokens,
        completion_tokens=extraction.completion_tokens,
        extractor_error=extraction.error,
    )


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def compute_metrics(outcomes: tuple[SampleOutcome, ...]) -> Metrics:
    """Aggregate per-sample outcomes into the reported metric set."""
    true_positives = sum(
        1
        for o in outcomes
        if o.expected_decision is Decision.APPROVE and o.predicted_decision is Decision.APPROVE
    )
    false_positives = sum(
        1
        for o in outcomes
        if o.expected_decision is not Decision.APPROVE and o.predicted_decision is Decision.APPROVE
    )
    true_negatives = sum(
        1
        for o in outcomes
        if o.expected_decision is not Decision.APPROVE
        and o.predicted_decision is not Decision.APPROVE
    )
    false_negatives = sum(
        1
        for o in outcomes
        if o.expected_decision is Decision.APPROVE and o.predicted_decision is not Decision.APPROVE
    )
    manual_reviews = sum(1 for o in outcomes if o.predicted_decision is Decision.MANUAL_REVIEW)
    adversarial = tuple(o for o in outcomes if o.adversarial is not AdversarialKind.NONE)
    adversarial_false_approvals = sum(
        1 for o in adversarial if o.predicted_decision is Decision.APPROVE
    )
    latencies = [o.latency_ms for o in outcomes]
    costs = [o.cost_usd for o in outcomes]
    per_field = tuple(
        FieldMetric(
            field=name,
            matches=sum(1 for o in outcomes if o.field_matches.get(name, False)),
            total=len(outcomes),
        )
        for name in FIELD_NAMES
    )
    covered = sum(1 for o in outcomes if o.covered)
    usage: dict[str, int] = {}
    for record in outcomes:
        usage[record.extractor_used] = usage.get(record.extractor_used, 0) + 1
    return Metrics(
        n=len(outcomes),
        per_field=per_field,
        extractor_usage=usage,
        approve_true_positives=true_positives,
        approve_false_positives=false_positives,
        approve_true_negatives=true_negatives,
        approve_false_negatives=false_negatives,
        manual_reviews=manual_reviews,
        false_approvals=false_positives,
        adversarial_n=len(adversarial),
        adversarial_false_approvals=adversarial_false_approvals,
        coverage=covered / len(outcomes) if outcomes else None,
        latency_mean_ms=mean(latencies) if latencies else 0.0,
        latency_p50_ms=_percentile(latencies, 0.50),
        latency_p95_ms=_percentile(latencies, 0.95),
        total_cost_usd=sum(costs, Decimal("0")) if costs else Decimal("0"),
        mean_cost_usd=(sum(costs, Decimal("0")) / len(costs)) if costs else Decimal("0"),
        prompt_tokens=sum(o.prompt_tokens for o in outcomes),
        completion_tokens=sum(o.completion_tokens for o in outcomes),
        extractor_errors=sum(1 for o in outcomes if o.extractor_error),
    )


def format_rate(value: MetricValue) -> str:
    """Format a rate for the text table (``n/a`` when undefined)."""
    return "n/a" if value is None else f"{value:.3f}"
