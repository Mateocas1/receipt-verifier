from datetime import datetime, timedelta
from decimal import Decimal

from receipt_verifier.extraction import FIELD_NAMES
from receipt_verifier.identifiers import quantize_amount
from receipt_verifier.metrics import (
    SampleOutcome,
    compute_metrics,
    evaluate_sample,
    expected_field_value,
    fields_match,
    format_rate,
    normalize_field_value,
)
from receipt_verifier.schema import (
    AR_TZ,
    AdversarialKind,
    Decision,
    VerdictReason,
)
from receipt_verifier.validate import ReceiptValidator
from tests.helpers import CVU_DESTINATION, NOW, extraction_from, make_label


def outcome(
    sample_id: str,
    *,
    expected: Decision,
    predicted: Decision,
    adversarial: AdversarialKind = AdversarialKind.NONE,
    covered: bool = True,
    failed_fields: tuple[str, ...] = (),
    latency_ms: float = 10.0,
    cost_usd: Decimal = Decimal("0"),
) -> SampleOutcome:
    return SampleOutcome(
        sample_id=sample_id,
        adversarial=adversarial,
        expected_decision=expected,
        predicted_decision=predicted,
        expected_reasons=((VerdictReason.AMOUNT_MISMATCH,) if expected is Decision.REJECT else ()),
        predicted_reasons=(
            (VerdictReason.AMOUNT_MISMATCH,) if predicted is Decision.REJECT else ()
        ),
        field_matches={name: name not in failed_fields for name in FIELD_NAMES},
        covered=covered,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
    )


TWO_BY_TWO = (
    outcome("a", expected=Decision.APPROVE, predicted=Decision.APPROVE, latency_ms=10.0),
    outcome(
        "b",
        expected=Decision.APPROVE,
        predicted=Decision.REJECT,
        failed_fields=("amount",),
        latency_ms=20.0,
    ),
    outcome(
        "c",
        expected=Decision.REJECT,
        predicted=Decision.APPROVE,
        adversarial=AdversarialKind.EDITED_AMOUNT,
        covered=False,
        latency_ms=30.0,
    ),
    outcome(
        "d",
        expected=Decision.REJECT,
        predicted=Decision.REJECT,
        adversarial=AdversarialKind.DUPLICATE_OPERATION_ID,
        covered=False,
        latency_ms=40.0,
    ),
)


class TestConfusionMetrics:
    def test_confusion_matrix(self) -> None:
        metrics = compute_metrics(TWO_BY_TWO)
        assert metrics.n == 4
        assert (
            metrics.approve_true_positives,
            metrics.approve_false_positives,
            metrics.approve_true_negatives,
            metrics.approve_false_negatives,
        ) == (1, 1, 1, 1)

    def test_precision_recall_f1(self) -> None:
        metrics = compute_metrics(TWO_BY_TWO)
        assert metrics.approve_precision == 0.5
        assert metrics.approve_recall == 0.5
        assert metrics.approve_f1 == 0.5

    def test_false_approval_rate_is_over_should_reject(self) -> None:
        metrics = compute_metrics(TWO_BY_TWO)
        assert metrics.false_approvals == 1
        assert metrics.false_approval_rate == 0.5

    def test_adversarial_subset_is_scored_separately(self) -> None:
        metrics = compute_metrics(TWO_BY_TWO)
        assert metrics.adversarial_n == 2
        assert metrics.adversarial_false_approvals == 1
        assert metrics.adversarial_false_approval_rate == 0.5

    def test_coverage_and_latency(self) -> None:
        metrics = compute_metrics(TWO_BY_TWO)
        assert metrics.coverage == 0.5
        assert metrics.latency_mean_ms == 25.0
        assert metrics.latency_p50_ms == 30.0
        assert metrics.latency_p95_ms == 40.0
        assert metrics.total_cost_usd == Decimal("0")

    def test_cost_is_summed(self) -> None:
        outcomes = (
            outcome(
                "a",
                expected=Decision.APPROVE,
                predicted=Decision.APPROVE,
                cost_usd=Decimal("0.01"),
            ),
            outcome(
                "b",
                expected=Decision.APPROVE,
                predicted=Decision.APPROVE,
                cost_usd=Decimal("0.03"),
            ),
        )
        metrics = compute_metrics(outcomes)
        assert metrics.total_cost_usd == Decimal("0.04")
        assert metrics.mean_cost_usd == Decimal("0.02")

    def test_per_field_accuracy(self) -> None:
        metrics = compute_metrics(TWO_BY_TWO)
        by_field = {metric.field: metric for metric in metrics.per_field}
        assert by_field["amount"].matches == 3
        assert by_field["amount"].total == 4
        assert by_field["amount"].accuracy == 0.75
        assert by_field["memo"].accuracy == 1.0

    def test_empty_input_is_safe(self) -> None:
        metrics = compute_metrics(())
        assert metrics.n == 0
        assert len(metrics.per_field) == len(FIELD_NAMES)
        assert all(metric.total == 0 and metric.accuracy is None for metric in metrics.per_field)
        assert metrics.coverage is None
        assert metrics.approve_precision is None
        assert metrics.approve_recall is None
        assert metrics.approve_f1 is None
        assert metrics.false_approval_rate is None
        assert metrics.adversarial_false_approval_rate is None
        assert format_rate(None) == "n/a"

    def test_manual_review_counts_as_not_approved(self) -> None:
        outcomes = (
            outcome("a", expected=Decision.APPROVE, predicted=Decision.APPROVE),
            outcome("b", expected=Decision.APPROVE, predicted=Decision.MANUAL_REVIEW),
            outcome(
                "c",
                expected=Decision.MANUAL_REVIEW,
                predicted=Decision.MANUAL_REVIEW,
                adversarial=AdversarialKind.DUPLICATE_OPERATION_ID,
            ),
            outcome(
                "d",
                expected=Decision.REJECT,
                predicted=Decision.APPROVE,
                adversarial=AdversarialKind.EDITED_AMOUNT,
            ),
        )
        metrics = compute_metrics(outcomes)
        assert metrics.manual_reviews == 2
        assert metrics.manual_review_rate == 0.5
        assert metrics.approval_rate == 0.25
        assert (metrics.approve_true_positives, metrics.approve_false_negatives) == (1, 1)
        assert (metrics.approve_true_negatives, metrics.approve_false_positives) == (1, 1)
        assert metrics.false_approvals == 1

    def test_no_predictions_means_undefined_precision(self) -> None:
        metrics = compute_metrics(
            (outcome("a", expected=Decision.APPROVE, predicted=Decision.REJECT),)
        )
        assert metrics.approve_precision is None
        assert metrics.approve_recall == 0.0


class TestFieldComparison:
    def test_amount_is_compared_at_cent_precision(self) -> None:
        assert fields_match("amount", Decimal("2500"), Decimal("2500.00"))
        assert not fields_match("amount", Decimal("2500.00"), Decimal("2500.01"))

    def test_datetime_is_compared_at_minute_precision(self) -> None:
        earlier = datetime(2025, 7, 1, 8, 0, 30, tzinfo=AR_TZ)
        later = datetime(2025, 7, 1, 8, 0, 55, tzinfo=AR_TZ)
        assert fields_match("transferred_at", earlier, later)
        assert not fields_match("transferred_at", earlier, earlier + timedelta(minutes=1))

    def test_alias_case_is_ignored(self) -> None:
        from receipt_verifier.schema import Destination, DestinationKind

        upper = Destination(kind=DestinationKind.ALIAS, value="Camila.Gomez.AR", holder="C G")
        lower = Destination(kind=DestinationKind.ALIAS, value="camila.gomez.ar", holder="C G")
        assert fields_match("destination", upper, lower)

    def test_text_is_casefolded_and_whitespace_collapsed(self) -> None:
        assert fields_match("memo", "  Pago   de julio ", "pago de julio")

    def test_missing_value_is_never_a_match(self) -> None:
        assert not fields_match("sender_bank", "Banco del Río", None)
        assert not fields_match("sender_bank", None, "Banco del Río")

    def test_two_absences_are_a_match(self) -> None:
        """No memo printed and no memo extracted agree; that is not a miss."""
        assert fields_match("memo", "", None)
        assert fields_match("memo", None, "")
        assert not fields_match("memo", "", "alquiler")
        assert not fields_match("memo", "alquiler", None)

    def test_text_comparison_folds_accents(self) -> None:
        assert fields_match("sender_name", "Sofía Ibáñez", "Sofia Ibanez")
        assert fields_match("sender_bank", "Banco del Río", "banco del rio")
        assert normalize_field_value("sender_name", "  Gómez  ") == "gomez"
        assert not fields_match("sender_name", "Sofía Ibáñez", "Sofia Ibanezx")

    def test_unknown_types_fall_back_to_equality(self) -> None:
        assert normalize_field_value("weird", 3) == 3

    def test_amount_normalization_quantizes(self) -> None:
        assert normalize_field_value("amount", Decimal("10.005")) == quantize_amount(
            Decimal("10.005")
        )


class TestFixtureIntegration:
    def test_evaluate_sample_records_the_comparison(self) -> None:
        label = make_label(destination=CVU_DESTINATION)
        extraction = extraction_from(label)
        validation = ReceiptValidator({CVU_DESTINATION.key}).validate(
            extraction, label.expectation, now=NOW
        )
        record = evaluate_sample(
            label,
            extraction,
            validation,
            latency_ms=3.5,
            cost_usd=Decimal("0"),
            min_confidence=0.5,
        )
        assert record.sample_id == label.sample_id
        assert record.predicted_decision is Decision.APPROVE
        assert all(record.field_matches.values())
        assert record.covered is True
        assert expected_field_value(label, "operation_id") == label.operation_id
