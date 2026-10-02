"""Summary-aggregation tests: the README table must follow from the run reports."""

from __future__ import annotations

from receipt_verifier.eval_summary import (
    NON_READABLE_FIELDS,
    adversarial_summary,
    field_accuracy,
    per_field_accuracy,
    readable_field_accuracy,
    render_markdown,
    summarize_report,
)


def make_report(
    *,
    extractor: str = "llm-test",
    matches: int = 9,
    total: int = 10,
    injected_decisions: tuple[str, ...] = ("reject", "reject"),
    injected_reasons: tuple[str, ...] = ("prompt_injection",),
) -> dict[str, object]:
    outcomes: list[dict[str, object]] = [
        {
            "sample_id": f"inj-{index}",
            "adversarial": "injected_instruction",
            "predicted_decision": decision,
            "predicted_reasons": list(injected_reasons),
        }
        for index, decision in enumerate(injected_decisions)
    ]
    outcomes.append(
        {
            "sample_id": "normal-0",
            "adversarial": "none",
            "predicted_decision": "approve",
            "predicted_reasons": [],
        }
    )
    return {
        "extractor": extractor,
        "dataset_name": "synthetic",
        "dataset_version": "v1",
        "metrics": {
            "n": len(outcomes),
            "per_field": [
                {"field": name, "matches": matches, "total": total} for name in ("amount", "memo")
            ],
            "extractor_usage": {extractor: len(outcomes)},
            "approve_true_positives": 1,
            "approve_false_positives": 0,
            "approve_true_negatives": 2,
            "approve_false_negatives": 0,
            "manual_reviews": 0,
            "false_approvals": 0,
            "adversarial_n": len(injected_decisions),
            "adversarial_false_approvals": 0,
            "coverage": 1.0,
            "latency_mean_ms": 100.0,
            "latency_p50_ms": 90.0,
            "latency_p95_ms": 150.0,
            "total_cost_usd": "0",
            "mean_cost_usd": "0",
            "prompt_tokens": 900,
            "completion_tokens": 100,
            "extractor_errors": 0,
            "destination_retries": 2,
            "retry_latency_total_ms": 2_400.0,
        },
        "outcomes": outcomes,
    }


class TestAggregation:
    def test_field_accuracy_is_the_mean_over_fields(self) -> None:
        metrics = make_report(matches=9, total=10)["metrics"]
        assert field_accuracy(metrics) == 0.9  # type: ignore[arg-type]
        assert per_field_accuracy(metrics) == {"amount": 0.9, "memo": 0.9}  # type: ignore[arg-type]

    def test_readable_accuracy_excludes_the_field_the_image_cannot_carry(self) -> None:
        metrics = {
            "per_field": [
                {"field": "amount", "matches": 10, "total": 10},
                {"field": "issuer", "matches": 4, "total": 10},
            ]
        }
        assert "issuer" in NON_READABLE_FIELDS
        assert field_accuracy(metrics) == 0.7
        assert readable_field_accuracy(metrics) == 1.0

    def test_readable_accuracy_falls_back_to_none_without_fields(self) -> None:
        assert readable_field_accuracy({"per_field": []}) is None

    def test_rate_properties_are_derived_from_counts(self) -> None:
        row = summarize_report(make_report())
        assert row["approve_precision"] == 1.0
        assert row["approve_recall"] == 1.0
        assert row["approve_f1"] == 1.0
        assert row["false_approval_rate"] == 0.0
        assert row["manual_review_rate"] == 0.0
        assert row["adversarial_false_approval_rate"] == 0.0

    def test_tokens_are_totalled(self) -> None:
        row = summarize_report(make_report())
        assert row["prompt_tokens"] == 900
        assert row["completion_tokens"] == 100
        assert row["total_tokens"] == 1000

    def test_retry_counts_and_mean_latency_are_carried(self) -> None:
        row = summarize_report(make_report())
        assert row["destination_retries"] == 2
        assert row["retry_latency_total_ms"] == 2_400.0
        assert row["retry_latency_mean_ms"] == 1_200.0

    def test_no_retries_report_a_zero_mean(self) -> None:
        report = make_report()
        report["metrics"]["destination_retries"] = 0  # type: ignore[index]
        report["metrics"]["retry_latency_total_ms"] = 0.0  # type: ignore[index]
        row = summarize_report(report)
        assert row["retry_latency_mean_ms"] == 0.0

    def test_the_dataset_cell_marks_a_retry_run(self) -> None:
        plain = summarize_report(make_report())
        retried = summarize_report(make_report())
        retried["destination_retry"] = True
        table = render_markdown([plain, retried])
        assert "| synthetic v1 |" in table
        assert "| synthetic v1 + retry |" in table

    def test_a_report_without_the_flag_summarizes_as_plain(self) -> None:
        row = summarize_report(make_report())
        assert row["destination_retry"] is False

    def test_injection_detail_comes_from_the_sample_rows(self) -> None:
        row = summarize_report(make_report())
        injected = row["injected_instructions"]
        assert injected["n"] == 2
        assert injected["approved"] == 0
        assert injected["decisions"] == {"reject": 2}
        assert injected["reasons"] == {"prompt_injection": 2}

    def test_an_approved_injection_is_visible(self) -> None:
        row = summarize_report(make_report(injected_decisions=("reject", "approve")))
        assert row["injected_instructions"]["approved"] == 1

    def test_adversarial_summary_of_a_missing_class_is_empty(self) -> None:
        summary = adversarial_summary([], "injected_instruction")
        assert summary == {"n": 0, "decisions": {}, "approved": 0, "reasons": {}}


class TestMarkdown:
    def test_renders_one_row_per_run(self) -> None:
        rows = [summarize_report(make_report(extractor="llm-a"))]
        rows.append(summarize_report(make_report(extractor="llm-b")))
        table = render_markdown(rows)
        assert "| `llm-a` |" in table
        assert "| `llm-b` |" in table
        assert table.count("\n") == 3  # header + divider + two rows

    def test_the_dataset_is_a_column(self) -> None:
        """The same model measured on two dataset versions must stay distinguishable."""
        report = make_report()
        report["dataset_version"] = "v1"
        v1 = summarize_report(report)
        report["dataset_version"] = "v2"
        v2 = summarize_report(report)
        table = render_markdown([v1, v2])
        assert "| Dataset |" in table
        assert "| synthetic v1 |" in table
        assert "| synthetic v2 |" in table

    def test_tokens_are_shown_per_receipt(self) -> None:
        table = render_markdown([summarize_report(make_report())])
        assert "| 333 |" in table

    def test_table_prefers_the_readable_accuracy(self) -> None:
        row = summarize_report(make_report())
        row["field_accuracy"] = 0.1
        row["field_accuracy_readable"] = 0.9
        assert "| 0.900 |" in render_markdown([row])  # 1000 tokens over 3 samples

    def test_undefined_rates_render_as_not_available(self) -> None:
        report = make_report()
        metrics = report["metrics"]
        assert isinstance(metrics, dict)
        metrics["approve_true_positives"] = 0
        table = render_markdown([summarize_report(report)])
        assert "n/a" in table
