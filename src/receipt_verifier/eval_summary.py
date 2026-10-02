"""Turn the harness reports into one small, committed comparison summary.

The live reports under ``reports/`` are large (a row per sample) and gitignored; the
summary is what a reviewer reads and what the README table is rendered from. Keeping the
aggregation here, instead of by hand, means the table cannot drift from the runs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

SUMMARIZABLE_FIELDS: tuple[str, ...] = (
    "n",
    "approve_precision",
    "approve_recall",
    "approve_f1",
    "false_approvals",
    "false_approval_rate",
    "adversarial_n",
    "adversarial_false_approvals",
    "adversarial_false_approval_rate",
    "manual_reviews",
    "manual_review_rate",
    "approval_rate",
    "coverage",
    "latency_mean_ms",
    "latency_p50_ms",
    "latency_p95_ms",
    "prompt_tokens",
    "completion_tokens",
    "extractor_errors",
)
"""Metric keys copied verbatim from the harness report into the summary."""


def _metric(metrics: Mapping[str, Any], key: str) -> Any:
    if key in metrics:
        return metrics[key]
    derived = _DERIVED.get(key)
    if derived is not None:
        return derived(metrics)
    return None


def _rate(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or not denominator:
        return None
    return float(numerator) / float(denominator)


def _sum_of(*values: Any) -> int:
    return sum(int(value or 0) for value in values)


def _precision(metrics: Mapping[str, Any]) -> float | None:
    return _rate(
        metrics.get("approve_true_positives"),
        _sum_of(metrics.get("approve_true_positives"), metrics.get("approve_false_positives")),
    )


def _recall(metrics: Mapping[str, Any]) -> float | None:
    return _rate(
        metrics.get("approve_true_positives"),
        _sum_of(metrics.get("approve_true_positives"), metrics.get("approve_false_negatives")),
    )


def _f1(metrics: Mapping[str, Any]) -> float | None:
    precision, recall = _precision(metrics), _recall(metrics)
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


_DERIVED: dict[str, Any] = {
    "approve_precision": _precision,
    "approve_recall": _recall,
    "approve_f1": _f1,
    "false_approval_rate": lambda m: _rate(
        m.get("false_approvals"),
        _sum_of(m.get("approve_false_positives"), m.get("approve_true_negatives")),
    ),
    "adversarial_false_approval_rate": lambda m: _rate(
        m.get("adversarial_false_approvals"), m.get("adversarial_n")
    ),
    "manual_review_rate": lambda m: _rate(m.get("manual_reviews"), m.get("n")),
    "approval_rate": lambda m: _rate(m.get("approve_true_positives"), m.get("n")),
}
"""Values the harness exposes as computed properties, so the JSON only holds inputs."""


def field_accuracy(metrics: Mapping[str, Any]) -> float | None:
    """Mean per-field exact match across the nine canonical fields."""
    rows = metrics.get("per_field") or []
    if not rows:
        return None
    return sum(float(row["matches"]) / float(row["total"]) for row in rows) / len(rows)


NON_READABLE_FIELDS: tuple[str, ...] = ("issuer",)
"""Fields the synthetic images cannot carry.

The renderer prints invented placeholder names ("Billetera A", "Banco Digital C") instead
of the issuer code, so no model can read ``issuer`` off the image; a high score there is an
inference from correlated layout and vocabulary, not evidence about real receipts. The
headline accuracy therefore excludes it, and the per-field table still reports it.
"""


def readable_field_accuracy(metrics: Mapping[str, Any]) -> float | None:
    """Mean per-field exact match over the fields the image can actually carry."""
    per_field = per_field_accuracy(metrics)
    readable = [
        value
        for name, value in per_field.items()
        if name not in NON_READABLE_FIELDS and value is not None
    ]
    if not readable:
        return None
    return sum(readable) / len(readable)


def per_field_accuracy(metrics: Mapping[str, Any]) -> dict[str, float | None]:
    rows = metrics.get("per_field") or []
    return {
        row["field"]: (float(row["matches"]) / float(row["total"]) if row["total"] else None)
        for row in rows
    }


def adversarial_summary(outcomes: Sequence[Mapping[str, Any]], kind: str) -> dict[str, Any]:
    """Decisions and predicted reasons for one adversarial class, from the sample rows."""
    rows = [row for row in outcomes if row.get("adversarial") == kind]
    decisions: dict[str, int] = {}
    reasons: dict[str, int] = {}
    for row in rows:
        decision = str(row.get("predicted_decision"))
        decisions[decision] = decisions.get(decision, 0) + 1
        for reason in row.get("predicted_reasons", ()):
            reasons[str(reason)] = reasons.get(str(reason), 0) + 1
    return {
        "n": len(rows),
        "decisions": decisions,
        "approved": decisions.get("approve", 0),
        "reasons": reasons,
    }


def summarize_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """One summary row per harness report, plus the adversarial detail it carries."""
    metrics: Mapping[str, Any] = report.get("metrics") or {}
    outcomes: Sequence[Mapping[str, Any]] = report.get("outcomes") or ()
    prompt_tokens = int(metrics.get("prompt_tokens") or 0)
    completion_tokens = int(metrics.get("completion_tokens") or 0)
    row: dict[str, Any] = {
        "extractor": report.get("extractor"),
        "dataset": f"{report.get('dataset_name')} {report.get('dataset_version')}",
        "field_accuracy": field_accuracy(metrics),
        "field_accuracy_readable": readable_field_accuracy(metrics),
        "per_field": per_field_accuracy(metrics),
        "total_tokens": prompt_tokens + completion_tokens,
        "total_cost_usd": str(metrics.get("total_cost_usd", "0")),
    }
    for key in SUMMARIZABLE_FIELDS:
        row[key] = _metric(metrics, key)
    row["extractor_usage"] = metrics.get("extractor_usage") or {}
    row["injected_instructions"] = adversarial_summary(outcomes, "injected_instruction")
    row["adversarial_by_kind"] = {
        kind: adversarial_summary(outcomes, kind)
        for kind in (
            "edited_amount",
            "wrong_destination",
            "duplicate_operation_id",
            "injected_instruction",
            "stale_date",
        )
    }
    return row


def _rate_cell(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _ms_cell(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.0f} ms"


def render_markdown(runs: Sequence[Mapping[str, Any]]) -> str:
    """The README comparison table, rendered from the summary rows."""
    header = (
        "| Extractor | Dataset | Field accuracy | approve precision | approve recall | "
        "False approvals (adversarial) | Manual review | Latency p50 / p95 | Tokens/receipt |"
    )
    divider = "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    lines = [header, divider]
    for run in runs:
        tokens = run.get("total_tokens") or 0
        n = run.get("n") or 0
        per_receipt = f"{tokens / n:.0f}" if n else "n/a"
        lines.append(
            "| `{extractor}` | {dataset} | {field} | {precision} | {recall} | {false_approvals} "
            "({adv}/{adv_n}) | {manual} | {p50} / {p95} | {tokens} |".format(
                extractor=run.get("extractor", "?"),
                dataset=run.get("dataset", "?"),
                field=_rate_cell(run.get("field_accuracy_readable") or run.get("field_accuracy")),
                precision=_rate_cell(run.get("approve_precision")),
                recall=_rate_cell(run.get("approve_recall")),
                false_approvals=run.get("false_approvals", "?"),
                adv=run.get("adversarial_false_approvals", "?"),
                adv_n=run.get("adversarial_n", "?"),
                manual=_rate_cell(run.get("manual_review_rate")),
                p50=_ms_cell(run.get("latency_p50_ms")),
                p95=_ms_cell(run.get("latency_p95_ms")),
                tokens=per_receipt,
            )
        )
    return "\n".join(lines)


__all__ = [
    "NON_READABLE_FIELDS",
    "SUMMARIZABLE_FIELDS",
    "adversarial_summary",
    "field_accuracy",
    "per_field_accuracy",
    "readable_field_accuracy",
    "render_markdown",
    "summarize_report",
]
