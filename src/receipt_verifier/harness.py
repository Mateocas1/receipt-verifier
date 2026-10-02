"""Evaluation harness: run an extractor over a dataset and aggregate the metrics.

The harness owns the clock (latency), the operation-id registry and the destination
allowlist, so extractors stay pure functions of the image bytes.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from time import perf_counter

from pydantic import BaseModel, ConfigDict

from receipt_verifier.dataset import Dataset
from receipt_verifier.extraction import ReceiptExtractor
from receipt_verifier.metrics import (
    Metrics,
    SampleOutcome,
    compute_metrics,
    evaluate_sample,
    format_rate,
)
from receipt_verifier.replay import receipt_hash
from receipt_verifier.validate import ReceiptValidator, ValidationPolicy


class EvaluationReport(BaseModel):
    """Full harness output: configuration, aggregate metrics and per-sample rows."""

    model_config = ConfigDict(frozen=True)

    extractor: str
    dataset_name: str
    dataset_version: str
    dataset_root: str
    dataset_seed: int
    evaluation_at: datetime
    noise: float = 0.0
    metrics: Metrics
    outcomes: tuple[SampleOutcome, ...]


def run_evaluation(
    dataset: Dataset,
    extractor: ReceiptExtractor,
    *,
    now: datetime | None = None,
    policy: ValidationPolicy | None = None,
    noise: float = 0.0,
    on_sample: Callable[[int, int], None] | None = None,
) -> EvaluationReport:
    """Extract, validate and score every label of a dataset, in dataset order.

    ``on_sample`` receives ``(done, total)`` after each sample; a live run against a real
    provider takes long enough that silence is indistinguishable from a hang.
    """
    active_policy = policy or ValidationPolicy()
    validator = ReceiptValidator(dataset.allowed_destinations, active_policy)
    evaluation_at = now or dataset.evaluation_at
    seen_operation_ids: dict[str, str] = {}
    outcomes: list[SampleOutcome] = []
    total = len(dataset.labels)

    for index, label in enumerate(dataset.labels, start=1):
        image = dataset.image_bytes(label)
        image_hash = receipt_hash(image)
        started = perf_counter()
        extraction = extractor.extract(image)
        latency_ms = (perf_counter() - started) * 1000.0
        validation = validator.validate(
            extraction,
            label.expectation,
            now=evaluation_at,
            seen_operation_ids=seen_operation_ids,
            receipt_hash=image_hash,
        )
        extracted_operation_id = extraction.operation_id.value
        if extracted_operation_id is not None:
            # Every receipt the pipeline has seen is remembered, whatever the verdict: a
            # receipt routed to a human is still one a human will act on, so a replay of
            # it must not look fresh.
            seen_operation_ids.setdefault(extracted_operation_id, image_hash)
        outcomes.append(
            evaluate_sample(
                label,
                extraction,
                validation,
                latency_ms=latency_ms,
                cost_usd=extraction.cost_usd,
                min_confidence=active_policy.min_field_confidence,
            )
        )
        if on_sample is not None:
            on_sample(index, total)

    frozen = tuple(outcomes)
    return EvaluationReport(
        extractor=extractor.name,
        dataset_name=dataset.manifest.name,
        dataset_version=dataset.manifest.version,
        dataset_root=str(dataset.root),
        dataset_seed=dataset.manifest.seed,
        evaluation_at=evaluation_at,
        noise=noise,
        metrics=compute_metrics(frozen),
        outcomes=frozen,
    )


def render_table(report: EvaluationReport) -> str:
    """Human-readable plain-text report (no third-party table dependency)."""
    metrics = report.metrics
    lines: list[str] = []
    lines.append(f"dataset       {report.dataset_name} {report.dataset_version}")
    lines.append(f"dataset root  {report.dataset_root}")
    lines.append(f"seed          {report.dataset_seed}")
    lines.append(f"evaluation_at {report.evaluation_at.isoformat()}")
    lines.append(f"extractor     {report.extractor} (noise={report.noise:.2f})")
    lines.append(f"samples       {metrics.n}")
    lines.append("")
    lines.append("per-field exact match")
    lines.append(f"  {'field':<16}{'accuracy':>10}{'matches':>12}")
    for field in metrics.per_field:
        accuracy = format_rate(field.accuracy)
        lines.append(f"  {field.field:<16}{accuracy:>10}{f'{field.matches}/{field.total}':>12}")
    lines.append("")
    lines.append("decisions (approve = positive class)")
    lines.append(
        f"  true positives {metrics.approve_true_positives:<4}"
        f"false positives {metrics.approve_false_positives:<4}"
        f"true negatives {metrics.approve_true_negatives:<4}"
        f"false negatives {metrics.approve_false_negatives:<4}"
    )
    rows: list[tuple[str, str]] = [
        ("approve precision", format_rate(metrics.approve_precision)),
        ("approve recall", format_rate(metrics.approve_recall)),
        ("approve f1", format_rate(metrics.approve_f1)),
        ("false approvals (count)", str(metrics.false_approvals)),
        ("false approval rate", format_rate(metrics.false_approval_rate)),
        ("manual reviews (count)", str(metrics.manual_reviews)),
        ("manual review rate", format_rate(metrics.manual_review_rate)),
        ("approval rate", format_rate(metrics.approval_rate)),
        (
            "adversarial false approvals",
            f"{metrics.adversarial_false_approvals}/{metrics.adversarial_n}",
        ),
        (
            "adversarial false approval rate",
            format_rate(metrics.adversarial_false_approval_rate),
        ),
        ("coverage", format_rate(metrics.coverage)),
        (
            "extractor usage",
            ", ".join(f"{name}={count}" for name, count in sorted(metrics.extractor_usage.items()))
            or "n/a",
        ),
        ("latency mean ms", f"{metrics.latency_mean_ms:.2f}"),
        ("latency p50 ms", f"{metrics.latency_p50_ms:.2f}"),
        ("latency p95 ms", f"{metrics.latency_p95_ms:.2f}"),
        ("total cost usd", f"{metrics.total_cost_usd:.6f}"),
        ("mean cost usd", f"{metrics.mean_cost_usd:.6f}"),
        ("prompt tokens", str(metrics.prompt_tokens)),
        ("completion tokens", str(metrics.completion_tokens)),
        ("total tokens", str(metrics.prompt_tokens + metrics.completion_tokens)),
        ("extractor errors", str(metrics.extractor_errors)),
    ]
    lines.append("")
    for name, value in rows:
        lines.append(f"  {name:<32}{value}")
    return "\n".join(lines)


def write_report(report: EvaluationReport, path: Path) -> None:
    """Write the report as JSON, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")


def default_report_path(
    extractor: str, noise: float = 0.0, *, reports_dir: Path = Path("reports")
) -> Path:
    """Conventional report location for an extractor run (relative to the cwd)."""
    suffix = f"-noise{noise:.2f}" if noise else ""
    return reports_dir / f"eval-{extractor}{suffix}.json"


__all__ = [
    "EvaluationReport",
    "default_report_path",
    "render_table",
    "run_evaluation",
    "write_report",
]
