"""End-to-end gate: the dummy extractor plus the validator must never approve an
adversarial receipt, and the harness must report that honestly."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

from receipt_verifier.dataset import Dataset, load_dataset, write_dataset
from receipt_verifier.extractors.dummy import DummyExtractor
from receipt_verifier.harness import EvaluationReport, run_evaluation
from receipt_verifier.schema import AR_TZ, AdversarialKind, Decision

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_DATASET = REPO_ROOT / "dataset" / "synthetic" / "v1"


@pytest.fixture(scope="module")
def small_dataset(tmp_path_factory: pytest.TempPathFactory) -> Dataset:
    from receipt_verifier.synthetic.generate import build_dataset

    root = tmp_path_factory.mktemp("synthetic-v1-mini")
    write_dataset(build_dataset(seed=1234, normal_per_issuer=2), root)
    return load_dataset(root)


@pytest.fixture(scope="module")
def clean_report(small_dataset: Dataset) -> EvaluationReport:
    extractor = DummyExtractor.from_dataset(small_dataset.root)
    return run_evaluation(small_dataset, extractor)


class TestAdversarialGate:
    def test_zero_false_approvals(self, clean_report: EvaluationReport) -> None:
        assert clean_report.metrics.false_approvals == 0
        assert clean_report.metrics.adversarial_false_approvals == 0
        assert clean_report.metrics.false_approval_rate == 0.0

    def test_no_adversarial_sample_is_ever_approved(self, clean_report: EvaluationReport) -> None:
        adversarial = [
            outcome
            for outcome in clean_report.outcomes
            if outcome.adversarial is not AdversarialKind.NONE
        ]
        assert adversarial
        approved = [
            outcome.sample_id
            for outcome in adversarial
            if outcome.predicted_decision is Decision.APPROVE
        ]
        assert approved == []

    def test_adversarial_kinds_route_to_their_expected_verdict(
        self, clean_report: EvaluationReport
    ) -> None:
        by_kind: dict[AdversarialKind, set[Decision]] = {}
        for outcome in clean_report.outcomes:
            if outcome.adversarial is not AdversarialKind.NONE:
                by_kind.setdefault(outcome.adversarial, set()).add(outcome.predicted_decision)
        assert by_kind[AdversarialKind.DUPLICATE_OPERATION_ID] == {Decision.MANUAL_REVIEW}
        for kind, decisions in by_kind.items():
            if kind is not AdversarialKind.DUPLICATE_OPERATION_ID:
                assert decisions == {Decision.REJECT}, kind

    def test_manual_review_count_is_reported(self, clean_report: EvaluationReport) -> None:
        assert clean_report.metrics.manual_reviews == 6
        assert clean_report.metrics.manual_review_rate == pytest.approx(6 / 42)

    def test_predicted_reasons_match_the_ground_truth_reason(
        self, clean_report: EvaluationReport
    ) -> None:
        for outcome in clean_report.outcomes:
            if outcome.adversarial is AdversarialKind.NONE:
                assert outcome.predicted_reasons == ()
            else:
                assert outcome.predicted_reasons == outcome.expected_reasons, outcome.sample_id

    def test_perfect_scores_at_zero_noise(self, clean_report: EvaluationReport) -> None:
        metrics = clean_report.metrics
        assert metrics.approve_precision == 1.0
        assert metrics.approve_recall == 1.0
        assert metrics.coverage == 1.0
        assert metrics.total_cost_usd == 0
        assert all(metric.accuracy == 1.0 for metric in metrics.per_field)

    def test_no_false_negatives_either(self, clean_report: EvaluationReport) -> None:
        assert clean_report.metrics.approve_false_negatives == 0


@pytest.fixture(scope="module")
def noisy_report(small_dataset: Dataset) -> EvaluationReport:
    extractor = DummyExtractor.from_dataset(small_dataset.root, noise=0.2, seed=99)
    return run_evaluation(small_dataset, extractor, noise=0.2)


class TestNoisyRunIsHonest:
    def test_coverage_drops(self, noisy_report: EvaluationReport) -> None:
        assert noisy_report.metrics.coverage is not None
        assert noisy_report.metrics.coverage < 1.0

    def test_reported_false_approvals_match_the_outcomes(
        self, noisy_report: EvaluationReport
    ) -> None:
        recounted = sum(
            1
            for outcome in noisy_report.outcomes
            if outcome.expected_decision is not Decision.APPROVE
            and outcome.predicted_decision is Decision.APPROVE
        )
        adversarial_recounted = sum(
            1
            for outcome in noisy_report.outcomes
            if outcome.adversarial is not AdversarialKind.NONE
            and outcome.predicted_decision is Decision.APPROVE
        )
        assert noisy_report.metrics.false_approvals == recounted
        assert noisy_report.metrics.adversarial_false_approvals == adversarial_recounted

    def test_latency_and_cost_are_reported(self, noisy_report: EvaluationReport) -> None:
        assert noisy_report.metrics.latency_mean_ms >= 0.0
        assert noisy_report.metrics.latency_p95_ms >= noisy_report.metrics.latency_p50_ms


@pytest.fixture(scope="module")
def committed_report() -> EvaluationReport:
    dataset = load_dataset(COMMITTED_DATASET)
    extractor = DummyExtractor.from_dataset(COMMITTED_DATASET)
    return run_evaluation(dataset, extractor)


@pytest.mark.skipif(not COMMITTED_DATASET.exists(), reason="committed dataset not present")
class TestCommittedDatasetGate:
    def test_zero_false_approvals_on_the_versioned_dataset(
        self, committed_report: EvaluationReport
    ) -> None:
        assert committed_report.metrics.false_approvals == 0
        assert committed_report.metrics.adversarial_false_approvals == 0

    def test_dataset_shape(self, committed_report: EvaluationReport) -> None:
        assert committed_report.metrics.n == 150
        assert committed_report.metrics.adversarial_n == 30
        assert committed_report.metrics.manual_reviews == 6
        assert committed_report.metrics.approve_precision == 1.0
        assert committed_report.metrics.approve_recall == 1.0

    def test_evaluation_clock_comes_from_the_manifest(
        self, committed_report: EvaluationReport
    ) -> None:
        dataset = load_dataset(COMMITTED_DATASET)
        assert committed_report.evaluation_at == dataset.manifest.evaluation_at


class TestManifestlessFolder:
    """Hand-curated (real, anonymized) folders ship no manifest: the clock is now."""

    def test_manifestless_folder_evaluates(self, tmp_path: Path) -> None:
        from receipt_verifier.synthetic.generate import build_dataset

        write_dataset(build_dataset(seed=5, normal_per_issuer=1), tmp_path)
        (tmp_path / "manifest.json").unlink()
        dataset = load_dataset(tmp_path, now=datetime(2025, 7, 1, 9, 0, tzinfo=AR_TZ))
        extractor = DummyExtractor.from_dataset(tmp_path)
        report = run_evaluation(dataset, extractor)
        assert report.metrics.n == dataset.manifest.sample_count
        assert report.evaluation_at == dataset.manifest.evaluation_at
        assert report.metrics.false_approvals == 0


@pytest.mark.skipif(not COMMITTED_DATASET.exists(), reason="committed dataset not present")
class TestEvaluateCli:
    def test_cli_exits_zero_and_reports_zero_false_approvals(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/evaluate.py",
                "--dataset",
                str(COMMITTED_DATASET),
                "--no-json",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert "false approvals (count)         0" in completed.stdout
        assert "adversarial false approvals     0/30" in completed.stdout

    def test_cli_writes_a_json_report(self, tmp_path: Path) -> None:
        target = tmp_path / "reports" / "eval-dummy.json"
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/evaluate.py",
                "--dataset",
                str(COMMITTED_DATASET),
                "--json",
                str(target),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        payload = target.read_text(encoding="utf-8")
        assert '"extractor": "dummy"' in payload
        assert '"false_approvals": 0' in payload
