"""Replay guardrail tests: every seen operation id is remembered, whatever the verdict.

The live sweep found the hole this file closes: an operation id was recorded only when the
receipt was *approved*, so a replay of a receipt that had been routed to `manual_review`
looked fresh and was approved (`glm5.3-flash`, `gemma4`).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from receipt_verifier.dataset import Dataset, DatasetBundle, load_dataset, write_dataset
from receipt_verifier.extraction import ExtractionResult
from receipt_verifier.harness import run_evaluation
from receipt_verifier.replay import SeenOperationIds, receipt_hash
from receipt_verifier.schema import (
    AR_TZ,
    DatasetManifest,
    Decision,
    VerdictReason,
)
from receipt_verifier.validate import ReceiptValidator
from tests.helpers import ALIAS_DESTINATION, NOW, extraction_from, make_label

IMAGE_A = b"\x89PNG\r\n\x1a\n" + b"receipt-a" * 20
IMAGE_B = b"\x89PNG\r\n\x1a\n" + b"receipt-b" * 20
HASH_A = receipt_hash(IMAGE_A)
HASH_B = receipt_hash(IMAGE_B)


class TestReceiptHash:
    def test_is_a_sha256_hex_digest(self) -> None:
        assert len(HASH_A) == 64
        assert set(HASH_A) <= set("0123456789abcdef")

    def test_same_bytes_hash_the_same(self) -> None:
        assert receipt_hash(IMAGE_A) == HASH_A

    def test_different_bytes_hash_differently(self) -> None:
        assert HASH_A != HASH_B


class TestSeenOperationIds:
    def test_records_the_hash_of_the_first_receipt(self) -> None:
        seen = SeenOperationIds()
        seen.record("MP-1", HASH_A)
        assert "MP-1" in seen
        assert seen.hash_for("MP-1") == HASH_A
        assert len(seen) == 1

    def test_a_later_receipt_does_not_replace_the_first_hash(self) -> None:
        """The reference hash must stay the first receipt's, or the identity check drifts."""
        seen = SeenOperationIds()
        seen.record("MP-1", HASH_A)
        seen.record("MP-1", HASH_B)
        assert seen.hash_for("MP-1") == HASH_A
        assert len(seen) == 1

    def test_unknown_id_has_no_hash(self) -> None:
        assert SeenOperationIds().hash_for("MP-1") is None

    def test_snapshot_is_an_independent_copy(self) -> None:
        seen = SeenOperationIds()
        seen.record("MP-1", HASH_A)
        snapshot = seen.snapshot()
        seen.record("MP-2", HASH_B)
        assert snapshot == {"MP-1": HASH_A}

    def test_bounded_registry_evicts_the_oldest(self) -> None:
        seen = SeenOperationIds(max_size=2)
        seen.record("MP-1", HASH_A)
        seen.record("MP-2", HASH_A)
        seen.record("MP-3", HASH_B)
        assert len(seen) == 2
        assert "MP-1" not in seen
        assert "MP-3" in seen

    def test_max_size_must_be_positive(self) -> None:
        try:
            SeenOperationIds(max_size=0)
        except ValueError as exc:
            assert "max_size" in str(exc)
        else:  # pragma: no cover - the constructor must reject zero
            raise AssertionError("max_size=0 must raise")


class TestValidatorReplay:
    """The validator is the only component that decides; these are its replay rules."""

    def validator(self) -> ReceiptValidator:
        return ReceiptValidator({ALIAS_DESTINATION.key})

    def test_seen_id_with_a_different_image_is_manual_review(self) -> None:
        label = make_label(operation_id="MP-DUPLICATED")
        result = self.validator().validate(
            extraction_from(label),
            label.expectation,
            now=NOW,
            seen_operation_ids={"MP-DUPLICATED": HASH_A},
            receipt_hash=HASH_B,
        )
        assert result.decision is Decision.MANUAL_REVIEW
        assert result.reasons == (VerdictReason.DUPLICATE_OPERATION_ID,)

    def test_seen_id_with_the_same_image_is_rejected(self) -> None:
        label = make_label(operation_id="MP-DUPLICATED")
        result = self.validator().validate(
            extraction_from(label),
            label.expectation,
            now=NOW,
            seen_operation_ids={"MP-DUPLICATED": HASH_A},
            receipt_hash=HASH_A,
        )
        assert result.decision is Decision.REJECT
        assert VerdictReason.DUPLICATE_OPERATION_ID in result.reasons
        assert VerdictReason.IDENTICAL_RECEIPT_REPLAY in result.reasons

    def test_without_a_receipt_hash_a_seen_id_still_never_approves(self) -> None:
        label = make_label(operation_id="MP-DUPLICATED")
        result = self.validator().validate(
            extraction_from(label),
            label.expectation,
            now=NOW,
            seen_operation_ids={"MP-DUPLICATED": HASH_A},
            receipt_hash=None,
        )
        assert result.decision is Decision.MANUAL_REVIEW
        assert result.reasons == (VerdictReason.DUPLICATE_OPERATION_ID,)

    def test_an_unseen_id_is_still_approved(self) -> None:
        label = make_label(operation_id="MP-FRESH")
        result = self.validator().validate(
            extraction_from(label),
            label.expectation,
            now=NOW,
            seen_operation_ids={"MP-OTHER": HASH_A},
            receipt_hash=HASH_B,
        )
        assert result.decision is Decision.APPROVE


class ScriptedExtractor:
    """Returns queued readings in order; used to reproduce the reviewed-then-replayed corner."""

    def __init__(self, results: tuple[ExtractionResult, ...]) -> None:
        self._results = list(results)
        self.calls = 0

    @property
    def name(self) -> str:
        return "scripted"

    def extract(self, image: bytes) -> ExtractionResult:
        result = self._results[self.calls]
        self.calls += 1
        return result


def two_sample_dataset(tmp_path: Path) -> Dataset:
    """An original receipt and a replay sharing one operation id."""
    original = make_label(sample_id="original", operation_id="MP-SHARED")
    replay = make_label(
        sample_id="replay",
        operation_id="MP-SHARED",
        expected_decision=Decision.MANUAL_REVIEW,
        reasons=(VerdictReason.DUPLICATE_OPERATION_ID,),
    )
    labels = (original, replay)
    manifest = DatasetManifest(
        name="replay-fixture",
        version="test",
        seed=1,
        generated_at=datetime(2025, 7, 1, 9, 0, tzinfo=AR_TZ),
        evaluation_at=NOW,
        max_receipt_age_seconds=604800,
        sample_count=len(labels),
        counts={"approve": 1, "reject": 0, "manual_review": 1},
        adversarial_counts={},
    )
    write_dataset(
        DatasetBundle(
            manifest=manifest,
            labels=labels,
            images={original.image: IMAGE_A, replay.image: IMAGE_B},
        ),
        tmp_path,
    )
    return load_dataset(tmp_path)


class TestReviewedThenReplayed:
    """The exact corner that produced the live false approvals."""

    def test_a_replay_of_a_reviewed_receipt_is_never_approved(self, tmp_path: Path) -> None:
        dataset = two_sample_dataset(tmp_path)
        first, second = dataset.labels
        # The first receipt is routed to a human because its issuer is unreadable; the
        # replay is otherwise perfect and matches the ledger.
        incomplete = extraction_from(first, issuer=None)
        complete = extraction_from(second)
        report = run_evaluation(dataset, ScriptedExtractor((incomplete, complete)))

        reviewed, replayed = report.outcomes
        assert reviewed.predicted_decision is Decision.MANUAL_REVIEW
        assert replayed.predicted_decision is Decision.MANUAL_REVIEW
        assert VerdictReason.DUPLICATE_OPERATION_ID in replayed.predicted_reasons
        assert report.metrics.false_approvals == 0

    def test_the_identical_image_is_rejected(self, tmp_path: Path) -> None:
        dataset = two_sample_dataset(tmp_path)
        first, second = dataset.labels
        # Same bytes twice: the registry must remember the first hash and reject the replay.
        bundle = DatasetBundle(
            manifest=dataset.manifest,
            labels=dataset.labels,
            images={first.image: IMAGE_A, second.image: IMAGE_A},
        )
        write_dataset(bundle, tmp_path)
        reloaded = load_dataset(tmp_path)
        report = run_evaluation(
            reloaded, ScriptedExtractor((extraction_from(first), extraction_from(second)))
        )
        _, replayed = report.outcomes
        assert replayed.predicted_decision is Decision.REJECT
        assert VerdictReason.IDENTICAL_RECEIPT_REPLAY in replayed.predicted_reasons
        assert report.metrics.false_approvals == 0

    def test_an_unrelated_receipt_is_unaffected(self, tmp_path: Path) -> None:
        dataset = two_sample_dataset(tmp_path)
        first, second = dataset.labels
        report = run_evaluation(
            dataset, ScriptedExtractor((extraction_from(first), extraction_from(second)))
        )
        approved, replayed = report.outcomes
        assert approved.predicted_decision is Decision.APPROVE
        assert replayed.predicted_decision is Decision.MANUAL_REVIEW
        assert report.metrics.approve_true_positives == 1
