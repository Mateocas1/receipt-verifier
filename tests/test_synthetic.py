from datetime import datetime
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

import pytest

from receipt_verifier.dataset import DatasetBundle, load_dataset, write_dataset
from receipt_verifier.identifiers import is_valid_alias, is_valid_cbu_or_cvu
from receipt_verifier.schema import (
    AR_TZ,
    AdversarialKind,
    DatasetManifest,
    Decision,
    DestinationKind,
    Issuer,
)
from receipt_verifier.synthetic.generate import (
    ADVERSARIAL_ORDER,
    DECISION_BY_KIND,
    DEFAULT_GENERATED_AT,
    DEFAULT_SEED,
    NORMAL_PER_ISSUER,
    REASON_BY_KIND,
    build_dataset,
)

COMMITTED_DATASET = Path(__file__).resolve().parents[1] / "dataset" / "synthetic" / "v1"


def digests(bundle: DatasetBundle) -> dict[str, str]:
    return {name: sha256(data).hexdigest() for name, data in bundle.images.items()}


class TestDeterminism:
    def test_same_seed_reproduces_labels_and_images(self) -> None:
        first = build_dataset(seed=42, normal_per_issuer=1)
        second = build_dataset(seed=42, normal_per_issuer=1)
        assert first.labels == second.labels
        assert first.manifest == second.manifest
        assert digests(first) == digests(second)

    def test_different_seed_produces_different_data(self) -> None:
        first = build_dataset(seed=1, normal_per_issuer=1)
        second = build_dataset(seed=2, normal_per_issuer=1)
        assert first.labels[0].operation_id != second.labels[0].operation_id
        assert digests(first) != digests(second)

    def test_naive_reference_timestamp_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            build_dataset(
                generated_at=datetime.fromisoformat("2025-07-01T09:00"), normal_per_issuer=1
            )

    def test_manifest_clock_matches_its_reference(self) -> None:
        bundle = build_dataset(normal_per_issuer=1)
        assert bundle.manifest.generated_at == DEFAULT_GENERATED_AT
        assert bundle.manifest.evaluation_at == bundle.manifest.generated_at
        assert bundle.manifest.seed == DEFAULT_SEED


@pytest.fixture(scope="module")
def bundle() -> DatasetBundle:
    return build_dataset(normal_per_issuer=1)


class TestComposition:
    def test_counts(self, bundle: DatasetBundle) -> None:
        normal = sum(1 for label in bundle.labels if label.adversarial is AdversarialKind.NONE)
        adversarial = len(bundle.labels) - normal
        assert normal == len(Issuer)
        assert adversarial == len(ADVERSARIAL_ORDER) * len(Issuer)
        expected_counts: dict[str, int] = {}
        for label in bundle.labels:
            expected_counts[label.expected_decision.value] = (
                expected_counts.get(label.expected_decision.value, 0) + 1
            )
        assert bundle.manifest.counts == expected_counts
        assert bundle.manifest.counts[Decision.APPROVE.value] == normal
        assert bundle.manifest.counts[Decision.MANUAL_REVIEW.value] == len(Issuer)

    def test_every_issuer_is_represented(self, bundle: DatasetBundle) -> None:
        assert {label.issuer for label in bundle.labels} == set(Issuer)

    def test_every_adversarial_kind_maps_to_its_reason(self, bundle: DatasetBundle) -> None:
        for kind in ADVERSARIAL_ORDER:
            samples = [label for label in bundle.labels if label.adversarial is kind]
            assert len(samples) == len(Issuer)
            for label in samples:
                assert label.expected_decision is DECISION_BY_KIND[kind]
                assert label.reasons == (REASON_BY_KIND[kind],)

    def test_duplicates_are_the_only_manual_review_samples(self, bundle: DatasetBundle) -> None:
        manual = [
            label for label in bundle.labels if label.expected_decision is Decision.MANUAL_REVIEW
        ]
        assert {label.adversarial for label in manual} == {AdversarialKind.DUPLICATE_OPERATION_ID}

    def test_normal_samples_are_approved(self, bundle: DatasetBundle) -> None:
        normal = [label for label in bundle.labels if label.adversarial is AdversarialKind.NONE]
        assert all(label.expected_decision is Decision.APPROVE for label in normal)
        assert all(not label.reasons for label in normal)

    def test_destinations_use_valid_formats(self, bundle: DatasetBundle) -> None:
        kinds = set()
        for label in bundle.labels:
            kinds.add(label.destination.kind)
            if label.destination.kind is DestinationKind.ALIAS:
                assert is_valid_alias(label.destination.value)
            else:
                assert is_valid_cbu_or_cvu(label.destination.value)
        assert kinds == {DestinationKind.ALIAS, DestinationKind.CVU}

    def test_amounts_are_positive_and_quantized(self, bundle: DatasetBundle) -> None:
        for label in bundle.labels:
            assert label.amount > Decimal("0")
            assert label.amount.as_tuple().exponent == -2

    def test_edited_amount_breaks_internal_consistency(self, bundle: DatasetBundle) -> None:
        edited = [
            label for label in bundle.labels if label.adversarial is AdversarialKind.EDITED_AMOUNT
        ]
        assert edited
        for label in edited:
            assert label.amount != label.amount_detail
            assert label.amount > label.amount_detail
            assert label.expectation.amount == label.amount_detail

    def test_duplicate_samples_reuse_an_earlier_operation_id(self, bundle: DatasetBundle) -> None:
        duplicates = [
            label
            for label in bundle.labels
            if label.adversarial is AdversarialKind.DUPLICATE_OPERATION_ID
        ]
        approved_ids = {
            label.operation_id
            for label in bundle.labels
            if label.adversarial is AdversarialKind.NONE
        }
        assert duplicates
        assert all(label.operation_id in approved_ids for label in duplicates)

    def test_wrong_destination_is_not_the_expected_one(self, bundle: DatasetBundle) -> None:
        wrong = [
            label
            for label in bundle.labels
            if label.adversarial is AdversarialKind.WRONG_DESTINATION
        ]
        assert all(label.destination.key != label.expectation.destination.key for label in wrong)

    def test_injected_instruction_samples_carry_instruction_text(
        self, bundle: DatasetBundle
    ) -> None:
        injected = [
            label
            for label in bundle.labels
            if label.adversarial is AdversarialKind.INJECTED_INSTRUCTION
        ]
        assert all(label.memo for label in injected)

    def test_stale_samples_are_older_than_the_window(self, bundle: DatasetBundle) -> None:
        stale = [
            label for label in bundle.labels if label.adversarial is AdversarialKind.STALE_DATE
        ]
        max_age = bundle.manifest.max_receipt_age_seconds
        assert all(
            (bundle.manifest.evaluation_at - label.transferred_at).total_seconds() > max_age
            for label in stale
        )

    def test_rendered_images_are_png(self, bundle: DatasetBundle) -> None:
        assert bundle.images
        assert all(data.startswith(b"\x89PNG\r\n\x1a\n") for data in bundle.images.values())

    def test_image_size_stays_reasonable(self, bundle: DatasetBundle) -> None:
        average = sum(len(data) for data in bundle.images.values()) / len(bundle.images)
        assert average < 120_000


class TestRoundTrip:
    def test_write_then_load(self, tmp_path: Path) -> None:
        bundle = build_dataset(seed=7, normal_per_issuer=1)
        write_dataset(bundle, tmp_path)
        loaded = load_dataset(tmp_path)
        assert loaded.labels == bundle.labels
        assert loaded.manifest == bundle.manifest
        assert loaded.allowed_destinations == frozenset(
            label.expectation.destination.key for label in bundle.labels
        )

    def test_defensive_image_path_is_preserved(self, tmp_path: Path) -> None:
        bundle = build_dataset(seed=7, normal_per_issuer=1)
        write_dataset(bundle, tmp_path)
        loaded = load_dataset(tmp_path)
        assert loaded.image_bytes(loaded.labels[0]).startswith(b"\x89PNG")

    def test_folder_without_manifest_derives_one(self, tmp_path: Path) -> None:
        bundle = build_dataset(seed=7, normal_per_issuer=1)
        write_dataset(bundle, tmp_path)
        (tmp_path / "manifest.json").unlink()
        clock = datetime(2025, 9, 1, 12, 0, tzinfo=AR_TZ)
        loaded = load_dataset(tmp_path, now=clock)
        assert loaded.manifest.version == "unversioned"
        assert loaded.manifest.evaluation_at == clock
        assert loaded.manifest.seed == 0
        assert loaded.manifest.sample_count == len(bundle.labels)
        assert loaded.manifest.counts == bundle.manifest.counts
        assert loaded.manifest.adversarial_counts == bundle.manifest.adversarial_counts


@pytest.mark.skipif(not COMMITTED_DATASET.exists(), reason="committed dataset not present")
class TestCommittedDataset:
    def test_manifest_matches_the_generator_contract(self) -> None:
        dataset = load_dataset(COMMITTED_DATASET)
        manifest = dataset.manifest
        assert isinstance(manifest, DatasetManifest)
        assert manifest.version == "v1"
        assert manifest.sample_count == len(dataset.labels)
        assert manifest.counts[Decision.APPROVE.value] == NORMAL_PER_ISSUER * len(Issuer)
        assert manifest.counts[Decision.REJECT.value] == (len(ADVERSARIAL_ORDER) - 1) * len(Issuer)
        assert manifest.counts[Decision.MANUAL_REVIEW.value] == len(Issuer)
        assert manifest.adversarial_counts == {
            kind.value: len(Issuer) for kind in ADVERSARIAL_ORDER
        }
        assert manifest.evaluation_at.tzinfo is not None

    def test_labels_are_reproducible_from_the_manifest(self) -> None:
        dataset = load_dataset(COMMITTED_DATASET)
        manifest = dataset.manifest
        rebuilt = build_dataset(
            seed=manifest.seed,
            generated_at=manifest.generated_at.astimezone(AR_TZ),
            version=manifest.version,
            normal_per_issuer=manifest.counts[Decision.APPROVE.value] // len(Issuer),
        )
        assert rebuilt.labels == dataset.labels
        assert rebuilt.manifest == manifest

    def test_every_committed_image_exists(self) -> None:
        dataset = load_dataset(COMMITTED_DATASET)
        for label in dataset.labels:
            path = dataset.image_path(label)
            assert path.is_file(), path
            assert path.stat().st_size > 0
