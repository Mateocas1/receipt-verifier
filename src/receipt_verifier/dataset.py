"""Dataset container: build-time bundle, on-disk format and loader."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from receipt_verifier.schema import AR_TZ, AdversarialKind, DatasetManifest, ReceiptLabel

LABELS_FILENAME = "labels.jsonl"
MANIFEST_FILENAME = "manifest.json"
IMAGES_DIRNAME = "images"
DEFAULT_MAX_RECEIPT_AGE = timedelta(days=7)


@dataclass(frozen=True)
class DatasetBundle:
    """Everything a generator produced: metadata, labels and rendered image bytes."""

    manifest: DatasetManifest
    labels: tuple[ReceiptLabel, ...]
    images: dict[str, bytes]


class Dataset(BaseModel):
    """A dataset loaded from disk, ready for evaluation."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    root: Path
    manifest: DatasetManifest
    labels: tuple[ReceiptLabel, ...]

    @property
    def evaluation_at(self) -> datetime:
        """Reference 'now' for deterministic date-window checks."""
        return self.manifest.evaluation_at

    @property
    def allowed_destinations(self) -> frozenset[str]:
        """Every destination the ledger knows about, as comparator keys."""
        return frozenset(label.expectation.destination.key for label in self.labels)

    def image_path(self, label: ReceiptLabel) -> Path:
        return self.root / label.image

    def image_bytes(self, label: ReceiptLabel) -> bytes:
        return self.image_path(label).read_bytes()


def write_dataset(bundle: DatasetBundle, root: Path) -> None:
    """Write a bundle as ``images/`` + ``labels.jsonl`` + ``manifest.json``."""
    root.mkdir(parents=True, exist_ok=True)
    for relative_path, data in sorted(bundle.images.items()):
        target = root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    labels_text = "".join(f"{label.model_dump_json()}\n" for label in bundle.labels)
    (root / LABELS_FILENAME).write_text(labels_text, encoding="utf-8")
    manifest_text = bundle.manifest.model_dump_json(indent=2) + "\n"
    (root / MANIFEST_FILENAME).write_text(manifest_text, encoding="utf-8")


def derive_manifest(
    root: Path, labels: tuple[ReceiptLabel, ...], *, now: datetime
) -> DatasetManifest:
    """Build a manifest for a folder that does not ship one (hand-curated real data).

    The evaluation clock is the wall clock: a real receipt must be judged against the
    moment it is verified, not against the moment the folder was assembled.
    """
    counts: dict[str, int] = {}
    adversarial_counts: dict[str, int] = {}
    for label in labels:
        counts[label.expected_decision.value] = counts.get(label.expected_decision.value, 0) + 1
        if label.adversarial is not AdversarialKind.NONE:
            adversarial_counts[label.adversarial.value] = (
                adversarial_counts.get(label.adversarial.value, 0) + 1
            )
    return DatasetManifest(
        name=root.name or "dataset",
        version="unversioned",
        seed=0,
        generated_at=now,
        evaluation_at=now,
        max_receipt_age_seconds=int(DEFAULT_MAX_RECEIPT_AGE.total_seconds()),
        sample_count=len(labels),
        counts=counts,
        adversarial_counts=adversarial_counts,
    )


def load_dataset(root: Path, *, now: datetime | None = None) -> Dataset:
    """Load a dataset directory written by :func:`write_dataset`.

    A folder without ``manifest.json`` is accepted: the manifest is derived from the
    labels and the current instant, which is how locally curated (real, anonymized)
    samples are evaluated. An empty folder (a frozen manifest and no labels yet) is a
    valid dataset with zero samples.
    """
    labels_path = root / LABELS_FILENAME
    raw_lines = (
        labels_path.read_text(encoding="utf-8").splitlines() if labels_path.is_file() else []
    )
    labels = tuple(ReceiptLabel.model_validate_json(line) for line in raw_lines if line.strip())
    manifest_path = root / MANIFEST_FILENAME
    if manifest_path.is_file():
        manifest = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    else:
        reference = now if now is not None else datetime.now(tz=AR_TZ)
        manifest = derive_manifest(root, labels, now=reference)
    return Dataset(root=root, manifest=manifest, labels=labels)
