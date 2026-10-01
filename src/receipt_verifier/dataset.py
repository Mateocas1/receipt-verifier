"""Dataset container: build-time bundle, on-disk format and loader."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from receipt_verifier.schema import DatasetManifest, ReceiptLabel

LABELS_FILENAME = "labels.jsonl"
MANIFEST_FILENAME = "manifest.json"
IMAGES_DIRNAME = "images"


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


def load_dataset(root: Path) -> Dataset:
    """Load a dataset directory written by :func:`write_dataset`."""
    manifest = DatasetManifest.model_validate_json(
        (root / MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    raw_lines = (root / LABELS_FILENAME).read_text(encoding="utf-8").splitlines()
    labels = tuple(ReceiptLabel.model_validate_json(line) for line in raw_lines if line.strip())
    return Dataset(root=root, manifest=manifest, labels=labels)
