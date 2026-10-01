"""Dummy extractor: replays the dataset ground truth, optionally with noise.

It is not a model. Its job is to prove the harness is wired correctly (perfect scores
at zero noise) and to let the validator be tested in isolation: any false approval it
produces is a validator bug, never an extraction error.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from random import Random
from typing import Self

from receipt_verifier.builders import build_extraction
from receipt_verifier.dataset import load_dataset
from receipt_verifier.extraction import FIELD_NAMES, ExtractionResult
from receipt_verifier.schema import Destination, Issuer, ReceiptLabel

DEFAULT_NOISE_SEED = 1717

_CORRUPTED_CONFIDENCE = 0.4


def image_digest(image: bytes) -> str:
    """Digest used to key ground truth by image content."""
    return sha256(image).hexdigest()


def _corrupt(name: str, value: object, rng: Random) -> object:
    """Distort a scalar value; structured values are only downgraded in confidence."""
    if isinstance(value, Decimal):
        return value + Decimal("1.00")
    if isinstance(value, datetime):
        return value - timedelta(hours=rng.randint(1, 6))
    if isinstance(value, Issuer | Destination):
        return value
    if isinstance(value, str):
        return f"{value}x"
    return value


class DummyExtractor:
    """Serves the label that belongs to a given image, with configurable noise."""

    def __init__(
        self,
        samples_by_digest: Mapping[str, ReceiptLabel],
        *,
        noise: float = 0.0,
        seed: int = DEFAULT_NOISE_SEED,
        name: str = "dummy",
    ) -> None:
        if not 0.0 <= noise <= 1.0:
            raise ValueError("noise must be within [0.0, 1.0]")
        self._samples = dict(samples_by_digest)
        self._noise = noise
        self._rng = Random(seed)
        self._name = name

    @classmethod
    def from_dataset(
        cls,
        dataset_dir: Path,
        *,
        noise: float = 0.0,
        seed: int = DEFAULT_NOISE_SEED,
        name: str = "dummy",
    ) -> Self:
        """Build an extractor backed by a dataset directory."""
        dataset = load_dataset(dataset_dir)
        samples = {image_digest(dataset.image_bytes(label)): label for label in dataset.labels}
        return cls(samples, noise=noise, seed=seed, name=name)

    @classmethod
    def from_labels(
        cls,
        labels: Mapping[str, ReceiptLabel],
        *,
        noise: float = 0.0,
        seed: int = DEFAULT_NOISE_SEED,
    ) -> Self:
        """Build an extractor from pre-computed image digests (used by tests)."""
        return cls(labels, noise=noise, seed=seed)

    @property
    def name(self) -> str:
        return self._name

    @property
    def noise(self) -> float:
        return self._noise

    def extract(self, image: bytes) -> ExtractionResult:
        digest = image_digest(image)
        try:
            label = self._samples[digest]
        except KeyError as exc:
            raise KeyError("no ground truth for this image; wrong dataset or extractor") from exc
        return self._extraction_for(label)

    def _extraction_for(self, label: ReceiptLabel) -> ExtractionResult:
        values: dict[str, object] = {
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
        confidences: dict[str, float] = dict.fromkeys(FIELD_NAMES, 1.0)
        for name in FIELD_NAMES:
            if self._noise > 0.0 and self._rng.random() < self._noise:
                if self._rng.random() < 0.5:
                    values[name] = None
                    confidences[name] = 0.0
                else:
                    values[name] = _corrupt(name, values[name], self._rng)
                    confidences[name] = _CORRUPTED_CONFIDENCE
        raw_text = " ".join(str(values[name]) for name in FIELD_NAMES if values[name] is not None)
        return build_extraction(
            self._name,
            values,
            confidences=confidences,
            raw_text=raw_text,
        )
