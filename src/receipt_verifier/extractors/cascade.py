"""Ordered extractor cascade: primary model, secondary model, local OCR.

The cascade is the resilience story of this service. Each stage is wrapped in its own
circuit breaker and deadline, and the cascade only accepts a stage's answer when the
deterministic usability rule is satisfied: every critical field present with at least
the policy's minimum confidence. A stage that errors, times out, is refused by its
breaker, or returns an incomplete reading does not stop the request — the next stage
gets its turn.

When nothing is usable the cascade still returns the *best partial* reading, with
``extractor="none"`` only if every stage failed outright. That keeps information for the
human reviewer while never fabricating one: the validator turns a partial reading into
``manual_review``, never into an approval.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from receipt_verifier.builders import build_extraction
from receipt_verifier.circuit import CircuitStatus, GuardedExtractor
from receipt_verifier.extraction import CRITICAL_FIELD_NAMES, ExtractionResult, ReceiptExtractor
from receipt_verifier.validate import ValidationPolicy

CASCADE_NAME = "cascade"
NO_EXTRACTOR = "none"


class CascadeExhausted(RuntimeError):
    """No stage produced a usable reading."""

    def __init__(self, failures: Mapping[str, str]) -> None:
        super().__init__(f"no extractor produced a usable reading: {dict(failures)}")
        self.failures = dict(failures)


@dataclass(frozen=True)
class CascadeOutcome:
    """What happened on the last :meth:`CascadeExtractor.extract` call."""

    used: str
    attempts: tuple[str, ...]
    failures: Mapping[str, str]
    usable: bool


@dataclass
class CascadeExtractor:
    """Tries extractors in order and returns the first usable reading."""

    extractors: Sequence[GuardedExtractor | ReceiptExtractor]
    policy: ValidationPolicy = field(default_factory=ValidationPolicy)
    _usage: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _failures: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _last: CascadeOutcome | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.extractors:
            raise ValueError("a cascade needs at least one extractor")
        self._guarded: tuple[GuardedExtractor, ...] = tuple(
            item if isinstance(item, GuardedExtractor) else GuardedExtractor(item)
            for item in self.extractors
        )

    @property
    def name(self) -> str:
        return CASCADE_NAME

    @property
    def stages(self) -> tuple[GuardedExtractor, ...]:
        return self._guarded

    def _is_usable(self, extraction: ExtractionResult) -> bool:
        fields = extraction.field_map()
        return all(
            fields[name].value is not None
            and fields[name].confidence >= self.policy.min_field_confidence
            for name in CRITICAL_FIELD_NAMES
        )

    @staticmethod
    def _completeness(extraction: ExtractionResult) -> int:
        fields = extraction.field_map()
        return sum(1 for name in CRITICAL_FIELD_NAMES if fields[name].value is not None)

    def extract(self, image: bytes) -> ExtractionResult:
        """Run the cascade and return the best reading available (never raises)."""
        attempts: list[str] = []
        failures: dict[str, str] = {}
        best: ExtractionResult | None = None
        for stage in self._guarded:
            attempts.append(stage.name)
            try:
                extraction = stage.extract(image)
            except Exception as exc:  # any stage failure must not break the request
                failures[stage.name] = type(exc).__name__
                self._failures[stage.name] = self._failures.get(stage.name, 0) + 1
                continue
            if best is None or self._completeness(extraction) > self._completeness(best):
                best = extraction
            if self._is_usable(extraction):
                self._usage[extraction.extractor] = self._usage.get(extraction.extractor, 0) + 1
                self._last = CascadeOutcome(
                    used=extraction.extractor,
                    attempts=tuple(attempts),
                    failures=dict(failures),
                    usable=True,
                )
                return extraction
        if best is not None:
            self._usage[best.extractor] = self._usage.get(best.extractor, 0) + 1
        self._last = CascadeOutcome(
            used=best.extractor if best is not None else NO_EXTRACTOR,
            attempts=tuple(attempts),
            failures=dict(failures),
            usable=False,
        )
        if best is not None:
            return best
        return build_extraction(NO_EXTRACTOR, {})

    def extract_or_raise(self, image: bytes) -> ExtractionResult:
        """Like :meth:`extract`, but raise when nothing was usable."""
        extraction = self.extract(image)
        outcome = self.last_outcome
        if outcome is not None and not outcome.usable:
            raise CascadeExhausted(outcome.failures)
        return extraction

    @property
    def last_outcome(self) -> CascadeOutcome | None:
        return self._last

    def usage_counts(self) -> dict[str, int]:
        """How many receipts each stage actually produced (for reports)."""
        return dict(self._usage)

    def failure_counts(self) -> dict[str, int]:
        return dict(self._failures)

    def statuses(self) -> tuple[CircuitStatus, ...]:
        return tuple(stage.status() for stage in self._guarded)


def build_cascade(
    extractors: Sequence[GuardedExtractor | ReceiptExtractor],
    *,
    policy: ValidationPolicy | None = None,
) -> CascadeExtractor:
    """Convenience constructor used by the settings layer."""
    return CascadeExtractor(extractors, policy=policy or ValidationPolicy())


__all__ = [
    "CASCADE_NAME",
    "NO_EXTRACTOR",
    "CascadeExhausted",
    "CascadeExtractor",
    "CascadeOutcome",
    "build_cascade",
]
