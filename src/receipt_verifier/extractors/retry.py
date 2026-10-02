"""Destination retry: one focused re-ask before the verifier gives up.

The first reading of a receipt can miss the destination the ledger expected. Two cases
are worth one extra call, and both are decidable in code:

* the destination is unusable (the model returned nothing, or something that fails the
  alias syntax / CBU-CVU check digits, which ``score_field`` already drops);
* the destination is well formed but is not one the configured allowlist knows.

The wrapper re-asks the same extractor for that one field with a focused prompt, then
applies the same allowlist rule before accepting the answer. Only the ``destination``
field is ever replaced: every other value, confidence and reason is the first reading's,
so a receipt rejected for amount, date, replay or injection can never be approved by a
retry. With no configured allowlist there is nothing to compare against, so the retry is
disabled instead of guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

from receipt_verifier.extraction import (
    DestinationRefiner,
    ExtractedField,
    ExtractionResult,
    ReceiptExtractor,
)
from receipt_verifier.schema import Destination

UNREADABLE = "unreadable"
NOT_ALLOWLISTED = "not_allowlisted"
REASONS = (UNREADABLE, NOT_ALLOWLISTED)


@dataclass
class DestinationRetryExtractor:
    """Wraps an extractor and gives a rejected destination exactly one focused re-ask."""

    inner: ReceiptExtractor
    allowed_destinations: frozenset[str] = frozenset()
    refiner: DestinationRefiner | None = None
    retries: int = field(default=0, init=False)
    adopted: int = field(default=0, init=False)
    failed_retries: int = field(default=0, init=False)
    retry_reasons: dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.refiner is None and isinstance(self.inner, DestinationRefiner):
            self.refiner = self.inner

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def can_retry(self) -> bool:
        """True when a focused re-ask is possible at all."""
        return self.refiner is not None and bool(self.allowed_destinations)

    def extract(self, image: bytes) -> ExtractionResult:
        """Read once, and re-ask for the destination only when the code cannot accept it."""
        reading = self.inner.extract(image)
        reason = self._reason(reading.destination)
        if reason is None or self.refiner is None:
            return reading
        self.retries += 1
        self.retry_reasons[reason] = self.retry_reasons.get(reason, 0) + 1
        started = perf_counter()
        try:
            focused = self.refiner.refine_destination(image)
        except Exception:  # a failed re-ask must not lose the first reading
            self.failed_retries += 1
            return reading
        return self._merged(reading, focused, (perf_counter() - started) * 1000.0)

    def _reason(self, destination: ExtractedField[Destination]) -> str | None:
        if not self.allowed_destinations:
            return None
        if destination.value is None:
            return UNREADABLE
        if destination.value.key not in self.allowed_destinations:
            return NOT_ALLOWLISTED
        return None

    def _merged(
        self,
        reading: ExtractionResult,
        focused: ExtractionResult,
        retry_latency_ms: float,
    ) -> ExtractionResult:
        """The first reading plus the focused call's cost, and its destination only if allowed."""
        updates: dict[str, object] = {
            "cost_usd": reading.cost_usd + focused.cost_usd,
            "prompt_tokens": reading.prompt_tokens + focused.prompt_tokens,
            "completion_tokens": reading.completion_tokens + focused.completion_tokens,
            "destination_retries": reading.destination_retries + 1,
            "retry_latency_ms": reading.retry_latency_ms + retry_latency_ms,
        }
        refined = focused.destination
        if refined.value is not None and refined.value.key in self.allowed_destinations:
            updates["destination"] = refined
            self.adopted += 1
        return reading.model_copy(update=updates)

    def retry_counts(self) -> dict[str, int]:
        """Focused calls issued per trigger, for reports."""
        return dict(self.retry_reasons)


def build_destination_retry(
    inner: ReceiptExtractor,
    allowed_destinations: frozenset[str],
    *,
    refiner: DestinationRefiner | None = None,
) -> DestinationRetryExtractor:
    """Convenience constructor used by the settings layer and the evaluation CLI."""
    return DestinationRetryExtractor(
        inner=inner,
        allowed_destinations=allowed_destinations,
        refiner=refiner,
    )


__all__ = [
    "NOT_ALLOWLISTED",
    "REASONS",
    "UNREADABLE",
    "DestinationRetryExtractor",
    "build_destination_retry",
]
