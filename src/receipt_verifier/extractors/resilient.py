"""Evaluation-only tolerance wrapper: one failed sample must not end the run.

The service survives a failing stage through the cascade, which falls through to the
next extractor. A single model evaluated on its own has no such net: a provider hiccup
that outlived its retries would abort a 150-receipt run at whichever sample hit it.

This wrapper turns that exception into an empty reading carrying the reason. The
validator then routes the sample to ``manual_review`` (``extraction_failed``) and the
metrics count the missing fields, so a flaky provider shows up as a worse number
instead of as a lost run. ``failures`` and ``last_error`` keep the count reportable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from receipt_verifier.builders import build_extraction
from receipt_verifier.extraction import ExtractionResult, ReceiptExtractor


@dataclass
class ResilientExtractor:
    """Wraps an extractor, converting a raised failure into an empty reading."""

    inner: ReceiptExtractor
    failures: int = field(default=0, init=False)
    last_error: str = field(default="", init=False)

    @property
    def name(self) -> str:
        return self.inner.name

    def extract(self, image: bytes) -> ExtractionResult:
        try:
            return self.inner.extract(image)
        except Exception as exc:  # any extractor failure is a measurement, not a crash
            self.failures += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            return build_extraction(self.inner.name, {}, error=self.last_error)


__all__ = ["ResilientExtractor"]
