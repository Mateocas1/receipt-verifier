"""Small builders that keep extractor implementations short and type-safe."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal

from receipt_verifier.extraction import FIELD_NAMES, ExtractionResult


def build_extraction(
    extractor: str,
    values: Mapping[str, object],
    *,
    confidences: Mapping[str, float] | None = None,
    raw_text: str = "",
    latency_ms: float = 0.0,
    cost_usd: Decimal = Decimal("0"),
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    error: str = "",
) -> ExtractionResult:
    """Build an :class:`ExtractionResult` from raw values.

    ``values`` maps canonical field names to raw Python values (a ``Decimal``, a
    ``datetime``, a :class:`~receipt_verifier.schema.Destination`, an ``Issuer`` or a
    ``str``); Pydantic coerces them to the declared field type. Missing keys become
    ``None`` with ``0.0`` confidence. Per-field ``confidences`` default to ``1.0`` for
    present values and ``0.0`` for absent ones. ``prompt_tokens``/``completion_tokens``
    are the provider usage this reading cost, when the extractor knows it.
    """
    explicit = confidences or {}
    payload: dict[str, object] = {
        "extractor": extractor,
        "raw_text": raw_text,
        "latency_ms": latency_ms,
        "cost_usd": cost_usd,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "error": error,
    }
    for name in FIELD_NAMES:
        raw = values.get(name)
        confidence = explicit.get(name, 1.0 if raw is not None else 0.0)
        payload[name] = {"value": raw, "confidence": confidence}
    return ExtractionResult.model_validate(payload)
