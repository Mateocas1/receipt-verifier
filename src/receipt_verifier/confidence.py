"""Deterministic per-field confidence.

Extractors propose values; this module decides how much the code trusts them. Nothing
that a model reports about itself reaches the verdict: the confidence of a field is
computed from evidence a program can check.

``confidence = source_quality * format_factor * corroboration_factor``

* **source_quality** — how reliable the engine is *for this field* (an OCR word
  confidence, a fixed LLM prior, or a per-field hint). Absent evidence defaults to the
  engine prior.
* **format_factor** — ``1.0`` when the value satisfies its strict format rule (CBU/CVU
  checksum, alias syntax, money parse, tz-aware timestamp, known issuer); ``0.6`` when it
  only parses loosely; a value that does not parse at all is dropped with confidence
  ``0.0``.
* **corroboration_factor** — ``1.0`` when the value can be found again in the receipt's
  raw text, ``0.75`` when it cannot (hallucination signal), ``1.0`` when there is no raw
  text to compare against.

The result is deliberately boring and reproducible: the same fields and the same raw text
always produce the same confidences.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Final

from receipt_verifier.builders import build_extraction
from receipt_verifier.extraction import FIELD_NAMES, ExtractionResult
from receipt_verifier.identifiers import (
    is_valid_alias,
    is_valid_cbu_or_cvu,
    parse_amount_text,
    quantize_amount,
)
from receipt_verifier.schema import AR_TZ, Destination, DestinationKind, Issuer

MAX_CONFIDENCE: Final = 1.0
UNPARSED_FORMAT_FACTOR: Final = 0.6
UNCORROBORATED_FACTOR: Final = 0.75
DEFAULT_SOURCE_QUALITY: Final = 0.85

_OPERATION_ID_RE: Final = re.compile(r"^[A-Za-z0-9]{2,10}-[A-Za-z0-9]{6,}$")
_ALNUM_RE: Final = re.compile(r"[^0-9a-z]+")
_DATE_FORMATS: Final = ("%d/%m/%Y %H:%M", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y", "%Y-%m-%dT%H:%M:%S%z")


@dataclass(frozen=True)
class SourceQuality:
    """How much the code trusts an engine, overall and per field."""

    default: float = DEFAULT_SOURCE_QUALITY
    per_field: Mapping[str, float] = field(default_factory=dict)

    def for_field(self, name: str) -> float:
        return min(MAX_CONFIDENCE, max(0.0, self.per_field.get(name, self.default)))


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return _ALNUM_RE.sub("", decomposed.encode("ascii", "ignore").decode("ascii").lower())


def to_decimal(value: object) -> Decimal | None:
    """Parse money from a ``Decimal``, a number or an Argentine-formatted string."""
    if isinstance(value, Decimal):
        return quantize_amount(value)
    if isinstance(value, int | float):
        return quantize_amount(Decimal(str(value)))
    if isinstance(value, str):
        parsed = parse_amount_text(value)
        if parsed is not None:
            return parsed
        try:
            return quantize_amount(Decimal(value.strip()))
        except InvalidOperation:
            return None
    return None


def to_datetime(value: object) -> datetime | None:
    """Parse a timestamp from a ``datetime`` or a printed/ISO string."""
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=AR_TZ)
    if isinstance(value, str):
        for fmt in _DATE_FORMATS:
            try:
                # Printed Argentine formats carry no offset; one is attached below.
                parsed = datetime.strptime(value.strip(), fmt)  # noqa: DTZ007
            except ValueError:
                continue
            return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=AR_TZ)
    return None


def to_destination(value: object) -> Destination | None:
    """Coerce a ``Destination`` or a mapping with kind/value/holder into one.

    A bare alias/CVU string cannot carry the holder name, so it is not accepted here; the
    holder travels as its own field and is filled in by the extractor.
    """
    if isinstance(value, Destination):
        return value
    if isinstance(value, Mapping):
        try:
            return Destination.model_validate(dict(value))
        except ValueError:
            return None
    return None


def to_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def to_issuer(value: object) -> Issuer | None:
    if isinstance(value, Issuer):
        return value
    if isinstance(value, str):
        try:
            return Issuer(value.strip().lower())
        except ValueError:
            return None
    return None


_COERCERS: dict[str, Callable[[object], object]] = {
    "amount": to_decimal,
    "amount_detail": to_decimal,
    "transferred_at": to_datetime,
    "sender_name": to_text,
    "sender_bank": to_text,
    "destination": to_destination,
    "operation_id": to_text,
    "issuer": to_issuer,
    "memo": to_text,
}
"""Coercion per canonical field: anything that fails becomes ``None`` (dropped)."""

_STRICT_FORMATS: dict[str, Callable[[object], bool]] = {
    "amount": lambda v: isinstance(v, Decimal) and v > 0,
    "amount_detail": lambda v: isinstance(v, Decimal) and v > 0,
    "transferred_at": lambda v: isinstance(v, datetime) and v.tzinfo is not None,
    "destination": lambda v: isinstance(v, Destination) and _destination_is_valid(v),
    "operation_id": lambda v: isinstance(v, str) and bool(_OPERATION_ID_RE.match(v)),
    "issuer": lambda v: isinstance(v, Issuer),
    "sender_name": lambda v: isinstance(v, str) and len(v) >= 3,
    "sender_bank": lambda v: isinstance(v, str) and len(v) >= 3,
    "memo": lambda v: isinstance(v, str),
}
"""Strict per-field format rules; failing one only lowers the confidence."""


def _destination_is_valid(destination: Destination) -> bool:
    if destination.kind is DestinationKind.ALIAS:
        return is_valid_alias(destination.value)
    return is_valid_cbu_or_cvu(destination.value)


def _corroboration_key(name: str, value: object) -> str | None:
    """Normalized form used to look the value up again in the raw text.

    ``None`` means "not corroboratable" (the field is not expected to appear verbatim).
    """
    if name == "destination" and isinstance(value, Destination):
        return _fold(value.value)
    if name == "issuer":
        return None
    if name == "operation_id" and isinstance(value, str):
        return _fold(value).upper()
    if isinstance(value, str):
        return _fold(value)
    return None


_MONEY_TOKEN_RE: Final = re.compile(r"\d{1,3}(?:\.\d{3})+(?:,\d{1,2})?|\d+(?:,\d{1,2})?")
_DATE_TOKEN_RE: Final = re.compile(r"(\d{2})/(\d{2})/(\d{4})(?:\s+(\d{2}):(\d{2}))?")


def _money_tokens(raw_text: str) -> set[Decimal]:
    tokens: set[Decimal] = set()
    for match in _MONEY_TOKEN_RE.finditer(raw_text):
        parsed = parse_amount_text(match.group(0))
        if parsed is not None:
            tokens.add(parsed)
    return tokens


def _is_corroborated(name: str, value: object, raw_text: str) -> bool | None:
    """Is ``value`` visible again in the receipt text? ``None`` when not applicable.

    Money and timestamps are compared as *parsed values*, never as digit substrings: a
    substring test would happily corroborate ``5,00`` inside ``2.500,00``.
    """
    if not raw_text.strip():
        return None
    if name in ("amount", "amount_detail"):
        return isinstance(value, Decimal) and quantize_amount(value) in _money_tokens(raw_text)
    if name == "transferred_at":
        if not isinstance(value, datetime):
            return None
        local = value.astimezone(AR_TZ)
        for day, month, year, hour, minute in _DATE_TOKEN_RE.findall(raw_text):
            if (int(year), int(month), int(day)) != (local.year, local.month, local.day):
                continue
            if not hour or (int(hour), int(minute)) == (local.hour, local.minute):
                return True
        return False
    key = _corroboration_key(name, value)
    if key is None:
        return None
    return not key or key in _fold(raw_text)


def score_field(
    name: str,
    value: object,
    *,
    source_quality: SourceQuality,
    raw_text: str,
) -> tuple[object | None, float]:
    """Coerce one field and compute its confidence. Unparsable values are dropped."""
    coercer = _COERCERS.get(name)
    if coercer is None:
        raise KeyError(f"unknown field: {name}")
    coerced = coercer(value)
    if coerced is None:
        return None, 0.0
    check = _STRICT_FORMATS.get(name)
    format_factor = 1.0 if check is None or check(coerced) else UNPARSED_FORMAT_FACTOR
    confidence = source_quality.for_field(name) * format_factor
    if _is_corroborated(name, coerced, raw_text) is False:
        confidence *= UNCORROBORATED_FACTOR
    return coerced, round(min(MAX_CONFIDENCE, max(0.0, confidence)), 4)


def score_extraction(
    extractor: str,
    values: Mapping[str, object],
    *,
    raw_text: str = "",
    source_quality: SourceQuality | None = None,
    latency_ms: float = 0.0,
    cost_usd: Decimal = Decimal("0"),
) -> ExtractionResult:
    """Turn raw candidate values into an :class:`ExtractionResult` with code-owned scores."""
    quality = source_quality or SourceQuality()
    scored: dict[str, object] = {}
    confidences: dict[str, float] = {}
    for name in FIELD_NAMES:
        coerced, confidence = score_field(
            name, values.get(name), source_quality=quality, raw_text=raw_text
        )
        scored[name] = coerced
        confidences[name] = confidence
    return build_extraction(
        extractor,
        scored,
        confidences=confidences,
        raw_text=raw_text,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
    )


def field_confidences(extraction: ExtractionResult) -> dict[str, float]:
    """Plain ``field -> confidence`` view, used by the service response."""
    return {name: field.confidence for name, field in extraction.field_map().items()}
