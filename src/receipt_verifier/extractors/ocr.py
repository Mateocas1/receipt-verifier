"""Local OCR extractor: text engine + per-issuer layout parsers.

The engine turns image bytes into positioned text lines (Tesseract through the
``tesserocr`` wheels, an optional dependency). The parser is pure: it reads those lines
and reconstructs the receipt fields using the wording and layout of each issuer, never
the image pixels.

Two properties matter more than raw accuracy here:

* **no invented values** — a field that cannot be located stays ``None`` and its
  confidence stays ``0.0``, which routes the receipt to ``manual_review``;
* **no trust in the engine** — parser output still goes through
  :mod:`receipt_verifier.confidence`, which is where the confidences are computed.
"""

from __future__ import annotations

import io
import os
import re
import threading
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Final, Protocol

from receipt_verifier.confidence import SourceQuality, score_extraction
from receipt_verifier.extraction import ExtractionResult
from receipt_verifier.identifiers import parse_destination_text
from receipt_verifier.schema import ISSUER_DISPLAY_NAMES, Issuer, issuer_for_display_name

OCR_EXTRACTOR_NAME: Final = "ocr"
DEFAULT_LANGUAGES: Final = "spa+eng"
DEFAULT_PAGE_SEGMENT_MODE: Final = 11
DEFAULT_UPSCALE: Final = 2.0
FUZZY_MATCH_THRESHOLD: Final = 0.75
ISSUER_MATCH_THRESHOLD: Final = 0.9
FUZZY_MIN_LENGTH: Final = 5
ROW_OVERLAP_RATIO: Final = 0.5
SEPARATE_COLUMN_GAP: Final = 30

LABELS: Final[dict[str, tuple[str, ...]]] = {
    "transferred_at": ("fecha y hora",),
    "sender_name": ("remitente",),
    "sender_bank": ("banco origen",),
    "destination": ("destino",),
    "destination_holder": ("titular destino",),
    "operation_id": ("n° de operación", "nro de operación", "numero de operacion"),
    "memo": ("concepto",),
}
"""Printed row labels of the six layouts, lowercased (compared folded)."""

ISSUER_LABELS: Final[dict[Issuer, tuple[str, str]]] = {
    Issuer.MP: ("importe transferido", "detalle del importe"),
    Issuer.UALA: ("monto", "total debitado"),
    Issuer.BRUBANK: ("importe", "importe acreditado"),
    Issuer.GALICIA: ("importe de la operación", "importe informado"),
    Issuer.SANTANDER: ("monto transferido", "monto original"),
    Issuer.BNA: ("importe transferido", "importe según detalle"),
}
"""``(amount label, detail label)`` per issuer; the pair identifies the issuer."""

MONEY_PATTERN: Final = re.compile(r"\d{1,3}(?:\.\d{3})*,\d{2}")
OPERATION_ID_PREFIX: Final = re.compile(r"^[A-Z]{2,10}-")
DATE_PATTERN: Final = re.compile(r"\d{2}/\d{2}/\d{4}(?:\s+\d{2}:\d{2})?")
OPERATION_ID_PATTERN: Final = re.compile(r"[A-Z]{2,10}-[0-9A-F]{6,}|\d{4}-\d{6,}")
OPERATION_ID_SEARCH: Final = re.compile(r"[A-Z]{2,10}-[A-Z0-9]{6,24}|\d{4}-\d{10}")
"""Search form: tolerates the letters that misread hex digits carry (``O`` for ``0``)."""
_HEX_DIGITS: Final = frozenset("0123456789ABCDEF")
_HEX_CONFUSIONS: Final = str.maketrans({"O": "0", "I": "1", "L": "1", "S": "5", "G": "6", "Z": "2"})
"""Letters that cannot be hexadecimal, mapped to the digit they are read as."""
_FOLD_RE: Final = re.compile(r"[^0-9a-z]+")


ALL_LABELS: Final[tuple[str, ...]] = tuple(
    [label for labels in LABELS.values() for label in labels]
    + [label for pair in ISSUER_LABELS.values() for label in pair]
)
"""Every printed label; used to stop a value walk before it eats the next field."""


def _looks_like_label(text: str) -> bool:
    folded = fold(text)
    return any(_similar(folded, fold(label)) for label in ALL_LABELS)


def _row_has_label(row: OcrRow) -> bool:
    """True when a row introduces a field ("Banco origen" [value]), not a value.

    Row-level text is not enough: in side-by-side layouts a row holds the next label
    *and* its value, and gluing that value onto the previous field is exactly the bug
    this guard prevents.
    """
    return _looks_like_label(row.text) or any(_looks_like_label(line.text) for line in row.lines)


class OcrUnavailable(RuntimeError):
    """The OCR engine (or its language data) is not installed."""


def fold(text: str) -> str:
    """Lowercase, strip accents and drop every separator: the comparison form."""
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_text = decomposed.encode("ascii", "ignore").decode("ascii").lower()
    return _FOLD_RE.sub("", ascii_text)


@dataclass(frozen=True)
class OcrLine:
    """One recognized text line with its pixel box and mean confidence (0..1)."""

    text: str
    left: int
    top: int
    right: int
    bottom: int
    confidence: float

    @property
    def height(self) -> int:
        return max(1, self.bottom - self.top)

    @property
    def width(self) -> int:
        return max(1, self.right - self.left)


@dataclass(frozen=True)
class OcrRow:
    """Lines that share a visual row, left to right."""

    lines: tuple[OcrLine, ...]

    @property
    def text(self) -> str:
        return " ".join(line.text for line in self.lines).strip()

    @property
    def folded(self) -> str:
        return fold(self.text)

    @property
    def top(self) -> int:
        return min(line.top for line in self.lines)

    @property
    def bottom(self) -> int:
        return max(line.bottom for line in self.lines)

    @property
    def height(self) -> int:
        return max(1, self.bottom - self.top)

    @property
    def confidence(self) -> float:
        return sum(line.confidence for line in self.lines) / len(self.lines)


@dataclass(frozen=True)
class OcrDocument:
    """Everything the engine saw: positioned lines in reading order."""

    lines: tuple[OcrLine, ...]

    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def mean_confidence(self) -> float:
        if not self.lines:
            return 0.0
        return sum(line.confidence for line in self.lines) / len(self.lines)

    def rows(self) -> tuple[OcrRow, ...]:
        """Group lines into visual rows (vertical overlap >= 50% of the shorter line)."""
        ordered = sorted(self.lines, key=lambda line: (line.top, line.left))
        rows: list[list[OcrLine]] = []
        for line in ordered:
            if rows and _vertically_aligned(rows[-1], line):
                rows[-1].append(line)
            else:
                rows.append([line])
        return tuple(OcrRow(tuple(sorted(row, key=lambda item: item.left))) for row in rows)


def _vertically_aligned(row: list[OcrLine], line: OcrLine) -> bool:
    top = max(min(item.top for item in row), line.top)
    bottom = min(max(item.bottom for item in row), line.bottom)
    overlap = max(0, bottom - top)
    smallest = min(line.height, min(item.height for item in row))
    return overlap >= ROW_OVERLAP_RATIO * smallest


class TextEngine(Protocol):
    """Anything that can turn image bytes into positioned text."""

    @property
    def name(self) -> str: ...

    def read(self, image: bytes) -> OcrDocument: ...


def _similar(left: str, right: str) -> bool:
    """Fuzzy label equality: exact for short strings, tolerant for longer ones.

    Deliberately *not* a prefix rule: "Importe" is a prefix of both "Importe" (brubank's
    amount) and "Importe acreditado" (its detail amount). Merged labels are handled in
    :func:`_match_label`, where a line has to start with the *whole* label.
    """
    if left == right:
        return True
    if min(len(left), len(right)) < FUZZY_MIN_LENGTH:
        return False
    return SequenceMatcher(None, left, right).ratio() >= FUZZY_MATCH_THRESHOLD


def _similar_strict(left: str, right: str) -> bool:
    """Near-exact equality, used for the issuer vote.

    Issuer identification must not accept prefixes: "Monto" is a prefix of both "Monto"
    (uala) and "Monto transferido" (Santander).
    """
    if left == right:
        return True
    if min(len(left), len(right)) < FUZZY_MIN_LENGTH:
        return False
    return SequenceMatcher(None, left, right).ratio() >= ISSUER_MATCH_THRESHOLD


def _label_pattern(label: str) -> re.Pattern[str]:
    """``Label: value`` on one line — the case where OCR merged the two columns."""
    words = [re.escape(word) for word in label.split()]
    return re.compile(r"^\s*" + r"\s*".join(words) + r"\s*:?\s*(?P<value>.+)$", re.IGNORECASE)


def _match_label(
    rows: Iterable[OcrRow], labels: Iterable[str], *, strict: bool = False
) -> OcrRow | None:
    """First row whose text (as a whole or per line) matches any candidate label."""
    compare = _similar_strict if strict else _similar
    for row in rows:
        for candidate in labels:
            folded = fold(candidate)
            if compare(row.folded, folded):
                return row
            for line in row.lines:
                line_fold = fold(line.text)
                # A line that *begins* with the whole label is that label plus its value
                # ("Remitente: Camila Gómez", "Detalle del importe: $ 2.500,00").
                if line_fold.startswith(folded) or compare(line_fold, folded):
                    return row
    return None


def _value_lines_in_row(row: OcrRow, labels: tuple[str, ...]) -> list[OcrLine]:
    """Lines of a row that carry the value of ``labels`` (its own row, right side)."""
    for candidate in labels:
        for line in row.lines:
            merged = _label_pattern(candidate).match(line.text)
            if merged is not None:
                # A line like "Remitente:" matches with the separator as its remainder;
                # that is not a value, so fall through to the column logic.
                cleaned = merged.group("value").strip().lstrip(":-.").strip()
                if cleaned:
                    return [
                        OcrLine(
                            cleaned, line.left, line.top, line.right, line.bottom, line.confidence
                        )
                    ]
    if len(row.lines) > 1:
        label_edge = (
            max(line.right for line in row.lines if _looks_like_label(line.text))
            if any(_looks_like_label(line.text) for line in row.lines)
            else min(line.right for line in row.lines)
        )
        right_side = [line for line in row.lines if line.left >= label_edge + SEPARATE_COLUMN_GAP]
        if right_side:
            return list(right_side)
    return []


def _same_column(line: OcrLine, reference: OcrLine) -> bool:
    """True when ``line`` sits in the same column as ``reference`` (left- or right-aligned)."""
    tolerance = max(10, int(0.6 * reference.height))
    return (
        abs(line.right - reference.right) <= tolerance
        or abs(line.left - reference.left) <= tolerance
    )


def _continuation_lines(
    rows: tuple[OcrRow, ...],
    start: int,
    reference: OcrLine,
    *,
    max_rows: int = 2,
) -> list[OcrLine]:
    """Lines directly below ``reference`` that continue the same wrapped value."""
    collected: list[OcrLine] = []
    bottom = reference.bottom
    for row in rows[start : start + max_rows]:
        if _row_has_label(row):
            break
        candidates = [line for line in row.lines if _same_column(line, reference)]
        if not candidates:
            break
        line = min(candidates, key=lambda item: abs(item.left - reference.left))
        gap = line.top - bottom
        if gap > 0.9 * reference.height or gap < -0.6 * reference.height:
            break
        collected.append(line)
        bottom = line.bottom
    return collected


def _value_after(
    rows: tuple[OcrRow, ...],
    label_row: OcrRow,
    labels: tuple[str, ...],
    *,
    max_extra_rows: int = 2,
) -> str | None:
    """Text that belongs to a label: its own row, plus any wrapped continuation lines.

    Wrapping is why more than one row may carry a value; the walk stops at the next
    printed label or when the column/vertical alignment no longer matches.
    """
    index = rows.index(label_row)
    lines = _value_lines_in_row(label_row, labels)
    if lines:
        reference = lines[-1]
        lines = lines + _continuation_lines(rows, index + 1, reference, max_rows=max_extra_rows)
        return " ".join(line.text.strip() for line in lines).strip() or None

    collected: list[OcrLine] = []
    for offset, row in enumerate(rows[index + 1 :], start=index + 1):
        if not row.text:
            continue
        if _row_has_label(row):
            break
        collected.extend(row.lines)
        collected.extend(_continuation_lines(rows, offset + 1, collected[-1], max_rows=1))
        break
    if not collected:
        return None
    return " ".join(line.text.strip() for line in collected).strip() or None


def identify_issuer(rows: tuple[OcrRow, ...]) -> Issuer | None:
    """Identify the issuer: the published header name first, then its layout wording.

    The header name is the published, 1:1 mapping (:data:`ISSUER_DISPLAY_NAMES`); the
    per-issuer amount/detail labels remain the fallback, which is what keeps receipts
    rendered before the mapping was published (``dataset/synthetic/v1``) readable.
    """
    named = _issuer_from_display_name(rows)
    if named is not None:
        return named
    scores: dict[Issuer, int] = {}
    for issuer, (amount_label, detail_label) in ISSUER_LABELS.items():
        scores[issuer] = sum(
            1
            for label in (amount_label, detail_label)
            if _match_label(rows, (label,), strict=True) is not None
        )
    best = max(scores.values(), default=0)
    if best == 0:
        return None
    winners = [issuer for issuer, score in scores.items() if score == best]
    return winners[0] if len(winners) == 1 else None


def _issuer_from_display_name(rows: tuple[OcrRow, ...]) -> Issuer | None:
    """The published issuer name printed in the header, tolerating OCR noise.

    An exact match wins outright; otherwise the single best name above the strict
    threshold is accepted, so a misread letter does not lose the field. A tie between two
    names is ambiguous and returns ``None`` rather than guessing.
    """
    candidates: set[Issuer] = set()
    for row in rows:
        for item in row.lines:
            exact = issuer_for_display_name(item.text)
            if exact is not None:
                return exact
            folded = fold(item.text)
            for issuer, name in ISSUER_DISPLAY_NAMES.items():
                if _similar_strict(folded, fold(name)):
                    candidates.add(issuer)
    if len(candidates) == 1:
        return next(iter(candidates))
    return None


def normalize_operation_id(token: str) -> str:
    """Repair obvious OCR misreads inside an alphanumeric operation id.

    ``O``/``I``/``L``/``S``/``G``/``Z`` cannot be hexadecimal, so in a token whose body is
    hexadecimal they can only be a misread digit. The mapping is applied only when it
    turns the body into a valid hexadecimal string; otherwise the token is returned
    untouched. Numeric ids (``1234-5678901234``) are never rewritten.
    """
    token = "".join(token.split()).upper()
    prefix_match = OPERATION_ID_PREFIX.match(token)
    if prefix_match is None:
        return token
    body = token[prefix_match.end() :]
    if not body or all(char in _HEX_DIGITS for char in body):
        return token
    repaired = body.translate(_HEX_CONFUSIONS)
    if all(char in _HEX_DIGITS for char in repaired):
        return f"{token[: prefix_match.end()]}{repaired}"
    return token


def extract_operation_id(rows: tuple[OcrRow, ...]) -> str | None:
    """Find the operation id, tolerating OCR splits inside the token.

    Candidates are searched one row at a time (never concatenated): gluing the next row
    on would turn "MP-A02F4C5E61AE" plus "CONCEPTO" into a longer fake identifier.
    """
    labels = LABELS["operation_id"]
    label_row = _match_label(rows, labels)
    if label_row is None:
        return None
    folded_labels = [fold(label) for label in labels]
    same_row = "".join(
        line.text
        for line in label_row.lines
        if not any(_similar(fold(line.text), candidate) for candidate in folded_labels)
    )
    index = rows.index(label_row)
    candidates = [same_row, *(row.text for row in rows[index + 1 : index + 3])]
    for candidate in candidates:
        match = OPERATION_ID_SEARCH.search(candidate.replace(" ", ""))
        if match is not None:
            return normalize_operation_id(match.group(0))
    return None


def _money_tokens(text: str) -> list[str]:
    return MONEY_PATTERN.findall(text)


def extract_amounts(rows: tuple[OcrRow, ...]) -> tuple[str | None, str | None, float]:
    """Headline and detail amounts, plus the confidence of the headline row.

    The headline is the money row with the tallest text (the big figure); the detail is
    the money that sits on a detail-label row. Keeping them apart is what makes an
    "edited amount" forgery visible to the validator.
    """
    detail_rows = [
        row
        for issuer_labels in ISSUER_LABELS.values()
        if (row := _match_label(rows, (issuer_labels[1],))) is not None
    ]
    detail: str | None = None
    for row in detail_rows:
        tokens = _money_tokens(row.text)
        if tokens:
            detail = tokens[-1]
            break

    headline_row: OcrRow | None = None
    for row in rows:
        if not _money_tokens(row.text) or row in detail_rows:
            continue
        if headline_row is None or row.height > headline_row.height:
            headline_row = row
    headline = _money_tokens(headline_row.text)[-1] if headline_row is not None else None
    if detail is None:
        detail = headline
    confidence = headline_row.confidence if headline_row is not None else 0.0
    return headline, detail, confidence


@dataclass(frozen=True)
class ParsedFields:
    """Raw values plus the per-field source quality derived from the OCR lines."""

    values: dict[str, object]
    source_quality: SourceQuality


def parse_document(document: OcrDocument) -> ParsedFields:
    """Reconstruct receipt fields from positioned text lines."""
    rows = document.rows()
    values: dict[str, object] = {}
    quality: dict[str, float] = {}

    def take(field: str, labels: tuple[str, ...]) -> str | None:
        row = _match_label(rows, labels)
        if row is None:
            return None
        value = _value_after(rows, row, labels)
        if value:
            quality[field] = row.confidence
        return value

    issuer = identify_issuer(rows)
    if issuer is not None:
        values["issuer"] = issuer
        quality["issuer"] = document.mean_confidence

    headline, detail, amount_confidence = extract_amounts(rows)
    if headline is not None:
        values["amount"] = headline
        quality["amount"] = amount_confidence
    if detail is not None:
        values["amount_detail"] = detail
        quality["amount_detail"] = amount_confidence

    date_match = DATE_PATTERN.search(document.text())
    if date_match:
        values["transferred_at"] = date_match.group(0)
        quality["transferred_at"] = document.mean_confidence

    for field in ("sender_name", "sender_bank", "memo"):
        found = take(field, LABELS[field])
        if found:
            values[field] = found

    destination_row = _match_label(rows, LABELS["destination"])
    if destination_row is not None:
        destination_text = _value_after(rows, destination_row, LABELS["destination"])
        parsed = parse_destination_text(destination_text) if destination_text else None
        holder = take("destination_holder", LABELS["destination_holder"])
        if parsed is not None:
            kind, raw = parsed
            values["destination"] = {"kind": kind, "value": raw, "holder": holder or ""}
            quality["destination"] = destination_row.confidence

    operation_id = extract_operation_id(rows)
    if operation_id is not None:
        values["operation_id"] = operation_id
        row = _match_label(rows, LABELS["operation_id"])
        quality["operation_id"] = row.confidence if row is not None else document.mean_confidence

    return ParsedFields(values=values, source_quality=SourceQuality(per_field=quality))


class OcrExtractor:
    """Turns image bytes into fields using a :class:`TextEngine` plus the layout parser."""

    def __init__(self, engine: TextEngine) -> None:
        self._engine = engine

    @property
    def name(self) -> str:
        return OCR_EXTRACTOR_NAME

    @property
    def engine(self) -> TextEngine:
        return self._engine

    def extract(self, image: bytes) -> ExtractionResult:
        document = self._engine.read(image)
        parsed = parse_document(document)
        return score_extraction(
            self.name,
            parsed.values,
            raw_text=document.text(),
            source_quality=parsed.source_quality,
        )


def discover_tessdata() -> Path | None:
    """Locate Tesseract language data: explicit env var first, then usual system paths."""
    configured = os.environ.get("OCR_TESSDATA")
    if configured:
        path = Path(configured)
        return path if path.is_dir() else None
    for candidate in (
        "/usr/share/tesseract-ocr/5/tessdata",
        "/usr/share/tesseract-ocr/4.00/tessdata",
        "/usr/share/tessdata",
        "/usr/local/share/tessdata",
    ):
        path = Path(candidate)
        if path.is_dir() and any(path.glob("*.traineddata")):
            return path
    return None


class TesserocrEngine:
    """Tesseract 5 through the ``tesserocr`` wheels (no system package required).

    One recognizer is reused and guarded by a lock: Tesseract's API object is not
    thread-safe and OCR is the slow path anyway.
    """

    def __init__(
        self,
        *,
        languages: str = DEFAULT_LANGUAGES,
        tessdata: Path | None = None,
        page_segment_mode: int = DEFAULT_PAGE_SEGMENT_MODE,
        upscale: float = DEFAULT_UPSCALE,
    ) -> None:
        try:
            import tesserocr
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise OcrUnavailable(
                "tesserocr is not installed; install the 'ocr' extra to enable local OCR"
            ) from exc
        self._tesserocr = tesserocr
        data = tessdata if tessdata is not None else discover_tessdata()
        kwargs: dict[str, object] = {"lang": languages}
        if data is not None:
            kwargs["path"] = str(data)
        try:
            self._api = tesserocr.PyTessBaseAPI(**kwargs)
        except RuntimeError as exc:
            raise OcrUnavailable(f"tesseract cannot load languages {languages!r}: {exc}") from exc
        self._api.SetPageSegMode(page_segment_mode)
        self._languages = languages
        if upscale < 1.0:
            raise ValueError("upscale must be >= 1.0")
        self._upscale = upscale
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return OCR_EXTRACTOR_NAME

    @property
    def languages(self) -> str:
        return self._languages

    @property
    def upscale(self) -> float:
        return self._upscale

    def read(self, image: bytes) -> OcrDocument:
        from PIL import Image

        with self._lock:
            picture: Image.Image = Image.open(io.BytesIO(image))
            picture.load()
            if self._upscale != 1.0:
                # Banking screenshots are routinely below 300 DPI; upscaling recovers
                # small labels and alphanumeric identifiers before recognition.
                picture = picture.resize(
                    (int(picture.width * self._upscale), int(picture.height * self._upscale)),
                    Image.Resampling.LANCZOS,
                )
            self._api.SetImage(picture)
            self._api.Recognize()
            iterator = self._api.GetIterator()
            lines: list[OcrLine] = []
            if iterator is not None:
                level = self._tesserocr.RIL.TEXTLINE
                while True:
                    text = (iterator.GetUTF8Text(level) or "").strip()
                    if text:
                        x0, y0, x1, y1 = iterator.BoundingBox(level)
                        confidence = float(iterator.Confidence(level)) / 100.0
                        lines.append(
                            OcrLine(
                                text=text,
                                left=int(x0),
                                top=int(y0),
                                right=int(x1),
                                bottom=int(y1),
                                confidence=max(0.0, min(1.0, confidence)),
                            )
                        )
                    if not iterator.Next(level):
                        break
        return OcrDocument(lines=tuple(lines))


def default_extractor(
    *,
    languages: str = DEFAULT_LANGUAGES,
    tessdata: Path | None = None,
    upscale: float = DEFAULT_UPSCALE,
) -> OcrExtractor:
    """Build the production OCR extractor, raising :class:`OcrUnavailable` if unusable."""
    return OcrExtractor(TesserocrEngine(languages=languages, tessdata=tessdata, upscale=upscale))


__all__ = [
    "DEFAULT_LANGUAGES",
    "DEFAULT_UPSCALE",
    "ISSUER_LABELS",
    "LABELS",
    "OCR_EXTRACTOR_NAME",
    "OcrDocument",
    "OcrExtractor",
    "OcrLine",
    "OcrRow",
    "OcrUnavailable",
    "ParsedFields",
    "TesserocrEngine",
    "TextEngine",
    "default_extractor",
    "discover_tessdata",
    "extract_amounts",
    "extract_operation_id",
    "fold",
    "identify_issuer",
    "normalize_operation_id",
    "parse_document",
]
