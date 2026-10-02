"""Render synthetic receipt PNGs.

The six styles imitate the *layout families* of common Argentine wallets and banks
(centered amount, card, table, boxed constancia, minimal, official header) but draw
generic text only: no logo, no trademark and no real institution name is reproduced.
Every image carries a synthetic-data footer so a leaked file can never be mistaken
for evidence.
"""

from __future__ import annotations

import io
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import Final, Literal

from PIL import Image, ImageDraw, ImageFont

from receipt_verifier.identifiers import render_amount_ars
from receipt_verifier.schema import (
    AR_TZ,
    ISSUER_DISPLAY_NAMES,
    Issuer,
    ReceiptLabel,
)

type RGB = tuple[int, int, int]
type FontType = ImageFont.FreeTypeFont | ImageFont.ImageFont
type LayoutName = Literal["band", "card", "boxed", "table", "minimal", "official"]
type AmountAlign = Literal["left", "center", "right"]

FOOTER_TEXT: Final = "Comprobante sintético · datos ficticios · sin validez legal"
DATE_FORMAT: Final = "%d/%m/%Y %H:%M"

_FONT_FILES: Final[dict[bool, tuple[str, ...]]] = {
    False: (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
    ),
    True: (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
    ),
}


@cache
def load_font(size: int, *, bold: bool = False) -> FontType:
    """Load a bundled system font, falling back to Pillow's embedded default."""
    for candidate in _FONT_FILES[bold]:
        path = Path(candidate)
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


@dataclass(frozen=True)
class IssuerStyle:
    """Everything that makes one issuer's receipt layout visually distinct."""

    issuer: Issuer
    document_label: str
    amount_label: str
    detail_label: str
    layout: LayoutName
    amount_align: AmountAlign
    accent: RGB
    header_bg: RGB
    header_text: RGB
    page_bg: RGB
    card_bg: RGB
    text: RGB
    muted: RGB
    width: int = 620
    height: int = 900
    header_height: int = 148
    margin: int = 32
    row_gap: int = 44
    amount_size: int = 46


STYLES: Final[dict[Issuer, IssuerStyle]] = {
    Issuer.MP: IssuerStyle(
        issuer=Issuer.MP,
        document_label="Comprobante de transferencia",
        amount_label="Importe transferido",
        detail_label="Detalle del importe",
        layout="band",
        amount_align="center",
        accent=(0, 158, 227),
        header_bg=(0, 116, 190),
        header_text=(255, 255, 255),
        page_bg=(240, 246, 251),
        card_bg=(255, 255, 255),
        text=(32, 40, 48),
        muted=(112, 126, 140),
    ),
    Issuer.UALA: IssuerStyle(
        issuer=Issuer.UALA,
        document_label="Transferencia enviada",
        amount_label="Monto",
        detail_label="Total debitado",
        layout="card",
        amount_align="left",
        accent=(124, 58, 237),
        header_bg=(124, 58, 237),
        header_text=(255, 255, 255),
        page_bg=(244, 241, 252),
        card_bg=(255, 255, 255),
        text=(35, 32, 46),
        muted=(120, 112, 140),
    ),
    Issuer.BRUBANK: IssuerStyle(
        issuer=Issuer.BRUBANK,
        document_label="Constancia de transferencia",
        amount_label="Importe",
        detail_label="Importe acreditado",
        layout="table",
        amount_align="right",
        accent=(16, 185, 129),
        header_bg=(15, 27, 45),
        header_text=(236, 244, 248),
        page_bg=(246, 248, 250),
        card_bg=(255, 255, 255),
        text=(20, 28, 38),
        muted=(108, 120, 132),
    ),
    Issuer.GALICIA: IssuerStyle(
        issuer=Issuer.GALICIA,
        document_label="Comprobante de transferencia inmediata",
        amount_label="Importe de la operación",
        detail_label="Importe informado",
        layout="minimal",
        amount_align="center",
        accent=(214, 69, 51),
        header_bg=(255, 255, 255),
        header_text=(30, 30, 30),
        page_bg=(255, 255, 255),
        card_bg=(249, 246, 243),
        text=(38, 34, 32),
        muted=(140, 130, 124),
    ),
    Issuer.SANTANDER: IssuerStyle(
        issuer=Issuer.SANTANDER,
        document_label="Comprobante de operación",
        amount_label="Monto transferido",
        detail_label="Monto original",
        layout="boxed",
        amount_align="left",
        accent=(204, 0, 0),
        header_bg=(204, 0, 0),
        header_text=(255, 255, 255),
        page_bg=(252, 246, 246),
        card_bg=(255, 255, 255),
        text=(40, 32, 32),
        muted=(130, 112, 112),
    ),
    Issuer.BNA: IssuerStyle(
        issuer=Issuer.BNA,
        document_label="Constancia de transferencia bancaria",
        amount_label="Importe transferido",
        detail_label="Importe según detalle",
        layout="official",
        amount_align="center",
        accent=(0, 92, 138),
        header_bg=(0, 92, 138),
        header_text=(255, 255, 255),
        page_bg=(238, 243, 246),
        card_bg=(255, 255, 255),
        text=(26, 34, 42),
        muted=(100, 112, 124),
        row_gap=46,
    ),
}
"""One style per issuer. Display names are invented placeholders, not brand names."""


def _fit(text: str, font: FontType, max_width: int) -> str:
    """Truncate ``text`` with an ellipsis so it fits ``max_width`` pixels."""
    if max_width <= 0:
        return ""
    if font.getlength(text) <= max_width:
        return text
    for end in range(len(text) - 1, 0, -1):
        candidate = f"{text[:end]}…"
        if font.getlength(candidate) <= max_width:
            return candidate
    return "…"


def _wrap(text: str, font: FontType, max_width: int, max_lines: int = 2) -> list[str]:
    """Wrap ``text`` to at most ``max_lines`` lines; only the overflow is ellipsized.

    Row values (especially the free-text concept) must be printed in full: a label that
    claims more text than the image holds would make every evaluation a lie.
    """
    remaining = text.split()
    lines: list[str] = []
    while remaining and len(lines) < max_lines:
        current = remaining.pop(0)
        while remaining:
            candidate = f"{current} {remaining[0]}"
            if font.getlength(candidate) > max_width:
                break
            current = candidate
            remaining.pop(0)
        lines.append(current)
    if remaining:
        lines[-1] = _fit(f"{lines[-1]} {' '.join(remaining)}", font, max_width)
    return lines or [""]


def _rows(label: ReceiptLabel) -> list[tuple[str, str]]:
    rows = [
        ("Fecha y hora", format_transferred_at(label.transferred_at)),
        ("Remitente", label.sender_name),
        ("Banco origen", label.sender_bank),
        ("Destino", f"{label.destination.kind.value.upper()} {label.destination.value}"),
        ("Titular destino", label.destination.holder),
        ("N° de operación", label.operation_id),
    ]
    if label.memo:
        rows.append(("Concepto", label.memo))
    return rows


def _draw_header(draw: ImageDraw.ImageDraw, style: IssuerStyle, display_name: str) -> int:
    """Draw the header with the published issuer name; return where the body starts."""
    width = style.width
    margin = style.margin
    if style.layout in ("band", "card", "table", "official"):
        draw.rectangle((0, 0, width, style.header_height), fill=style.header_bg)
        title_font = load_font(22, bold=True)
        sub_font = load_font(15)
        draw.text((margin, 34), display_name, font=title_font, fill=style.header_text)
        draw.text((margin, 68), style.document_label, font=sub_font, fill=style.header_text)
        draw.rectangle((0, style.header_height, width, style.header_height + 4), fill=style.accent)
        return style.header_height + 28
    if style.layout == "minimal":
        title_font = load_font(19, bold=True)
        sub_font = load_font(14)
        draw.text((margin, 40), display_name, font=title_font, fill=style.text)
        draw.text((margin, 70), style.document_label, font=sub_font, fill=style.muted)
        draw.rectangle((margin, 104, width - margin, 108), fill=style.accent)
        return 132
    # boxed
    draw.rectangle((16, 16, width - 16, style.header_height), outline=style.accent, width=2)
    title_font = load_font(21, bold=True)
    sub_font = load_font(14)
    draw.text((width // 2, 46), display_name, font=title_font, fill=style.text, anchor="mm")
    draw.text((width // 2, 80), style.document_label, font=sub_font, fill=style.muted, anchor="mm")
    return style.header_height + 26


def _draw_amount(
    draw: ImageDraw.ImageDraw, style: IssuerStyle, label: ReceiptLabel, top: int
) -> int:
    """Draw the headline amount plus the receipt's detail line; return the next y."""
    width = style.width
    margin = style.margin
    content_width = width - 2 * margin

    if style.layout == "card":
        draw.rounded_rectangle(
            (margin - 8, top - 12, width - margin + 8, top + style.amount_size + 62),
            radius=14,
            fill=style.card_bg,
        )
    amount_font = load_font(style.amount_size, bold=True)
    label_font = load_font(15)
    detail_font = load_font(16)
    amount_text = f"$ {render_amount_ars(label.amount)}"
    detail_text = f"{style.detail_label}: $ {render_amount_ars(label.amount_detail)}"
    if style.layout == "official":
        box_top = top - 10
        draw.rectangle(
            (margin, box_top, width - margin, box_top + 104), outline=style.accent, width=1
        )
    anchor = {"left": "lm", "center": "mm", "right": "rm"}[style.amount_align]
    x = {"left": margin + 16, "center": width // 2, "right": width - margin - 16}[
        style.amount_align
    ]
    label_y = top + 4
    if style.amount_align == "center":
        draw.text(
            (width // 2, label_y),
            style.amount_label,
            font=label_font,
            fill=style.muted,
            anchor="mm",
        )
    else:
        draw.text(
            (x, label_y), style.amount_label, font=label_font, fill=style.muted, anchor=anchor
        )
    draw.text((x, top + 38), amount_text, font=amount_font, fill=style.text, anchor=anchor)
    detail_y = top + 74
    if style.amount_align == "left":
        draw.text(
            (x, detail_y),
            _fit(detail_text, detail_font, content_width - 32),
            font=detail_font,
            fill=style.muted,
        )
    else:
        draw.text(
            (width // 2, detail_y), detail_text, font=detail_font, fill=style.muted, anchor="mm"
        )
    return top + 116


def _draw_rows(draw: ImageDraw.ImageDraw, style: IssuerStyle, label: ReceiptLabel, top: int) -> int:
    """Draw the field rows; return the y coordinate after the last row.

    A value that needs two lines is printed on two lines: the label sidecar must never
    claim more text than the image holds.
    """
    width = style.width
    margin = style.margin
    content_width = width - 2 * margin
    label_font = load_font(16)
    value_font = load_font(18)
    y = top
    rows = _rows(label)
    for index, (name, value) in enumerate(rows):
        value_width = content_width - 8 if style.layout == "minimal" else int(content_width * 0.62)
        value_lines = _wrap(value, value_font, value_width)
        extra_lines = len(value_lines) - 1
        if style.layout == "table":
            if index % 2 == 0:
                draw.rectangle(
                    (margin - 8, y - 20, width - margin + 8, y + 20 + 18 * extra_lines),
                    fill=style.card_bg,
                )
            draw.text((margin, y), name, font=label_font, fill=style.muted, anchor="lm")
            for offset, line in enumerate(value_lines):
                draw.text(
                    (width - margin, y + 18 * offset),
                    line,
                    font=value_font,
                    fill=style.text,
                    anchor="rm",
                )
            if index < len(rows) - 1:
                baseline = y + 22 + 18 * extra_lines
                draw.line(
                    (margin, baseline, width - margin, baseline), fill=(230, 234, 238), width=1
                )
        elif style.layout == "official":
            draw.text((margin, y), f"{name}:", font=label_font, fill=style.muted, anchor="lm")
            for offset, line in enumerate(value_lines):
                draw.text(
                    (margin + 190, y + 18 * offset),
                    line,
                    font=value_font,
                    fill=style.text,
                    anchor="lm",
                )
            baseline = y + 22 + 18 * extra_lines
            draw.line((margin, baseline, width - margin, baseline), fill=(214, 222, 228), width=1)
        elif style.layout == "minimal":
            draw.text(
                (margin, y - 12), name.upper(), font=load_font(12), fill=style.muted, anchor="lm"
            )
            for offset, line in enumerate(value_lines):
                draw.text(
                    (margin, y + 12 + 20 * offset),
                    line,
                    font=value_font,
                    fill=style.text,
                    anchor="lm",
                )
        elif style.layout in ("card", "band"):
            draw.text((margin, y), name, font=label_font, fill=style.muted, anchor="lm")
            for offset, line in enumerate(value_lines):
                draw.text(
                    (width - margin, y + 18 * offset),
                    line,
                    font=value_font,
                    fill=style.text,
                    anchor="rm",
                )
        else:  # boxed
            draw.text((margin, y), name, font=label_font, fill=style.muted, anchor="lm")
            for offset, line in enumerate(value_lines):
                draw.text(
                    (margin + 200, y + 18 * offset),
                    line,
                    font=value_font,
                    fill=style.text,
                    anchor="lm",
                )
        y += style.row_gap if style.layout != "minimal" else style.row_gap + 8
        y += 20 * extra_lines
    return y


def render_receipt(
    label: ReceiptLabel, *, display_names: Mapping[Issuer, str] | None = None
) -> bytes:
    """Render one label as an optimized PNG and return its bytes.

    ``display_names`` is the published issuer table; the v1 override exists only so the
    frozen dataset keeps reproducing byte for byte.
    """
    names = display_names or ISSUER_DISPLAY_NAMES
    style = STYLES[label.issuer]
    image = Image.new("RGB", (style.width, style.height), style.page_bg)
    draw = ImageDraw.Draw(image)

    if style.layout in ("card", "boxed", "official", "table"):
        draw.rectangle(
            (
                style.margin - 16,
                style.header_height + 40,
                style.width - style.margin + 16,
                style.height - 96,
            ),
            fill=style.card_bg,
        )

    top = _draw_header(draw, style, names[label.issuer])
    top = _draw_amount(draw, style, label, top)
    bottom = _draw_rows(draw, style, label, top + 24)

    footer_font = load_font(13)
    draw.line(
        (style.margin, style.height - 78, style.width - style.margin, style.height - 78),
        fill=style.muted,
        width=1,
    )
    draw.text(
        (style.width // 2, style.height - 54),
        FOOTER_TEXT,
        font=footer_font,
        fill=style.muted,
        anchor="mm",
    )
    if style.layout == "official" and bottom < style.height - 110:
        draw.rectangle(
            (style.margin, style.height - 132, style.width - style.margin, style.height - 96),
            outline=style.accent,
            width=1,
        )
        draw.text(
            (style.width // 2, style.height - 114),
            "CONSTANCIA VÁLIDA — DOCUMENTO SINTÉTICO",
            font=footer_font,
            fill=style.muted,
            anchor="mm",
        )

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def format_transferred_at(value: datetime) -> str:
    """Render a timestamp the way the synthetic receipts display it (AR local time)."""
    return value.astimezone(AR_TZ).strftime(DATE_FORMAT)
