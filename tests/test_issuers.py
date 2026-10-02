"""Published issuer mapping: the printed name, the code, and who uses which.

The synthetic fixtures cannot print a trademark, but a model (or a person) still needs a
deterministic way to go from what is printed to the issuer code. That is the point of this
table: one published, 1:1 mapping used by the renderer, the OCR reader and the extraction
prompt, so nothing can drift.
"""

from __future__ import annotations

from pathlib import Path

from receipt_verifier.extractors.llm import SYSTEM_PROMPT
from receipt_verifier.extractors.ocr import OcrDocument, OcrLine, identify_issuer
from receipt_verifier.schema import (
    ISSUER_DISPLAY_NAMES,
    LEGACY_DISPLAY_NAMES_V1,
    Issuer,
    display_names_for_version,
    issuer_for_display_name,
)
from receipt_verifier.synthetic.generate import build_dataset


def line(text: str, left: int = 30, top: int = 10) -> OcrLine:
    return OcrLine(
        text=text, left=left, top=top, right=left + 200, bottom=top + 18, confidence=0.95
    )


class TestPublishedTable:
    def test_every_issuer_has_a_display_name(self) -> None:
        assert set(ISSUER_DISPLAY_NAMES) == set(Issuer)

    def test_the_mapping_is_one_to_one(self) -> None:
        names = list(ISSUER_DISPLAY_NAMES.values())
        assert len(names) == len(set(names))

    def test_names_are_stable_and_generic(self) -> None:
        """Pinned on purpose: renaming one would silently change the published mapping."""
        assert ISSUER_DISPLAY_NAMES == {
            Issuer.MP: "Billetera Alfa",
            Issuer.UALA: "Billetera Beta",
            Issuer.BRUBANK: "Banco Digital Gamma",
            Issuer.GALICIA: "Banco Delta",
            Issuer.SANTANDER: "Banco Epsilon",
            Issuer.BNA: "Banco Publico Zeta",
        }

    def test_a_printed_name_maps_back_to_its_code(self) -> None:
        for issuer, name in ISSUER_DISPLAY_NAMES.items():
            assert issuer_for_display_name(name) is issuer

    def test_lookup_ignores_case_and_extra_spaces(self) -> None:
        assert issuer_for_display_name("  billetera   alfa ") is Issuer.MP

    def test_an_unknown_name_has_no_code(self) -> None:
        assert issuer_for_display_name("Comprobante de transferencia") is None

    def test_version_selects_the_published_scheme(self) -> None:
        assert display_names_for_version("v2") == ISSUER_DISPLAY_NAMES
        assert display_names_for_version("v1") == LEGACY_DISPLAY_NAMES_V1

    def test_the_v1_scheme_stays_frozen_and_different(self) -> None:
        """v1 images are committed; its names must not move, and v2 must differ from it."""
        assert LEGACY_DISPLAY_NAMES_V1[Issuer.MP] == "Billetera A"
        assert set(LEGACY_DISPLAY_NAMES_V1) == set(Issuer)
        assert all(
            LEGACY_DISPLAY_NAMES_V1[issuer] != ISSUER_DISPLAY_NAMES[issuer] for issuer in Issuer
        )


class TestRendererUsesTheTable:
    def test_v2_images_carry_the_published_name_and_v1_keeps_its_own(self, tmp_path: Path) -> None:
        v1 = build_dataset(seed=1, version="v1", normal_per_issuer=1)
        v2 = build_dataset(seed=1, version="v2", normal_per_issuer=1)
        # Same labels (the name is presentation only), different bytes.
        assert v1.labels == v2.labels
        assert set(v1.images) == set(v2.images)
        assert all(v1.images[path] != v2.images[path] for path in v1.images)


class TestReaderUsesTheTable:
    def test_a_receipt_is_identified_by_its_printed_name_alone(self) -> None:
        for issuer, name in ISSUER_DISPLAY_NAMES.items():
            document = OcrDocument(lines=(line(name),))
            assert identify_issuer(document.rows()) is issuer

    def test_a_name_plus_body_text_still_identifies_the_issuer(self) -> None:
        document = OcrDocument(
            lines=(line(ISSUER_DISPLAY_NAMES[Issuer.GALICIA], top=10), line("Importe", top=60))
        )
        assert identify_issuer(document.rows()) is Issuer.GALICIA

    def test_a_v1_receipt_without_a_published_name_still_identifies(self) -> None:
        """v1 stays readable: the layout vocabulary remains the fallback."""
        document = OcrDocument(
            lines=(
                line(LEGACY_DISPLAY_NAMES_V1[Issuer.UALA], top=10),
                line("Monto", left=420, top=60),
                line("18.500,31", left=420, top=78),
                line("Total debitado", left=420, top=110),
                line("18.500,31", left=420, top=128),
            )
        )
        assert identify_issuer(document.rows()) is Issuer.UALA


class TestPromptPublishesTheMapping:
    def test_the_extraction_prompt_carries_every_published_name(self) -> None:
        for issuer, name in ISSUER_DISPLAY_NAMES.items():
            assert name in SYSTEM_PROMPT, name
            assert issuer.value in SYSTEM_PROMPT, issuer
