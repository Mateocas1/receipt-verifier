from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from receipt_verifier.confidence import (
    DEFAULT_SOURCE_QUALITY,
    SourceQuality,
    field_confidences,
    score_extraction,
    score_field,
    to_datetime,
    to_decimal,
    to_destination,
    to_issuer,
)
from receipt_verifier.identifiers import parse_destination_text, render_amount_ars
from receipt_verifier.schema import AR_TZ, Destination, DestinationKind, Issuer

RAW = (
    "Billetera A\nImporte transferido\n$ 2.500,00\nDetalle del importe: $ 2.500,00\n"
    "30/06/2025 18:20\nCamila Gómez\nBanco del Río\nALIAS camila.gomez.ar\n"
    "Lautaro Ojeda\nMP-6E81DA675F9D\nAlquiler julio"
)

DESTINATION = Destination(
    kind=DestinationKind.ALIAS, value="camila.gomez.ar", holder="Lautaro Ojeda"
)
TRANSFERRED_AT = datetime(2025, 6, 30, 18, 20, tzinfo=AR_TZ)


class TestCoercers:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("$ 2.500,00", Decimal("2500.00")),
            ("2500,00", Decimal("2500.00")),
            ("2500.00", Decimal("2500.00")),
            (Decimal("2500"), Decimal("2500.00")),
            (2500, Decimal("2500.00")),
            ("no es un monto", None),
            (None, None),
            ("-5,00", Decimal("-5.00")),
        ],
    )
    def test_to_decimal(self, raw: object, expected: Decimal | None) -> None:
        assert to_decimal(raw) == expected

    def test_to_datetime_accepts_printed_and_iso_forms(self) -> None:
        assert to_datetime("30/06/2025 18:20") == TRANSFERRED_AT
        assert to_datetime("30/06/2025") == datetime(2025, 6, 30, tzinfo=AR_TZ)
        assert to_datetime(TRANSFERRED_AT) == TRANSFERRED_AT
        assert to_datetime("ayer") is None

    def test_naive_datetime_is_anchored_to_argentina(self) -> None:
        naive = datetime.fromisoformat("2025-06-30T18:20")
        assert naive.tzinfo is None
        assert to_datetime(naive).utcoffset() == timedelta(hours=-3)

    def test_to_destination_accepts_models_and_mappings(self) -> None:
        assert to_destination(DESTINATION) == DESTINATION
        mapping = {"kind": "alias", "value": "camila.gomez.ar", "holder": "Lautaro Ojeda"}
        assert to_destination(mapping) == DESTINATION
        assert to_destination({"kind": "cvu", "value": "123", "holder": "x"}) is None
        assert to_destination("camila.gomez.ar") is None

    def test_to_issuer(self) -> None:
        assert to_issuer("MP") is Issuer.MP
        assert to_issuer(Issuer.BNA) is Issuer.BNA
        assert to_issuer("banco raro") is None

    def test_parse_destination_text_round_trip(self) -> None:
        assert parse_destination_text("ALIAS camila.gomez.ar") == ("alias", "camila.gomez.ar")
        assert parse_destination_text("CVU 2852240785731632146398") == (
            "cvu",
            "2852240785731632146398",
        )
        assert parse_destination_text("camila.gomez.ar") == ("alias", "camila.gomez.ar")
        assert parse_destination_text("") is None

    def test_rendered_amount_is_corroboration_ready(self) -> None:
        assert render_amount_ars(Decimal("2500")) == "2.500,00"

    def test_date_corroboration_matches_the_printed_minute(self) -> None:
        _, confidence = score_field(
            "transferred_at", "30/06/2025 18:20", source_quality=SourceQuality(0.9), raw_text=RAW
        )
        assert confidence == 0.9

    def test_date_corroboration_rejects_another_minute(self) -> None:
        _, confidence = score_field(
            "transferred_at", "30/06/2025 19:20", source_quality=SourceQuality(0.9), raw_text=RAW
        )
        assert confidence == pytest.approx(0.9 * 0.75)


class TestScoreField:
    def test_corroborated_value_gets_the_source_prior(self) -> None:
        value, confidence = score_field(
            "amount", "$ 2.500,00", source_quality=SourceQuality(0.9), raw_text=RAW
        )
        assert value == Decimal("2500.00")
        assert confidence == 0.9

    def test_uncorroborated_value_is_discounted(self) -> None:
        value, confidence = score_field(
            "amount", "$ 9.999,00", source_quality=SourceQuality(0.9), raw_text=RAW
        )
        assert value == Decimal("9999.00")
        assert confidence == pytest.approx(0.9 * 0.75)

    def test_format_failure_lowers_confidence(self) -> None:
        _, confidence = score_field(
            "operation_id", "no-es-un-id-largo", source_quality=SourceQuality(0.9), raw_text=RAW
        )
        assert confidence == pytest.approx(0.9 * 0.6 * 0.75)

    def test_unparsable_value_is_dropped(self) -> None:
        value, confidence = score_field(
            "transferred_at", "no es fecha", source_quality=SourceQuality(), raw_text=RAW
        )
        assert value is None
        assert confidence == 0.0

    def test_no_raw_text_means_no_corroboration_penalty(self) -> None:
        _, confidence = score_field(
            "amount", "2.500,00", source_quality=SourceQuality(0.9), raw_text=""
        )
        assert confidence == 0.9

    def test_per_field_source_quality_wins(self) -> None:
        quality = SourceQuality(default=0.9, per_field={"amount": 0.4})
        _, confidence = score_field("amount", "2.500,00", source_quality=quality, raw_text=RAW)
        assert confidence == 0.4
        _, other = score_field("memo", "Alquiler julio", source_quality=quality, raw_text=RAW)
        assert other == 0.9

    def test_source_quality_is_clamped(self) -> None:
        quality = SourceQuality(default=1.0, per_field={"amount": 1.5})
        assert quality.for_field("amount") == 1.0
        assert SourceQuality(default=-1.0).for_field("memo") == 0.0

    def test_unknown_field_is_a_programming_error(self) -> None:
        with pytest.raises(KeyError):
            score_field("nope", "x", source_quality=SourceQuality(), raw_text="")

    def test_destination_with_a_bad_checksum_is_dropped_not_kept(self) -> None:
        broken = {
            "kind": "cvu",
            "value": "2852240785731632146399",
            "holder": "Lautaro Ojeda",
        }
        value, confidence = score_field(
            "destination", broken, source_quality=SourceQuality(0.9), raw_text=RAW
        )
        assert value is None
        assert confidence == 0.0

    def test_digit_substrings_do_not_corroborate_an_amount(self) -> None:
        # "5,00" must not be considered present just because "2.500,00" contains 500.
        _, confidence = score_field(
            "amount", "5,00", source_quality=SourceQuality(0.9), raw_text=RAW
        )
        assert confidence == pytest.approx(0.9 * 0.75)

    def test_format_failure_keeps_the_value_for_the_validator(self) -> None:
        # A parseable but implausible value survives with a lower score: the validator,
        # not the confidence, decides what to do with it.
        value, confidence = score_field(
            "amount", "-5,00", source_quality=SourceQuality(0.9), raw_text=RAW
        )
        assert value == Decimal("-5.00")
        assert confidence == pytest.approx(0.9 * 0.6 * 0.75)

    def test_issuer_is_not_corroborated_against_raw_text(self) -> None:
        _, confidence = score_field(
            "issuer", "mp", source_quality=SourceQuality(0.9), raw_text="nada que ver"
        )
        assert confidence == 0.9


class TestScoreExtraction:
    def test_full_ocr_like_payload(self) -> None:
        result = score_extraction(
            "ocr",
            {
                "amount": "$ 2.500,00",
                "amount_detail": "$ 2.500,00",
                "transferred_at": "30/06/2025 18:20",
                "sender_name": "Camila Gómez",
                "sender_bank": "Banco del Río",
                "destination": DESTINATION,
                "operation_id": "MP-6E81DA675F9D",
                "issuer": "mp",
                "memo": "Alquiler julio",
            },
            raw_text=RAW,
            source_quality=SourceQuality(0.9),
        )
        assert result.extractor == "ocr"
        confidences = field_confidences(result)
        assert all(confidence >= 0.5 for confidence in confidences.values())
        assert result.amount.value == Decimal("2500.00")
        assert result.issuer.value is Issuer.MP
        assert result.transferred_at.value == TRANSFERRED_AT

    def test_empty_payload_yields_zero_confidences(self) -> None:
        result = score_extraction("llm", {}, raw_text="")
        assert result.field_map()["amount"].value is None
        assert all(confidence == 0.0 for confidence in field_confidences(result).values())

    def test_missing_fields_are_not_invented(self) -> None:
        result = score_extraction(
            "llm",
            {"amount": "2500.00", "sender_name": "Camila Gómez"},
            raw_text=RAW,
            source_quality=SourceQuality(DEFAULT_SOURCE_QUALITY),
        )
        assert result.amount.value == Decimal("2500.00")
        assert result.destination.value is None
        assert field_confidences(result)["destination"] == 0.0

    def test_low_quality_source_stays_below_the_default_threshold(self) -> None:
        result = score_extraction(
            "llm",
            {"amount": "2500.00", "destination": DESTINATION, "issuer": "mp"},
            raw_text="",
            source_quality=SourceQuality(0.4),
        )
        assert field_confidences(result)["amount"] == pytest.approx(0.4)
