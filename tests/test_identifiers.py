from decimal import Decimal

import pytest

from receipt_verifier.identifiers import (
    cbu_check_digits,
    cuit_check_digit,
    is_valid_alias,
    is_valid_cbu_or_cvu,
    is_valid_cuit,
    make_cbu_or_cvu,
    make_cuit,
    parse_amount_text,
    quantize_amount,
    render_amount_ars,
)


class TestCuit:
    def test_generated_cuit_is_valid(self) -> None:
        cuit = make_cuit("20", "12345678")
        assert len(cuit) == 11
        assert is_valid_cuit(cuit)

    def test_human_readable_dashes_are_accepted(self) -> None:
        cuit = make_cuit("27", "87654321")
        formatted = f"{cuit[:2]}-{cuit[2:10]}-{cuit[10]}"
        assert is_valid_cuit(formatted)

    def test_tampered_check_digit_is_rejected(self) -> None:
        cuit = make_cuit("30", "71234567")
        wrong = f"{cuit[:10]}{(int(cuit[10]) + 1) % 10}"
        assert not is_valid_cuit(wrong)

    @pytest.mark.parametrize("value", ["", "123", "abcdefghijk", "2012345678"])
    def test_malformed_values_are_rejected(self, value: str) -> None:
        assert not is_valid_cuit(value)

    def test_check_digit_requires_ten_digits(self) -> None:
        with pytest.raises(ValueError, match="10 digits"):
            cuit_check_digit("123")


class TestCbu:
    def test_generated_cbu_is_valid(self) -> None:
        cbu = make_cbu_or_cvu("0070001", "0000000012345")
        assert len(cbu) == 22
        assert is_valid_cbu_or_cvu(cbu)

    def test_tampered_digit_is_rejected(self) -> None:
        cbu = make_cbu_or_cvu("0140002", "0000000098765")
        broken = f"{cbu[:5]}9{cbu[6:]}"
        assert not is_valid_cbu_or_cvu(broken)

    def test_check_digits_are_stable(self) -> None:
        assert cbu_check_digits("0070001", "0000000012345") == (6, 9)

    @pytest.mark.parametrize("value", ["", "0070001", "0" * 22, "285224078573163214639a"])
    def test_malformed_values_are_rejected(self, value: str) -> None:
        assert not is_valid_cbu_or_cvu(value)

    def test_block_lengths_are_enforced(self) -> None:
        with pytest.raises(ValueError, match="7 digits"):
            cbu_check_digits("00700", "0000000012345")
        with pytest.raises(ValueError, match="13 digits"):
            cbu_check_digits("0070001", "12345")


class TestAlias:
    @pytest.mark.parametrize(
        "value",
        ["camila.gomez.ar", "lautaro_ojeda.07", "mp.pagos-2024"],
    )
    def test_valid_aliases(self, value: str) -> None:
        assert is_valid_alias(value)

    @pytest.mark.parametrize(
        "value",
        ["", "abc", "1camila.gomez", "camila gomez", "2852240785731632146398", "a" * 21],
    )
    def test_invalid_aliases(self, value: str) -> None:
        assert not is_valid_alias(value)


class TestAmounts:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (Decimal("2500"), "2.500,00"),
            (Decimal("2500.5"), "2.500,50"),
            (Decimal("1234567.89"), "1.234.567,89"),
            (Decimal("0.01"), "0,01"),
        ],
    )
    def test_render_amount_ars(self, value: Decimal, expected: str) -> None:
        assert render_amount_ars(value) == expected

    @pytest.mark.parametrize(
        "text",
        ["$ 25.000,00", "25000,00", "ARS 25.000,00", " 25.000,00 "],
    )
    def test_parse_round_trip(self, text: str) -> None:
        assert parse_amount_text(text) == Decimal("25000.00")

    @pytest.mark.parametrize("text", ["", "abc", "25.000,00 extra", "25,000.00"])
    def test_unparseable_text_returns_none(self, text: str) -> None:
        assert parse_amount_text(text) is None

    def test_quantize_uses_half_up(self) -> None:
        assert quantize_amount(Decimal("3.005")) == Decimal("3.01")
        assert quantize_amount(Decimal("2.004")) == Decimal("2.00")
