"""Argentine identifier helpers: CUIT and CBU/CVU checksums and money formatting.

All functions here are pure so they can be reused by both the generator (to emit
coherent fake data) and the schema (to validate labels).
"""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Final

CENTS: Final = Decimal("0.01")

# CBU/CVU: two 11-digit blocks. The first block is (bank 3 digits + branch 4 digits)
# plus its check digit; the second block is the 13-digit account plus its check digit.
CBU_BLOCK1_WEIGHTS: Final = (7, 1, 3, 9, 7, 1, 3)
CBU_BLOCK2_WEIGHTS: Final = (3, 9, 7, 1, 3, 9, 7, 1, 3, 9, 7, 1, 3)
CUIT_WEIGHTS: Final = (5, 4, 3, 2, 7, 6, 5, 4, 3, 2)

_ALIAS_RE: Final = re.compile(r"^[A-Za-z][A-Za-z0-9._-]{5,19}$")
_DIGITS_RE: Final = re.compile(r"^\d+$")


def _weighted_sum(digits: str, weights: tuple[int, ...]) -> int:
    return sum(int(digit) * weight for digit, weight in zip(digits, weights, strict=True))


def _mod11_check_digit(digits: str, weights: tuple[int, ...]) -> int:
    """CBU/CVU check digit: 10 - (weighted sum mod 10), and 0 when the result is 10."""
    remainder = _weighted_sum(digits, weights) % 10
    check = 10 - remainder
    return 0 if check == 10 else check


def cbu_check_digits(bank_branch: str, account: str) -> tuple[int, int]:
    """Return the two CBU/CVU check digits for a 7-digit bank+branch and 13-digit account."""
    if len(bank_branch) != 7 or not _DIGITS_RE.match(bank_branch):
        raise ValueError("bank_branch must be exactly 7 digits")
    if len(account) != 13 or not _DIGITS_RE.match(account):
        raise ValueError("account must be exactly 13 digits")
    return (
        _mod11_check_digit(bank_branch, CBU_BLOCK1_WEIGHTS),
        _mod11_check_digit(account, CBU_BLOCK2_WEIGHTS),
    )


def is_valid_cbu_or_cvu(value: str) -> bool:
    """True when ``value`` is a 22-digit CBU or CVU with valid check digits.

    The all-zero value satisfies the checksum but is not a real account identifier,
    so it is rejected as degenerate.
    """
    if len(value) != 22 or not _DIGITS_RE.match(value):
        return False
    if value == "0" * 22:
        return False
    expected_1, expected_2 = cbu_check_digits(value[:7], value[8:21])
    return int(value[7]) == expected_1 and int(value[21]) == expected_2


def make_cbu_or_cvu(bank_branch: str, account: str) -> str:
    """Build a 22-digit CBU/CVU from 7-digit bank+branch and 13-digit account blocks."""
    check_1, check_2 = cbu_check_digits(bank_branch, account)
    return f"{bank_branch}{check_1}{account}{check_2}"


def cuit_check_digit(prefix: str) -> int:
    """Modulo-11 check digit for a 10-digit CUIT prefix.

    Person types are ``20``/``23``/``24``/``27`` and company types ``30``/``33``/``34``.
    """
    if len(prefix) != 10 or not _DIGITS_RE.match(prefix):
        raise ValueError("CUIT prefix must be exactly 10 digits")
    remainder = _weighted_sum(prefix, CUIT_WEIGHTS) % 11
    check = 11 - remainder
    if check == 11:
        return 0
    if check == 10:
        # The canonical algorithm maps 10 to 9 for the person-type prefixes we generate.
        return 9
    return check


def make_cuit(prefix: str, body: str) -> str:
    """Build an 11-digit CUIT with a valid check digit."""
    if len(body) != 8 or not _DIGITS_RE.match(body):
        raise ValueError("CUIT body must be exactly 8 digits")
    base = f"{prefix}{body}"
    return f"{base}{cuit_check_digit(base)}"


def is_valid_cuit(value: str) -> bool:
    """True when ``value`` is an 11-digit CUIT with a valid check digit."""
    value = value.replace("-", "")
    if len(value) != 11 or not _DIGITS_RE.match(value):
        return False
    return cuit_check_digit(value[:10]) == int(value[10])


def is_valid_alias(value: str) -> bool:
    """True when ``value`` looks like a bank alias: 6-20 chars, letter-first, no spaces."""
    return bool(_ALIAS_RE.match(value)) and not value.isdigit()


def quantize_amount(value: Decimal) -> Decimal:
    """Round a money value to two decimals with banker-free half-up rounding."""
    return value.quantize(CENTS, rounding=ROUND_HALF_UP)


def render_amount_ars(value: Decimal) -> str:
    """Render a Decimal as an Argentine-formatted amount, e.g. ``25.000,00``."""
    quantized = quantize_amount(value)
    integer, _, fraction = f"{quantized:.2f}".partition(".")
    negative = integer.startswith("-")
    integer = integer.lstrip("-")
    grouped = f"{int(integer):,}".replace(",", ".")
    return f"{'-' if negative else ''}{grouped},{fraction}"


_AMOUNT_TOKEN_RE: Final = re.compile(r"^-?\d{1,3}(?:\.\d{3})*(?:,\d{1,2})?$|^-?\d+(?:,\d{1,2})?$")

_DESTINATION_KINDS: Final = frozenset({"alias", "cvu", "cbu"})


def parse_destination_text(text: str) -> tuple[str, str] | None:
    """Parse a destination as printed on a receipt: ``ALIAS name`` / ``CVU 2852...``.

    Also accepts a bare alias or a bare 22-digit CBU/CVU. Returns
    ``(kind_value, value)`` (``kind_value`` in ``alias``/``cvu``/``cbu``) or ``None`` when
    the text holds no destination. Kept free of schema imports so the schema itself can
    use this module.
    """
    cleaned = text.strip().strip(":").strip()
    if not cleaned:
        return None
    head, _, tail = cleaned.partition(" ")
    kind = head.strip().lower()
    if kind in _DESTINATION_KINDS and tail.strip():
        # OCR happily splits "2852 240785731632146398" or "camila .ojeda.pago".
        return kind, "".join(tail.split()).strip(",.;")
    token = "".join(cleaned.split()).strip(",.;")
    if is_valid_cbu_or_cvu(token):
        return "cvu", token
    return ("alias", token) if is_valid_alias(token) else None


def parse_amount_text(text: str) -> Decimal | None:
    """Parse an Argentine-formatted amount such as ``$ 25.000,00`` back into a Decimal.

    Returns ``None`` when the text does not contain a single parseable amount.
    """
    cleaned = text.strip().replace("$", "").replace("ARS", "").replace("\u00a0", " ").strip()
    cleaned = cleaned.replace(" ", "")
    if not cleaned or not _AMOUNT_TOKEN_RE.match(cleaned):
        return None
    normalized = cleaned.replace(".", "").replace(",", ".")
    try:
        return quantize_amount(Decimal(normalized))
    except InvalidOperation:
        return None
