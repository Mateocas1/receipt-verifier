from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from receipt_verifier.schema import (
    AR_TZ,
    AdversarialKind,
    Decision,
    Destination,
    DestinationKind,
    LedgerEntry,
    ReceiptLabel,
    RejectReason,
)
from tests.helpers import ALIAS_DESTINATION, NOW, make_label


class TestDestination:
    def test_alias_is_case_insensitive_for_comparison(self) -> None:
        upper = Destination(kind=DestinationKind.ALIAS, value="Camila.Gomez.AR", holder="C G")
        lower = Destination(kind=DestinationKind.ALIAS, value="camila.gomez.ar", holder="C G")
        assert upper.key == lower.key

    @pytest.mark.parametrize(
        "value",
        ["corta", "1camila.gomez", "camila gomez", "2852240785731632146397"],
    )
    def test_invalid_alias_is_rejected(self, value: str) -> None:
        with pytest.raises(ValidationError):
            Destination(kind=DestinationKind.ALIAS, value=value, holder="Titular")

    def test_cvu_with_wrong_check_digits_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="check digits"):
            Destination(
                kind=DestinationKind.CVU,
                value="2852240785731632146399",
                holder="Titular",
            )


class TestReceiptLabel:
    def test_minimal_valid_label(self) -> None:
        label = make_label()
        assert label.expected_decision is Decision.APPROVE
        assert label.amount == Decimal("2500.00")
        assert label.amount_detail == label.amount

    def test_naive_datetime_is_rejected(self) -> None:
        naive = datetime.fromisoformat("2025-07-01T09:00")
        assert naive.tzinfo is None
        with pytest.raises(ValidationError, match="timezone-aware"):
            make_label(transferred_at=naive)

    @pytest.mark.parametrize("offset", [timedelta(hours=2), timedelta(0)])
    def test_non_argentine_offset_is_rejected(self, offset: timedelta) -> None:
        moment = datetime(2025, 7, 1, 9, 0, tzinfo=timezone(offset))
        with pytest.raises(ValidationError, match="America/Argentina/Buenos_Aires"):
            make_label(transferred_at=moment)

    def test_argentina_offset_is_accepted(self) -> None:
        label = make_label(transferred_at=datetime(2025, 7, 1, 9, 0, tzinfo=AR_TZ))
        assert label.transferred_at.utcoffset() == timedelta(hours=-3)

    def test_approve_with_reasons_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="approved sample cannot"):
            make_label(reasons=(RejectReason.AMOUNT_MISMATCH,))

    def test_reject_without_reasons_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must carry at least one reason"):
            make_label(
                expected_decision=Decision.REJECT,
                adversarial=AdversarialKind.STALE_DATE,
            )

    def test_adversarial_mutation_must_be_rejected(self) -> None:
        with pytest.raises(ValidationError, match="adversarial sample must be rejected"):
            make_label(adversarial=AdversarialKind.EDITED_AMOUNT)

    def test_amounts_are_quantized_to_cents(self) -> None:
        label = make_label(amount=Decimal("1000"), amount_detail=Decimal("1000"))
        assert label.amount == Decimal("1000.00")
        assert str(label.amount) == "1000.00"

    def test_non_positive_amount_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_label(amount=Decimal("0"))

    def test_absolute_image_path_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="relative"):
            ReceiptLabel(
                **{
                    **make_label().model_dump(),
                    "image": "/etc/passwd.png",
                }
            )

    def test_non_png_image_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"\.png"):
            ReceiptLabel(**{**make_label().model_dump(), "image": "images/x.jpg"})

    def test_critical_fields_cover_the_expected_set(self) -> None:
        assert set(make_label().critical_fields) == {
            "amount",
            "transferred_at",
            "sender_name",
            "destination",
            "operation_id",
            "issuer",
        }


class TestLedgerEntry:
    def test_ledger_accepts_a_different_amount_than_the_receipt(self) -> None:
        ledger = LedgerEntry(
            payment_id="pay-1",
            amount=Decimal("250.00"),
            destination=ALIAS_DESTINATION,
            requested_at=NOW,
        )
        label = make_label(amount=Decimal("2500.00"), expectation=ledger)
        assert label.amount != label.expectation.amount

    def test_adversarial_kind_round_trips(self) -> None:
        label = make_label(
            expected_decision=Decision.REJECT,
            reasons=(RejectReason.STALE_DATE,),
            adversarial=AdversarialKind.STALE_DATE,
        )
        assert label.adversarial is AdversarialKind.STALE_DATE
