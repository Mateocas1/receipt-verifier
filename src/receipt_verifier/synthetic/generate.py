"""Build the synthetic labeled dataset.

A single seed drives a ``random.Random`` instance, so the same seed and the same
reference timestamp always produce byte-identical labels and images (given the same
Pillow version). Samples are emitted in a fixed order: 20 normal receipts per issuer,
then the adversarial set grouped by mutation kind.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from decimal import Decimal

from receipt_verifier.dataset import DatasetBundle
from receipt_verifier.schema import (
    AR_TZ,
    AdversarialKind,
    DatasetManifest,
    Decision,
    Destination,
    Issuer,
    LedgerEntry,
    ReceiptLabel,
    RejectReason,
)
from receipt_verifier.synthetic.fake_data import FakeDataFactory
from receipt_verifier.synthetic.render import render_receipt

DEFAULT_SEED = 20250701
DEFAULT_VERSION = "v1"
DEFAULT_GENERATED_AT = datetime(2025, 7, 1, 9, 0, tzinfo=AR_TZ)
"""Fixed by default so the committed dataset is reproducible without extra flags."""

NORMAL_PER_ISSUER = 20
MAX_RECEIPT_AGE = timedelta(days=7)
STALE_RECEIPT_AGE = timedelta(days=45)

ADVERSARIAL_ORDER: tuple[AdversarialKind, ...] = (
    AdversarialKind.EDITED_AMOUNT,
    AdversarialKind.WRONG_DESTINATION,
    AdversarialKind.DUPLICATE_OPERATION_ID,
    AdversarialKind.INJECTED_INSTRUCTION,
    AdversarialKind.STALE_DATE,
)

REASON_BY_KIND: dict[AdversarialKind, RejectReason] = {
    AdversarialKind.EDITED_AMOUNT: RejectReason.AMOUNT_MISMATCH,
    AdversarialKind.WRONG_DESTINATION: RejectReason.DESTINATION_MISMATCH,
    AdversarialKind.DUPLICATE_OPERATION_ID: RejectReason.DUPLICATE_OPERATION_ID,
    AdversarialKind.INJECTED_INSTRUCTION: RejectReason.PROMPT_INJECTION,
    AdversarialKind.STALE_DATE: RejectReason.STALE_DATE,
}


def _payment_id(index: int) -> str:
    return f"pay-{index:05d}"


def _normal_sample(
    factory: FakeDataFactory,
    *,
    issuer: Issuer,
    index: int,
    evaluation_at: datetime,
    payment_index: int,
) -> ReceiptLabel:
    amount = factory.amount()
    destination = factory.destination()
    transferred_at = evaluation_at - timedelta(minutes=factory.minutes_ago(max_days=6))
    sample_id = f"{issuer.value}-normal-{index:04d}"
    return ReceiptLabel(
        sample_id=sample_id,
        image=f"images/{sample_id}.png",
        issuer=issuer,
        amount=amount,
        amount_detail=amount,
        transferred_at=transferred_at,
        sender_name=factory.person_name(),
        sender_bank=factory.sender_bank(),
        destination=destination,
        operation_id=factory.operation_id(issuer),
        memo=factory.memo(),
        expected_decision=Decision.APPROVE,
        reasons=(),
        adversarial=AdversarialKind.NONE,
        expectation=LedgerEntry(
            payment_id=_payment_id(payment_index),
            amount=amount,
            destination=destination,
            requested_at=transferred_at,
        ),
    )


def _adversarial_sample(
    factory: FakeDataFactory,
    *,
    issuer: Issuer,
    kind: AdversarialKind,
    ordinal: int,
    evaluation_at: datetime,
    payment_index: int,
    normal_samples: tuple[ReceiptLabel, ...],
    borrowed_destination: Destination,
) -> ReceiptLabel:
    amount = factory.amount()
    destination = factory.destination()
    transferred_at = evaluation_at - timedelta(minutes=factory.minutes_ago(max_days=6))
    memo = factory.memo()
    operation_id = factory.operation_id(issuer)
    ledger_amount = amount
    ledger_destination = destination

    if kind is AdversarialKind.EDITED_AMOUNT:
        # The forger raised the headline amount but the detail line kept the original figure.
        amount = Decimal("10") * amount
    elif kind is AdversarialKind.WRONG_DESTINATION:
        # Half of the cases point at a destination the ledger also knows (so an allowlist
        # alone would not save us); the other half point somewhere unknown entirely.
        destination = borrowed_destination if ordinal % 2 == 0 else factory.destination()
    elif kind is AdversarialKind.DUPLICATE_OPERATION_ID:
        operation_id = normal_samples[ordinal % len(normal_samples)].operation_id
    elif kind is AdversarialKind.INJECTED_INSTRUCTION:
        memo = factory.injection_memo()
    elif kind is AdversarialKind.STALE_DATE:
        transferred_at = evaluation_at - STALE_RECEIPT_AGE

    sample_id = f"{issuer.value}-{kind.value}-{ordinal:02d}"
    return ReceiptLabel(
        sample_id=sample_id,
        image=f"images/{sample_id}.png",
        issuer=issuer,
        amount=amount,
        amount_detail=ledger_amount if kind is AdversarialKind.EDITED_AMOUNT else amount,
        transferred_at=transferred_at,
        sender_name=factory.person_name(),
        sender_bank=factory.sender_bank(),
        destination=destination,
        operation_id=operation_id,
        memo=memo,
        expected_decision=Decision.REJECT,
        reasons=(REASON_BY_KIND[kind],),
        adversarial=kind,
        expectation=LedgerEntry(
            payment_id=_payment_id(payment_index),
            amount=ledger_amount,
            destination=ledger_destination,
            requested_at=transferred_at,
        ),
    )


def build_dataset(
    *,
    seed: int = DEFAULT_SEED,
    generated_at: datetime | None = None,
    version: str = DEFAULT_VERSION,
    normal_per_issuer: int = NORMAL_PER_ISSUER,
) -> DatasetBundle:
    """Build the dataset in memory: manifest, labels and rendered PNG bytes."""
    reference = generated_at if generated_at is not None else DEFAULT_GENERATED_AT
    if reference.tzinfo is None:
        raise ValueError("generated_at must be timezone-aware")

    rng = random.Random(seed)
    factory = FakeDataFactory(rng)
    labels: list[ReceiptLabel] = []
    normal_samples: list[ReceiptLabel] = []

    for issuer in Issuer:
        for index in range(1, normal_per_issuer + 1):
            label = _normal_sample(
                factory,
                issuer=issuer,
                index=index,
                evaluation_at=reference,
                payment_index=len(labels) + 1,
            )
            labels.append(label)
            normal_samples.append(label)

    adversarial_labels: list[ReceiptLabel] = []
    for kind in ADVERSARIAL_ORDER:
        for ordinal, issuer in enumerate(Issuer):
            borrowed = normal_samples[
                (ordinal * 5 + len(normal_samples) // 2) % len(normal_samples)
            ]
            label = _adversarial_sample(
                factory,
                issuer=issuer,
                kind=kind,
                ordinal=ordinal,
                evaluation_at=reference,
                payment_index=len(labels) + len(adversarial_labels) + 1,
                normal_samples=tuple(normal_samples),
                borrowed_destination=borrowed.destination,
            )
            adversarial_labels.append(label)

    labels.extend(adversarial_labels)
    images = {label.image: render_receipt(label) for label in labels}

    normal_count = sum(1 for label in labels if label.adversarial is AdversarialKind.NONE)
    adversarial_counts = {kind.value: 0 for kind in ADVERSARIAL_ORDER}
    for label in adversarial_labels:
        adversarial_counts[label.adversarial.value] += 1

    manifest = DatasetManifest(
        name="synthetic",
        version=version,
        seed=seed,
        generated_at=reference,
        evaluation_at=reference,
        max_receipt_age_seconds=int(MAX_RECEIPT_AGE.total_seconds()),
        sample_count=len(labels),
        counts={
            Decision.APPROVE.value: normal_count,
            Decision.REJECT.value: len(adversarial_labels),
        },
        adversarial_counts=adversarial_counts,
    )
    return DatasetBundle(manifest=manifest, labels=tuple(labels), images=images)
