"""Receipt verifier: schema, synthetic dataset, extractors, validator and evaluation harness."""
from receipt_verifier.builders import build_extraction
from receipt_verifier.extraction import (
    CRITICAL_FIELD_NAMES,
    FIELD_NAMES,
    ExtractedField,
    ExtractionResult,
    ReceiptExtractor,
)
from receipt_verifier.identifiers import (
    cuit_check_digit,
    is_valid_cbu_or_cvu,
    is_valid_cuit,
    parse_amount_text,
    render_amount_ars,
)
from receipt_verifier.schema import (
    AR_TZ,
    AdversarialKind,
    DatasetManifest,
    Decision,
    Destination,
    DestinationKind,
    Issuer,
    LedgerEntry,
    ReceiptLabel,
    RejectReason,
)

__all__ = [
    "AR_TZ",
    "CRITICAL_FIELD_NAMES",
    "FIELD_NAMES",
    "AdversarialKind",
    "DatasetManifest",
    "Decision",
    "Destination",
    "DestinationKind",
    "ExtractedField",
    "ExtractionResult",
    "Issuer",
    "LedgerEntry",
    "ReceiptExtractor",
    "ReceiptLabel",
    "RejectReason",
    "build_extraction",
    "cuit_check_digit",
    "is_valid_cbu_or_cvu",
    "is_valid_cuit",
    "parse_amount_text",
    "render_amount_ars",
]
