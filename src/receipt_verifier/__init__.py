"""Receipt verifier: schema, synthetic dataset, extractors, validator and evaluation harness."""

from receipt_verifier.builders import build_extraction
from receipt_verifier.dataset import Dataset, DatasetBundle, load_dataset, write_dataset
from receipt_verifier.extraction import (
    CRITICAL_FIELD_NAMES,
    FIELD_NAMES,
    ExtractedField,
    ExtractionResult,
    ReceiptExtractor,
)
from receipt_verifier.extractors.dummy import DummyExtractor
from receipt_verifier.harness import EvaluationReport, render_table, run_evaluation
from receipt_verifier.identifiers import (
    cuit_check_digit,
    is_valid_cbu_or_cvu,
    is_valid_cuit,
    parse_amount_text,
    render_amount_ars,
)
from receipt_verifier.metrics import Metrics, SampleOutcome, compute_metrics
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
from receipt_verifier.validate import (
    DEFAULT_INJECTION_PATTERNS,
    ReceiptValidator,
    ValidationPolicy,
    ValidationResult,
)

__all__ = [
    "AR_TZ",
    "CRITICAL_FIELD_NAMES",
    "DEFAULT_INJECTION_PATTERNS",
    "FIELD_NAMES",
    "AdversarialKind",
    "Dataset",
    "DatasetBundle",
    "DatasetManifest",
    "Decision",
    "Destination",
    "DestinationKind",
    "DummyExtractor",
    "EvaluationReport",
    "ExtractedField",
    "ExtractionResult",
    "Issuer",
    "LedgerEntry",
    "Metrics",
    "ReceiptExtractor",
    "ReceiptLabel",
    "ReceiptValidator",
    "RejectReason",
    "SampleOutcome",
    "ValidationPolicy",
    "ValidationResult",
    "build_extraction",
    "compute_metrics",
    "cuit_check_digit",
    "is_valid_cbu_or_cvu",
    "is_valid_cuit",
    "load_dataset",
    "parse_amount_text",
    "render_amount_ars",
    "render_table",
    "run_evaluation",
    "write_dataset",
]
