"""Extractor implementations: dummy replay, local OCR, vision LLM and the cascade."""

from receipt_verifier.extractors.cascade import (
    CASCADE_NAME,
    NO_EXTRACTOR,
    CascadeExhausted,
    CascadeExtractor,
    CascadeOutcome,
    build_cascade,
)
from receipt_verifier.extractors.dummy import DummyExtractor, image_digest
from receipt_verifier.extractors.llm import (
    LLM_EXTRACTOR_NAME,
    HttpLlmTransport,
    LlmConfig,
    LlmPrices,
    LlmResponseError,
    LlmTransportError,
    VisionLlmExtractor,
)
from receipt_verifier.extractors.ocr import (
    OCR_EXTRACTOR_NAME,
    OcrDocument,
    OcrExtractor,
    OcrUnavailable,
    TesserocrEngine,
    TextEngine,
    default_extractor,
)

__all__ = [
    "CASCADE_NAME",
    "LLM_EXTRACTOR_NAME",
    "NO_EXTRACTOR",
    "OCR_EXTRACTOR_NAME",
    "CascadeExhausted",
    "CascadeExtractor",
    "CascadeOutcome",
    "DummyExtractor",
    "HttpLlmTransport",
    "LlmConfig",
    "LlmPrices",
    "LlmResponseError",
    "LlmTransportError",
    "OcrDocument",
    "OcrExtractor",
    "OcrUnavailable",
    "TesserocrEngine",
    "TextEngine",
    "VisionLlmExtractor",
    "build_cascade",
    "default_extractor",
    "image_digest",
]
