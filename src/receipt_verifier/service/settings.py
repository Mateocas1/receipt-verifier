"""Service configuration, read once from the environment.

Secrets live here and nowhere else: the API key is excluded from the dataclass
representation, so it cannot leak through a log line that prints settings.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path

from receipt_verifier.circuit import (
    DEFAULT_FAILURE_THRESHOLD,
    DEFAULT_OPEN_SECONDS,
    CircuitBreaker,
    GuardedExtractor,
)
from receipt_verifier.extraction import ReceiptExtractor
from receipt_verifier.extractors.cascade import CascadeExtractor
from receipt_verifier.extractors.llm import (
    DEFAULT_BASE_URL,
    HttpLlmTransport,
    LlmConfig,
    LlmPrices,
    VisionLlmExtractor,
)
from receipt_verifier.extractors.ocr import (
    DEFAULT_LANGUAGES,
    OcrExtractor,
    OcrUnavailable,
    TesserocrEngine,
    discover_tessdata,
)
from receipt_verifier.validate import ValidationPolicy

DEFAULT_MAX_IMAGE_BYTES = 8 * 1024 * 1024
DEFAULT_EXTRACTOR_TIMEOUT_SECONDS = 20.0


class NoExtractorConfigured(RuntimeError):
    """The environment describes no usable extraction stage."""


def parse_allowed_destinations(raw: str) -> frozenset[str]:
    """Parse ``alias:name``/``cvu:digits`` entries (bare values infer their kind)."""
    keys: set[str] = set()
    for entry in raw.replace(";", ",").split(","):
        token = entry.strip()
        if not token:
            continue
        head, _, tail = token.partition(":")
        if tail and head.lower() in {"alias", "cvu", "cbu"}:
            value = tail.strip().lower() if head.lower() == "alias" else tail.strip()
            keys.add(f"{head.lower()}:{value}")
            continue
        digits = token.replace("-", "").replace(" ", "")
        if digits.isdigit() and len(digits) == 22:
            keys.add(f"cvu:{digits}")
        else:
            keys.add(f"alias:{token.lower()}")
    return frozenset(keys)


@dataclass(frozen=True)
class Settings:
    """Everything the app needs, with safe defaults for a local run."""

    service_token: str = field(default="", repr=False)
    allowed_destinations: frozenset[str] = frozenset()
    max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES
    extractor_timeout_seconds: float = DEFAULT_EXTRACTOR_TIMEOUT_SECONDS
    breaker_failure_threshold: int = DEFAULT_FAILURE_THRESHOLD
    breaker_open_seconds: float = DEFAULT_OPEN_SECONDS
    policy: ValidationPolicy = field(default_factory=ValidationPolicy)
    seen_operation_ids_max: int = 10_000
    llm: LlmConfig = field(default_factory=LlmConfig)
    ocr_languages: str = DEFAULT_LANGUAGES
    ocr_tessdata: Path | None = None
    enable_llm: bool = True
    enable_ocr: bool = True

    @property
    def token_configured(self) -> bool:
        return bool(self.service_token)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        source = env if env is not None else os.environ

        def text(name: str, default: str = "") -> str:
            return source.get(name, default).strip()

        def number(name: str, default: float) -> float:
            raw = text(name)
            if not raw:
                return default
            try:
                return float(raw)
            except ValueError:
                return default

        def count(name: str, default: int) -> int:
            return int(number(name, float(default)))

        def flag(name: str, default: bool) -> bool:
            raw = text(name).lower()
            if not raw:
                return default
            return raw in {"1", "true", "yes", "on"}

        llm = LlmConfig(
            base_url=text("LLM_BASE_URL", DEFAULT_BASE_URL) or DEFAULT_BASE_URL,
            api_key=text("LLM_API_KEY"),
            primary_model=text("VISION_MODEL_PRIMARY"),
            secondary_model=text("VISION_MODEL_SECONDARY"),
            timeout_seconds=number("LLM_TIMEOUT_SECONDS", 30.0),
            prices=LlmPrices(
                input_per_1k=_decimal(source.get("LLM_INPUT_PRICE_PER_1K", "")),
                output_per_1k=_decimal(source.get("LLM_OUTPUT_PRICE_PER_1K", "")),
            ),
        )
        tessdata = text("OCR_TESSDATA")
        return cls(
            service_token=text("RECEIPT_VERIFIER_TOKEN"),
            allowed_destinations=parse_allowed_destinations(
                text("RECEIPT_VERIFIER_ALLOWED_DESTINATIONS")
            ),
            max_image_bytes=count("RECEIPT_VERIFIER_MAX_IMAGE_BYTES", DEFAULT_MAX_IMAGE_BYTES),
            extractor_timeout_seconds=number(
                "EXTRACTOR_TIMEOUT_SECONDS", DEFAULT_EXTRACTOR_TIMEOUT_SECONDS
            ),
            breaker_failure_threshold=count("BREAKER_FAILURE_THRESHOLD", DEFAULT_FAILURE_THRESHOLD),
            breaker_open_seconds=number("BREAKER_OPEN_SECONDS", DEFAULT_OPEN_SECONDS),
            llm=llm,
            ocr_languages=text("OCR_LANGUAGES", DEFAULT_LANGUAGES) or DEFAULT_LANGUAGES,
            ocr_tessdata=Path(tessdata) if tessdata else None,
            enable_llm=flag("RECEIPT_VERIFIER_ENABLE_LLM", True),
            enable_ocr=flag("RECEIPT_VERIFIER_ENABLE_OCR", True),
        )

    def build_extractor(self) -> ReceiptExtractor:
        """Assemble the cascade from the configured stages, skipping unusable ones."""
        stages: list[GuardedExtractor] = []
        if self.enable_llm and self.llm.api_key:
            if self.llm.primary_model:
                stages.append(
                    self._guard(
                        VisionLlmExtractor(
                            model=self.llm.primary_model,
                            transport=HttpLlmTransport(self.llm),
                            timeout_seconds=self.llm.timeout_seconds,
                            prices=self.llm.prices,
                            name="llm-primary",
                        )
                    )
                )
            if self.llm.secondary_model:
                stages.append(
                    self._guard(
                        VisionLlmExtractor(
                            model=self.llm.secondary_model,
                            transport=HttpLlmTransport(self.llm),
                            timeout_seconds=self.llm.timeout_seconds,
                            prices=self.llm.prices,
                            name="llm-secondary",
                        )
                    )
                )
        if self.enable_ocr:
            # No local OCR on this host must not break the request path: keep whatever
            # the LLM stages provide.
            with contextlib.suppress(OcrUnavailable):
                stages.append(self._guard(self._build_ocr()))
        if not stages:
            raise NoExtractorConfigured(
                "no extractor is configured: set LLM_API_KEY with VISION_MODEL_PRIMARY, "
                "or install the OCR extra with Tesseract language data"
            )
        return CascadeExtractor(stages, policy=self.policy)

    def _build_ocr(self) -> OcrExtractor:
        return OcrExtractor(
            TesserocrEngine(
                languages=self.ocr_languages,
                tessdata=self.ocr_tessdata or discover_tessdata(),
            )
        )

    def _guard(self, extractor: ReceiptExtractor) -> GuardedExtractor:
        return GuardedExtractor(
            extractor,
            timeout_seconds=self.extractor_timeout_seconds,
            breaker=CircuitBreaker(
                name=extractor.name,
                failure_threshold=self.breaker_failure_threshold,
                open_seconds=self.breaker_open_seconds,
            ),
        )


def _decimal(raw: str) -> Decimal:
    token = raw.strip()
    if not token:
        return Decimal("0")
    try:
        return Decimal(token)
    except InvalidOperation:
        return Decimal("0")


__all__ = [
    "DEFAULT_EXTRACTOR_TIMEOUT_SECONDS",
    "DEFAULT_MAX_IMAGE_BYTES",
    "NoExtractorConfigured",
    "Settings",
    "parse_allowed_destinations",
]
