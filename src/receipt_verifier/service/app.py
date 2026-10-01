"""FastAPI application: ``POST /v1/receipts`` and ``GET /healthz``.

Design constraints that show up directly in this module:

* the bearer token is compared in constant time, and a service without a configured
  token answers 503 instead of pretending to be protected;
* images are read into memory, validated by magic bytes and never written or logged;
* the verdict comes from the deterministic validator. The HTTP layer only maps a
  validation result onto a response and never overrides it.
"""

from __future__ import annotations

import hmac
from datetime import datetime
from time import perf_counter
from typing import Annotated, Final, Protocol

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status

from receipt_verifier.circuit import CircuitStatus
from receipt_verifier.extraction import ReceiptExtractor
from receipt_verifier.extractors.cascade import CascadeExtractor
from receipt_verifier.schema import AR_TZ, LedgerEntry, VerdictReason
from receipt_verifier.service.imaging import (
    ImageRejected,
    decode_base64_image,
    validate_image,
)
from receipt_verifier.service.registry import SeenOperationIds
from receipt_verifier.service.schemas import (
    PaymentInput,
    ReceiptRequest,
    ReceiptResponse,
    response_from,
)
from receipt_verifier.service.settings import Settings
from receipt_verifier.validate import ReceiptValidator

SERVICE_NAME: Final = "receipt-verifier"


class FormReader(Protocol):
    """The slice of a parsed multipart form this module needs.

    Declared with a single argument on purpose: Starlette's ``FormData.get`` is generic
    over the default value, which does not unify with a plain protocol signature.
    """

    def get(self, key: str) -> object: ...


SERVICE_VERSION: Final = "0.2.0"


def create_app(
    settings: Settings | None = None,
    extractor: ReceiptExtractor | None = None,
) -> FastAPI:
    """Build the app. Tests inject a settings object and a stub extractor."""
    resolved = settings or Settings.from_env()
    app = FastAPI(
        title=SERVICE_NAME,
        version=SERVICE_VERSION,
        description=(
            "Verifica comprobantes de transferencia argentinos: extracción en cascada "
            "(LLM visión -> OCR local) y decisión determinista en código."
        ),
    )
    app.state.settings = resolved
    app.state.extractor = extractor
    app.state.seen = SeenOperationIds(max_size=resolved.seen_operation_ids_max)
    app.state.validator = ReceiptValidator(resolved.allowed_destinations, resolved.policy)

    @app.get("/healthz")
    def healthz() -> dict[str, object]:
        current = app.state.extractor
        statuses: list[CircuitStatus] = (
            list(current.statuses()) if isinstance(current, CascadeExtractor) else []
        )
        return {
            "status": "ok",
            "service": SERVICE_NAME,
            "version": SERVICE_VERSION,
            "token_configured": resolved.token_configured,
            "allowed_destinations": len(resolved.allowed_destinations),
            "extractors": [
                {"name": item.name, "state": item.state.value, "failures": item.failures}
                for item in statuses
            ],
        }

    def require_token(
        request: Request,
        authorization: Annotated[str | None, Header()] = None,
    ) -> None:
        """Bearer auth with a constant-time comparison and no default token."""
        active: Settings = request.app.state.settings
        if not active.token_configured:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="service token is not configured",
            )
        scheme, _, token = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="missing bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if not hmac.compare_digest(token.strip().encode(), active.service_token.encode()):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    def extractor_for_request(request: Request) -> ReceiptExtractor:
        existing: ReceiptExtractor | None = request.app.state.extractor
        if existing is not None:
            return existing
        try:
            built = resolved.build_extractor()
        except Exception as exc:  # missing credentials, missing OCR data, ...
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"no extractor available: {exc}",
            ) from exc
        request.app.state.extractor = built
        return built

    def payment_from_form(form: FormReader) -> PaymentInput | None:
        amount = _field(form, "payment_amount")
        destination = _field(form, "payment_destination")
        if not amount or not destination:
            return None
        try:
            return PaymentInput.model_validate(
                {
                    "payment_id": _field(form, "payment_id") or "unverified",
                    "amount": amount,
                    "destination": {
                        "kind": _field(form, "payment_destination_kind") or "alias",
                        "value": destination,
                    },
                }
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"invalid payment fields: {exc}") from exc

    async def read_payload(request: Request) -> tuple[bytes, PaymentInput | None]:
        """Accept multipart ``file`` or a JSON body with ``image_base64``."""
        content_type = (request.headers.get("content-type") or "").lower()
        if content_type.startswith("multipart/form-data"):
            form = await request.form()
            upload = form.get("file")
            if upload is None or isinstance(upload, str):
                raise HTTPException(status_code=400, detail="multipart field 'file' is required")
            return await upload.read(), payment_from_form(form)
        if content_type.startswith("application/json"):
            try:
                payload = ReceiptRequest.model_validate(await request.json())
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"invalid JSON body: {exc}") from exc
            image = decode_base64_image(payload.image_base64, max_bytes=resolved.max_image_bytes)
            return image, payload.payment
        raise HTTPException(
            status_code=415,
            detail="send multipart/form-data with 'file' or application/json with 'image_base64'",
        )

    def _field(form: FormReader, key: str) -> str | None:
        value = form.get(key)
        if value is None or isinstance(value, str):
            return value
        return None  # an uploaded file where a payment field was expected

    @app.post(
        "/v1/receipts",
        dependencies=[Depends(require_token)],
        response_model=ReceiptResponse,
        summary="Extract and verify one receipt image",
    )
    async def verify_receipt(request: Request) -> ReceiptResponse:
        started = perf_counter()
        try:
            image, payment = await read_payload(request)
            media_type = validate_image(image, max_bytes=resolved.max_image_bytes)
        except ImageRejected as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

        extractor = extractor_for_request(request)
        extraction = extractor.extract(image)

        now = datetime.now(tz=AR_TZ)
        expectation: LedgerEntry | None = payment.to_ledger_entry(now=now) if payment else None
        seen: SeenOperationIds = request.app.state.seen
        validation = request.app.state.validator.validate(
            extraction,
            expectation,
            now=now,
            seen_operation_ids=seen.snapshot(),
        )
        reasons = list(validation.reasons)
        if extraction.extractor == "none" and VerdictReason.EXTRACTION_FAILED not in reasons:
            # Every stage failed: say so explicitly, on top of the missing-field reason.
            reasons.insert(0, VerdictReason.EXTRACTION_FAILED)
        if validation.approved and extraction.operation_id.value:
            seen.add(extraction.operation_id.value)

        attempted: tuple[str, ...] = ()
        if isinstance(extractor, CascadeExtractor) and extractor.last_outcome is not None:
            attempted = extractor.last_outcome.attempts
        return response_from(
            extraction,
            decision=validation.decision,
            reasons=tuple(reasons),
            checks=dict(validation.checks),
            latency_ms=(perf_counter() - started) * 1000.0,
            media_type=media_type,
            attempted=attempted,
        )

    return app


__all__ = ["SERVICE_NAME", "SERVICE_VERSION", "create_app"]
