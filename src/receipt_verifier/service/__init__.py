"""HTTP service: configuration, image intake, API models and the FastAPI app."""

from receipt_verifier.replay import SeenOperationIds
from receipt_verifier.service.app import SERVICE_NAME, SERVICE_VERSION, create_app
from receipt_verifier.service.imaging import (
    SUPPORTED_MEDIA_TYPES,
    ImageRejected,
    ImageTooLarge,
    MalformedImagePayload,
    UnsupportedImageType,
    decode_base64_image,
    sniff_media_type,
    validate_image,
)
from receipt_verifier.service.schemas import (
    DestinationInput,
    PaymentInput,
    ReceiptRequest,
    ReceiptResponse,
)
from receipt_verifier.service.settings import (
    NoExtractorConfigured,
    Settings,
    parse_allowed_destinations,
)

__all__ = [
    "SERVICE_NAME",
    "SERVICE_VERSION",
    "SUPPORTED_MEDIA_TYPES",
    "DestinationInput",
    "ImageRejected",
    "ImageTooLarge",
    "MalformedImagePayload",
    "NoExtractorConfigured",
    "PaymentInput",
    "ReceiptRequest",
    "ReceiptResponse",
    "SeenOperationIds",
    "Settings",
    "UnsupportedImageType",
    "create_app",
    "decode_base64_image",
    "parse_allowed_destinations",
    "sniff_media_type",
    "validate_image",
]
