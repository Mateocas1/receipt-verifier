"""Image intake rules: size, real type by magic bytes, base64 decoding.

The declared ``Content-Type`` is never trusted: what matters is what the bytes are, and
what happens to them is nothing at all — everything stays in memory.
"""

from __future__ import annotations

import base64
import binascii
import re

MAX_IMAGE_BYTES = 8 * 1024 * 1024

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"
WEBP_RIFF = b"RIFF"
WEBP_TAG = b"WEBP"

SUPPORTED_MEDIA_TYPES = ("image/png", "image/jpeg", "image/webp")

_DATA_URL_RE = re.compile(r"^data:(?P<media>[a-z/+.-]+)?;?base64,", re.IGNORECASE)


class ImageRejected(ValueError):
    """The payload is not an image this service accepts."""

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class ImageTooLarge(ImageRejected):
    def __init__(self, size: int, limit: int) -> None:
        super().__init__(f"image is {size} bytes, limit is {limit}", status_code=413)
        self.size = size
        self.limit = limit


class UnsupportedImageType(ImageRejected):
    def __init__(self, media_type: str | None) -> None:
        super().__init__(
            f"unsupported image type {media_type or 'unknown'}; "
            f"expected one of {', '.join(SUPPORTED_MEDIA_TYPES)}",
            status_code=415,
        )
        self.media_type = media_type


class MalformedImagePayload(ImageRejected):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=400)


def sniff_media_type(data: bytes) -> str | None:
    """Real media type from the first bytes, or ``None`` when unsupported."""
    if data.startswith(PNG_MAGIC):
        return "image/png"
    if data.startswith(JPEG_MAGIC):
        return "image/jpeg"
    if data[:4] == WEBP_RIFF and data[8:12] == WEBP_TAG:
        return "image/webp"
    return None


def validate_image(data: bytes, *, max_bytes: int = MAX_IMAGE_BYTES) -> str:
    """Return the media type of an acceptable image, or raise a typed rejection."""
    if not data:
        raise MalformedImagePayload("image payload is empty")
    if len(data) > max_bytes:
        raise ImageTooLarge(len(data), max_bytes)
    media_type = sniff_media_type(data)
    if media_type is None:
        raise UnsupportedImageType(None)
    return media_type


def decode_base64_image(payload: str, *, max_bytes: int = MAX_IMAGE_BYTES) -> bytes:
    """Decode a base64 (optionally ``data:`` prefixed) image, bounded by ``max_bytes``.

    The size limit is applied to the encoded length first, so an oversized body is
    rejected before it is ever expanded in memory.
    """
    token = _DATA_URL_RE.sub("", payload.strip())
    if not token:
        raise MalformedImagePayload("image_base64 is empty")
    encoded_limit = (max_bytes // 3 + 1) * 4
    if len(token) > encoded_limit:
        raise ImageTooLarge(len(token) * 3 // 4, max_bytes)
    compact = re.sub(r"\s+", "", token)
    try:
        data = base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise MalformedImagePayload(f"image_base64 is not valid base64: {exc}") from exc
    if not data:
        raise MalformedImagePayload("image_base64 decoded to nothing")
    return data


__all__ = [
    "MAX_IMAGE_BYTES",
    "SUPPORTED_MEDIA_TYPES",
    "ImageRejected",
    "ImageTooLarge",
    "MalformedImagePayload",
    "UnsupportedImageType",
    "decode_base64_image",
    "sniff_media_type",
    "validate_image",
]
