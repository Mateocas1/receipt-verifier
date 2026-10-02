"""Vision-capability probe for an OpenAI-compatible provider.

"Does this model accept images?" is not in the model list, so the only honest answer
is a probe: send one tiny synthetic PNG and see whether the model reads it. This module
keeps the pure parts (parsing the inventory, choosing candidates, building the image,
judging the answer) separate from the network call, so the harness can test them
without a provider.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from time import perf_counter

from PIL import Image, ImageDraw, ImageFont

from receipt_verifier.extractors.llm import (
    SYSTEM_PROMPT,
    LlmTransport,
    LlmTransportError,
    media_type_of,
)

PROBE_TEXT: str = "TOTAL 25.000,00"
"""One unmistakable amount for the model to read back."""

PROBE_MARKER: str = "25000"
"""Digits the answer must contain to count as "this model can see the image"."""

PROBE_PROMPT: str = (
    "Extract the fields from this image. Only the headline amount is printed; every "
    "other field is unreadable. Reply with a single JSON object."
)
"""The probe asks for the real extraction shape, so its verdict measures the real use."""

NON_CHAT_MODELS: dict[str, str] = {
    "rerank": "reranking endpoint, not a chat model",
    "whisper": "audio transcription",
    "kokoro": "text-to-speech",
    "qwen3-embedding": "embeddings",
    "flux-2-klein": "text-to-image generation (image output, not input)",
    "qwen-image-2.1": "text-to-image generation (image output, not input)",
}
"""Listed models that cannot answer a vision chat request, with the reason recorded."""


@dataclass(frozen=True)
class VisionProbe:
    """What one model did with the synthetic probe image."""

    model: str
    vision: bool
    latency_ms: float
    answer: str = ""
    error: str = ""


def parse_model_ids(body: object) -> tuple[str, ...]:
    """Read model ids from an OpenAI-compatible ``/models`` body.

    Accepts ``{"data": [{"id": ...}]}`` and a bare list of ids, ignoring anything
    without a usable id.
    """
    raw: object = body.get("data") if isinstance(body, dict) else body
    if not isinstance(raw, list):
        return ()
    ids: list[str] = []
    for entry in raw:
        if isinstance(entry, str) and entry.strip():
            ids.append(entry.strip())
        elif isinstance(entry, dict):
            value = entry.get("id")
            if isinstance(value, str) and value.strip():
                ids.append(value.strip())
    return tuple(ids)


def chat_candidates(model_ids: Sequence[str]) -> tuple[str, ...]:
    """Models worth probing, in inventory order, minus the documented non-chat ones."""
    return tuple(model_id for model_id in model_ids if model_id not in NON_CHAT_MODELS)


def build_probe_image(text: str = PROBE_TEXT) -> bytes:
    """A small white PNG with one high-contrast line: the cheapest possible probe."""
    image = Image.new("RGB", (520, 160), "white")
    draw = ImageDraw.Draw(image)
    font = _probe_font(44)
    box = draw.textbbox((0, 0), text, font=font)
    position = ((image.width - (box[2] - box[0])) // 2, (image.height - (box[3] - box[1])) // 2)
    draw.text(position, text, fill="black", font=font)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _probe_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has no size argument
        return ImageFont.load_default()


def looks_like_amount(answer: str) -> bool:
    """True when the model transcribed the probe amount, digits or printed form."""
    compact = answer.replace(".", "").replace(",", "").replace(" ", "")
    return PROBE_MARKER in compact


def run_vision_probe(
    *,
    model: str,
    transport: LlmTransport,
    image: bytes,
    timeout_seconds: float,
) -> VisionProbe:
    """Send the probe image once and classify the answer. A failure is a result, not a raise."""
    started = perf_counter()
    try:
        completion = transport.complete(
            model=model,
            system=SYSTEM_PROMPT,
            prompt=PROBE_PROMPT,
            image_base64=base64.b64encode(image).decode("ascii"),
            media_type=media_type_of(image),
            timeout_seconds=timeout_seconds,
        )
    except LlmTransportError as exc:
        return VisionProbe(
            model=model,
            vision=False,
            latency_ms=(perf_counter() - started) * 1000.0,
            error=f"{type(exc).__name__}: {exc}",
        )
    latency_ms = (perf_counter() - started) * 1000.0
    return VisionProbe(
        model=model,
        vision=looks_like_amount(completion.text),
        latency_ms=latency_ms,
        answer=completion.text.strip()[:200],
    )


__all__ = [
    "NON_CHAT_MODELS",
    "PROBE_MARKER",
    "PROBE_PROMPT",
    "PROBE_TEXT",
    "VisionProbe",
    "build_probe_image",
    "chat_candidates",
    "looks_like_amount",
    "parse_model_ids",
    "run_vision_probe",
]
