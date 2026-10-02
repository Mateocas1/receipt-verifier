"""List the provider's models and probe which ones actually accept an image.

    LLM_API_KEY=... EVAL_RPM=20 uv run python scripts/probe_vision_models.py

The model list is not a capability list: the only honest answer to "does this model read
images?" is to send one. The probe uses a tiny synthetic PNG (`reports/vision-probe.png`)
asking for the amount printed on it, paced through the same rolling-window limiter the
evaluation uses because the provider key's window budget is shared with other agents.

Writes `reports/vision-probe.json` (gitignored) with the inventory, the skipped non-chat
models and one row per probe. Never prints or writes the API key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import httpx

from receipt_verifier.extractors.llm import HttpLlmTransport, build_http_client
from receipt_verifier.ratelimit import limiter_from_env, parse_rpm
from receipt_verifier.service.settings import Settings
from receipt_verifier.vision_probe import (
    NON_CHAT_MODELS,
    VisionProbe,
    build_probe_image,
    chat_candidates,
    parse_model_ids,
    run_vision_probe,
)

DEFAULT_OUTPUT: Final = Path("reports/vision-probe.json")
DEFAULT_IMAGE_PATH: Final = Path("reports/vision-probe.png")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-out", type=Path, default=DEFAULT_IMAGE_PATH)
    parser.add_argument(
        "--models",
        default=None,
        help="comma-separated ids to probe instead of every chat candidate in the inventory",
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="per-request deadline")
    return parser.parse_args(argv)


def fetch_inventory(client: httpx.Client, *, timeout: float) -> tuple[str, ...]:
    """Read ``GET /models``; a failure here is fatal because there is nothing to probe."""
    try:
        response = client.get("/models", timeout=timeout)
        response.raise_for_status()
        body = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise SystemExit(f"could not list models from the provider: {exc}") from exc
    return parse_model_ids(body)


def probe_models(
    *,
    models: tuple[str, ...],
    client: httpx.Client,
    settings: Settings,
    image: bytes,
    timeout: float,
) -> tuple[tuple[VisionProbe, ...], float]:
    """Probe every model in order through one shared limiter (concurrency 1)."""
    limiter = limiter_from_env()
    transport = HttpLlmTransport(settings.llm, client=client, limiter=limiter)
    probes: list[VisionProbe] = []
    for model in models:
        probe = run_vision_probe(
            model=model, transport=transport, image=image, timeout_seconds=timeout
        )
        probes.append(probe)
        verdict = "vision" if probe.vision else "no"
        detail = probe.answer or probe.error
        print(f"{model:<20} {verdict:<8} {probe.latency_ms:8.1f} ms  {detail}")
    return tuple(probes), transport.rate_limit_waits_seconds()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = Settings.from_env()
    if not settings.llm.api_key:
        raise SystemExit("LLM_API_KEY is required to probe the provider")

    image = build_probe_image()
    args.image_out.parent.mkdir(parents=True, exist_ok=True)
    args.image_out.write_bytes(image)

    client = build_http_client(settings.llm)
    try:
        listed = fetch_inventory(client, timeout=args.timeout)
        if args.models:
            candidates = tuple(token.strip() for token in args.models.split(",") if token.strip())
        else:
            candidates = chat_candidates(listed)
        print(f"listed {len(listed)} models; probing {len(candidates)}")
        probes, waits = probe_models(
            models=candidates,
            client=client,
            settings=settings,
            image=image,
            timeout=args.timeout,
        )
    finally:
        client.close()

    skipped = [
        {"model": model, "reason": NON_CHAT_MODELS[model]}
        for model in listed
        if model in NON_CHAT_MODELS
    ]
    report = {
        "probed_at": datetime.now(tz=UTC).isoformat(),
        "base_url": settings.llm.base_url,
        "max_tokens": settings.llm.max_tokens,
        "rpm": parse_rpm(os.environ.get("EVAL_RPM")),
        "probe_image_sha256": hashlib.sha256(image).hexdigest(),
        "listed_models": list(listed),
        "skipped": skipped,
        "probes": [
            {
                "model": probe.model,
                "vision": probe.vision,
                "latency_ms": round(probe.latency_ms, 2),
                "answer": probe.answer,
                "error": probe.error,
            }
            for probe in probes
        ],
        "vision_models": [probe.model for probe in probes if probe.vision],
        "rate_limit_wait_seconds": round(waits, 2),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\njson report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
