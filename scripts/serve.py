"""Run the receipt verifier service.

RECEIPT_VERIFIER_TOKEN=dev \\\n
uv run python scripts/serve.py --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
from typing import Final

from receipt_verifier.service.app import SERVICE_VERSION, create_app
from receipt_verifier.service.settings import NoExtractorConfigured, Settings

DEFAULT_HOST: Final = "127.0.0.1"
DEFAULT_PORT: Final = 8000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--reload", action="store_true", help="development reload")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = Settings.from_env()
    if not settings.token_configured:
        print("warning: RECEIPT_VERIFIER_TOKEN is not set; /v1/receipts will answer 503")
    try:
        settings.build_extractor()
    except NoExtractorConfigured as exc:
        print(f"warning: {exc}")
    print(f"receipt-verifier {SERVICE_VERSION} on http://{args.host}:{args.port}")

    import uvicorn

    uvicorn.run(
        create_app(settings),
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
