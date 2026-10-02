"""Generate the synthetic labeled dataset.

uv run python scripts/generate_synthetic.py --out dataset/synthetic/v2
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Final

from receipt_verifier.dataset import write_dataset
from receipt_verifier.schema import AR_TZ
from receipt_verifier.synthetic.generate import (
    DEFAULT_GENERATED_AT,
    DEFAULT_SEED,
    DEFAULT_VERSION,
    NORMAL_PER_ISSUER,
    build_dataset,
)

DEFAULT_OUT: Final = Path("dataset/synthetic/v2")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output dataset directory")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="deterministic RNG seed")
    parser.add_argument(
        "--version",
        default=DEFAULT_VERSION,
        help="dataset version tag written into manifest.json",
    )
    parser.add_argument(
        "--normal-per-issuer",
        type=int,
        default=NORMAL_PER_ISSUER,
        help="number of non-adversarial receipts generated per issuer",
    )
    parser.add_argument(
        "--generated-at",
        default=None,
        help=(
            "ISO-8601 reference timestamp. Defaults to the fixed constant "
            f"{DEFAULT_GENERATED_AT.isoformat()} so the committed dataset is reproducible."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    generated_at = (
        datetime.fromisoformat(args.generated_at).astimezone(AR_TZ) if args.generated_at else None
    )
    bundle = build_dataset(
        seed=args.seed,
        generated_at=generated_at,
        version=args.version,
        normal_per_issuer=args.normal_per_issuer,
    )
    write_dataset(bundle, args.out)

    total_bytes = sum(len(data) for data in bundle.images.values())
    print(f"wrote {bundle.manifest.sample_count} samples to {args.out}")
    print(f"  decisions: {bundle.manifest.counts}")
    print(f"  adversarial: {bundle.manifest.adversarial_counts}")
    print(f"  image bytes: {total_bytes / (1024 * 1024):.2f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
