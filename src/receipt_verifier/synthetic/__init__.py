"""Synthetic dataset generation: fake data, layout styles and label rendering."""

from receipt_verifier.synthetic.generate import (
    DEFAULT_GENERATED_AT,
    DEFAULT_SEED,
    DEFAULT_VERSION,
    NORMAL_PER_ISSUER,
    build_dataset,
)
from receipt_verifier.synthetic.render import STYLES, IssuerStyle, render_receipt

__all__ = [
    "DEFAULT_GENERATED_AT",
    "DEFAULT_SEED",
    "DEFAULT_VERSION",
    "NORMAL_PER_ISSUER",
    "STYLES",
    "IssuerStyle",
    "build_dataset",
    "render_receipt",
]
