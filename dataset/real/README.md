# Real receipts slot (`dataset/real/`)

This folder is reserved for **anonymized** real receipts. Nothing else belongs here.

> **Originals are never committed.** Not to this folder, not to another branch, not to a
> PR, not to a gist, not to an issue. If an original ever lands in Git history it must be
> treated as a data leak: rotate the affected identifiers and rewrite history.

## Rules

1. Anonymization happens **locally, before** anything touches the repository.
2. Only files inside `dataset/real/anonymized/` may be committed. `.gitignore` blocks the
   rest of `dataset/real/`, plus PDFs, HEIC files and any `*.original.*` file as defence in
   depth.
3. Every committed sample is a **PNG** plus one line in `labels.jsonl`, following the schema
   in `src/receipt_verifier/schema.py` (`ReceiptLabel`).
4. The `sample_id` must not encode the source file name, the customer, or the date it was
   collected. Use `real-0001`, `real-0002`, …

## What to replace

| Field on the receipt | Replacement rule |
| --- | --- |
| Sender name | Fictional full name from a fixed, non-repeating list. |
| Destination holder | Different fictional full name (never the same as the sender). |
| Destination alias | Fictional alias, 6–20 chars, `[A-Za-z][A-Za-z0-9._-]+`. |
| CVU / CBU | Synthetic 22-digit value with **valid** check digits (`make_cbu_or_cvu`). |
| CUIT / CUIL | Synthetic 11-digit value with a valid check digit (`make_cuit`). |
| `operation_id` | Synthetic token with the same shape/prefix family as the issuer's real one. |
| Free-text memo | Generic business text ("Pago de servicios"); drop anything personal. |
| Institution name/brand | Leave the layout; the synthetic dataset uses neutral placeholder names instead of real brands. |
| Amount | Keep it **as printed** (amounts are not identifying) unless it is unusual enough to be identifying. |
| Date/time | Keep the printed offset (`-03:00`); you may shift it by a whole number of days. |

## How to keep values coherent

- Replace **the same** original value with **the same** fake value across every occurrence
  in the image and in the metadata (alias in the header *and* in the detail line).
- Recompute explicit checksums after replacing CBU/CVU/CUIT digits, otherwise the validator
  rejects a sample for a reason that is an anonymization artifact, not a real finding.
- Never reuse a fake value across two different real receipts: duplicates are adversarial
  signals in this dataset (`duplicate_operation_id`), and accidental reuse would create a
  fake duplicate.
- Preserve image dimensions, orientation and format. Do not re-render, upscale, or
  beautify: text quality is part of what the extractor sees.
- Redact or blur any real logo you cannot remove without changing the layout, and note it in
  the memo of the label file.

## Adding samples

```
dataset/real/anonymized/
├── labels.jsonl          # one ReceiptLabel JSON object per line
└── images/
    ├── real-0001.png
    └── real-0002.png
```

1. Produce the anonymized PNG locally.
2. Append a `ReceiptLabel` line with `"image": "images/real-000N.png"`, the real
   `expected_decision` and `reasons` you verified by hand, and the `expectation` block copied
   from the payment ledger (never from the receipt).
3. Run the harness against the folder:

   ```bash
   uv run python scripts/evaluate.py --dataset dataset/real/anonymized --extractor dummy
   ```

   The dummy extractor reads `labels.jsonl`, so this checks the labels, the validator and the
   harness wiring. Evaluate a real extractor by wiring a new `ReceiptExtractor`
   implementation into `scripts/evaluate.py`.
4. Sanity-check the diff for leaked strings before committing:

   ```bash
   git diff --cached | grep -Ei 'cuit|cbu|cvu|alias|[0-9]{11}|[0-9]{22}' || true
   ```

## Status

No real anonymized sample is committed yet. The owner of the receipts is expected to add
them; until then all evaluation numbers in the README come from synthetic data only.
