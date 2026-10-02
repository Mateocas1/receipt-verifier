"""Replay state: every operation id the pipeline has seen, with the receipt it came from.

The registry is what stops a receipt from being approved twice. It only works if it
remembers *every* operation id that was extracted, not just the approved ones: a receipt
routed to `manual_review` is still a receipt that a human will eventually act on, so a
second arrival carrying the same operation id is a replay whether or not the first one was
approved. The live sweep proved the point — with approval-only recording, a replay of a
reviewed receipt looked fresh and was approved.

Each entry keeps the hash of the image that first carried the operation id, so the
validator can tell a resubmission of the same document (byte-identical, reject) from a
different document reusing the number (route to a human).

The registry is deliberately *not* durable: a restart forgets what it saw, which is an
honest limit of this slice (the real ledger owns this state, not the service process).
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict


def receipt_hash(image: bytes) -> str:
    """Stable identity of one receipt image: SHA-256 over the exact bytes received."""
    return hashlib.sha256(image).hexdigest()


class SeenOperationIds:
    """Thread-safe, FIFO-bounded map of operation id to the first receipt hash seen."""

    def __init__(self, max_size: int = 10_000) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self._max_size = max_size
        self._seen: OrderedDict[str, str] = OrderedDict()
        self._lock = threading.Lock()

    def __contains__(self, operation_id: object) -> bool:
        with self._lock:
            return operation_id in self._seen

    def __len__(self) -> int:
        with self._lock:
            return len(self._seen)

    def record(self, operation_id: str, receipt_hash: str) -> None:
        """Remember an operation id; the first hash recorded for it wins.

        Keeping the first hash is what makes the identity check meaningful: if a later,
        different receipt could overwrite it, a replayed original would stop looking like
        the document that was seen first.
        """
        with self._lock:
            if operation_id not in self._seen:
                self._seen[operation_id] = receipt_hash
            self._seen.move_to_end(operation_id)
            while len(self._seen) > self._max_size:
                self._seen.popitem(last=False)

    def hash_for(self, operation_id: str) -> str | None:
        with self._lock:
            return self._seen.get(operation_id)

    def snapshot(self) -> dict[str, str]:
        """Consistent copy for one validation call."""
        with self._lock:
            return dict(self._seen)


__all__ = ["SeenOperationIds", "receipt_hash"]
