"""Bounded in-memory registry of operation ids that were already approved.

It exists so a replayed receipt is routed to a human instead of being approved twice.
It is deliberately *not* durable: a restart forgets what it saw, which is an honest
limit of this slice (the real ledger owns this state, not the service process).
"""

from __future__ import annotations

import threading
from collections import OrderedDict


class SeenOperationIds:
    """Thread-safe FIFO set with a hard size limit."""

    def __init__(self, max_size: int = 10_000) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self._max_size = max_size
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._lock = threading.Lock()

    def __contains__(self, value: str) -> bool:
        with self._lock:
            return value in self._seen

    def __len__(self) -> int:
        with self._lock:
            return len(self._seen)

    def add(self, value: str) -> None:
        with self._lock:
            self._seen[value] = None
            self._seen.move_to_end(value)
            while len(self._seen) > self._max_size:
                self._seen.popitem(last=False)

    def snapshot(self) -> frozenset[str]:
        """Consistent copy for one validation call."""
        with self._lock:
            return frozenset(self._seen)


__all__ = ["SeenOperationIds"]
