"""Resilience primitives: circuit breaker and per-call timeout.

Both exist so that one sick extractor cannot take the service down or make it hang: a
broken engine is skipped for a while, and every call has a hard deadline.

A timed-out call is *abandoned*, not killed — Python cannot interrupt a thread that is
stuck inside a C extension (an OCR engine, for instance). The worker thread is left to
finish in the background while the caller moves on with the rest of the cascade.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from enum import StrEnum
from time import monotonic
from typing import Final

from receipt_verifier.extraction import ExtractionResult, ReceiptExtractor

DEFAULT_FAILURE_THRESHOLD: Final = 3
DEFAULT_OPEN_SECONDS: Final = 30.0
DEFAULT_TIMEOUT_SECONDS: Final = 20.0


class CircuitState(StrEnum):
    """Breaker state: ``closed`` passes traffic, ``open`` refuses, ``half_open`` probes."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """Raised when a call is refused because the breaker is open."""

    def __init__(self, name: str, retry_in: float) -> None:
        super().__init__(f"circuit {name!r} is open, retry in {retry_in:.1f}s")
        self.name = name
        self.retry_in = retry_in


class ExtractorTimeout(RuntimeError):
    """Raised when an extractor call exceeds its deadline."""

    def __init__(self, name: str, timeout_seconds: float) -> None:
        super().__init__(f"extractor {name!r} exceeded {timeout_seconds:.1f}s")
        self.name = name
        self.timeout_seconds = timeout_seconds


@dataclass(frozen=True)
class CircuitStatus:
    """Snapshot used by ``/healthz`` and the cascade report."""

    name: str
    state: CircuitState
    failures: int
    failure_threshold: int
    retry_in_seconds: float


@dataclass
class CircuitBreaker:
    """Counts consecutive failures and refuses traffic while open."""

    name: str
    failure_threshold: int = DEFAULT_FAILURE_THRESHOLD
    open_seconds: float = DEFAULT_OPEN_SECONDS
    clock: Callable[[], float] = monotonic
    failures: int = 0
    state: CircuitState = CircuitState.CLOSED
    opened_at: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if self.open_seconds < 0:
            raise ValueError("open_seconds must be >= 0")

    def _retry_in(self, now: float) -> float:
        return max(0.0, self.opened_at + self.open_seconds - now)

    def allow(self) -> bool:
        """True when a call may proceed (also performs the half-open transition).

        While half-open only one probe may be in flight; the probe window is the same
        length as the open window, so a probe that never reports back does not wedge the
        breaker shut forever.
        """
        with self._lock:
            if self.state is CircuitState.CLOSED:
                return True
            now = self.clock()
            if self._retry_in(now) > 0:
                return False
            self.state = CircuitState.HALF_OPEN
            self.opened_at = now
            return True

    def record_success(self) -> None:
        with self._lock:
            self.failures = 0
            self.state = CircuitState.CLOSED
            self.opened_at = 0.0

    def record_failure(self) -> None:
        with self._lock:
            self.failures += 1
            if self.state is CircuitState.HALF_OPEN or self.failures >= self.failure_threshold:
                self.state = CircuitState.OPEN
                self.opened_at = self.clock()

    def status(self) -> CircuitStatus:
        with self._lock:
            now = self.clock()
            retry_in = self._retry_in(now) if self.state is CircuitState.OPEN else 0.0
            return CircuitStatus(
                name=self.name,
                state=self.state,
                failures=self.failures,
                failure_threshold=self.failure_threshold,
                retry_in_seconds=retry_in,
            )


_EXECUTOR: ThreadPoolExecutor | None = None
_EXECUTOR_LOCK = threading.Lock()


def _executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    with _EXECUTOR_LOCK:
        if _EXECUTOR is None:
            _EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="extractor")
        return _EXECUTOR


def shutdown_executor() -> None:
    """Release the shared worker pool (used by tests and graceful shutdown)."""
    global _EXECUTOR
    with _EXECUTOR_LOCK:
        if _EXECUTOR is not None:
            _EXECUTOR.shutdown(wait=False, cancel_futures=True)
            _EXECUTOR = None


def run_with_timeout[T](name: str, func: Callable[[], T], timeout_seconds: float) -> T:
    """Run ``func`` on a worker thread and give up after ``timeout_seconds``."""
    if timeout_seconds <= 0:
        return func()
    future = _executor().submit(func)
    try:
        return future.result(timeout=timeout_seconds)
    except FutureTimeoutError as exc:
        future.cancel()
        raise ExtractorTimeout(name, timeout_seconds) from exc


class GuardedExtractor:
    """Wraps an extractor with a breaker and a deadline, so the cascade can trust it."""

    def __init__(
        self,
        inner: ReceiptExtractor,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        self._inner = inner
        self._timeout = timeout_seconds
        self._breaker = breaker or CircuitBreaker(name=inner.name)

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def inner(self) -> ReceiptExtractor:
        return self._inner

    @property
    def breaker(self) -> CircuitBreaker:
        return self._breaker

    @property
    def timeout_seconds(self) -> float:
        return self._timeout

    def status(self) -> CircuitStatus:
        return self._breaker.status()

    def extract(self, image: bytes) -> ExtractionResult:
        """Call the wrapped extractor, or raise if the breaker refuses the call."""
        if not self._breaker.allow():
            status = self._breaker.status()
            raise CircuitOpenError(self.name, status.retry_in_seconds)
        try:
            result = run_with_timeout(self.name, lambda: self._inner.extract(image), self._timeout)
        except Exception:
            self._breaker.record_failure()
            raise
        self._breaker.record_success()
        return result
