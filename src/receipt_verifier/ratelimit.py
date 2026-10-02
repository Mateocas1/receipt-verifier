"""Client-side request pacing for a shared provider key.

The evaluation talks to a provider whose window budget is shared with other agents
(``60 requests/window`` for the whole key), so the harness must never burst. Two
guarantees live here:

* :class:`RequestLimiter` keeps a rolling per-minute request budget (default
  :data:`DEFAULT_RPM`) and blocks instead of failing when the budget is exhausted;
* :func:`retry_after_seconds` reads how long a ``429`` wants us to wait, so a retry
  rides the window out rather than being recorded as an extractor failure.

Both are pure of any network concern and inject clock/sleep, so tests never wait.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from datetime import UTC
from email.utils import parsedate_to_datetime
from typing import Final

DEFAULT_RPM: Final = 20
DEFAULT_WINDOW_SECONDS: Final = 60.0

#: Below this an ``x-ratelimit-reset`` value is a duration; above it, a Unix epoch.
_EPOCH_THRESHOLD: Final = 1_000_000_000.0

_RETRY_AFTER_HEADERS: Final = ("retry-after",)
_RESET_HEADERS: Final = ("x-ratelimit-reset-seconds", "x-ratelimit-reset", "ratelimit-reset")


def parse_rpm(raw: str | None, default: int = DEFAULT_RPM) -> int:
    """Parse ``EVAL_RPM``-style text; anything but a positive integer yields ``default``."""
    if raw is None:
        return default
    try:
        value = int(raw.strip())
    except (AttributeError, ValueError):
        return default
    return value if value > 0 else default


class RequestLimiter:
    """Rolling-window limiter: at most ``rpm`` acquisitions per ``window_seconds``.

    ``acquire`` blocks (through the injected ``sleep``) until a slot frees up, which is
    what makes a run pace itself instead of tripping the provider's own limit. A lock
    serializes concurrent callers, so the budget holds even under threads.
    """

    def __init__(
        self,
        rpm: int = DEFAULT_RPM,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        now: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._rpm = max(1, int(rpm))
        self._window = max(1.0, float(window_seconds))
        self._now = now
        self._sleep = sleep
        self._lock = threading.Lock()
        self._stamps: list[float] = []
        self._waited = 0.0

    @property
    def rpm(self) -> int:
        return self._rpm

    def _prune(self, now: float) -> None:
        cutoff = now - self._window
        while self._stamps and self._stamps[0] <= cutoff:
            self._stamps.pop(0)

    def seconds_until_slot(self) -> float:
        """Seconds until one more request fits in the window (0 when it fits now)."""
        with self._lock:
            now = self._now()
            self._prune(now)
            if len(self._stamps) < self._rpm:
                return 0.0
            return max(0.0, self._stamps[0] + self._window - now)

    def acquire(self) -> None:
        """Take one slot, sleeping while the window is full."""
        while True:
            with self._lock:
                now = self._now()
                self._prune(now)
                if len(self._stamps) < self._rpm:
                    self._stamps.append(now)
                    return
                wait = max(0.0, self._stamps[0] + self._window - now)
            self._waited += wait
            self._sleep(wait)

    def waited_seconds(self) -> float:
        """Total time this limiter spent sleeping, for the run summary."""
        with self._lock:
            return self._waited


def _header(headers: Mapping[str, str], names: tuple[str, ...]) -> str | None:
    for name in names:
        for key, value in headers.items():
            if key.lower() == name and str(value).strip():
                return str(value).strip()
    return None


def _parse_seconds(raw: str) -> float | None:
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value >= 0 else None


def retry_after_seconds(headers: Mapping[str, str], *, now: float | None = None) -> float | None:
    """Wait a ``429`` asks for, in seconds, or ``None`` when it says nothing usable.

    ``Retry-After`` accepts either a delay (``2``) or an HTTP-date. Provider reset
    headers are read next; a value large enough to be a Unix epoch is treated as an
    absolute instant, a smaller one as a duration.
    """
    value = _header(headers, _RETRY_AFTER_HEADERS)
    if value is not None:
        seconds = _parse_seconds(value)
        if seconds is not None:
            return seconds
        try:
            moment = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            moment = None
        if moment is not None:
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=UTC)
            reference = now if now is not None else time.time()
            return max(0.0, moment.timestamp() - reference)
    reset = _header(headers, _RESET_HEADERS)
    if reset is not None:
        seconds = _parse_seconds(reset)
        if seconds is None:
            return None
        if seconds >= _EPOCH_THRESHOLD:
            reference = now if now is not None else time.time()
            return max(0.0, seconds - reference)
        return seconds
    return None


__all__ = [
    "DEFAULT_RPM",
    "DEFAULT_WINDOW_SECONDS",
    "RequestLimiter",
    "parse_rpm",
    "retry_after_seconds",
]
