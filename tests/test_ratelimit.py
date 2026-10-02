"""Rate-limit guardrail tests: clocks and sleeps are injected, nothing really waits."""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

from receipt_verifier.ratelimit import (
    DEFAULT_RPM,
    RequestLimiter,
    limiter_from_env,
    parse_rpm,
    retry_after_seconds,
)


class FakeClock:
    """A monotonic clock that only advances when the limiter sleeps."""

    def __init__(self) -> None:
        self.time = 0.0
        self.sleeps: list[float] = []
        self._lock = threading.Lock()

    def now(self) -> float:
        return self.time

    def sleep(self, seconds: float) -> None:
        with self._lock:
            self.sleeps.append(seconds)
            self.time += seconds


def limiter(rpm: int, clock: FakeClock) -> RequestLimiter:
    return RequestLimiter(rpm=rpm, window_seconds=60.0, now=clock.now, sleep=clock.sleep)


class TestParseRpm:
    def test_default_when_absent_or_blank(self) -> None:
        assert parse_rpm(None) == DEFAULT_RPM
        assert parse_rpm("") == DEFAULT_RPM
        assert parse_rpm("   ") == DEFAULT_RPM

    def test_rejects_non_positive_and_garbage(self) -> None:
        assert parse_rpm("0") == DEFAULT_RPM
        assert parse_rpm("-3") == DEFAULT_RPM
        assert parse_rpm("abc") == DEFAULT_RPM

    def test_accepts_positive_integers(self) -> None:
        assert parse_rpm("25") == 25
        assert parse_rpm(" 7 ") == 7

    def test_limiter_from_env_uses_eval_rpm(self) -> None:
        assert limiter_from_env({}).rpm == DEFAULT_RPM
        assert limiter_from_env({"EVAL_RPM": "5"}).rpm == 5
        assert limiter_from_env({"EVAL_RPM": "nonsense"}).rpm == DEFAULT_RPM


class TestRequestLimiter:
    def test_default_budget_is_twenty_per_minute(self) -> None:
        assert DEFAULT_RPM == 20

    def test_allows_rpm_requests_then_waits_the_window(self) -> None:
        clock = FakeClock()
        limiter_ = limiter(2, clock)
        limiter_.acquire()
        limiter_.acquire()
        assert clock.sleeps == []
        limiter_.acquire()
        assert clock.sleeps == [60.0]
        assert limiter_.waited_seconds() == 60.0

    def test_seconds_until_slot_is_zero_when_free(self) -> None:
        clock = FakeClock()
        limiter_ = limiter(1, clock)
        assert limiter_.seconds_until_slot() == 0.0
        limiter_.acquire()
        assert limiter_.seconds_until_slot() == 60.0
        clock.time += 30.0
        assert limiter_.seconds_until_slot() == 30.0
        clock.time += 30.0
        assert limiter_.seconds_until_slot() == 0.0

    def test_old_stamps_leave_the_window(self) -> None:
        clock = FakeClock()
        limiter_ = limiter(1, clock)
        limiter_.acquire()
        clock.time += 61.0
        limiter_.acquire()
        assert clock.sleeps == []

    def test_concurrent_callers_still_honour_the_budget(self) -> None:
        clock = FakeClock()
        limiter_ = limiter(1, clock)
        barrier = threading.Barrier(2)

        def acquire() -> None:
            barrier.wait()
            limiter_.acquire()

        threads = [threading.Thread(target=acquire) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        # Exactly one caller takes the only slot; the other waits out the window.
        assert clock.sleeps == [60.0]


class TestRetryAfter:
    def test_integer_seconds(self) -> None:
        assert retry_after_seconds({"Retry-After": "2"}) == 2.0

    def test_header_lookup_is_case_insensitive(self) -> None:
        assert retry_after_seconds({"retry-after": "5"}) == 5.0

    def test_http_date(self) -> None:
        moment = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        header = format_datetime(moment + timedelta(seconds=30))
        assert retry_after_seconds({"Retry-After": header}, now=moment.timestamp()) == 30.0

    def test_ratelimit_reset_headers(self) -> None:
        assert retry_after_seconds({"x-ratelimit-reset-seconds": "7"}) == 7.0
        reset_at = 1_800_000_000.0
        assert (
            retry_after_seconds({"x-ratelimit-reset": str(int(reset_at))}, now=reset_at - 3) == 3.0
        )

    def test_absent_or_unparseable(self) -> None:
        assert retry_after_seconds({}) is None
        assert retry_after_seconds({"Retry-After": "soon"}) is None
