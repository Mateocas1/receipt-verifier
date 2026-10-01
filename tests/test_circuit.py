from time import sleep

import pytest

from receipt_verifier.circuit import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    ExtractorTimeout,
    GuardedExtractor,
    run_with_timeout,
    shutdown_executor,
)
from tests.helpers import StubExtractor

IMAGE = b"\x89PNG-irrelevant"


class FakeClock:
    """Controllable clock so the breaker tests never sleep for real."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TestCircuitBreaker:
    def test_closed_by_default(self) -> None:
        breaker = CircuitBreaker(name="ocr")
        assert breaker.allow()
        assert breaker.status().state is CircuitState.CLOSED

    def test_success_resets_the_failure_count(self) -> None:
        breaker = CircuitBreaker(name="ocr", failure_threshold=3)
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_success()
        assert breaker.status().failures == 0
        breaker.record_failure()
        assert breaker.status().state is CircuitState.CLOSED

    def test_opens_after_the_threshold(self) -> None:
        clock = FakeClock()
        breaker = CircuitBreaker(name="llm", failure_threshold=2, open_seconds=30.0, clock=clock)
        breaker.record_failure()
        assert breaker.allow()
        breaker.record_failure()
        assert breaker.status().state is CircuitState.OPEN
        assert not breaker.allow()
        assert breaker.status().retry_in_seconds == 30.0

    def test_refuses_while_open_then_probes_once(self) -> None:
        clock = FakeClock()
        breaker = CircuitBreaker(name="llm", failure_threshold=1, open_seconds=10.0, clock=clock)
        breaker.record_failure()
        assert not breaker.allow()
        clock.advance(9.9)
        assert not breaker.allow()
        clock.advance(0.2)
        assert breaker.allow()  # the single probe
        assert breaker.status().state is CircuitState.HALF_OPEN
        assert not breaker.allow()  # probe already in flight
        clock.advance(100.0)
        assert breaker.allow()  # a probe that never reports back does not wedge it

    def test_probe_success_closes(self) -> None:
        clock = FakeClock()
        breaker = CircuitBreaker(name="llm", failure_threshold=1, open_seconds=5.0, clock=clock)
        breaker.record_failure()
        clock.advance(5.0)
        assert breaker.allow()
        breaker.record_success()
        assert breaker.status().state is CircuitState.CLOSED
        assert breaker.status().failures == 0
        assert breaker.status().retry_in_seconds == 0.0

    def test_probe_failure_reopens_the_window(self) -> None:
        clock = FakeClock()
        breaker = CircuitBreaker(name="llm", failure_threshold=1, open_seconds=5.0, clock=clock)
        breaker.record_failure()
        clock.advance(5.0)
        assert breaker.allow()
        breaker.record_failure()
        assert breaker.status().state is CircuitState.OPEN
        assert not breaker.allow()
        clock.advance(5.0)
        assert breaker.allow()

    @pytest.mark.parametrize(("threshold", "open_seconds"), [(0, 10.0), (1, -1.0)])
    def test_invalid_configuration(self, threshold: int, open_seconds: float) -> None:
        with pytest.raises(ValueError):
            CircuitBreaker(name="x", failure_threshold=threshold, open_seconds=open_seconds)


class TestTimeoutGuard:
    def test_fast_call_returns(self) -> None:
        assert run_with_timeout("fast", lambda: 42, 1.0) == 42

    def test_slow_call_raises(self) -> None:
        with pytest.raises(ExtractorTimeout) as excinfo:
            run_with_timeout("slow", lambda: sleep(0.5) or 1, 0.01)
        assert excinfo.value.name == "slow"

    def test_exception_propagates(self) -> None:
        def boom() -> None:
            raise ValueError("nope")

        with pytest.raises(ValueError, match="nope"):
            run_with_timeout("boom", boom, 1.0)

    def test_zero_timeout_disables_the_guard(self) -> None:
        assert run_with_timeout("inline", lambda: "ok", 0.0) == "ok"


class TestGuardedExtractor:
    def test_passes_results_through(self) -> None:
        inner = StubExtractor("ocr")
        guarded = GuardedExtractor(inner, timeout_seconds=1.0)
        result = guarded.extract(IMAGE)
        assert result.extractor == "ocr"
        assert guarded.name == "ocr"
        assert inner.calls == 1
        assert guarded.status().state is CircuitState.CLOSED

    def test_failures_open_the_breaker_and_stop_calls(self) -> None:
        inner = StubExtractor("llm", error=RuntimeError("boom"))
        guarded = GuardedExtractor(
            inner, timeout_seconds=1.0, breaker=CircuitBreaker(name="llm", failure_threshold=2)
        )
        for _ in range(2):
            with pytest.raises(RuntimeError):
                guarded.extract(IMAGE)
        assert guarded.status().state is CircuitState.OPEN
        with pytest.raises(CircuitOpenError) as excinfo:
            guarded.extract(IMAGE)
        assert excinfo.value.name == "llm"
        assert inner.calls == 2  # the open breaker never reached the extractor

    def test_timeout_counts_as_a_failure(self) -> None:
        inner = StubExtractor("ocr", delay=0.3)
        guarded = GuardedExtractor(
            inner, timeout_seconds=0.01, breaker=CircuitBreaker(name="ocr", failure_threshold=1)
        )
        with pytest.raises(ExtractorTimeout):
            guarded.extract(IMAGE)
        assert guarded.status().state is CircuitState.OPEN

    def test_recovery_after_the_window(self) -> None:
        clock = FakeClock()
        inner = StubExtractor("llm", error=RuntimeError("boom"))
        guarded = GuardedExtractor(
            inner,
            timeout_seconds=1.0,
            breaker=CircuitBreaker(name="llm", failure_threshold=1, open_seconds=5.0, clock=clock),
        )
        with pytest.raises(RuntimeError):
            guarded.extract(IMAGE)
        with pytest.raises(CircuitOpenError):
            guarded.extract(IMAGE)
        inner.set_error(None)  # the provider came back
        clock.advance(5.0)
        assert guarded.extract(IMAGE).extractor == "llm"
        assert guarded.status().state is CircuitState.CLOSED

    def test_shutdown_executor_is_reusable(self) -> None:
        shutdown_executor()
        assert run_with_timeout("again", lambda: "ok", 1.0) == "ok"
