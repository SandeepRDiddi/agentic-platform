"""Fallbacks (trust layer: "model + agent").

* CircuitBreaker  – stop hammering a failing model/provider; probe again after a cool-down.
* run_with_fallbacks – ordered chain of attempts (model chain, agent chain).
* DEGRADED_ANSWER – what the user sees when every path is exhausted.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar

from agentic.common.errors import CircuitOpen

log = logging.getLogger(__name__)
T = TypeVar("T")

DEGRADED_ANSWER = (
    "I couldn't complete this request reliably right now. Nothing was changed. "
    "Please retry in a few minutes or contact support if it persists."
)


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_seconds: float = 30.0) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self._failures: dict[str, int] = {}
        self._opened_at: dict[str, float] = {}

    def state(self, key: str) -> str:
        opened = self._opened_at.get(key)
        if opened is None:
            return "closed"
        return "half_open" if time.monotonic() - opened >= self.recovery_seconds else "open"

    def before(self, key: str) -> None:
        if self.state(key) == "open":
            raise CircuitOpen(f"circuit open for {key}")

    def success(self, key: str) -> None:
        self._failures.pop(key, None)
        self._opened_at.pop(key, None)

    def failure(self, key: str) -> None:
        n = self._failures.get(key, 0) + 1
        self._failures[key] = n
        if n >= self.failure_threshold or self.state(key) == "half_open":
            self._opened_at[key] = time.monotonic()
            log.error("circuit OPEN for %s after %d failures", key, n)


async def run_with_fallbacks(
    attempts: Sequence[tuple[str, Callable[[], Awaitable[T]]]],
    *,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    breaker: CircuitBreaker | None = None,
) -> tuple[T, str, bool]:
    """Try each (name, thunk) in order. Returns (value, name_used, used_fallback)."""
    last: BaseException | None = None
    for i, (name, thunk) in enumerate(attempts):
        try:
            if breaker:
                breaker.before(name)
            value = await thunk()
            if breaker:
                breaker.success(name)
            return value, name, i > 0
        except retry_on as e:
            last = e
            if breaker and not isinstance(e, CircuitOpen):
                breaker.failure(name)
            log.warning("attempt %s failed (%s: %s); trying next", name, type(e).__name__, e)
    assert last is not None
    raise last
