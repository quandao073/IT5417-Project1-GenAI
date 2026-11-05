"""Reliability of the LLM calls of the KG search (docs/v2-kg-design.md, N3).

`Guard.call` runs a call under a hard wall-clock limit and counts consecutive failures: after
`failures` of them in a row the circuit is open and every call fails at once for `cooldown_s`
seconds, so a dead or hanging provider costs the first few queries the limit and the rest almost
nothing. Callers turn the `LLMError` into their deterministic route. An invalid reply
(`LLMInvalidOutput`) is the model's fault, not an outage, and does not count.

`GuardedClient` puts a Guard in front of any client with the parser's `chat(messages, schema)`
contract, so the unchanged `QueryParser` gets the same protection.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

from src.query.llm_parser import LLMError, LLMInvalidOutput


class Guard:
    def __init__(self, failures: int = 3, cooldown_s: float = 60.0,
                 clock: Callable[[], float] = time.monotonic):
        self.failures, self.cooldown_s, self._clock = failures, cooldown_s, clock
        self._consecutive, self._open_until = 0, 0.0
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=16, thread_name_prefix="llm-guard")
        self.stats = {"calls": 0, "failures": 0, "short_circuited": 0}

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._clock() < self._open_until

    def _record(self, ok: bool) -> None:
        with self._lock:
            if ok:
                self._consecutive = 0
                return
            self.stats["failures"] += 1
            self._consecutive += 1
            if self._consecutive >= self.failures:
                self._open_until = self._clock() + self.cooldown_s
                self._consecutive = self.failures - 1   # one failure after the cooldown re-opens

    def call(self, fn: Callable, timeout_s: float):
        """fn() under the wall-clock limit; LLMError if the circuit is open, on timeout, or when
        fn raises one."""
        with self._lock:
            if self._clock() < self._open_until:
                self.stats["short_circuited"] += 1
                raise LLMError("circuit open: the language model failed repeatedly")
            self.stats["calls"] += 1
        future = self._pool.submit(fn)
        try:
            result = future.result(timeout=timeout_s)
        except FutureTimeout as exc:
            future.cancel()
            self._record(False)
            raise LLMError(f"timeout after {timeout_s:g} s") from exc
        except LLMInvalidOutput:
            self._record(True)
            raise
        except LLMError:
            self._record(False)
            raise
        except Exception as exc:  # noqa: BLE001 - any other failure of the provider is an outage
            self._record(False)
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc
        self._record(True)
        return result


class GuardedClient:
    """A parser-contract client whose `chat` runs under a Guard and a time limit."""

    def __init__(self, inner, guard: Guard, timeout_s: float):
        self.inner, self.guard, self.timeout_s = inner, guard, timeout_s
        self.model = inner.model

    def chat(self, messages: list[dict], schema: dict) -> tuple[str, float]:
        return self.guard.call(lambda: self.inner.chat(messages, schema), self.timeout_s)

    def probe(self):
        return self.inner.probe()

    def version(self):
        return self.inner.version()

    @property
    def usage(self):
        return getattr(self.inner, "usage", None)
