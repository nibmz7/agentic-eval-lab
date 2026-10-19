"""Fault injection for testing retry and recovery paths.

Retry logic you haven't seen fire hasn't been tested. `FlakyStore` lets a test
make a specific backend call fail, either before it does any work (a crash) or
after it has committed (the work happened but the caller never hears back).
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict, deque
from typing import Any, Literal

from .store import SupportStore

FaultMode = Literal["before", "after"]


class InjectedFault(sqlite3.OperationalError):
    """Looks like a transient database error to everything above the store."""


class FlakyStore(SupportStore):
    def __init__(self, path: str = ":memory:"):
        super().__init__(path)
        self._faults: dict[str, deque[FaultMode]] = defaultdict(deque)

    def inject(self, method: str, *modes: FaultMode) -> None:
        """Queue faults for the next calls to `method`, one per call, in order."""
        self._faults[method].extend(modes)

    def _next_fault(self, method: str) -> FaultMode | None:
        queue = self._faults.get(method)
        return queue.popleft() if queue else None

    def process_refund(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._with_fault("process_refund", super().process_refund, *args, **kwargs)

    def escalate_ticket(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._with_fault("escalate_ticket", super().escalate_ticket, *args, **kwargs)

    def _with_fault(self, method: str, fn: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        fault = self._next_fault(method)
        if fault == "before":
            raise InjectedFault(f"injected fault before {method}")
        result = fn(*args, **kwargs)
        if fault == "after":
            raise InjectedFault(f"injected fault after {method} committed")
        return result
