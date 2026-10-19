"""Cross-layer trace records required by the paper's invariant checker."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from time import monotonic_ns

from .types import ExecutionTag


@dataclass(frozen=True, slots=True)
class TraceRecord:
    tag: ExecutionTag
    action: str
    phase: str
    request_id: str | None = None
    token_range: tuple[int, int] | None = None
    physical_block: int | None = None
    timestamp_ns: int = field(default_factory=monotonic_ns)
    detail: str = ""


class TraceRecorder:
    """Thread-safe append-only trace shared by serving and KV adapters."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._records: list[TraceRecord] = []

    def record(self, record: TraceRecord) -> None:
        with self._lock:
            self._records.append(record)

    def snapshot(self) -> tuple[TraceRecord, ...]:
        with self._lock:
            return tuple(self._records)
