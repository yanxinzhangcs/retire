"""Deterministic backend for protocol races and CPU-only development."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator, Sequence
from typing import Any

from .types import (
    EngineFingerprint,
    FenceReceipt,
    ReclaimReceipt,
    RecoveryPlan,
    RequestSpec,
    ReuseProbe,
)


_DONE = object()


class ScriptedBackend:
    """Behaves like a streaming engine while allowing output after abort ACK."""

    def __init__(self, fingerprint: EngineFingerprint) -> None:
        self.fingerprint = fingerprint
        self._queues: defaultdict[str, asyncio.Queue[Any]] = defaultdict(asyncio.Queue)
        self.fence_calls: list[tuple[str, int, tuple[str, ...]]] = []
        self.reclaim_calls: list[tuple[str, int]] = []
        self.generated_specs: list[RequestSpec] = []
        self.next_probe: ReuseProbe | None = None

    async def generate(self, spec: RequestSpec) -> AsyncIterator[Any]:
        self.generated_specs.append(spec)
        queue = self._queues[spec.request_id]
        while True:
            item = await queue.get()
            if item is _DONE:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    async def discard_unlaunched(self, spec: RequestSpec) -> None:
        del spec

    async def fence(
        self, scope_id: str, retired_epoch: int, request_ids: Sequence[str]
    ) -> FenceReceipt:
        ids = tuple(request_ids)
        self.fence_calls.append((scope_id, retired_epoch, ids))
        # Deliberately do not stop the queues. This models delayed engine output
        # and proves the epoch gate, rather than cooperative cancellation, is safe.
        return FenceReceipt(
            scope_id=scope_id,
            retired_epoch=retired_epoch,
            requested_request_ids=ids,
            acknowledged_request_ids=ids,
            launch_fence_installed=True,
        )

    async def probe_reuse(
        self,
        scope_id: str,
        retired_epoch: int,
        source_request_id: str | None,
        block_aligned_tokens: int,
    ) -> ReuseProbe:
        if self.next_probe is None:
            return ReuseProbe.unavailable(source_request_id)
        probe = self.next_probe
        self.next_probe = None
        return probe

    async def reclaim(
        self, scope_id: str, retired_epoch: int
    ) -> ReclaimReceipt:
        self.reclaim_calls.append((scope_id, retired_epoch))
        return ReclaimReceipt(
            scope_id=scope_id,
            retired_epoch=retired_epoch,
            safe_to_reuse=True,
            detail="all scripted old writers drained",
        )

    async def prepare_resume(
        self,
        spec: RequestSpec,
        retired_epoch: int,
        recovery: RecoveryPlan,
    ) -> None:
        return None

    async def prepare_termination(self, scope_id: str, retired_epoch: int) -> None:
        del scope_id, retired_epoch

    def emit(self, request_id: str, payload: Any) -> None:
        self._queues[request_id].put_nowait(payload)

    def fail(self, request_id: str, error: BaseException) -> None:
        self._queues[request_id].put_nowait(error)

    def finish(self, request_id: str) -> None:
        self._queues[request_id].put_nowait(_DONE)
