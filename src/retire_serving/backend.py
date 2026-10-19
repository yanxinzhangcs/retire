"""Framework-neutral inference backend contract."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, Protocol

from .types import (
    EngineFingerprint,
    FenceReceipt,
    ReclaimReceipt,
    RecoveryPlan,
    RequestSpec,
    ReuseProbe,
)


class InferenceBackend(Protocol):
    """Port implemented by vLLM now and other serving engines later."""

    fingerprint: EngineFingerprint

    def advance_epoch(
        self, scope_id: str, new_epoch: int, generation: int
    ) -> None: ...

    def generate(self, spec: RequestSpec) -> AsyncIterator[Any]: ...

    async def discard_unlaunched(self, spec: RequestSpec) -> None:
        """Release state prepared for a request revoked before admission."""
        ...

    async def fence(
        self, scope_id: str, retired_epoch: int, request_ids: Sequence[str]
    ) -> FenceReceipt: ...

    async def probe_reuse(
        self,
        scope_id: str,
        retired_epoch: int,
        source_request_id: str | None,
        block_aligned_tokens: int,
    ) -> ReuseProbe: ...

    async def prepare_resume(
        self,
        spec: RequestSpec,
        retired_epoch: int,
        recovery: RecoveryPlan,
    ) -> None: ...

    async def prepare_termination(self, scope_id: str, retired_epoch: int) -> None:
        """Release recovery state when retirement has no successor."""
        ...

    async def reclaim(
        self, scope_id: str, retired_epoch: int
    ) -> ReclaimReceipt: ...
