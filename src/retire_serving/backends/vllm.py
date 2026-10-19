"""Adapters between the protocol runtime and vLLM's asynchronous engine."""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Sequence
from typing import Any, Protocol, runtime_checkable

from ..types import (
    EngineFingerprint,
    FenceReceipt,
    ParticipantFenceReceipt,
    ReclaimReceipt,
    RecoveryPlan,
    RequestSpec,
    ReuseProbe,
)


@runtime_checkable
class VLLMInternalHooks(Protocol):
    """Hooks supplied by the eventual vLLM fork.

    The public vLLM API cannot prove physical KV ownership or writer drain.
    Keeping those operations in this explicit port prevents the wrapper from
    making an unsafe reuse claim.
    """

    async def bind_request(self, spec: RequestSpec) -> bool | None: ...

    def advance_epoch(
        self, scope_id: str, new_epoch: int, generation: int
    ) -> None: ...

    async def observe_output(self, spec: RequestSpec, output: Any) -> None: ...

    async def request_complete(self, spec: RequestSpec) -> None: ...

    async def install_scope_fence(
        self, scope_id: str, retired_epoch: int, request_ids: Sequence[str]
    ) -> Sequence[ParticipantFenceReceipt] | None: ...

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

    async def prepare_termination(self, scope_id: str, retired_epoch: int) -> None: ...

    async def reclaim_scope(
        self, scope_id: str, retired_epoch: int
    ) -> ReclaimReceipt: ...


class VLLMAsyncBackend:
    """Duck-typed adapter for vLLM V1 `AsyncLLM`/`AsyncLLMEngine`.

    vLLM remains an optional dependency so the control-plane and race tests run
    on CPU. A real engine instance is passed in by the application.
    """

    def __init__(
        self,
        engine: Any,
        *,
        fingerprint: EngineFingerprint,
        hooks: VLLMInternalHooks | None = None,
    ) -> None:
        self.engine = engine
        self.fingerprint = fingerprint
        self.hooks = hooks

    def advance_epoch(
        self, scope_id: str, new_epoch: int, generation: int
    ) -> None:
        """Publish a version change without entering EngineCore's work queue."""
        advance = getattr(self.hooks, "advance_epoch", None)
        if advance is not None:
            advance(scope_id, new_epoch, generation)

    @property
    def requires_participant_fence(self) -> bool:
        configured = getattr(self.hooks, "requires_participant_fence", None)
        if configured is not None:
            return bool(configured)
        return (
            len(self.fingerprint.participant_coordinates) > 1
            or self.fingerprint.speculative_mode != "disabled"
        )

    async def generate(self, spec: RequestSpec) -> AsyncIterator[Any]:
        try:
            if self.hooks is not None:
                admitted = await self.hooks.bind_request(spec)
                if admitted is False:
                    return
            cache_salt = spec.cache_salt
            encode_cache_salt = getattr(
                self.hooks,
                "encode_request_cache_salt",
                None,
            )
            if encode_cache_salt is not None:
                cache_salt = encode_cache_salt(spec)
            prompt = {
                "prompt_token_ids": list(spec.prompt_token_ids),
                "cache_salt": cache_salt,
            }
            stream = self.engine.generate(
                prompt,
                spec.sampling_params,
                request_id=spec.request_id,
            )
            if inspect.isawaitable(stream) and not hasattr(stream, "__aiter__"):
                stream = await stream
            observe = getattr(self.hooks, "observe_output", None)
            needs_observation = getattr(self.hooks, "needs_output_observation", None)
            if observe is not None and needs_observation is not None:
                if not bool(needs_observation(spec)):
                    observe = None
            if hasattr(stream, "__aiter__"):
                async for output in stream:
                    if observe is not None:
                        await observe(spec, output)
                    yield output
            else:
                # Useful for lightweight engine doubles and defensive API support.
                if observe is not None:
                    await observe(spec, stream)
                yield stream
        finally:
            complete = getattr(self.hooks, "request_complete", None)
            if complete is not None:
                await complete(spec)

    async def discard_unlaunched(self, spec: RequestSpec) -> None:
        """Close prepared KV ownership when ADVANCE wins before admission."""

        complete = getattr(self.hooks, "request_complete", None)
        if complete is not None:
            await complete(spec)

    async def fence(
        self, scope_id: str, retired_epoch: int, request_ids: Sequence[str]
    ) -> FenceReceipt:
        ids = tuple(request_ids)
        participants: tuple[ParticipantFenceReceipt, ...] = ()
        if self.hooks is not None:
            result = await self.hooks.install_scope_fence(
                scope_id,
                retired_epoch,
                ids,
            )
            if result is not None:
                participants = tuple(result)

        abort = getattr(self.engine, "abort", None)
        if abort is None:
            abort = getattr(self.engine, "abort_request", None)
        if abort is None and ids:
            raise TypeError("vLLM engine exposes neither abort nor abort_request")

        acknowledged: list[str] = []
        for request_id in ids:
            result = abort(request_id)
            if inspect.isawaitable(result):
                await result
            acknowledged.append(request_id)
        return FenceReceipt(
            scope_id=scope_id,
            retired_epoch=retired_epoch,
            requested_request_ids=ids,
            acknowledged_request_ids=tuple(acknowledged),
            launch_fence_installed=True,
            participants=participants,
        )

    async def probe_reuse(
        self,
        scope_id: str,
        retired_epoch: int,
        source_request_id: str | None,
        block_aligned_tokens: int,
    ) -> ReuseProbe:
        if self.hooks is None:
            # The public prefix-cache API is intentionally not treated as an
            # exactness certificate.
            return ReuseProbe.unavailable(source_request_id)
        return await self.hooks.probe_reuse(
            scope_id,
            retired_epoch,
            source_request_id,
            block_aligned_tokens,
        )

    async def prepare_resume(
        self,
        spec: RequestSpec,
        retired_epoch: int,
        recovery: RecoveryPlan,
    ) -> None:
        if self.hooks is not None:
            await self.hooks.prepare_resume(spec, retired_epoch, recovery)

    async def prepare_termination(self, scope_id: str, retired_epoch: int) -> None:
        prepare = getattr(self.hooks, "prepare_termination", None)
        if prepare is not None:
            await prepare(scope_id, retired_epoch)

    async def reclaim(
        self, scope_id: str, retired_epoch: int
    ) -> ReclaimReceipt:
        if self.hooks is not None:
            return await self.hooks.reclaim_scope(scope_id, retired_epoch)
        return ReclaimReceipt(
            scope_id=scope_id,
            retired_epoch=retired_epoch,
            safe_to_reuse=True,
            detail="public vLLM owns and reclaims its blocks; no manual reuse",
        )
