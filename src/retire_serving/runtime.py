"""End-to-end scope-epoch runtime over an asynchronous inference backend."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from itertools import count
from time import monotonic_ns
from typing import Any, TypeVar

from .align import AlignmentPlanner
from .backend import InferenceBackend
from .control import ScopeRegistry
from .trace import TraceRecord, TraceRecorder
from .types import (
    AdvanceReceipt,
    ExecutionTag,
    FenceReceipt,
    NumericalContract,
    OutputEnvelope,
    ProtocolEvent,
    ProtocolStage,
    ReclaimReceipt,
    RecoveryPlan,
    RequestSpec,
    SpeculativeOutput,
    SupersessionDescriptor,
)

_END = object()
_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class _Failure:
    error: BaseException


@dataclass(frozen=True, slots=True)
class ExecutionStats:
    received_outputs: int
    published_outputs: int
    stale_outputs_dropped: int


class ExecutionHandle:
    """A stream whose final publication point is guarded by the scope epoch."""

    def __init__(
        self,
        spec: RequestSpec,
        registry: ScopeRegistry,
        trace: TraceRecorder,
        *,
        record_committed_outputs: bool = True,
    ) -> None:
        self.spec = spec
        self._registry = registry
        self._trace = trace
        self._record_committed_outputs = record_committed_outputs
        self._queue: asyncio.Queue[OutputEnvelope | _Failure | object] = (
            asyncio.Queue()
        )
        self._done = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._received = 0
        self._published = 0
        self._dropped = 0

    @property
    def request_id(self) -> str:
        return self.spec.request_id

    @property
    def scope_id(self) -> str:
        return self.spec.scope_id

    @property
    def epoch(self) -> int:
        return self.spec.epoch

    @property
    def tenant_id(self) -> str:
        return self.spec.tenant_id

    def _attach(self, task: asyncio.Task[None]) -> None:
        if self._task is not None:
            raise RuntimeError("execution task already attached")
        self._task = task

    def _put_output(self, payload: Any) -> None:
        self._received += 1
        envelope = OutputEnvelope(
            scope_id=self.scope_id,
            epoch=self.epoch,
            request_id=self.request_id,
            operation_id=self.spec.operation_id,
            generation=self.spec.generation,
            payload=payload,
            tenant_id=self.tenant_id,
        )
        # Early rejection bounds stale queue growth. The dequeue gate below is
        # still mandatory for output queued before ADVANCE but consumed after it.
        if self._registry.is_current(
            self.scope_id,
            self.epoch,
            self.tenant_id,
        ):
            self._queue.put_nowait(envelope)
        else:
            self._dropped += 1
            self._record_publication("drop", "stale at frontend enqueue")

    def _record_publication(self, phase: str, detail: str = "") -> None:
        self._trace.record(
            TraceRecord(
                tag=ExecutionTag(
                    scope_id=self.scope_id,
                    epoch=self.epoch,
                    operation_id=self.spec.operation_id,
                    object_type="output",
                    generation=self.spec.generation,
                    tenant_id=self.tenant_id,
                ),
                action="output_publication",
                phase=phase,
                request_id=self.request_id,
                detail=detail,
            )
        )

    def _put_failure(self, error: BaseException) -> None:
        self._queue.put_nowait(_Failure(error))

    def _finish(self) -> None:
        self._queue.put_nowait(_END)
        self._done.set()

    def __aiter__(self) -> ExecutionHandle:
        return self

    async def __anext__(self) -> Any:
        while True:
            item = await self._queue.get()
            if item is _END:
                raise StopAsyncIteration

            # This check is the API-visible commit point. There is no await
            # between the check and return, so ADVANCE is totally ordered with
            # publication in the event loop.
            current = self._registry.is_current(
                self.scope_id,
                self.epoch,
                self.tenant_id,
            )
            if isinstance(item, _Failure):
                if current:
                    raise item.error
                self._dropped += 1
                self._record_publication("drop", "stale failure")
                continue
            if not current:
                self._dropped += 1
                self._record_publication("drop", "stale at API commit")
                continue
            self._published += 1
            if self._record_committed_outputs:
                self._record_publication("commit")
            return item.payload

    def _drop_speculative(self, detail: str) -> None:
        self._received += 1
        self._dropped += 1
        self._record_publication("drop", detail)

    async def wait(self) -> ExecutionStats:
        await self._done.wait()
        return self.stats

    @property
    def stats(self) -> ExecutionStats:
        return ExecutionStats(
            received_outputs=self._received,
            published_outputs=self._published,
            stale_outputs_dropped=self._dropped,
        )


@dataclass(slots=True)
class Supersession:
    advance: AdvanceReceipt
    fence: FenceReceipt
    recovery: RecoveryPlan
    successor: ExecutionHandle
    events: list[ProtocolEvent]
    _reclaim_task: asyncio.Task[ReclaimReceipt] = field(repr=False)

    async def wait_reclaim(self) -> ReclaimReceipt:
        return await self._reclaim_task


class RetireServing:
    """The focused `ADVANCE -> FENCE -> ALIGN -> RESUME -> RECLAIM` runtime."""

    def __init__(
        self,
        backend: InferenceBackend,
        *,
        numerical_contract: NumericalContract | None = None,
        trace: TraceRecorder | None = None,
        record_committed_outputs: bool = True,
    ) -> None:
        self.backend = backend
        self.registry = ScopeRegistry()
        self.contract = numerical_contract or NumericalContract.strict_clean_recomputation()
        backend_trace = getattr(backend, "trace", None)
        self.trace = trace or backend_trace or TraceRecorder()
        self.record_committed_outputs = record_committed_outputs
        self.planner = AlignmentPlanner(backend.fingerprint.alignment_block_size)
        self._ids = count()
        self._control_ids = count()
        self._handles: dict[str, ExecutionHandle] = {}
        self._supersession_locks: defaultdict[str, asyncio.Lock] = defaultdict(
            asyncio.Lock
        )

    @staticmethod
    def _new_request_id(scope_id: str, epoch: int, operation_id: int) -> str:
        return f"{scope_id}:e{epoch}:r{operation_id}"

    @staticmethod
    async def _before_deadline(
        operation: Callable[[], Awaitable[_T]],
        deadline_ns: int | None,
    ) -> _T:
        if deadline_ns is None:
            return await operation()
        remaining_seconds = (deadline_ns - monotonic_ns()) / 1_000_000_000
        if remaining_seconds <= 0:
            raise TimeoutError("supersession deadline expired")
        try:
            return await asyncio.wait_for(operation(), timeout=remaining_seconds)
        except TimeoutError as exc:
            raise TimeoutError("supersession deadline expired") from exc

    async def start(
        self,
        scope_id: str,
        prompt_token_ids: list[int] | tuple[int, ...],
        sampling_params: Any,
        *,
        request_id: str | None = None,
        tenant_id: str = "default",
    ) -> ExecutionHandle:
        epoch = self.registry.ensure_scope(scope_id, tenant_id)
        spec = self._make_spec(
            scope_id=scope_id,
            epoch=epoch,
            prompt_token_ids=tuple(prompt_token_ids),
            sampling_params=sampling_params,
            request_id=request_id,
            cache_salt=f"{tenant_id}:{scope_id}:e{epoch}",
            tenant_id=tenant_id,
        )
        return self._launch(spec)

    def _make_spec(
        self,
        *,
        scope_id: str,
        epoch: int,
        prompt_token_ids: tuple[int, ...],
        sampling_params: Any,
        request_id: str | None = None,
        cache_salt: str,
        tenant_id: str,
    ) -> RequestSpec:
        operation_id = next(self._ids)
        return RequestSpec(
            scope_id=scope_id,
            epoch=epoch,
            request_id=request_id
            or self._new_request_id(scope_id, epoch, operation_id),
            operation_id=operation_id,
            prompt_token_ids=prompt_token_ids,
            cache_salt=cache_salt,
            sampling_params=sampling_params,
            generation=epoch,
            tenant_id=tenant_id,
        )

    def _launch(self, spec: RequestSpec) -> ExecutionHandle:
        self.registry.register(spec)
        handle = ExecutionHandle(
            spec,
            self.registry,
            self.trace,
            record_committed_outputs=self.record_committed_outputs,
        )
        self._handles[spec.request_id] = handle
        self.trace.record(
            TraceRecord(
                tag=spec.tag,
                action="request_launch",
                phase="commit",
                request_id=spec.request_id,
            )
        )
        task = asyncio.create_task(self._pump(handle))
        handle._attach(task)
        return handle

    async def _pump(self, handle: ExecutionHandle) -> None:
        try:
            # A request task can be created immediately before ADVANCE but not
            # enter the backend until the event loop schedules it.  Recheck
            # launch authority at that boundary so a queued old-epoch task
            # cannot enter the serving engine after the scope has advanced.
            if not self.registry.is_current(
                handle.scope_id,
                handle.epoch,
                handle.tenant_id,
            ):
                discard = getattr(self.backend, "discard_unlaunched", None)
                if discard is not None:
                    await discard(handle.spec)
                return
            async for output in self.backend.generate(handle.spec):
                if isinstance(output, SpeculativeOutput):
                    if not output.verified:
                        handle._drop_speculative("unverified speculative output")
                        continue
                    if output.generation != handle.spec.generation:
                        handle._drop_speculative("stale speculative generation")
                        continue
                    output = output.payload
                handle._put_output(output)
        except Exception as exc:  # noqa: BLE001 - stream errors are data-plane output
            handle._put_failure(exc)
        finally:
            self.registry.complete(
                handle.scope_id,
                handle.epoch,
                handle.request_id,
            )
            handle._finish()

    async def supersede(
        self,
        scope_id: str,
        revised_prompt_token_ids: list[int] | tuple[int, ...],
        sampling_params: Any,
        *,
        request_id: str | None = None,
        tenant_id: str = "default",
    ) -> Supersession:
        async with self._supersession_locks[scope_id]:
            old_epoch = self.registry.current_epoch(scope_id, tenant_id)
            descriptor = SupersessionDescriptor(
                scope_id=scope_id,
                old_epoch=old_epoch,
                new_epoch=old_epoch + 1,
                revised_prompt_token_ids=tuple(revised_prompt_token_ids),
                contract=self.contract,
                tenant_id=tenant_id,
            )
            return await self._supersede_locked(
                descriptor,
                sampling_params,
                request_id=request_id,
            )

    async def supersede_descriptor(
        self,
        descriptor: SupersessionDescriptor,
        sampling_params: Any,
        *,
        request_id: str | None = None,
    ) -> Supersession:
        """Execute an explicit paper descriptor without inferring its epochs."""

        async with self._supersession_locks[descriptor.scope_id]:
            return await self._supersede_locked(
                descriptor,
                sampling_params,
                request_id=request_id,
            )

    async def terminate(
        self,
        scope_id: str,
        *,
        tenant_id: str = "default",
    ) -> ReclaimReceipt:
        """Retire a scope without launching a successor.

        Termination is the terminal form of a version transition.  Advancing
        authority before fencing closes the launch-after-abort race that a raw
        request-local abort leaves open when admission is still in flight.
        """

        async with self._supersession_locks[scope_id]:
            old_epoch = self.registry.current_epoch(scope_id, tenant_id)
            advance = self.registry.advance(
                scope_id,
                expected_epoch=old_epoch,
                new_epoch=old_epoch + 1,
                tenant_id=tenant_id,
            )
            advance_backend = getattr(self.backend, "advance_epoch", None)
            if advance_backend is not None:
                advance_backend(scope_id, advance.new_epoch, advance.new_epoch)

            control_operation_id = next(self._control_ids)
            events = [
                ProtocolEvent(
                    ProtocolStage.ADVANCE,
                    scope_id,
                    advance.new_epoch,
                    detail=f"terminated epoch {advance.old_epoch}",
                    tenant_id=tenant_id,
                )
            ]
            self.trace.record(
                TraceRecord(
                    tag=ExecutionTag(
                        scope_id=scope_id,
                        epoch=advance.new_epoch,
                        operation_id=control_operation_id,
                        object_type="control",
                        tenant_id=tenant_id,
                    ),
                    action=ProtocolStage.ADVANCE.value,
                    phase="complete",
                    detail=f"terminated epoch {advance.old_epoch}",
                )
            )

            fence = await self.backend.fence(
                scope_id,
                advance.old_epoch,
                advance.retired_request_ids,
            )
            self._validate_fence(scope_id, advance.old_epoch, fence)
            events.append(
                ProtocolEvent(
                    ProtocolStage.FENCE,
                    scope_id,
                    advance.new_epoch,
                    detail=f"fenced {len(fence.acknowledged_request_ids)} requests",
                    tenant_id=tenant_id,
                )
            )
            self.trace.record(
                TraceRecord(
                    tag=ExecutionTag(
                        scope_id=scope_id,
                        epoch=advance.new_epoch,
                        operation_id=control_operation_id,
                        object_type="control",
                        tenant_id=tenant_id,
                    ),
                    action=ProtocolStage.FENCE.value,
                    phase="complete",
                    detail=f"fenced {len(fence.acknowledged_request_ids)} requests",
                )
            )
            prepare_termination = getattr(
                self.backend,
                "prepare_termination",
                None,
            )
            if prepare_termination is not None:
                await prepare_termination(scope_id, advance.old_epoch)
            return await self._reclaim_after_drain(
                advance,
                events,
                control_operation_id=control_operation_id,
            )

    def _validate_fence(
        self,
        scope_id: str,
        retired_epoch: int,
        fence: FenceReceipt,
    ) -> None:
        if not fence.safe:
            raise RuntimeError(
                "backend did not establish a safe distributed fence for "
                f"{scope_id} epoch {retired_epoch}"
            )
        requires_participant_fence = bool(
            getattr(
                self.backend,
                "requires_participant_fence",
                len(self.backend.fingerprint.participant_coordinates) > 1
                or self.backend.fingerprint.speculative_mode != "disabled",
            )
        )
        if requires_participant_fence and not fence.participants:
            raise RuntimeError(
                "multi-group, multi-PP/TP, or speculative execution requires "
                "participant-level fence receipts"
            )
        if fence.participants:
            expected = set(self.backend.fingerprint.participant_coordinates)
            observed = {item.coordinate for item in fence.participants}
            if observed != expected or len(observed) != len(fence.participants):
                raise RuntimeError(
                    "distributed fence did not cover every PP stage, TP rank, "
                    "and KV group"
                )

    async def _supersede_locked(
        self,
        descriptor: SupersessionDescriptor,
        sampling_params: Any,
        *,
        request_id: str | None,
    ) -> Supersession:
        scope_id = descriptor.scope_id
        revised = descriptor.revised_prompt_token_ids
        events: list[ProtocolEvent] = []
        control_operation_id = next(self._control_ids)

        def record_stage(stage: ProtocolStage, detail: str) -> None:
            self.trace.record(
                TraceRecord(
                    tag=ExecutionTag(
                        scope_id=scope_id,
                        epoch=descriptor.new_epoch,
                        operation_id=control_operation_id,
                        object_type="control",
                        tenant_id=descriptor.tenant_id,
                    ),
                    action=stage.value,
                    phase="complete",
                    detail=detail,
                )
            )

        # ADVANCE is deliberately completed before any await.
        advance = self.registry.advance(
            scope_id,
            expected_epoch=descriptor.old_epoch,
            new_epoch=descriptor.new_epoch,
            tenant_id=descriptor.tenant_id,
        )
        # This notification must remain synchronous and precede the first
        # await after the registry transition.  A serving backend may use it
        # to update an out-of-band device mailbox while an old model step is
        # still executing; the later FENCE remains responsible for drain and
        # reclamation proof.
        advance_backend = getattr(self.backend, "advance_epoch", None)
        if advance_backend is not None:
            advance_backend(
                scope_id,
                descriptor.new_epoch,
                descriptor.new_epoch,
            )
        events.append(
            ProtocolEvent(
                ProtocolStage.ADVANCE,
                scope_id,
                advance.new_epoch,
                detail=f"retired epoch {advance.old_epoch}",
                tenant_id=descriptor.tenant_id,
            )
        )
        record_stage(ProtocolStage.ADVANCE, f"retired epoch {advance.old_epoch}")

        fence = await self._before_deadline(
            lambda: self.backend.fence(
                scope_id,
                advance.old_epoch,
                advance.retired_request_ids,
            ),
            descriptor.deadline_ns,
        )
        self._validate_fence(scope_id, advance.old_epoch, fence)
        events.append(
            ProtocolEvent(
                ProtocolStage.FENCE,
                scope_id,
                advance.new_epoch,
                detail=f"fenced {len(fence.acknowledged_request_ids)} requests",
                tenant_id=descriptor.tenant_id,
            )
        )
        record_stage(
            ProtocolStage.FENCE,
            f"fenced {len(fence.acknowledged_request_ids)} requests",
        )

        source_request_id, common_prefix = self.planner.choose_source(
            advance.retired_contexts,
            revised,
        )
        aligned = common_prefix - (common_prefix % self.planner.block_size)
        probe = await self._before_deadline(
            lambda: self.backend.probe_reuse(
                scope_id,
                advance.old_epoch,
                source_request_id,
                aligned,
            ),
            descriptor.deadline_ns,
        )
        recovery = self.planner.plan(
            source_request_id=source_request_id,
            common_prefix_tokens=common_prefix,
            revised_tokens=len(revised),
            contract=descriptor.contract,
            current_fingerprint=self.backend.fingerprint,
            current_tenant_id=descriptor.tenant_id,
            probe=probe,
        )
        events.append(
            ProtocolEvent(
                ProtocolStage.ALIGN,
                scope_id,
                advance.new_epoch,
                detail=recovery.certificate.reason,
                tenant_id=descriptor.tenant_id,
            )
        )
        record_stage(ProtocolStage.ALIGN, recovery.certificate.reason)

        source_handle = (
            self._handles.get(source_request_id)
            if source_request_id is not None
            else None
        )
        successor_salt = (
            source_handle.spec.cache_salt
            if recovery.certificate.admissible and source_handle is not None
            else f"{descriptor.tenant_id}:{scope_id}:e{advance.new_epoch}"
        )
        successor_spec = self._make_spec(
            scope_id=scope_id,
            epoch=advance.new_epoch,
            prompt_token_ids=revised,
            sampling_params=sampling_params,
            request_id=request_id,
            cache_salt=successor_salt,
            tenant_id=descriptor.tenant_id,
        )
        await self._before_deadline(
            lambda: self.backend.prepare_resume(
                successor_spec,
                advance.old_epoch,
                recovery,
            ),
            descriptor.deadline_ns,
        )
        successor = self._launch(successor_spec)
        events.append(
            ProtocolEvent(
                ProtocolStage.RESUME,
                scope_id,
                advance.new_epoch,
                detail=f"{recovery.tier.value}; reuse={recovery.reuse_tokens}",
                tenant_id=descriptor.tenant_id,
            )
        )
        record_stage(
            ProtocolStage.RESUME,
            f"{recovery.tier.value}; reuse={recovery.reuse_tokens}",
        )

        reclaim_task = asyncio.create_task(
            self._reclaim_after_drain(
                advance,
                events,
                control_operation_id=control_operation_id,
            )
        )
        return Supersession(
            advance=advance,
            fence=fence,
            recovery=recovery,
            successor=successor,
            events=events,
            _reclaim_task=reclaim_task,
        )

    async def _reclaim_after_drain(
        self,
        advance: AdvanceReceipt,
        events: list[ProtocolEvent],
        *,
        control_operation_id: int,
    ) -> ReclaimReceipt:
        handles = [
            self._handles[request_id]
            for request_id in advance.retired_request_ids
            if request_id in self._handles
        ]
        if handles:
            await asyncio.gather(*(handle.wait() for handle in handles))
        receipt = await self.backend.reclaim(
            advance.scope_id,
            advance.old_epoch,
        )
        if not receipt.safe_to_reuse:
            raise RuntimeError(
                f"backend did not certify reclaim of {advance.scope_id} "
                f"epoch {advance.old_epoch}: {receipt.detail}"
            )
        self.registry.mark_reclaimed(
            advance.scope_id,
            advance.old_epoch,
            advance.tenant_id,
        )
        events.append(
            ProtocolEvent(
                ProtocolStage.RECLAIM,
                advance.scope_id,
                advance.new_epoch,
                detail=receipt.detail,
                tenant_id=advance.tenant_id,
            )
        )
        self.trace.record(
            TraceRecord(
                tag=ExecutionTag(
                    scope_id=advance.scope_id,
                    epoch=advance.new_epoch,
                    operation_id=control_operation_id,
                    object_type="control",
                    tenant_id=advance.tenant_id,
                ),
                action=ProtocolStage.RECLAIM.value,
                phase="complete",
                detail=receipt.detail,
            )
        )
        return receipt
