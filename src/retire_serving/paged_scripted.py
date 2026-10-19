"""Integrated deterministic request server with paged KV ownership.

This backend is the executable reference path for protocol fault injection. It
uses the same ``InferenceBackend`` port as vLLM, but its token outputs are
scripted so races and block ownership can be checked deterministically.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from .kv import KVBlockPool, KVOwner, KVSlotState, WriterTicket
from .scripted import ScriptedBackend
from .trace import TraceRecord, TraceRecorder
from .types import (
    EngineFingerprint,
    ExecutionTag,
    ReclaimReceipt,
    RecoveryPlan,
    RequestSpec,
    ReuseProbe,
    ReuseTier,
)


class PagedScriptedBackend(ScriptedBackend):
    """A minimal integrated server for the paper's first software milestone."""

    def __init__(
        self,
        fingerprint: EngineFingerprint,
        *,
        num_kv_slots: int = 256,
        numerical_attestation: str = "strict-clean-recomputation-v1",
        trace: TraceRecorder | None = None,
    ) -> None:
        super().__init__(fingerprint)
        self.pool = KVBlockPool(num_kv_slots)
        self.numerical_attestation = numerical_attestation
        self.trace = trace or TraceRecorder()
        self.request_slots: dict[str, tuple[int, ...]] = {}
        self.request_owners: dict[str, KVOwner] = {}
        self.request_specs: dict[str, RequestSpec] = {}
        self._prepared_reuse: dict[str, tuple[int, ...]] = {}
        self._writer_events: dict[WriterTicket, asyncio.Event] = {}

    @staticmethod
    def _owner(spec: RequestSpec) -> KVOwner:
        return KVOwner(
            spec.scope_id,
            spec.epoch,
            spec.operation_id,
            spec.tenant_id,
        )

    def _trace_slot(
        self,
        spec: RequestSpec,
        *,
        action: str,
        phase: str,
        slot_id: int,
        token_range: tuple[int, int],
    ) -> None:
        generation = self.pool.snapshot()[slot_id].generation
        self.trace.record(
            TraceRecord(
                tag=ExecutionTag(
                    scope_id=spec.scope_id,
                    epoch=spec.epoch,
                    operation_id=spec.operation_id,
                    object_type="kv_block",
                    generation=generation,
                    tenant_id=spec.tenant_id,
                ),
                action=action,
                phase=phase,
                request_id=spec.request_id,
                token_range=token_range,
                physical_block=slot_id,
            )
        )

    def _materialize_request(self, spec: RequestSpec) -> None:
        owner = self._owner(spec)
        self.request_owners[spec.request_id] = owner
        self.request_specs[spec.request_id] = spec
        block_size = self.fingerprint.block_size
        total_blocks = (len(spec.prompt_token_ids) + block_size - 1) // block_size
        inherited = self._prepared_reuse.pop(spec.request_id, ())
        first_new_block = len(inherited)
        new_count = total_blocks - first_new_block
        new_slots = self.pool.allocate(owner, new_count) if new_count else ()
        for block_index, slot_id in enumerate(new_slots, start=first_new_block):
            start = block_index * block_size
            end = min(start + block_size, len(spec.prompt_token_ids))
            token_range = (start, end)
            self._trace_slot(
                spec,
                action="kv_write",
                phase="start",
                slot_id=slot_id,
                token_range=token_range,
            )
            ticket = self.pool.begin_write(slot_id, owner)
            payload = tuple(spec.prompt_token_ids[start:end])
            if not self.pool.complete_write(ticket, payload):
                raise RuntimeError("current request KV write was rejected")
            if end - start == block_size:
                self.pool.seal(
                    slot_id,
                    owner,
                    token_range=token_range,
                    fingerprint=self.fingerprint,
                )
            self._trace_slot(
                spec,
                action="kv_write",
                phase="complete",
                slot_id=slot_id,
                token_range=token_range,
            )
        self.request_slots[spec.request_id] = inherited + new_slots

    def begin_late_write(self, request_id: str, block_index: int) -> WriterTicket:
        """Inject an old writer into a mutable block for reclamation tests."""

        spec = self.request_specs[request_id]
        owner = self.request_owners[request_id]
        slot_id = self.request_slots[request_id][block_index]
        snapshot = self.pool.snapshot()[slot_id]
        token_range = snapshot.token_range or (
            block_index * self.fingerprint.block_size,
            min(
                (block_index + 1) * self.fingerprint.block_size,
                len(spec.prompt_token_ids),
            ),
        )
        ticket = self.pool.begin_write(slot_id, owner)
        self._writer_events[ticket] = asyncio.Event()
        self._trace_slot(
            spec,
            action="late_kv_write",
            phase="start",
            slot_id=slot_id,
            token_range=token_range,
        )
        return ticket

    def complete_late_write(self, ticket: WriterTicket, payload: Any) -> bool:
        installed = self.pool.complete_write(ticket, payload)
        spec = next(
            spec
            for spec in self.request_specs.values()
            if self._owner(spec) == ticket.owner
        )
        block_index = self.request_slots[spec.request_id].index(ticket.slot_id)
        token_range = (
            block_index * self.fingerprint.block_size,
            min(
                (block_index + 1) * self.fingerprint.block_size,
                len(spec.prompt_token_ids),
            ),
        )
        self.trace.record(
            TraceRecord(
                tag=ExecutionTag(
                    scope_id=ticket.owner.scope_id,
                    epoch=ticket.owner.epoch,
                    operation_id=ticket.owner.operation_id,
                    object_type="kv_block",
                    generation=ticket.generation,
                    tenant_id=ticket.owner.tenant_id,
                ),
                action="late_kv_install",
                phase="commit" if installed else "drop",
                request_id=spec.request_id,
                token_range=token_range,
                physical_block=ticket.slot_id,
            )
        )
        event = self._writer_events[ticket]
        event.set()
        return installed

    async def generate(self, spec: RequestSpec) -> AsyncIterator[Any]:
        self._materialize_request(spec)
        async for output in super().generate(spec):
            yield output

    async def probe_reuse(
        self,
        scope_id: str,
        retired_epoch: int,
        source_request_id: str | None,
        block_aligned_tokens: int,
    ) -> ReuseProbe:
        if source_request_id is None or block_aligned_tokens <= 0:
            return ReuseProbe.unavailable(source_request_id)
        owner = self.request_owners.get(source_request_id)
        slots = self.request_slots.get(source_request_id, ())
        block_count = block_aligned_tokens // self.fingerprint.block_size
        candidate = slots[:block_count]
        snapshots = self.pool.snapshot()
        valid = (
            owner is not None
            and owner.scope_id == scope_id
            and owner.epoch == retired_epoch
            and len(candidate) == block_count
            and all(
                snapshots[slot_id].state is KVSlotState.ACTIVE
                and snapshots[slot_id].owner == owner
                and snapshots[slot_id].immutable
                and not snapshots[slot_id].pending_writers
                and snapshots[slot_id].fingerprint == self.fingerprint
                for slot_id in candidate
            )
        )
        if not valid:
            return ReuseProbe.unavailable(source_request_id)
        return ReuseProbe(
            source_request_id=source_request_id,
            tier=ReuseTier.HBM,
            available_tokens=block_aligned_tokens,
            no_pending_writer=True,
            source_fingerprint=self.fingerprint,
            numerical_attestation=self.numerical_attestation,
            source_tenant_id=owner.tenant_id,
        )

    async def prepare_resume(
        self,
        spec: RequestSpec,
        retired_epoch: int,
        recovery: RecoveryPlan,
    ) -> None:
        if not recovery.certificate.admissible:
            return
        source_id = recovery.certificate.source_request_id
        if source_id is None:
            raise RuntimeError("admissible recovery has no source request")
        source_owner = self.request_owners[source_id]
        target_owner = self._owner(spec)
        block_count = recovery.reuse_tokens // self.fingerprint.block_size
        source_slots = self.request_slots[source_id][:block_count]
        adopted = self.pool.adopt_immutable(
            source_slots,
            source=source_owner,
            target=target_owner,
            fingerprint=self.fingerprint,
            through_token=recovery.reuse_tokens,
        )
        self._prepared_reuse[spec.request_id] = adopted
        for block_index, slot_id in enumerate(adopted):
            token_range = (
                block_index * self.fingerprint.block_size,
                (block_index + 1) * self.fingerprint.block_size,
            )
            self._trace_slot(
                spec,
                action="kv_handoff",
                phase="commit",
                slot_id=slot_id,
                token_range=token_range,
            )

    async def reclaim(
        self, scope_id: str, retired_epoch: int
    ) -> ReclaimReceipt:
        before = self.pool.snapshot()
        owners = {
            owner
            for owner in self.request_owners.values()
            if owner.scope_id == scope_id and owner.epoch == retired_epoch
        }
        quarantined: list[int] = []
        for owner in owners:
            quarantined.extend(self.pool.retire(owner))
        waiting = [
            event
            for ticket, event in self._writer_events.items()
            if ticket.slot_id in quarantined and not event.is_set()
        ]
        if waiting:
            await asyncio.gather(*(event.wait() for event in waiting))
        after = self.pool.snapshot()
        reclaimed = sum(
            old.state is not KVSlotState.FREE and new.state is KVSlotState.FREE
            for old, new in zip(before, after)
        )
        safe = all(
            after[slot_id].state is KVSlotState.FREE for slot_id in quarantined
        )
        return ReclaimReceipt(
            scope_id=scope_id,
            retired_epoch=retired_epoch,
            safe_to_reuse=safe,
            reclaimed_blocks=reclaimed,
            detail=(
                "retired blocks reclaimed or handed to successor"
                if safe
                else f"waiting for writers in slots {sorted(quarantined)}"
            ),
        )
