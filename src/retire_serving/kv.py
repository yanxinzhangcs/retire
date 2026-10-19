"""Executable KV ownership model for generation and quarantine mechanisms."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import count
from threading import RLock
from typing import Any

from .types import EngineFingerprint


class KVSlotState(str, Enum):
    FREE = "free"
    ACTIVE = "active"
    QUARANTINED = "quarantined"


@dataclass(frozen=True, slots=True)
class KVOwner:
    scope_id: str
    epoch: int
    operation_id: int = 0
    tenant_id: str = "default"


@dataclass(frozen=True, slots=True)
class WriterTicket:
    slot_id: int
    generation: int
    owner: KVOwner
    operation_id: int


@dataclass(slots=True)
class _Slot:
    slot_id: int
    generation: int = 0
    state: KVSlotState = KVSlotState.FREE
    owner: KVOwner | None = None
    payload: Any = None
    pending_writers: set[int] = field(default_factory=set)
    immutable: bool = False
    token_range: tuple[int, int] | None = None
    fingerprint: EngineFingerprint | None = None


@dataclass(frozen=True, slots=True)
class KVSlotSnapshot:
    slot_id: int
    generation: int
    state: KVSlotState
    owner: KVOwner | None
    pending_writers: tuple[int, ...]
    immutable: bool
    token_range: tuple[int, int] | None
    fingerprint: EngineFingerprint | None


class KVBlockPool:
    """Small reference model for the vLLM block-pool changes.

    A retiring slot with an in-flight writer is quarantined, not returned to
    the free list. Therefore a successor receives a different physical slot.
    Generation checks are a second line of defense against late completions.
    """

    def __init__(self, num_slots: int) -> None:
        if num_slots <= 0:
            raise ValueError("num_slots must be positive")
        self._lock = RLock()
        self._slots = [_Slot(slot_id=i) for i in range(num_slots)]
        self._operations = count()

    def allocate(self, owner: KVOwner, count_: int = 1) -> tuple[int, ...]:
        with self._lock:
            free = [slot for slot in self._slots if slot.state is KVSlotState.FREE]
            if len(free) < count_:
                raise MemoryError("insufficient non-quarantined KV slots")
            allocated: list[int] = []
            for slot in free[:count_]:
                slot.generation += 1
                slot.state = KVSlotState.ACTIVE
                slot.owner = owner
                slot.payload = None
                slot.pending_writers.clear()
                slot.immutable = False
                slot.token_range = None
                slot.fingerprint = None
                allocated.append(slot.slot_id)
            return tuple(allocated)

    def begin_write(self, slot_id: int, owner: KVOwner) -> WriterTicket:
        with self._lock:
            slot = self._slots[slot_id]
            if slot.state is not KVSlotState.ACTIVE or slot.owner != owner:
                raise RuntimeError("writer does not own an active KV slot")
            if slot.immutable:
                raise RuntimeError("cannot write an immutable KV slot")
            operation_id = next(self._operations)
            slot.pending_writers.add(operation_id)
            return WriterTicket(
                slot_id=slot_id,
                generation=slot.generation,
                owner=owner,
                operation_id=operation_id,
            )

    def seal(
        self,
        slot_id: int,
        owner: KVOwner,
        *,
        token_range: tuple[int, int],
        fingerprint: EngineFingerprint,
    ) -> None:
        """Mark one complete, writer-free KV block immutable and reusable."""

        start, end = token_range
        if start < 0 or end <= start:
            raise ValueError("token_range must be non-empty and ordered")
        with self._lock:
            slot = self._slots[slot_id]
            if slot.state is not KVSlotState.ACTIVE or slot.owner != owner:
                raise RuntimeError("only the active owner can seal a KV slot")
            if slot.pending_writers:
                raise RuntimeError("cannot seal a KV slot with a pending writer")
            slot.immutable = True
            slot.token_range = token_range
            slot.fingerprint = fingerprint

    def adopt_immutable(
        self,
        slot_ids: tuple[int, ...],
        *,
        source: KVOwner,
        target: KVOwner,
        fingerprint: EngineFingerprint,
        through_token: int,
    ) -> tuple[int, ...]:
        """Hand complete immutable blocks to a successor without copying bytes."""

        with self._lock:
            if source.tenant_id != target.tenant_id:
                raise RuntimeError("cross-tenant KV handoff is forbidden")
            selected = [self._slots[slot_id] for slot_id in slot_ids]
            for slot in selected:
                if slot.state is not KVSlotState.ACTIVE or slot.owner != source:
                    raise RuntimeError("reuse source no longer owns an active slot")
                if not slot.immutable or slot.pending_writers:
                    raise RuntimeError("reuse source is mutable or still has a writer")
                if slot.fingerprint != fingerprint:
                    raise RuntimeError("reuse source fingerprint mismatch")
                if slot.token_range is None or slot.token_range[1] > through_token:
                    raise RuntimeError("reuse source extends beyond certified prefix")
            for slot in selected:
                # Ownership handoff advances the physical generation while
                # preserving immutable KV bytes. Any old ticket is now stale.
                slot.generation += 1
                slot.owner = target
            return tuple(slot.slot_id for slot in selected)

    def complete_write(self, ticket: WriterTicket, payload: Any) -> bool:
        """Return True only when bytes become reachable from the current owner."""

        with self._lock:
            slot = self._slots[ticket.slot_id]
            if (
                slot.generation != ticket.generation
                or ticket.operation_id not in slot.pending_writers
            ):
                return False
            slot.pending_writers.remove(ticket.operation_id)
            if slot.state is KVSlotState.ACTIVE and slot.owner == ticket.owner:
                slot.payload = payload
                return True
            # A real DMA may have touched the quarantined bytes, but no successor
            # can reference them. Release only after the final old writer drains.
            if slot.state is KVSlotState.QUARANTINED and not slot.pending_writers:
                self._release(slot)
            return False

    def retire(self, owner: KVOwner) -> tuple[int, ...]:
        quarantined: list[int] = []
        with self._lock:
            for slot in self._slots:
                if slot.owner != owner or slot.state is not KVSlotState.ACTIVE:
                    continue
                if slot.pending_writers:
                    slot.state = KVSlotState.QUARANTINED
                    quarantined.append(slot.slot_id)
                else:
                    self._release(slot)
        return tuple(quarantined)

    def read(self, slot_id: int, owner: KVOwner) -> Any:
        with self._lock:
            slot = self._slots[slot_id]
            if slot.state is not KVSlotState.ACTIVE or slot.owner != owner:
                raise RuntimeError("KV slot is not reachable by this owner")
            return slot.payload

    def snapshot(self) -> tuple[KVSlotSnapshot, ...]:
        with self._lock:
            return tuple(
                KVSlotSnapshot(
                    slot_id=slot.slot_id,
                    generation=slot.generation,
                    state=slot.state,
                    owner=slot.owner,
                    pending_writers=tuple(sorted(slot.pending_writers)),
                    immutable=slot.immutable,
                    token_range=slot.token_range,
                    fingerprint=slot.fingerprint,
                )
                for slot in self._slots
            )

    @staticmethod
    def _release(slot: _Slot) -> None:
        if slot.pending_writers:
            raise RuntimeError("cannot release a slot with pending writers")
        slot.state = KVSlotState.FREE
        slot.owner = None
        slot.payload = None
        slot.immutable = False
        slot.token_range = None
        slot.fingerprint = None
