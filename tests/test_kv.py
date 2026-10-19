from __future__ import annotations

import pytest

from retire_serving import EngineFingerprint, KVBlockPool, KVOwner, KVSlotState


def test_quarantine_prevents_late_writer_from_corrupting_successor() -> None:
    pool = KVBlockPool(2)
    old = KVOwner("scope", 0)
    new = KVOwner("scope", 1)

    (old_slot,) = pool.allocate(old)
    old_writer = pool.begin_write(old_slot, old)
    assert pool.retire(old) == (old_slot,)
    assert pool.snapshot()[old_slot].state is KVSlotState.QUARANTINED

    (new_slot,) = pool.allocate(new)
    assert new_slot != old_slot
    new_writer = pool.begin_write(new_slot, new)
    assert pool.complete_write(new_writer, b"new")

    # The old transfer may finish physically, but only inside its quarantined
    # slot; it can never become reachable from epoch 1.
    assert not pool.complete_write(old_writer, b"late-old")
    assert pool.read(new_slot, new) == b"new"
    assert pool.snapshot()[old_slot].state is KVSlotState.FREE


def test_quarantined_capacity_is_not_allocatable_until_writer_drains() -> None:
    pool = KVBlockPool(1)
    old = KVOwner("scope", 2)
    new = KVOwner("scope", 3)
    (slot,) = pool.allocate(old)
    writer = pool.begin_write(slot, old)
    pool.retire(old)

    with pytest.raises(MemoryError):
        pool.allocate(new)
    pool.complete_write(writer, b"discarded")
    assert pool.allocate(new) == (slot,)


def test_complete_immutable_block_handoff_preserves_bytes_and_advances_generation() -> None:
    pool = KVBlockPool(2)
    fingerprint = EngineFingerprint(model="model", dtype="bf16", block_size=4)
    old = KVOwner("scope", 0, operation_id=10)
    new = KVOwner("scope", 1, operation_id=11)
    (slot,) = pool.allocate(old)
    writer = pool.begin_write(slot, old)
    assert pool.complete_write(writer, b"prefix-kv")
    pool.seal(slot, old, token_range=(0, 4), fingerprint=fingerprint)
    generation = pool.snapshot()[slot].generation

    assert pool.adopt_immutable(
        (slot,),
        source=old,
        target=new,
        fingerprint=fingerprint,
        through_token=4,
    ) == (slot,)
    snapshot = pool.snapshot()[slot]
    assert snapshot.owner == new
    assert snapshot.generation == generation + 1
    assert snapshot.immutable
    assert pool.read(slot, new) == b"prefix-kv"


def test_immutable_handoff_rejects_partial_or_pending_blocks() -> None:
    pool = KVBlockPool(2)
    fingerprint = EngineFingerprint(model="model", block_size=4)
    old = KVOwner("scope", 0)
    new = KVOwner("scope", 1)
    first, second = pool.allocate(old, 2)
    first_writer = pool.begin_write(first, old)
    assert pool.complete_write(first_writer, b"complete")
    pool.seal(first, old, token_range=(0, 4), fingerprint=fingerprint)
    pool.begin_write(second, old)

    with pytest.raises(RuntimeError, match="beyond certified prefix"):
        pool.adopt_immutable(
            (first,),
            source=old,
            target=new,
            fingerprint=fingerprint,
            through_token=3,
        )
    with pytest.raises(RuntimeError, match="mutable or still has a writer"):
        pool.adopt_immutable(
            (second,),
            source=old,
            target=new,
            fingerprint=fingerprint,
            through_token=8,
        )


def test_immutable_handoff_rejects_cross_tenant_ownership() -> None:
    pool = KVBlockPool(1)
    fingerprint = EngineFingerprint(model="model", block_size=4)
    old = KVOwner("scope", 0, tenant_id="tenant-a")
    attacker = KVOwner("scope", 1, tenant_id="tenant-b")
    (slot,) = pool.allocate(old)
    writer = pool.begin_write(slot, old)
    assert pool.complete_write(writer, b"tenant-a-kv")
    pool.seal(slot, old, token_range=(0, 4), fingerprint=fingerprint)

    with pytest.raises(RuntimeError, match="cross-tenant"):
        pool.adopt_immutable(
            (slot,),
            source=old,
            target=attacker,
            fingerprint=fingerprint,
            through_token=4,
        )
