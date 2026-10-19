from __future__ import annotations

import asyncio

import pytest

from retire_serving import (
    EngineFingerprint,
    PagedScriptedBackend,
    RetireServing,
    ReuseTier,
    check_invariants,
)


async def wait_materialized(backend: PagedScriptedBackend, request_id: str) -> None:
    for _ in range(100):
        if request_id in backend.request_slots:
            return
        await asyncio.sleep(0)
    raise TimeoutError(f"request {request_id} was not materialized")


@pytest.mark.asyncio
async def test_integrated_paged_backend_hands_off_certified_prefix_and_reclaims_suffix() -> None:
    fingerprint = EngineFingerprint(model="scripted", dtype="bf16", block_size=4)
    backend = PagedScriptedBackend(fingerprint, num_kv_slots=8)
    runtime = RetireServing(backend)
    old = await runtime.start("scope", [1, 2, 3, 4, 5, 6, 7, 8], {})
    await wait_materialized(backend, old.request_id)
    old_slots = backend.request_slots[old.request_id]
    old_prefix_generation = backend.pool.snapshot()[old_slots[0]].generation

    transition = await runtime.supersede("scope", [1, 2, 3, 4, 9, 10, 11, 12], {})
    assert transition.recovery.tier is ReuseTier.HBM
    assert transition.recovery.reuse_tokens == 4
    await wait_materialized(backend, transition.successor.request_id)
    successor_slots = backend.request_slots[transition.successor.request_id]
    assert successor_slots[0] == old_slots[0]
    prefix = backend.pool.snapshot()[successor_slots[0]]
    assert prefix.owner == backend.request_owners[transition.successor.request_id]
    assert prefix.generation == old_prefix_generation + 1
    assert backend.pool.read(
        successor_slots[0], backend.request_owners[transition.successor.request_id]
    ) == (1, 2, 3, 4)

    backend.emit(old.request_id, "late-old")
    backend.finish(old.request_id)
    backend.emit(transition.successor.request_id, "current")
    backend.finish(transition.successor.request_id)
    assert [item async for item in old] == []
    assert [item async for item in transition.successor] == ["current"]
    reclaim = await transition.wait_reclaim()
    assert reclaim.safe_to_reuse

    records = backend.trace.snapshot()
    assert any(record.action == "kv_handoff" for record in records)
    kv_records = [record for record in records if record.tag.object_type == "kv_block"]
    assert all(record.physical_block is not None for record in kv_records)
    assert all(record.token_range is not None for record in kv_records)
    assert [
        record.action
        for record in records
        if record.tag.object_type == "control"
    ] == ["ADVANCE", "FENCE", "ALIGN", "RESUME", "RECLAIM"]
    assert check_invariants(records).passed


@pytest.mark.asyncio
async def test_integrated_resume_precedes_late_writer_reclaim_without_corruption() -> None:
    fingerprint = EngineFingerprint(model="scripted", dtype="bf16", block_size=4)
    backend = PagedScriptedBackend(fingerprint, num_kv_slots=8)
    runtime = RetireServing(backend)
    old = await runtime.start("scope", [1, 2, 3, 4, 5, 6], {})
    await wait_materialized(backend, old.request_id)
    ticket = backend.begin_late_write(old.request_id, block_index=1)

    transition = await runtime.supersede("scope", [1, 2, 3, 4, 9, 10], {})
    await wait_materialized(backend, transition.successor.request_id)
    successor_owner = backend.request_owners[transition.successor.request_id]
    successor_slots = backend.request_slots[transition.successor.request_id]
    backend.emit(transition.successor.request_id, "current")
    backend.finish(transition.successor.request_id)
    backend.finish(old.request_id)
    assert [item async for item in transition.successor] == ["current"]

    reclaim_task = asyncio.create_task(transition.wait_reclaim())
    for _ in range(100):
        if backend.pool.snapshot()[ticket.slot_id].state.value == "quarantined":
            break
        await asyncio.sleep(0)
    else:
        raise TimeoutError("old writer did not enter quarantine")
    assert not reclaim_task.done()
    assert backend.pool.read(successor_slots[0], successor_owner) == (1, 2, 3, 4)

    # The old completion lands only in its quarantined generation and releases
    # that slot. It cannot reach either successor block.
    assert not backend.complete_late_write(ticket, b"late-old")
    receipt = await reclaim_task
    assert receipt.safe_to_reuse
    assert backend.pool.read(successor_slots[0], successor_owner) == (1, 2, 3, 4)
    assert check_invariants(backend.trace.snapshot()).passed
