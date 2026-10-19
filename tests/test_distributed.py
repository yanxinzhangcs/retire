from __future__ import annotations

import asyncio

import pytest

from retire_serving import (
    DistributedRetirementBackend,
    EngineFingerprint,
    KVCacheGroupFingerprint,
    ParticipantCoordinate,
    RetireServing,
    ReuseTier,
    ScriptedRetirementParticipant,
    SpeculativeOutput,
)
from retire_serving.scripted import ScriptedBackend


def distributed_fingerprint() -> EngineFingerprint:
    return EngineFingerprint(
        model="distributed-model",
        revision="test",
        dtype="bf16",
        tensor_parallel_size=2,
        block_size=4,
        kv_cache_groups=(
            KVCacheGroupFingerprint("attention", 4, dtype="bf16"),
            KVCacheGroupFingerprint("recurrent", 8, dtype="fp32"),
        ),
        speculative_mode="draft-verify",
        draft_model="draft-model",
    )


def make_distributed_backend() -> tuple[
    ScriptedBackend,
    DistributedRetirementBackend,
    tuple[ScriptedRetirementParticipant, ...],
]:
    fingerprint = distributed_fingerprint()
    data_plane = ScriptedBackend(fingerprint)
    participants = tuple(
        ScriptedRetirementParticipant(
            coordinate,
            fingerprint=fingerprint,
            tier=(
                ReuseTier.DRAM
                if coordinate.kv_group_id == "recurrent"
                else ReuseTier.HBM
            ),
            committed_token_limit=(
                16
                if coordinate == ParticipantCoordinate(1, "attention")
                else None
            ),
        )
        for coordinate in fingerprint.participant_coordinates
    )
    return (
        data_plane,
        DistributedRetirementBackend(
            data_plane,
            participants,
            fingerprint=fingerprint,
        ),
        participants,
    )


async def wait_generated(data_plane: ScriptedBackend, request_id: str) -> None:
    for _ in range(100):
        if any(spec.request_id == request_id for spec in data_plane.generated_specs):
            return
        await asyncio.sleep(0)
    raise TimeoutError(f"request {request_id} did not enter the data plane")


@pytest.mark.asyncio
async def test_unified_distributed_path_intersects_all_groups_and_tiers() -> None:
    data_plane, backend, participants = make_distributed_backend()
    runtime = RetireServing(backend)
    old = await runtime.start(
        "agent-scope",
        list(range(24)),
        {},
        tenant_id="tenant-a",
    )
    await wait_generated(data_plane, old.request_id)

    revised = list(range(20)) + [900, 901, 902, 903]
    transition = await runtime.supersede(
        "agent-scope",
        revised,
        {},
        tenant_id="tenant-a",
    )

    assert len(transition.fence.participants) == 4
    assert all(receipt.safe for receipt in transition.fence.participants)
    assert transition.recovery.certificate.admissible
    assert transition.recovery.reuse_tokens == 16
    assert transition.recovery.tier is ReuseTier.MIXED
    assert len(transition.recovery.group_plans) == 4
    assert {plan.coordinate for plan in transition.recovery.group_plans} == set(
        backend.fingerprint.participant_coordinates
    )
    assert all(plan.reuse_tokens == 16 for plan in transition.recovery.group_plans)

    # Draft-only and stale-generation results never cross the API commit gate.
    data_plane.emit(
        transition.successor.request_id,
        SpeculativeOutput("draft-only", generation=1, verified=False),
    )
    data_plane.emit(
        transition.successor.request_id,
        SpeculativeOutput("stale-verify", generation=0, verified=True),
    )
    data_plane.emit(
        transition.successor.request_id,
        SpeculativeOutput("verified-current", generation=1, verified=True),
    )
    data_plane.finish(transition.successor.request_id)
    data_plane.emit(
        old.request_id,
        SpeculativeOutput("verified-but-retired", generation=0, verified=True),
    )
    data_plane.finish(old.request_id)

    assert [item async for item in old] == []
    assert [item async for item in transition.successor] == ["verified-current"]
    assert transition.successor.stats.stale_outputs_dropped == 2
    reclaim = await transition.wait_reclaim()
    assert reclaim.safe_to_reuse
    assert len(reclaim.participants) == 4
    assert all(item.safe_to_reuse for item in reclaim.participants)

    dram_participants = [
        participant
        for participant in participants
        if participant.tier is ReuseTier.DRAM
    ]
    assert len(dram_participants) == 2
    assert all(
        transition.successor.request_id in participant.restored_payloads
        for participant in dram_participants
    )

    # The DRAM-restored successor is itself a valid source for the next epoch;
    # tiering is part of the normal path rather than a one-shot side channel.
    second = await runtime.supersede(
        "agent-scope",
        list(range(16)) + [800, 801, 802, 803, 804, 805, 806, 807],
        {},
        tenant_id="tenant-a",
    )
    assert second.recovery.certificate.admissible
    assert second.recovery.reuse_tokens == 16
    assert second.recovery.tier is ReuseTier.MIXED
    data_plane.finish(second.successor.request_id)
    await second.wait_reclaim()


def test_distributed_backend_rejects_missing_tp_or_kv_group() -> None:
    data_plane, _, participants = make_distributed_backend()
    with pytest.raises(ValueError, match="cover every PP stage, TP rank"):
        DistributedRetirementBackend(
            data_plane,
            participants[:-1],
            fingerprint=distributed_fingerprint(),
        )


def test_participant_coordinates_cover_pipeline_tensor_and_groups() -> None:
    fingerprint = EngineFingerprint(
        model="pipeline-model",
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        kv_cache_groups=(
            KVCacheGroupFingerprint("attention", 4),
            KVCacheGroupFingerprint("recurrent", 8),
        ),
    )

    assert len(fingerprint.participant_coordinates) == 8
    assert {
        (
            item.pipeline_parallel_rank,
            item.tensor_parallel_rank,
            item.kv_group_id,
        )
        for item in fingerprint.participant_coordinates
    } == {
        (pp, tp, group)
        for pp in range(2)
        for tp in range(2)
        for group in ("attention", "recurrent")
    }


@pytest.mark.asyncio
async def test_legacy_whole_engine_receipt_cannot_certify_multi_tp() -> None:
    fingerprint = EngineFingerprint(
        model="multi-tp",
        tensor_parallel_size=2,
        block_size=4,
    )
    backend = ScriptedBackend(fingerprint)
    runtime = RetireServing(backend)
    old = await runtime.start("scope", list(range(8)), {})

    with pytest.raises(RuntimeError, match="participant-level fence receipts"):
        await runtime.supersede("scope", [0, 1, 2, 3, 9, 10, 11, 12], {})

    backend.emit(old.request_id, "late-old")
    backend.finish(old.request_id)
    assert [item async for item in old] == []


@pytest.mark.asyncio
async def test_unsafe_speculative_fence_fails_closed_after_advance() -> None:
    fingerprint = EngineFingerprint(
        model="speculative-model",
        block_size=4,
        speculative_mode="draft-verify",
        draft_model="draft",
    )
    data_plane = ScriptedBackend(fingerprint)
    participant = ScriptedRetirementParticipant(
        ParticipantCoordinate(0, "group-0"),
        fingerprint=fingerprint,
    )
    backend = DistributedRetirementBackend(data_plane, (participant,))
    runtime = RetireServing(backend)
    old = await runtime.start("scope", list(range(8)), {})
    await wait_generated(data_plane, old.request_id)
    participant.speculation_drained = False

    with pytest.raises(RuntimeError, match="safe distributed fence"):
        await runtime.supersede("scope", [0, 1, 2, 3, 9, 10, 11, 12], {})

    assert runtime.registry.current_epoch("scope") == 1
    data_plane.emit(
        old.request_id,
        SpeculativeOutput("late-old", generation=0, verified=True),
    )
    data_plane.finish(old.request_id)
    assert [item async for item in old] == []


@pytest.mark.asyncio
async def test_one_bad_group_attestation_forces_cold_fallback_for_all_groups() -> None:
    data_plane, backend, participants = make_distributed_backend()
    participants[-1].numerical_attestation = "wrong-contract"
    runtime = RetireServing(backend)
    old = await runtime.start("scope", list(range(24)), {})
    await wait_generated(data_plane, old.request_id)

    transition = await runtime.supersede(
        "scope",
        list(range(20)) + [999, 998, 997, 996],
        {},
    )
    assert transition.recovery.tier is ReuseTier.COLD
    assert transition.recovery.reuse_tokens == 0
    assert not transition.recovery.certificate.numerical_valid
    assert not any(
        transition.successor.request_id in participant.prepared_requests
        for participant in participants
    )

    data_plane.finish(old.request_id)
    data_plane.finish(transition.successor.request_id)
    await transition.wait_reclaim()
