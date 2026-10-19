from __future__ import annotations

import hashlib

import pytest

from retire_serving import (
    AlignmentCertificate,
    EngineFingerprint,
    GroupRecoveryPlan,
    ParticipantCoordinate,
    RecoveryPlan,
    RequestSpec,
    ReuseTier,
    VLLMDistributedParticipant,
)


class FakeDistributedCore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def call_utility_async(
        self,
        method: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        self.calls.append((method, payload))
        base = {
            "pipeline_parallel_rank": payload["pipeline_parallel_rank"],
            "tensor_parallel_rank": payload["tensor_parallel_rank"],
            "kv_group_id": payload["kv_group_id"],
        }
        if method == "retire_participant_fence":
            return {
                **base,
                "acknowledged": True,
                "writer_drained": True,
                "speculation_drained": True,
                "codec_drained": True,
                "transfer_drained": True,
                "safe_point_sequence": 7,
            }
        if method == "retire_participant_probe":
            decoded = b"decoded-kv"
            encoded = b"encoded-kv"
            return {
                **base,
                "source_request_id": payload["source_request_id"],
                "source_tenant_id": "tenant-a",
                "tier": "dram",
                "available_tokens": 16,
                "committed_tokens": 16,
                "writer_drained": True,
                "speculation_drained": True,
                "codec_drained": True,
                "transfer_drained": True,
                "fingerprint_match": True,
                "numerical_attestation": "strict-clean-recomputation-v1",
                "codec": {
                    "name": "exact-codec",
                    "version": "1",
                    "lossless": True,
                    "encoded_sha256": hashlib.sha256(encoded).hexdigest(),
                    "decoded_sha256": hashlib.sha256(decoded).hexdigest(),
                    "operation_drained": True,
                    "integrity_verified": True,
                },
            }
        if method == "retire_participant_prepare":
            return {**base, "prepared": True}
        if method == "retire_participant_reclaim":
            return {**base, "safe_to_reuse": True, "reclaimed_blocks": 1}
        raise AssertionError(method)


class FakeEngine:
    def __init__(self) -> None:
        self.engine_core = FakeDistributedCore()


def plan(coordinate: ParticipantCoordinate) -> RecoveryPlan:
    return RecoveryPlan(
        tier=ReuseTier.DRAM,
        reuse_tokens=16,
        recompute_tokens=4,
        certificate=AlignmentCertificate(
            source_request_id="old",
            common_prefix_tokens=20,
            block_aligned_tokens=16,
            structural_valid=True,
            numerical_valid=True,
            physical_valid=True,
            admissible=True,
            reason="fixture",
        ),
        group_plans=(GroupRecoveryPlan(coordinate, ReuseTier.DRAM, 16),),
    )


@pytest.mark.asyncio
async def test_vllm_participant_maps_all_four_utility_receipts() -> None:
    engine = FakeEngine()
    coordinate = ParticipantCoordinate(0, "group-0")
    fingerprint = EngineFingerprint(
        model="model",
        block_size=16,
        speculative_mode="draft-verify",
    )
    participant = VLLMDistributedParticipant(
        engine,
        coordinate,
        fingerprint=fingerprint,
    )
    old = RequestSpec(
        scope_id="scope",
        epoch=0,
        request_id="old",
        operation_id=0,
        prompt_token_ids=tuple(range(20)),
        cache_salt="tenant-a:scope:e0",
        sampling_params={},
        tenant_id="tenant-a",
    )
    successor = RequestSpec(
        scope_id="scope",
        epoch=1,
        request_id="new",
        operation_id=1,
        prompt_token_ids=tuple(range(20)),
        cache_salt="tenant-a:scope:e0",
        sampling_params={},
        generation=1,
        tenant_id="tenant-a",
    )
    await participant.bind_request(old)
    fence = await participant.fence("scope", 0, ("old",))
    evidence = await participant.probe_reuse("scope", 0, "old", 16)
    await participant.prepare_resume(successor, 0, plan(coordinate))
    reclaim = await participant.reclaim("scope", 0)

    assert fence.safe
    assert evidence.tier is ReuseTier.DRAM
    assert evidence.codec_valid
    assert evidence.source_tenant_id == "tenant-a"
    assert evidence.committed_tokens == 16
    assert reclaim.safe_to_reuse
    assert [name for name, _ in engine.engine_core.calls] == [
        "retire_participant_fence",
        "retire_participant_probe",
        "retire_participant_prepare",
        "retire_participant_reclaim",
    ]


@pytest.mark.asyncio
async def test_vllm_participant_missing_fields_fails_closed() -> None:
    class EmptyCore:
        async def call_utility_async(
            self,
            method: str,
            payload: dict[str, object],
        ) -> dict[str, object]:
            del method
            return {
                "pipeline_parallel_rank": payload["pipeline_parallel_rank"],
                "tensor_parallel_rank": payload["tensor_parallel_rank"],
                "kv_group_id": payload["kv_group_id"],
            }

    engine = FakeEngine()
    engine.engine_core = EmptyCore()
    participant = VLLMDistributedParticipant(
        engine,
        ParticipantCoordinate(0, "group-0"),
        fingerprint=EngineFingerprint(model="model"),
    )
    receipt = await participant.fence("scope", 0, ())
    assert not receipt.safe
