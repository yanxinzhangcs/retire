from __future__ import annotations

from typing import Any

import pytest

from retire_serving import (
    AlignmentCertificate,
    EngineFingerprint,
    RecoveryPlan,
    RequestSpec,
    ReuseTier,
    TieredPinnedVLLMHooks,
)


class TieredCore:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def call_utility_async(self, method: str, *args: Any) -> dict[str, Any]:
        self.calls.append(method)
        if method == "retire_fence_and_pin":
            return {
                "safe_point": True,
                "safe_point_sequence": 1,
                "pinned_blocks": 1,
                "participants": [
                    {
                        "pipeline_parallel_rank": 0,
                        "tensor_parallel_rank": 0,
                        "kv_group_id": "group-0",
                        "acknowledged": True,
                        "writer_drained": True,
                        "speculation_drained": True,
                        "codec_drained": True,
                        "transfer_drained": True,
                        "safe_point_sequence": 1,
                    }
                ],
                "sources": {
                    args[1][0]: {
                        "pinned_tokens": 16,
                        "committed_tokens": 16,
                        "group_block_ids": [[7]],
                        "group_tokens": [16],
                        "writer_drained": True,
                        "all_blocks_resident": True,
                    }
                },
            }
        if method == "retire_offload_pins":
            return {
                "codec": "zlib",
                "operation_drained": True,
                "workers": [
                    {
                        "tensor_parallel_rank": 0,
                        "pipeline_parallel_rank": 0,
                        "operation_drained": True,
                        "encoded_sha256": "a" * 64,
                        "decoded_sha256": "b" * 64,
                        "raw_bytes": 128,
                        "encoded_bytes": 64,
                    }
                ],
            }
        if method == "retire_offload_pins_capacity":
            return {
                "codec": "zlib",
                "operation_drained": True,
                "capacity_reclaimed": True,
                "released_hbm_blocks": 1,
                "workers": [
                    {
                        "tensor_parallel_rank": 0,
                        "pipeline_parallel_rank": 0,
                        "operation_drained": True,
                        "encoded_sha256": "a" * 64,
                        "decoded_sha256": "b" * 64,
                        "raw_bytes": 128,
                        "encoded_bytes": 64,
                    }
                ],
            }
        if method == "retire_trim_pins":
            return {
                "kept_blocks": 1,
                "released_blocks": 0,
                "kept_group_block_ids": [[7]],
            }
        if method == "retire_expect_request_blocks":
            return {"request_id": args[0], "armed": True}
        if method == "retire_cancel_request_expectation":
            return {"request_id": args[0], "removed": True}
        if method == "retire_restore_pins":
            return {
                "operation_drained": True,
                "integrity_verified": True,
                "workers": [{"restored_bytes": 128}],
            }
        if method == "retire_restore_capacity_pins":
            return {
                "operation_drained": True,
                "integrity_verified": True,
                "allocated_hbm_blocks": 1,
                "restored_group_block_ids": [[11]],
                "workers": [{"restored_bytes": 128, "remapped_blocks": 1}],
            }
        if method == "retire_verify_request_blocks":
            return {
                "installed_group_block_ids": [[7]],
                "same_physical_blocks": True,
            }
        if method == "retire_release_pins":
            return {
                "released_blocks": 1,
                "safe_point_sequence": 2,
                "worker_receipts": [
                    {
                        "pipeline_parallel_rank": 0,
                        "tensor_parallel_rank": 0,
                        "writer_drained": True,
                    }
                ],
            }
        raise AssertionError(method)


class Engine:
    def __init__(self) -> None:
        self.engine_core = TieredCore()


def test_tiered_hook_defaults_to_identity_codec() -> None:
    hooks = TieredPinnedVLLMHooks(
        Engine(),
        fingerprint=EngineFingerprint(model="model"),
        numerical_attestation="contract",
    )

    assert hooks.codec == "identity"


@pytest.mark.asyncio
async def test_tiered_hook_certifies_offload_and_restores_before_resume() -> None:
    engine = Engine()
    hooks = TieredPinnedVLLMHooks(
        engine,
        fingerprint=EngineFingerprint(model="model"),
        numerical_attestation="contract",
        codec="zlib",
    )
    await hooks.install_scope_fence("scope", 0, ("old",))
    probe = await hooks.probe_reuse("scope", 0, "old", 16)
    assert probe.tier is ReuseTier.DRAM
    assert probe.codec_certificate is not None
    assert probe.codec_certificate.admissible

    recovery = RecoveryPlan(
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
    )
    successor = RequestSpec(
        scope_id="scope",
        epoch=1,
        request_id="new",
        operation_id=1,
        prompt_token_ids=tuple(range(20)),
        cache_salt="default:scope:e0",
        sampling_params={},
        generation=1,
    )
    await hooks.prepare_resume(successor, 0, recovery)
    assert engine.engine_core.calls[:5] == [
        "retire_fence_and_pin",
        "retire_offload_pins",
        "retire_trim_pins",
        "retire_expect_request_blocks",
        "retire_restore_pins",
    ]


@pytest.mark.asyncio
async def test_tiered_hook_trims_certified_transfer_before_offload() -> None:
    engine = Engine()
    hooks = TieredPinnedVLLMHooks(
        engine,
        fingerprint=EngineFingerprint(model="model", block_size=16),
        numerical_attestation="contract",
        max_reuse_tokens=16,
    )

    await hooks.install_scope_fence("scope", 0, ("old",))

    assert engine.engine_core.calls[:3] == [
        "retire_fence_and_pin",
        "retire_trim_pins",
        "retire_offload_pins",
    ]
    receipt = hooks.fence_receipts[0]
    assert receipt["pre_offload_trim"]["kept_group_block_ids"] == [[7]]


@pytest.mark.asyncio
async def test_tiered_hook_reclaims_capacity_and_arms_remapped_successor() -> None:
    engine = Engine()
    hooks = TieredPinnedVLLMHooks(
        engine,
        fingerprint=EngineFingerprint(model="model", block_size=16),
        numerical_attestation="contract",
        codec="zlib",
        reclaim_hbm_capacity=True,
    )
    await hooks.install_scope_fence("scope", 0, ("old",))
    probe = await hooks.probe_reuse("scope", 0, "old", 16)
    recovery = RecoveryPlan(
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
    )
    successor = RequestSpec(
        scope_id="scope",
        epoch=1,
        request_id="new",
        operation_id=1,
        prompt_token_ids=tuple(range(20)),
        cache_salt="default:scope:e0",
        sampling_params={},
        generation=1,
    )

    await hooks.prepare_resume(successor, 0, recovery)

    assert probe.tier is ReuseTier.DRAM
    assert engine.engine_core.calls[:4] == [
        "retire_fence_and_pin",
        "retire_offload_pins_capacity",
        "retire_restore_capacity_pins",
        "retire_expect_request_blocks",
    ]
    assert hooks.fence_receipts[0]["active_pinned_blocks"] == 1
    assert hooks.fence_receipts[0]["sources"]["old"]["group_block_ids"] == [
        [11]
    ]
