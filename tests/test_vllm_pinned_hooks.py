from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from retire_serving import (
    AlignmentCertificate,
    EngineFingerprint,
    PinnedVLLMHooks,
    RecoveryPlan,
    RequestSpec,
    ReuseTier,
    VLLMAsyncBackend,
)


class FakeCoreClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    async def call_utility_async(self, method: str, *args: object) -> object:
        self.calls.append((method, args))
        if method == "retire_fence_and_pin":
            request_ids = args[1]
            assert isinstance(request_ids, list)
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
                    request_ids[0]: {
                        "pinned_tokens": 16,
                        "committed_tokens": 16,
                        "group_block_ids": [[41]],
                        "group_tokens": [16],
                        "writer_drained": True,
                        "all_blocks_resident": True,
                    }
                },
            }
        if method == "retire_verify_request_blocks":
            return {
                "installed_group_block_ids": [[41]],
                "same_physical_blocks": True,
            }
        if method == "retire_trim_pins":
            return {
                "kept_blocks": 1,
                "released_blocks": 0,
                "kept_group_block_ids": [[41]],
            }
        if method == "retire_expect_request_blocks":
            return {"request_id": args[0], "armed": True}
        if method == "retire_cancel_request_expectation":
            return {"request_id": args[0], "removed": True}
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


class FakePatchedEngine:
    def __init__(self) -> None:
        self.engine_core = FakeCoreClient()

    async def generate(
        self, prompt: dict[str, object], sampling_params: object, request_id: str
    ) -> AsyncIterator[dict[str, object]]:
        del prompt, sampling_params
        yield {"request_id": request_id}


class DelayedVerifyCoreClient(FakeCoreClient):
    def __init__(self) -> None:
        super().__init__()
        self.verify_started = asyncio.Event()
        self.allow_verify = asyncio.Event()

    async def call_utility_async(self, method: str, *args: object) -> object:
        if method == "retire_verify_request_blocks":
            self.calls.append((method, args))
            self.verify_started.set()
            await self.allow_verify.wait()
            return {
                "installed_group_block_ids": [[41]],
                "same_physical_blocks": True,
            }
        return await super().call_utility_async(method, *args)


def admissible_plan(source_request_id: str) -> RecoveryPlan:
    return RecoveryPlan(
        tier=ReuseTier.HBM,
        reuse_tokens=16,
        recompute_tokens=4,
        certificate=AlignmentCertificate(
            source_request_id=source_request_id,
            common_prefix_tokens=19,
            block_aligned_tokens=16,
            structural_valid=True,
            numerical_valid=True,
            physical_valid=True,
            admissible=True,
            reason="fixture",
        ),
    )


@pytest.mark.asyncio
async def test_pinned_hook_fences_verifies_same_blocks_and_releases() -> None:
    engine = FakePatchedEngine()
    fingerprint = EngineFingerprint(model="fake", block_size=16)
    hooks = PinnedVLLMHooks(
        engine,
        fingerprint=fingerprint,
        numerical_attestation="token-contract",
        tenant_pin_quota_blocks=8,
    )
    await hooks.install_scope_fence("scope", 0, ("old",))
    probe = await hooks.probe_reuse("scope", 0, "old", 16)
    assert probe.no_pending_writer
    assert probe.available_tokens == 16
    assert probe.numerical_attestation == "token-contract"

    successor = RequestSpec(
        scope_id="scope",
        epoch=1,
        request_id="new",
        operation_id=7,
        prompt_token_ids=(1, 2, 3),
        cache_salt="same",
        sampling_params=object(),
    )
    await hooks.prepare_resume(successor, 0, admissible_plan("old"))
    backend = VLLMAsyncBackend(engine, fingerprint=fingerprint, hooks=hooks)
    assert [output async for output in backend.generate(successor)] == [
        {"request_id": "new"}
    ]
    reclaim = await hooks.reclaim_scope("scope", 0)
    assert reclaim.safe_to_reuse
    assert hooks.verifications[0].same_physical_blocks
    assert hooks.verifications[0].kv_group_id == "group-0"
    assert [name for name, _ in engine.engine_core.calls] == [
        "retire_fence_and_pin",
        "retire_trim_pins",
        "retire_expect_request_blocks",
        "retire_verify_request_blocks",
        "retire_release_pins",
    ]
    fence_args = engine.engine_core.calls[0][1]
    assert fence_args[2:] == ("default", 8)
    trim_args = engine.engine_core.calls[1][1]
    assert trim_args == ("scope:e0", "old", 16)
    expectation_args = engine.engine_core.calls[2][1]
    assert expectation_args == ("new", [[41]])


@pytest.mark.asyncio
async def test_adoption_audit_does_not_block_first_output() -> None:
    engine = FakePatchedEngine()
    engine.engine_core = DelayedVerifyCoreClient()
    hooks = PinnedVLLMHooks(
        engine,
        fingerprint=EngineFingerprint(model="fake", block_size=16),
        numerical_attestation="token-contract",
    )
    await hooks.install_scope_fence("scope", 0, ("old",))
    successor = RequestSpec(
        scope_id="scope",
        epoch=1,
        request_id="new",
        operation_id=7,
        prompt_token_ids=(1, 2, 3),
        cache_salt="same",
        sampling_params=object(),
    )
    await hooks.prepare_resume(successor, 0, admissible_plan("old"))
    backend = VLLMAsyncBackend(engine, fingerprint=hooks.fingerprint, hooks=hooks)
    stream = backend.generate(successor).__aiter__()

    first = await asyncio.wait_for(anext(stream), timeout=0.1)
    assert first == {"request_id": "new"}
    await asyncio.wait_for(engine.engine_core.verify_started.wait(), timeout=0.1)
    assert not any(
        method == "retire_release_pins" for method, _ in engine.engine_core.calls
    )

    engine.engine_core.allow_verify.set()
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    reclaim = await hooks.reclaim_scope("scope", 0)
    assert reclaim.safe_to_reuse


@pytest.mark.asyncio
async def test_prepared_successor_revoked_before_admission_releases_pins() -> None:
    engine = FakePatchedEngine()
    fingerprint = EngineFingerprint(model="fake", block_size=16)
    hooks = PinnedVLLMHooks(
        engine,
        fingerprint=fingerprint,
        numerical_attestation="token-contract",
    )
    await hooks.install_scope_fence("scope", 0, ("old",))
    successor = RequestSpec(
        scope_id="scope",
        epoch=1,
        request_id="never-admitted",
        operation_id=7,
        prompt_token_ids=(1, 2, 3),
        cache_salt="same",
        sampling_params=object(),
    )
    await hooks.prepare_resume(successor, 0, admissible_plan("old"))
    backend = VLLMAsyncBackend(engine, fingerprint=fingerprint, hooks=hooks)

    await backend.discard_unlaunched(successor)
    reclaim = await hooks.reclaim_scope("scope", 0)

    assert reclaim.safe_to_reuse
    assert [name for name, _ in engine.engine_core.calls] == [
        "retire_fence_and_pin",
        "retire_trim_pins",
        "retire_expect_request_blocks",
        "retire_cancel_request_expectation",
        "retire_release_pins",
    ]


@pytest.mark.asyncio
async def test_empty_terminal_frame_does_not_trigger_adoption_verification() -> None:
    engine = FakePatchedEngine()
    hooks = PinnedVLLMHooks(
        engine,
        fingerprint=EngineFingerprint(model="fake", block_size=16),
        numerical_attestation="token-contract",
    )
    await hooks.install_scope_fence("scope", 0, ("old",))
    successor = RequestSpec(
        scope_id="scope",
        epoch=1,
        request_id="new",
        operation_id=7,
        prompt_token_ids=(1, 2, 3),
        cache_salt="same",
        sampling_params=object(),
    )
    await hooks.prepare_resume(successor, 0, admissible_plan("old"))

    class EmptyOutput:
        outputs = ()

    await hooks.observe_output(successor, EmptyOutput())
    await hooks.request_complete(successor)

    names = [name for name, _ in engine.engine_core.calls]
    assert "retire_verify_request_blocks" not in names
    assert "retire_cancel_request_expectation" in names


@pytest.mark.asyncio
async def test_terminal_retirement_releases_pins_without_a_successor() -> None:
    engine = FakePatchedEngine()
    fingerprint = EngineFingerprint(model="fake", block_size=16)
    hooks = PinnedVLLMHooks(
        engine,
        fingerprint=fingerprint,
        numerical_attestation="token-contract",
    )
    await hooks.install_scope_fence("scope", 0, ("old",))
    backend = VLLMAsyncBackend(engine, fingerprint=fingerprint, hooks=hooks)

    await backend.prepare_termination("scope", 0)
    reclaim = await hooks.reclaim_scope("scope", 0)

    assert reclaim.safe_to_reuse
    assert [name for name, _ in engine.engine_core.calls] == [
        "retire_fence_and_pin",
        "retire_release_pins",
    ]


@pytest.mark.asyncio
async def test_zero_reuse_cap_applies_to_group_evidence() -> None:
    engine = FakePatchedEngine()
    hooks = PinnedVLLMHooks(
        engine,
        fingerprint=EngineFingerprint(model="fake", block_size=16),
        numerical_attestation="token-contract",
        max_reuse_tokens=0,
    )
    await hooks.install_scope_fence("scope", 0, ("old",))
    probe = await hooks.probe_reuse("scope", 0, "old", 16)
    assert probe.available_tokens == 0
    assert probe.group_evidence[0].available_tokens == 0


@pytest.mark.asyncio
async def test_cold_fast_path_skips_physical_pin_and_reclaims_after_drain() -> None:
    engine = FakePatchedEngine()
    hooks = PinnedVLLMHooks(
        engine,
        fingerprint=EngineFingerprint(model="fake", block_size=16),
        numerical_attestation="token-contract",
        max_reuse_tokens=0,
        cold_fast_path=True,
    )
    participants = await hooks.install_scope_fence("scope", 0, ("old",))
    probe = await hooks.probe_reuse("scope", 0, "old", 16)
    reclaim = await hooks.reclaim_scope("scope", 0)
    assert participants == ()
    assert not hooks.requires_participant_fence
    assert probe.available_tokens == 0
    assert not probe.no_pending_writer
    assert reclaim.safe_to_reuse
    assert engine.engine_core.calls == []


def test_cold_request_does_not_enable_per_output_adoption_observation() -> None:
    hooks = PinnedVLLMHooks(
        FakePatchedEngine(),
        fingerprint=EngineFingerprint(model="fake", block_size=16),
        numerical_attestation="token-contract",
        max_reuse_tokens=0,
        cold_fast_path=True,
    )
    request = RequestSpec(
        scope_id="scope",
        epoch=0,
        request_id="cold",
        operation_id=0,
        prompt_token_ids=(1, 2, 3),
        cache_salt="cold",
        sampling_params=object(),
    )
    assert not hooks.needs_output_observation(request)


def test_cold_fast_path_rejects_a_nonzero_reuse_cap() -> None:
    with pytest.raises(ValueError, match="max_reuse_tokens=0"):
        PinnedVLLMHooks(
            FakePatchedEngine(),
            fingerprint=EngineFingerprint(model="fake", block_size=16),
            numerical_attestation="token-contract",
            max_reuse_tokens=16,
            cold_fast_path=True,
        )
