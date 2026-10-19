from __future__ import annotations

import asyncio
from time import monotonic_ns

import pytest

from retire_serving import (
    EngineFingerprint,
    NumericalContract,
    ProtocolStage,
    RetireServing,
    ReuseProbe,
    ReuseTier,
    StaleEpochError,
    SupersessionDescriptor,
    TenantIsolationError,
)
from retire_serving.scripted import ScriptedBackend


def make_backend(block_size: int = 16) -> ScriptedBackend:
    return ScriptedBackend(
        EngineFingerprint(
            model="Qwen3-32B",
            revision="test",
            dtype="bf16",
            block_size=block_size,
            attention_backend="flash-attention",
        )
    )


@pytest.mark.asyncio
async def test_output_is_gated_again_when_stalled_consumer_reads() -> None:
    backend = make_backend(block_size=4)
    runtime = RetireServing(backend)
    old = await runtime.start("scope", [1, 2, 3, 4, 5], {})

    # This output reaches the frontend queue while e0 is current, but the
    # consumer intentionally does not read it yet.
    backend.emit(old.request_id, "queued-old")
    await asyncio.sleep(0)

    transition = await runtime.supersede("scope", [1, 2, 3, 4, 9], {})
    backend.emit(old.request_id, "late-old")
    backend.finish(old.request_id)
    backend.emit(transition.successor.request_id, "current")
    backend.finish(transition.successor.request_id)

    assert [item async for item in old] == []
    assert [item async for item in transition.successor] == ["current"]
    assert old.stats.received_outputs == 2
    assert old.stats.stale_outputs_dropped == 2

    await transition.wait_reclaim()
    assert [event.stage for event in transition.events] == list(ProtocolStage)


@pytest.mark.asyncio
async def test_terminal_scope_retirement_closes_launch_and_output_authority() -> None:
    backend = make_backend(block_size=4)
    runtime = RetireServing(backend)
    old = await runtime.start("scope", [1, 2, 3, 4, 5], {})

    backend.emit(old.request_id, "queued-before-termination")
    await asyncio.sleep(0)

    termination = asyncio.create_task(runtime.terminate("scope"))
    await asyncio.sleep(0)
    assert runtime.registry.current_epoch("scope") == 1
    assert backend.fence_calls == [("scope", 0, (old.request_id,))]

    backend.emit(old.request_id, "late-after-termination")
    backend.finish(old.request_id)
    receipt = await termination

    assert receipt.safe_to_reuse
    assert [item async for item in old] == []
    assert old.stats.stale_outputs_dropped == 2
    assert len(backend.generated_specs) == 1
    assert backend.reclaim_calls == [("scope", 0)]


@pytest.mark.asyncio
async def test_one_scope_fences_multiple_native_vllm_requests() -> None:
    backend = make_backend()
    runtime = RetireServing(backend)
    first = await runtime.start("workflow", list(range(32)), {})
    second = await runtime.start("workflow", list(range(40)), {})

    transition = await runtime.supersede("workflow", list(range(36)), {})
    assert set(transition.fence.acknowledged_request_ids) == {
        first.request_id,
        second.request_id,
    }
    assert first.epoch == second.epoch == 0
    assert transition.successor.epoch == 1

    backend.finish(first.request_id)
    backend.finish(second.request_id)
    backend.finish(transition.successor.request_id)
    await transition.wait_reclaim()


@pytest.mark.asyncio
async def test_public_backend_evidence_defaults_to_cold_recomputation() -> None:
    backend = make_backend()
    runtime = RetireServing(backend)
    old = await runtime.start("scope", list(range(64)), {})

    transition = await runtime.supersede("scope", list(range(55)) + [999], {})
    assert transition.recovery.tier is ReuseTier.COLD
    assert transition.recovery.reuse_tokens == 0
    assert transition.recovery.recompute_tokens == 56
    assert not transition.recovery.certificate.admissible

    backend.finish(old.request_id)
    backend.finish(transition.successor.request_id)
    await transition.wait_reclaim()


@pytest.mark.asyncio
async def test_attested_hbm_prefix_is_reused_at_block_granularity() -> None:
    backend = make_backend()
    runtime = RetireServing(backend)
    old = await runtime.start("scope", list(range(64)), {})
    backend.next_probe = ReuseProbe(
        source_request_id=old.request_id,
        tier=ReuseTier.HBM,
        available_tokens=64,
        no_pending_writer=True,
        source_fingerprint=backend.fingerprint,
        numerical_attestation="strict-clean-recomputation-v1",
    )

    revised = list(range(55)) + [999]
    transition = await runtime.supersede("scope", revised, {})
    assert transition.recovery.tier is ReuseTier.HBM
    assert transition.recovery.reuse_tokens == 48
    assert transition.recovery.recompute_tokens == 8
    assert transition.recovery.certificate.admissible

    backend.finish(old.request_id)
    backend.finish(transition.successor.request_id)
    await transition.wait_reclaim()


@pytest.mark.asyncio
async def test_completed_shorter_prefix_is_reused_during_partial_prefill() -> None:
    backend = make_backend()
    runtime = RetireServing(backend)
    old = await runtime.start("scope", list(range(128)), {})
    backend.next_probe = ReuseProbe(
        source_request_id=old.request_id,
        tier=ReuseTier.HBM,
        available_tokens=32,
        no_pending_writer=True,
        source_fingerprint=backend.fingerprint,
        numerical_attestation="strict-clean-recomputation-v1",
    )

    revised = list(range(95)) + [999]
    transition = await runtime.supersede("scope", revised, {})
    assert transition.recovery.tier is ReuseTier.HBM
    assert transition.recovery.certificate.block_aligned_tokens == 80
    assert transition.recovery.reuse_tokens == 32
    assert transition.recovery.recompute_tokens == 64
    assert transition.recovery.certificate.admissible

    backend.finish(old.request_id)
    backend.finish(transition.successor.request_id)
    await transition.wait_reclaim()


@pytest.mark.asyncio
async def test_pending_writer_forces_cold_fallback() -> None:
    backend = make_backend()
    runtime = RetireServing(
        backend,
        numerical_contract=NumericalContract.strict_clean_recomputation(),
    )
    old = await runtime.start("scope", list(range(64)), {})
    backend.next_probe = ReuseProbe(
        source_request_id=old.request_id,
        tier=ReuseTier.HBM,
        available_tokens=64,
        no_pending_writer=False,
        source_fingerprint=backend.fingerprint,
        numerical_attestation="strict-clean-recomputation-v1",
    )

    transition = await runtime.supersede("scope", list(range(56)), {})
    assert transition.recovery.tier is ReuseTier.COLD
    assert "pending writer" in transition.recovery.certificate.reason

    backend.finish(old.request_id)
    backend.finish(transition.successor.request_id)
    await transition.wait_reclaim()


@pytest.mark.asyncio
async def test_explicit_descriptor_rejects_replay_after_epoch_advance() -> None:
    backend = make_backend(block_size=4)
    runtime = RetireServing(backend)
    old = await runtime.start("scope", [1, 2, 3, 4], {})
    descriptor = SupersessionDescriptor(
        scope_id="scope",
        old_epoch=0,
        new_epoch=1,
        revised_prompt_token_ids=(1, 2, 3, 9),
        contract=runtime.contract,
    )
    transition = await runtime.supersede_descriptor(descriptor, {})

    with pytest.raises(StaleEpochError):
        await runtime.supersede_descriptor(descriptor, {})

    backend.finish(old.request_id)
    backend.finish(transition.successor.request_id)
    await transition.wait_reclaim()


@pytest.mark.asyncio
async def test_queued_old_epoch_task_is_fenced_before_backend_launch() -> None:
    backend = make_backend(block_size=4)
    runtime = RetireServing(backend)
    old = await runtime.start("scope", [1, 2, 3, 4], {})

    # Do not yield after start: the old pump exists but has not entered the
    # backend. ADVANCE must revoke its launch authority.
    transition = await runtime.supersede("scope", [1, 2, 3, 9], {})
    await asyncio.sleep(0)
    launched_ids = {spec.request_id for spec in backend.generated_specs}
    assert old.request_id not in launched_ids

    backend.finish(transition.successor.request_id)
    await transition.wait_reclaim()


@pytest.mark.asyncio
async def test_concurrent_supersessions_are_serialized_by_scope() -> None:
    backend = make_backend(block_size=4)
    runtime = RetireServing(backend)
    old = await runtime.start("scope", [1, 2, 3, 4], {})

    first, second = await asyncio.gather(
        runtime.supersede("scope", [1, 2, 3, 8], {}),
        runtime.supersede("scope", [1, 2, 3, 9], {}),
    )
    assert first.advance.new_epoch == 1
    assert second.advance.new_epoch == 2
    assert runtime.registry.current_epoch("scope") == 2

    backend.finish(old.request_id)
    backend.finish(first.successor.request_id)
    backend.finish(second.successor.request_id)
    await first.wait_reclaim()
    await second.wait_reclaim()


@pytest.mark.asyncio
async def test_scope_namespace_is_bound_to_exactly_one_tenant() -> None:
    backend = make_backend(block_size=4)
    runtime = RetireServing(backend)
    old = await runtime.start(
        "shared-name",
        [1, 2, 3, 4],
        {},
        tenant_id="tenant-a",
    )

    with pytest.raises(TenantIsolationError):
        await runtime.start(
            "shared-name",
            [1, 2, 3, 4],
            {},
            tenant_id="tenant-b",
        )
    with pytest.raises(TenantIsolationError):
        await runtime.supersede(
            "shared-name",
            [1, 2, 3, 9],
            {},
            tenant_id="tenant-b",
        )

    backend.finish(old.request_id)
    await old.wait()


@pytest.mark.asyncio
async def test_production_fast_path_can_disable_per_output_commit_tracing() -> None:
    backend = make_backend(block_size=4)
    runtime = RetireServing(backend, record_committed_outputs=False)
    handle = await runtime.start("scope", [1, 2, 3, 4], {})
    backend.emit(handle.request_id, "current")
    backend.finish(handle.request_id)

    assert [item async for item in handle] == ["current"]
    assert handle.stats.published_outputs == 1
    assert not any(
        record.action == "output_publication" and record.phase == "commit"
        for record in runtime.trace.snapshot()
    )


@pytest.mark.asyncio
async def test_expired_descriptor_advances_then_fails_closed_before_fence() -> None:
    backend = make_backend(block_size=4)
    runtime = RetireServing(backend)
    old = await runtime.start("deadline-scope", [1, 2, 3, 4], {})
    descriptor = SupersessionDescriptor(
        scope_id="deadline-scope",
        old_epoch=0,
        new_epoch=1,
        revised_prompt_token_ids=(1, 2, 3, 9),
        contract=runtime.contract,
        deadline_ns=monotonic_ns() - 1,
    )

    with pytest.raises(TimeoutError, match="deadline expired"):
        await runtime.supersede_descriptor(descriptor, {})
    assert runtime.registry.current_epoch("deadline-scope") == 1
    assert backend.fence_calls == []

    backend.emit(old.request_id, "late-old")
    backend.finish(old.request_id)
    assert [item async for item in old] == []
