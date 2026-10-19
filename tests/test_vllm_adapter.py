from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from retire_serving import EngineFingerprint, RetireServing, VLLMAsyncBackend


class FakeAsyncLLM:
    def __init__(self) -> None:
        self.aborted: list[str] = []
        self.prompts: list[dict[str, object]] = []

    async def generate(
        self, prompt: dict[str, object], sampling_params: object, request_id: str
    ) -> AsyncIterator[dict[str, object]]:
        self.prompts.append(prompt)
        yield {"request_id": request_id, "tokens": prompt["prompt_token_ids"]}

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


class RejectingHooks:
    def __init__(self) -> None:
        self.completed: list[str] = []

    async def bind_request(self, _spec: object) -> bool:
        return False

    async def request_complete(self, spec: object) -> None:
        self.completed.append(spec.request_id)


class EncodingHooks:
    async def bind_request(self, _spec: object) -> bool:
        return True

    def encode_request_cache_salt(self, _spec: object) -> str:
        return "salt-with-authority"


class CompletionHooks:
    def __init__(self) -> None:
        self.completed: list[str] = []

    async def bind_request(self, _spec: object) -> bool:
        raise AssertionError("revoked request must not reach admission")

    async def request_complete(self, spec: object) -> None:
        self.completed.append(spec.request_id)


@pytest.mark.asyncio
async def test_real_vllm_adapter_shape_without_importing_vllm() -> None:
    engine = FakeAsyncLLM()
    backend = VLLMAsyncBackend(
        engine,
        fingerprint=EngineFingerprint(model="fake", block_size=16),
    )
    runtime = RetireServing(backend)

    old = await runtime.start("scope", [1, 2, 3], sampling_params=object())
    assert [output async for output in old] == [
        {"request_id": old.request_id, "tokens": [1, 2, 3]}
    ]

    transition = await runtime.supersede(
        "scope", [1, 2, 4], sampling_params=object()
    )
    assert engine.aborted == []  # completed requests need no physical abort
    assert [output async for output in transition.successor]
    await transition.wait_reclaim()


@pytest.mark.asyncio
async def test_stale_successor_is_dropped_before_engine_admission() -> None:
    engine = FakeAsyncLLM()
    hooks = RejectingHooks()
    backend = VLLMAsyncBackend(
        engine,
        fingerprint=EngineFingerprint(model="fake", block_size=16),
        hooks=hooks,
    )
    runtime = RetireServing(backend)
    handle = await runtime.start("scope", [1, 2, 3], sampling_params=object())
    assert [output async for output in handle] == []
    assert hooks.completed == [handle.request_id]
    assert engine.prompts == []


@pytest.mark.asyncio
async def test_authority_metadata_travels_with_request_admission() -> None:
    engine = FakeAsyncLLM()
    backend = VLLMAsyncBackend(
        engine,
        fingerprint=EngineFingerprint(model="fake", block_size=16),
        hooks=EncodingHooks(),
    )
    runtime = RetireServing(backend)
    handle = await runtime.start("scope", [1, 2, 3], sampling_params=object())
    assert [output async for output in handle]
    assert engine.prompts[0]["cache_salt"] == "salt-with-authority"


@pytest.mark.asyncio
async def test_revoked_before_admission_releases_prepared_backend_state() -> None:
    engine = FakeAsyncLLM()
    hooks = CompletionHooks()
    backend = VLLMAsyncBackend(
        engine,
        fingerprint=EngineFingerprint(model="fake", block_size=16),
        hooks=hooks,
    )
    runtime = RetireServing(backend)

    old = await runtime.start("scope", [1, 2, 3], sampling_params=object())
    runtime.registry.advance(
        "scope",
        expected_epoch=0,
        new_epoch=1,
    )
    await old.wait()

    assert hooks.completed == [old.request_id]
    assert engine.prompts == []
