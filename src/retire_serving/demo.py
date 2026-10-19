"""Demonstrate stale-output rejection and certified reference KV handoff on CPU.

This example uses synthetic tokens and scripted output, not a language model.
"""

from __future__ import annotations

import asyncio

from .invariants import check_invariants
from .paged_scripted import PagedScriptedBackend
from .runtime import RetireServing
from .types import EngineFingerprint, ReuseTier


async def _materialized(backend: PagedScriptedBackend, request_id: str) -> None:
    async with asyncio.timeout(5):
        while request_id not in backend.request_slots:
            await asyncio.sleep(0)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


async def _run() -> None:
    backend = PagedScriptedBackend(
        EngineFingerprint(model="scripted-demo", dtype="reference", block_size=4),
        num_kv_slots=8,
    )
    runtime = RetireServing(backend)
    old = await runtime.start("coding-task", [1, 2, 3, 4, 5, 6, 7, 8], {})
    await _materialized(backend, old.request_id)
    prefix_block = backend.request_slots[old.request_id][0]
    backend.emit(old.request_id, "old output queued before the instruction changed")
    await asyncio.sleep(0)

    transition = await runtime.supersede(
        "coding-task", [1, 2, 3, 4, 9, 10, 11, 12], {}
    )
    await _materialized(backend, transition.successor.request_id)
    inherited_block = backend.request_slots[transition.successor.request_id][0]
    successor_owner = backend.request_owners[transition.successor.request_id]
    _require(
        backend.pool.read(inherited_block, successor_owner) == (1, 2, 3, 4),
        "inherited prefix contents changed",
    )

    backend.emit(old.request_id, "late output after cancellation acknowledgement")
    backend.finish(old.request_id)
    backend.emit(transition.successor.request_id, "revised plan output")
    backend.finish(transition.successor.request_id)
    old_visible = [item async for item in old]
    new_visible = [item async for item in transition.successor]
    reclaim = await transition.wait_reclaim()
    stages = [event.stage.value for event in transition.events]

    _require(old_visible == [], "obsolete output escaped")
    _require(new_visible == ["revised plan output"], "successor output missing")
    _require(transition.recovery.tier is ReuseTier.HBM, "prefix was not inherited")
    _require(transition.recovery.reuse_tokens == 4, "unexpected reuse boundary")
    _require(prefix_block == inherited_block, "prefix block was not handed off")
    _require(reclaim.safe_to_reuse, "old resources are not safe to reclaim")
    _require(
        stages == ["ADVANCE", "FENCE", "ALIGN", "RESUME", "RECLAIM"],
        "incomplete version transition",
    )
    _require(check_invariants(backend.trace.snapshot()).passed, "protocol violation")

    print(f"old API-visible outputs: {old_visible}")
    print(f"new API-visible outputs: {new_visible}")
    print(f"certified prefix tokens reused: {transition.recovery.reuse_tokens}")
    print(f"same prefix block inherited: {prefix_block == inherited_block}")
    print(f"old resources safely reclaimed: {reclaim.safe_to_reuse}")
    print("protocol:", " -> ".join(stages))


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
