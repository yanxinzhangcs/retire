"""API-only supersession example; no physical KV reuse is certified here."""

from __future__ import annotations

import argparse
import asyncio

from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

from retire_serving import EngineFingerprint, RetireServing, VLLMAsyncBackend


async def consume(label: str, handle: object) -> None:
    async for output in handle:  # type: ignore[attr-defined]
        text = output.outputs[0].text if output.outputs else ""
        print(f"[{label} e{handle.epoch}] {text!r}")  # type: ignore[attr-defined]


async def run(model: str) -> None:
    args = AsyncEngineArgs(
        model=model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        enable_prefix_caching=True,
    )
    engine = AsyncLLM.from_engine_args(args)
    tokenizer = AutoTokenizer.from_pretrained(model)
    backend = VLLMAsyncBackend(
        engine,
        fingerprint=EngineFingerprint(
            model=model,
            dtype="bfloat16",
            tensor_parallel_size=1,
            block_size=16,
        ),
    )
    runtime = RetireServing(backend)
    params = SamplingParams(temperature=0, max_tokens=128)

    original = tokenizer.encode(
        "Write a plan for improving a parser. Redesign its public API.",
        add_special_tokens=True,
    )
    revised = tokenizer.encode(
        "Write a plan for improving a parser. Keep its public API unchanged.",
        add_special_tokens=True,
    )

    old = await runtime.start("agent-run-1", original, params)
    old_consumer = asyncio.create_task(consume("old", old))
    await asyncio.sleep(0.1)

    transition = await runtime.supersede("agent-run-1", revised, params)
    await consume("new", transition.successor)
    await old_consumer
    await transition.wait_reclaim()

    print("protocol:", " -> ".join(event.stage.value for event in transition.events))
    print("recovery:", transition.recovery.certificate.reason)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parsed = parser.parse_args()
    asyncio.run(run(parsed.model))


if __name__ == "__main__":
    main()
