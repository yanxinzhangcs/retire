# How Should LLM Servers Handle Agent Interruptions?

A user changes an instruction while a coding agent is generating a plan. The
agent abandons its current approach and submits a revised prompt. That sounds
like a request cancellation followed by another request, but several things
are still in flight: buffered tokens, model execution, and writes to state
that the next request might reuse.

[RETIRE](../README.md) treats this event as a handoff between execution versions.
It makes the runtime's interruption decision effective across output
publication, in-flight work, and state ownership. The design is described in
[Serving a Revisable World: Versioned Execution for Interruptible Agents](https://arxiv.org/abs/2610.01160).

## A concrete mid-execution revision

Suppose the old request contains repository context followed by “replace this
API.” The new instruction is “keep the API and optimize its implementation.”
The repository context still matches, but the old plan no longer applies.

The agent runtime decides which work to replace and supplies the revised token
context. RETIRE starts at the serving boundary, where that decision must take
effect across output, computation, and memory.

Once the runtime replaces the old plan, its buffered output and outstanding
state writes must lose permission to affect the application or the successor.
The successor must also be able to make progress while isolated obsolete work
drains.

The transition can incur two costs: waiting for obsolete computation to reach
a stopping boundary, and recomputing context that remains useful. Bounded
retirement and certified state inheritance reduce these costs within the same
version transition.

## Making an interruption take effect across serving

Cancellation tells a scheduler to stop work. It does not by itself prove that
every buffered output has been discarded or every state writer has closed.
Likewise, a matching KV-cache prefix does not prove that no old writer can
modify its pages or that reuse meets the successor's numerical requirements.

RETIRE assigns an execution version to a runtime-selected scope. The scope can
contain one request or several requests belonging to the abandoned plan. An
authority check at publication rejects output belonging to an obsolete version.

The handoff has five stages:

1. **ADVANCE:** select the new version and revoke the old version's authority.
2. **FENCE:** stop new obsolete work and obtain the required writer-closure evidence.
3. **ALIGN:** identify the completed common prefix and validate its reuse contract.
4. **RESUME:** admit the revised execution with the certified state, or recompute.
5. **RECLAIM:** release old resources once their remaining users have drained.

Logical invalidation happens before physical cleanup. Only the state needed by
the successor must be ready for its admission; isolated old resources can be
reclaimed asynchronously.

## Interruption, revision, preemption, and recovery

These terms describe related problems, but their contracts differ.

| Term | Meaning in this project |
| --- | --- |
| Agent interruption | A runtime decision to replace live work with a revised execution |
| Mid-execution revision | A change of execution version and publication authority |
| Request cancellation | Termination of a request; it need not have a successor |
| Preemption or pause/resume | Suspending work that may later continue with the same identity |
| KV-state recovery | Restoring or inheriting state after validating its reuse contract |
| Workflow crash recovery | Restoring an agent's durable application state; outside this runtime's scope |

RETIRE does not choose a new plan, reverse an external tool action, or infer
which previous reasoning remains semantically valid. The agent runtime makes
those application decisions. RETIRE enforces the resulting serving transition.

## Try the handoff

Follow the [quick start](../README.md#try-it-locally), then run:

```bash
python -m retire_serving.demo
```

The deterministic example uses two synthetic token sequences with a shared
four-token prefix. It queues obsolete output, revises the task, injects another
late old output, and checks that only the successor's output is visible. It
also checks that the successor inherits the same certified prefix block and
that old resources can be reclaimed safely.

This is an executable reference model of the protocol. The
[vLLM integration](vllm-integration.md) connects the same runtime to model
serving through explicit backend hooks.
