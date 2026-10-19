# vLLM Integration

RETIRE keeps its versioned runtime independent of the serving engine. The
included vLLM integration has an API-only adapter and a patched core path.

## API-only adapter

[`VLLMAsyncBackend`](../src/retire_serving/backends/vllm.py) wraps a vLLM V1
asynchronous engine. Without internal hooks, it supports request supersession
and API-visible output gating but cannot certify physical KV ownership or
writer closure. The runtime therefore uses conservative cold recomputation.

The optional [`revise_vllm.py`](../examples/revise_vllm.py) example demonstrates
this path. In a separate environment with a compatible vLLM installation and a
model available to it:

```bash
python -m pip install -e .
python examples/revise_vllm.py --model YOUR_MODEL
```

The research patches below target **vLLM 0.13.0**. vLLM is not installed by the
CPU quick start, and compatibility with other versions is not implied.

## Patched core and certified inheritance

[`PinnedVLLMHooks`](../src/retire_serving/backends/vllm_pinned.py) connects the
runtime to core-side fencing, pinning, and physical-block verification. The
patch files are in [`patches/`](../patches/). In a dedicated environment with
vLLM 0.13.0 installed, apply the core patch with:

```bash
bash scripts/apply_vllm_retire_patch.sh
```

The script checks the installed version before modifying its source. It uses
the active Python environment, or the directory supplied in
`VLLM_SITE_PACKAGES`. Keep the RETIRE package available to vLLM worker processes.

The optional model-layer patches add safe points for the model implementations
named in their filenames. Set `RETIRE_LAYER_HORIZON` to a positive number when
applying them and retain that setting when running the integration. These hooks
do not provide arbitrary instruction-level cancellation.

| Stage | Backend responsibility |
| --- | --- |
| ADVANCE | Publish the selected scope version and reject obsolete admissions |
| FENCE | Close writers at supported boundaries and retain candidate prefix blocks |
| ALIGN | Validate tokens, configuration, numerical contract, ownership, and participant receipts |
| RESUME | Install certified state and verify that the successor adopts the expected blocks |
| RECLAIM | Release old references after their users drain |

To enable physical reuse, an integration must provide a numerical attestation
and any required reuse boundary for its exact execution configuration. The
`numerical_attestation` argument records that evidence; assigning it a string
does not establish correctness. Token equality and matching block IDs alone
do not establish numerical equivalence. Unsupported or uncertified state must
take the cold path.

The optional shared epoch mailbox requires explicit setup through
`RETIRE_EPOCH_MAILBOX_PATH`. Distributed integrations require receipts covering
every configured participant. These are research integration interfaces, not
a general production deployment recipe.

## Validation in this repository

The CPU test suite checks the adapters with deterministic engine doubles,
including delayed writers, participant validation, ownership transfer, and
fallback paths. All three patches were also checked against the public vLLM
`v0.13.0` source tag, and the resulting Python files passed syntax compilation.
These checks do not run GPU inference or reproduce paper measurements. The
paper is the source for the evaluation methodology and results.
