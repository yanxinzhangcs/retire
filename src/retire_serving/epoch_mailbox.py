"""Out-of-band scope-version mailbox for the vLLM research integration.

The frontend writes packed ``generation:epoch`` words through a shared mmap.
Each GPU worker independently mirrors the fixed-size table to device memory.
The mmap is deliberately independent of EngineCore RPCs so ADVANCE remains
visible while a long model step is running.
"""

from __future__ import annotations

import mmap
import os
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Callable
from typing import Any

_WORD = struct.Struct("<Q")
_DEFAULT_CAPACITY = 4096
_DEFAULT_LAYER_SAFE_POINT_MIN_TOKENS = 32
_POLL_SECONDS = float(os.environ.get("RETIRE_MAILBOX_POLL_SECONDS", "0.001"))
_AUTHORITY_SALT_MARKER = "|__RETIRE_AUTHORITY_V1__|"


def pack_version(epoch: int, generation: int) -> int:
    """Pack two unsigned 32-bit protocol fields into one publication word."""
    if not 0 <= epoch < 2**32:
        raise ValueError("epoch must fit in uint32")
    if not 0 <= generation < 2**32:
        raise ValueError("generation must fit in uint32")
    return (generation << 32) | epoch


def unpack_version(version: int) -> tuple[int, int]:
    return version & 0xFFFFFFFF, version >> 32


def encode_cache_salt_authority(
    cache_salt: str | None,
    scope_slot: int,
    expected_version: int,
) -> str:
    """Carry immutable admission authority in the request IPC payload."""
    if scope_slot < 0 or expected_version < 0:
        raise ValueError("authority fields must be non-negative")
    base = "" if cache_salt is None else cache_salt
    was_none = int(cache_salt is None)
    return (
        f"{base}{_AUTHORITY_SALT_MARKER}"
        f"{scope_slot}:{expected_version}:{was_none}"
    )


def decode_cache_salt_authority(
    encoded: str | None,
) -> tuple[str | None, tuple[int, int] | None]:
    """Remove request authority before vLLM computes prefix-cache hashes."""
    if not isinstance(encoded, str) or _AUTHORITY_SALT_MARKER not in encoded:
        return encoded, None
    base, payload = encoded.rsplit(_AUTHORITY_SALT_MARKER, 1)
    try:
        slot_text, version_text, none_text = payload.split(":")
        slot = int(slot_text)
        version = int(version_text)
        was_none = int(none_text)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid RETIRE cache-salt authority metadata") from exc
    if slot < 0 or version < 0 or was_none not in {0, 1}:
        raise ValueError("invalid RETIRE cache-salt authority fields")
    return (None if was_none else base), (slot, version)


def mailbox_path() -> Path | None:
    value = os.environ.get("RETIRE_EPOCH_MAILBOX_PATH")
    return Path(value) if value else None


def mailbox_capacity() -> int:
    return int(os.environ.get("RETIRE_EPOCH_MAILBOX_CAPACITY", _DEFAULT_CAPACITY))


def layer_safe_point_horizon() -> int:
    value = os.environ.get("RETIRE_LAYER_HORIZON", "0")
    try:
        horizon = int(value)
    except ValueError as exc:
        raise ValueError("RETIRE_LAYER_HORIZON must be an integer") from exc
    return horizon


def layer_safe_point_min_tokens() -> int:
    """Small steps retire at their existing scheduler boundary.

    Device publication predicates and layer-level host probes are reserved for
    model steps large enough to violate the configured retirement-latency
    budget.  Decode and small draft/verify steps therefore keep the exact same
    correctness boundary without paying a probe or tag-tensor cost per layer.
    """
    value = os.environ.get(
        "RETIRE_LAYER_SAFE_POINT_MIN_TOKENS",
        str(_DEFAULT_LAYER_SAFE_POINT_MIN_TOKENS),
    )
    try:
        minimum = int(value)
    except ValueError as exc:
        raise ValueError(
            "RETIRE_LAYER_SAFE_POINT_MIN_TOKENS must be an integer"
        ) from exc
    if minimum <= 0:
        raise ValueError("RETIRE_LAYER_SAFE_POINT_MIN_TOKENS must be positive")
    return minimum


class EpochMailbox:
    """Process-safe fixed-word storage; slot allocation is frontend-local."""

    def __init__(self, path: Path, capacity: int = _DEFAULT_CAPACITY) -> None:
        if capacity <= 0:
            raise ValueError("mailbox capacity must be positive")
        self.path = path
        self.capacity = capacity
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            size = (capacity + 1) * _WORD.size
            if os.fstat(fd).st_size != size:
                os.ftruncate(fd, size)
            self._mapping = mmap.mmap(fd, size, access=mmap.ACCESS_WRITE)
        finally:
            os.close(fd)
        self._lock = threading.Lock()
        self._scope_slots: dict[str, int] = {}

    def close(self) -> None:
        self._mapping.close()

    def reset(self) -> None:
        with self._lock:
            self._mapping[:] = b"\0" * len(self._mapping)
            self._scope_slots.clear()

    @property
    def _sequence_slot(self) -> int:
        return self.capacity

    def _bump_sequence_locked(self) -> None:
        offset = self._sequence_slot * _WORD.size
        sequence = _WORD.unpack_from(self._mapping, offset)[0]
        _WORD.pack_into(self._mapping, offset, (sequence + 1) % (2**64))

    def read_sequence(self) -> int:
        return _WORD.unpack_from(
            self._mapping,
            self._sequence_slot * _WORD.size,
        )[0]

    def bind_scope(self, scope_id: str, epoch: int, generation: int) -> int:
        with self._lock:
            slot = self._scope_slots.get(scope_id)
            if slot is None:
                slot = len(self._scope_slots)
                if slot >= self.capacity:
                    raise RuntimeError("scope-version mailbox is full")
                self._scope_slots[scope_id] = slot
            _WORD.pack_into(
                self._mapping,
                slot * _WORD.size,
                pack_version(epoch, generation),
            )
            self._bump_sequence_locked()
            return slot

    def slot_for(self, scope_id: str) -> int:
        with self._lock:
            try:
                return self._scope_slots[scope_id]
            except KeyError as exc:
                raise KeyError(f"scope is not bound to mailbox: {scope_id}") from exc

    def publish(self, scope_id: str, epoch: int, generation: int) -> int:
        slot = self.slot_for(scope_id)
        self.write_slot(slot, pack_version(epoch, generation))
        return slot

    def write_slot(self, slot: int, version: int) -> None:
        if not 0 <= slot < self.capacity:
            raise IndexError("mailbox slot out of range")
        with self._lock:
            _WORD.pack_into(self._mapping, slot * _WORD.size, version)
            self._bump_sequence_locked()

    def read_slot(self, slot: int) -> int:
        if not 0 <= slot < self.capacity:
            raise IndexError("mailbox slot out of range")
        return _WORD.unpack_from(self._mapping, slot * _WORD.size)[0]

    def snapshot(self) -> bytes:
        return self._mapping[: self.capacity * _WORD.size]


_FRONTEND_LOCK = threading.Lock()
_FRONTEND_MAILBOX: EpochMailbox | None = None


def get_frontend_mailbox() -> EpochMailbox:
    global _FRONTEND_MAILBOX
    path = mailbox_path()
    if path is None:
        raise RuntimeError("RETIRE_EPOCH_MAILBOX_PATH is not configured")
    with _FRONTEND_LOCK:
        if _FRONTEND_MAILBOX is None or _FRONTEND_MAILBOX.path != path:
            if _FRONTEND_MAILBOX is not None:
                _FRONTEND_MAILBOX.close()
            _FRONTEND_MAILBOX = EpochMailbox(path, mailbox_capacity())
        return _FRONTEND_MAILBOX


def reset_frontend_mailbox(path: Path, capacity: int = _DEFAULT_CAPACITY) -> None:
    os.environ["RETIRE_EPOCH_MAILBOX_PATH"] = str(path)
    os.environ["RETIRE_EPOCH_MAILBOX_CAPACITY"] = str(capacity)
    mailbox = get_frontend_mailbox()
    mailbox.reset()


def host_check_version(
    scope_slot: int, expected_version: int
) -> dict[str, int | bool]:
    """Check admission against the authoritative mmap publication.

    The GPU mirror can lag the mmap relay briefly.  EngineCore uses this
    direct host check *after* inserting a request into the scheduler: an
    already advanced request is finished immediately, while any later
    ADVANCE is ordered after insertion and its FENCE can therefore see it.
    """

    observed = get_frontend_mailbox().read_slot(scope_slot)
    return {
        "scope_slot": scope_slot,
        "expected_version": expected_version,
        "observed_version": observed,
        "admission_safe": observed == expected_version,
        "superseded_before_admission": observed > expected_version,
    }


@dataclass
class WorkerEpochState:
    """One worker's host relay, device table, and kernel audit counters."""

    mailbox: EpochMailbox
    device: Any
    table: Any
    counters: Any
    host_words: Any
    stream: Any
    stop_event: threading.Event
    wake_event: threading.Event = field(default_factory=threading.Event)
    relay_active_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    copies: int = 0
    last_copy_ns: int = 0
    step_authorities: tuple[tuple[int, int], ...] = ()
    step_fully_tagged: bool = False
    step_scheduled_tokens: int = 0
    step_requires_device_predicates: bool = False
    operator_safe_points: int = 0
    operator_early_exits: int = 0
    boundary_retired_steps: int = 0
    device_guarded_steps: int = 0
    speculative_steps: int = 0
    speculative_draft_tokens: int = 0

    def start(self) -> None:
        self.thread = threading.Thread(
            target=self._relay,
            name="retire-epoch-relay",
            daemon=True,
        )
        self.thread.start()

    def _relay(self) -> None:
        import numpy as np
        import torch

        torch.cuda.set_device(self.device)
        previous_sequence = -1
        while not self.stop_event.is_set():
            # Short steps close authority in EngineCore and do not need a
            # device mirror.  Sleep without periodic wakeups until admission
            # or a long guarded model step explicitly activates the relay.
            self.relay_active_event.wait()
            if self.stop_event.is_set():
                break
            sequence_before = self.mailbox.read_sequence()
            if sequence_before != previous_sequence:
                snapshot = self.mailbox.snapshot()
                sequence_after = self.mailbox.read_sequence()
                if sequence_before != sequence_after:
                    continue
                words = np.frombuffer(snapshot, dtype=np.uint64).astype(
                    np.int64, copy=False
                )
                self.host_words.copy_(torch.from_numpy(words.copy()))
                with torch.cuda.stream(self.stream):
                    self.table.copy_(self.host_words, non_blocking=True)
                previous_sequence = sequence_after
                self.copies += 1
                self.last_copy_ns = time.monotonic_ns()
            # Normal detection remains periodic, but an admission RPC can
            # wake the relay immediately instead of waiting a full polling
            # period before the successor crosses its device barrier.
            self.wake_event.wait(_POLL_SECONDS)
            self.wake_event.clear()

    def activate_relay(self) -> None:
        self.relay_active_event.set()
        self.wake_event.set()

    def deactivate_relay(self) -> None:
        self.relay_active_event.clear()

    def finish_step(self) -> None:
        """Release the relay after a device-guarded model step completes."""
        self.deactivate_relay()

    def audit(self) -> dict[str, int]:
        import torch

        torch.cuda.synchronize(self.device)
        values = self.counters.cpu().tolist()
        return {
            "mailbox_copies": self.copies,
            "last_copy_ns": self.last_copy_ns,
            "attention_tile_exits": int(values[0]),
            "kv_store_rejections": int(values[1]),
            "output_commit_rejections": int(values[2]),
            "operator_safe_points": self.operator_safe_points,
            "operator_early_exits": self.operator_early_exits,
            "boundary_retired_steps": self.boundary_retired_steps,
            "device_guarded_steps": self.device_guarded_steps,
            "speculative_steps": self.speculative_steps,
            "speculative_draft_tokens": self.speculative_draft_tokens,
        }

    def bind_step_authority(
        self,
        request_ids: list[str],
        request_tags: dict[str, tuple[int, int]],
        scheduled_spec_decode_tokens: dict[str, list[int]] | None = None,
        scheduled_token_count: int = 0,
    ) -> None:
        """Bind the authority set consumed by model-layer launch horizons.

        A whole-layer early exit is legal only when every request in the model
        step is tagged. Mixed tagged/stock batches keep executing and rely on
        the per-sequence attention, KV, and output publication gates.
        """
        tags = [request_tags.get(request_id) for request_id in request_ids]
        self.step_fully_tagged = bool(tags) and all(tag is not None for tag in tags)
        self.step_authorities = (
            tuple(sorted({(int(tag[0]), int(tag[1])) for tag in tags if tag}))
            if self.step_fully_tagged
            else ()
        )
        self.step_scheduled_tokens = int(scheduled_token_count)
        has_tags = any(tag is not None for tag in tags)
        self.step_requires_device_predicates = bool(
            has_tags
            and self.step_scheduled_tokens >= layer_safe_point_min_tokens()
        )
        if self.step_requires_device_predicates:
            self.activate_relay()
        else:
            self.deactivate_relay()
        if has_tags:
            if self.step_requires_device_predicates:
                self.device_guarded_steps += 1
            else:
                self.boundary_retired_steps += 1
        if scheduled_spec_decode_tokens:
            draft_tokens = sum(
                len(scheduled_spec_decode_tokens.get(request_id, ()))
                for request_id in request_ids
            )
            if draft_tokens:
                self.speculative_steps += 1
                self.speculative_draft_tokens += draft_tokens

    def step_is_locally_retired(self) -> bool:
        return self.step_fully_tagged and bool(self.step_authorities) and all(
            self.mailbox.read_slot(scope_slot) != expected_version
            for scope_slot, expected_version in self.step_authorities
        )

    def wait_for_version(
        self, scope_slot: int, expected_version: int, timeout_s: float = 5.0
    ) -> dict[str, int | bool]:
        """Admission barrier for a successor after out-of-band ADVANCE."""
        deadline = time.monotonic() + timeout_s
        observed = -1
        self.activate_relay()
        try:
            while time.monotonic() < deadline:
                self.stream.synchronize()
                observed = int(self.table[scope_slot].item())
                if observed == expected_version:
                    return {
                        "scope_slot": scope_slot,
                        "expected_version": expected_version,
                        "observed_version": observed,
                        "admission_safe": True,
                        "superseded_before_admission": False,
                    }
                if observed > expected_version:
                    return {
                        "scope_slot": scope_slot,
                        "expected_version": expected_version,
                        "observed_version": observed,
                        "admission_safe": False,
                        "superseded_before_admission": True,
                    }
                self.wake_event.set()
                time.sleep(_POLL_SECONDS)
        finally:
            self.deactivate_relay()
        raise TimeoutError(
            "device epoch relay did not reach successor version: "
            f"slot={scope_slot}, expected={expected_version}, observed={observed}"
        )

    def wait_for_versions(
        self,
        authorities: list[tuple[int, int]],
        timeout_s: float = 5.0,
    ) -> list[dict[str, int | bool]]:
        """Batch an admission barrier over one mailbox relay snapshot.

        A concurrent request group should not pay one EngineCore/worker RPC per
        logical scope.  The worker synchronizes the relay stream once, reads
        every requested slot from the same monotonic snapshot, and retries the
        group only while at least one expected version has not arrived yet.
        """
        if not authorities:
            return []
        normalized = [(int(slot), int(version)) for slot, version in authorities]
        deadline = time.monotonic() + timeout_s
        observed = [-1] * len(normalized)
        self.activate_relay()
        try:
            while time.monotonic() < deadline:
                self.stream.synchronize()
                observed = [int(self.table[slot].item()) for slot, _ in normalized]
                receipts = [
                    {
                        "scope_slot": slot,
                        "expected_version": expected,
                        "observed_version": value,
                        "admission_safe": value == expected,
                        "superseded_before_admission": value > expected,
                    }
                    for (slot, expected), value in zip(
                        normalized, observed, strict=True
                    )
                ]
                if all(
                    receipt["admission_safe"]
                    or receipt["superseded_before_admission"]
                    for receipt in receipts
                ):
                    return receipts
                self.wake_event.set()
                time.sleep(_POLL_SECONDS)
        finally:
            self.deactivate_relay()
        raise TimeoutError(
            "device epoch relay did not reach batched successor versions: "
            f"expected={normalized}, observed={observed}"
        )


_WORKER_STATE: WorkerEpochState | None = None


def start_worker_epoch_relay(device: Any) -> WorkerEpochState | None:
    """Start once when the opt-in mailbox environment variable is present."""
    global _WORKER_STATE
    path = mailbox_path()
    if path is None:
        return None
    if _WORKER_STATE is not None:
        return _WORKER_STATE

    import torch

    mailbox = EpochMailbox(path, mailbox_capacity())
    table = torch.zeros(mailbox.capacity, dtype=torch.int64, device=device)
    counters = torch.zeros(3, dtype=torch.int64, device=device)
    host_words = torch.empty(
        mailbox.capacity, dtype=torch.int64, device="cpu", pin_memory=True
    )
    state = WorkerEpochState(
        mailbox=mailbox,
        device=device,
        table=table,
        counters=counters,
        host_words=host_words,
        # CUDA uses numerically smaller priorities for higher-priority work.
        # The driver clamps this request to its supported priority range.
        stream=torch.cuda.Stream(device=device, priority=-1),
        stop_event=threading.Event(),
    )
    _WORKER_STATE = state
    state.start()
    return state


def get_worker_epoch_state() -> WorkerEpochState | None:
    return _WORKER_STATE


def worker_epoch_audit() -> dict[str, int]:
    if _WORKER_STATE is None:
        return {
            "mailbox_copies": 0,
            "last_copy_ns": 0,
            "attention_tile_exits": 0,
            "kv_store_rejections": 0,
            "output_commit_rejections": 0,
            "operator_safe_points": 0,
            "operator_early_exits": 0,
            "boundary_retired_steps": 0,
            "device_guarded_steps": 0,
            "speculative_steps": 0,
            "speculative_draft_tokens": 0,
        }
    return _WORKER_STATE.audit()


def worker_wait_for_version(
    scope_slot: int, expected_version: int
) -> dict[str, int | bool]:
    if _WORKER_STATE is None:
        raise RuntimeError("worker epoch relay is not initialized")
    return _WORKER_STATE.wait_for_version(scope_slot, expected_version)


def worker_wait_for_versions(
    authorities: list[tuple[int, int]],
) -> list[dict[str, int | bool]]:
    if _WORKER_STATE is None:
        raise RuntimeError("worker epoch relay is not initialized")
    return _WORKER_STATE.wait_for_versions(authorities)


def _worker_layer_safe_point_now() -> bool:
    """Make one TP-consistent exit decision for an already-armed step."""
    state = _WORKER_STATE
    if state is None or not state.step_fully_tagged:
        return False

    import torch
    from vllm.distributed import get_tp_group

    # Pipeline stages poll the same monotonic authority independently.  A PP
    # collective here would deadlock: a downstream stage can be waiting for
    # the upstream intermediate while the upstream stage waits in the
    # collective.  Stage-local TP agreement is sufficient.  Once an upstream
    # stage observes retirement, its shape-correct intermediate is discardable;
    # downstream stages observe the same newer version before publication, and
    # the KV/output gates remain the final correctness boundary.

    state.operator_safe_points += 1
    tp_group = get_tp_group()
    if tp_group.world_size == 1 and not state.step_is_locally_retired():
        # A host-side current-version observation is enough to keep launching.
        # If ADVANCE races immediately after this check, the next horizon
        # catches it, so the additional launch depth remains bounded by N.
        return False

    torch.cuda.current_stream(state.device).synchronize()
    decision = torch.tensor(
        int(state.step_is_locally_retired()),
        dtype=torch.int32,
        device=state.device,
    )
    if tp_group.world_size > 1:
        torch.distributed.all_reduce(
            decision,
            op=torch.distributed.ReduceOp.MIN,
            group=tp_group.device_group,
        )
    should_exit = bool(decision.item())
    if should_exit:
        state.operator_early_exits += 1
    return should_exit


def worker_layer_safe_point_plan() -> tuple[int, Callable[[], bool]] | None:
    """Build one cheap layer-loop plan per model forward.

    Returning ``None`` makes decode and small verify steps use vLLM's existing
    model-step boundary.  Long prefill steps return a horizon and a predicate;
    the model loop invokes that predicate only at the selected layers.
    """
    horizon = layer_safe_point_horizon()
    state = _WORKER_STATE
    if (
        horizon <= 0
        or state is None
        or not state.step_fully_tagged
        or not state.step_requires_device_predicates
    ):
        return None
    return horizon, _worker_layer_safe_point_now


def worker_layer_safe_point(layer_index: int) -> bool:
    """Backward-compatible single-call layer-horizon helper.

    This path is opt-in through ``RETIRE_LAYER_HORIZON``. Synchronizing the
    model stream every N layers prevents Python from enqueueing the entire
    transformer after an epoch advance. All TP ranks reduce their local
    mailbox decision before any rank may skip the remaining layers.
    """
    horizon = layer_safe_point_horizon()
    state = _WORKER_STATE
    if (
        horizon <= 0
        or layer_index <= 0
        or layer_index % horizon
        or state is None
        or not state.step_fully_tagged
        or not state.step_requires_device_predicates
    ):
        return False
    return _worker_layer_safe_point_now()
