import sys
import threading
import types
from pathlib import Path

import retire_serving.epoch_mailbox as epoch_mailbox
from retire_serving.epoch_mailbox import (
    EpochMailbox,
    WorkerEpochState,
    decode_cache_salt_authority,
    encode_cache_salt_authority,
    get_frontend_mailbox,
    host_check_version,
    pack_version,
    reset_frontend_mailbox,
    unpack_version,
)


def test_pack_version_round_trip() -> None:
    version = pack_version(17, 29)
    assert unpack_version(version) == (17, 29)


def test_cache_salt_authority_round_trip_preserves_original_salt() -> None:
    encoded = encode_cache_salt_authority("tenant:scope:e3", 7, 19)
    assert decode_cache_salt_authority(encoded) == (
        "tenant:scope:e3",
        (7, 19),
    )
    encoded_none = encode_cache_salt_authority(None, 1, 2)
    assert decode_cache_salt_authority(encoded_none) == (None, (1, 2))


def test_mailbox_scope_publication_is_visible_across_mappings(
    tmp_path: Path,
) -> None:
    path = tmp_path / "epoch.mailbox"
    frontend = EpochMailbox(path, capacity=8)
    worker = EpochMailbox(path, capacity=8)
    try:
        slot = frontend.bind_scope("agent-a", 3, 3)
        assert slot == 0
        assert worker.read_slot(slot) == pack_version(3, 3)

        frontend.publish("agent-a", 4, 5)
        assert worker.read_slot(slot) == pack_version(4, 5)
    finally:
        frontend.close()
        worker.close()


def test_mailbox_reuses_scope_slot(tmp_path: Path) -> None:
    mailbox = EpochMailbox(tmp_path / "epoch.mailbox", capacity=2)
    try:
        assert mailbox.bind_scope("a", 0, 0) == 0
        assert mailbox.bind_scope("a", 1, 1) == 0
        assert mailbox.bind_scope("b", 0, 0) == 1
    finally:
        mailbox.close()


def test_mailbox_sequence_changes_only_when_authority_changes(
    tmp_path: Path,
) -> None:
    mailbox = EpochMailbox(tmp_path / "epoch.mailbox", capacity=4)
    try:
        assert mailbox.read_sequence() == 0
        slot = mailbox.bind_scope("scope", 0, 0)
        after_bind = mailbox.read_sequence()
        assert after_bind == 1
        assert mailbox.snapshot() == mailbox.snapshot()
        assert mailbox.read_sequence() == after_bind
        mailbox.publish("scope", 1, 1)
        assert mailbox.read_sequence() == after_bind + 1
        assert mailbox.read_slot(slot) == pack_version(1, 1)
    finally:
        mailbox.close()


def test_host_admission_check_reads_authoritative_publication(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "host-check.mailbox"
    monkeypatch.setenv("RETIRE_EPOCH_MAILBOX_PATH", str(path))
    monkeypatch.setenv("RETIRE_EPOCH_MAILBOX_CAPACITY", "4")
    reset_frontend_mailbox(path, capacity=4)
    mailbox = get_frontend_mailbox()
    slot = mailbox.bind_scope("scope", 3, 3)

    current = host_check_version(slot, pack_version(3, 3))
    assert current["admission_safe"]
    assert not current["superseded_before_admission"]

    mailbox.publish("scope", 4, 4)
    stale = host_check_version(slot, pack_version(3, 3))
    assert not stale["admission_safe"]
    assert stale["superseded_before_admission"]
    assert stale["observed_version"] == pack_version(4, 4)


def test_batched_worker_admission_uses_one_relay_snapshot(tmp_path: Path) -> None:
    class FakeValue:
        def __init__(self, value: int) -> None:
            self.value = value

        def item(self) -> int:
            return self.value

    class FakeStream:
        synchronizations = 0

        def synchronize(self) -> None:
            self.synchronizations += 1

    mailbox = EpochMailbox(tmp_path / "batch-admission.mailbox", capacity=4)
    try:
        expected_a = pack_version(1, 1)
        expected_b = pack_version(2, 2)
        stream = FakeStream()
        state = WorkerEpochState(
            mailbox=mailbox,
            device=None,
            table=[FakeValue(expected_a), FakeValue(expected_b)],
            counters=None,
            host_words=None,
            stream=stream,
            stop_event=threading.Event(),
        )
        rows = state.wait_for_versions([(0, expected_a), (1, expected_b)])
        assert stream.synchronizations == 1
        assert state.wake_event.is_set()
        assert not state.relay_active_event.is_set()
        assert [row["admission_safe"] for row in rows] == [True, True]
        assert not any(row["superseded_before_admission"] for row in rows)
    finally:
        mailbox.close()


def test_whole_step_exit_requires_every_request_to_be_tagged(tmp_path: Path) -> None:
    mailbox = EpochMailbox(tmp_path / "epoch.mailbox", capacity=4)
    try:
        slot_a = mailbox.bind_scope("a", 0, 0)
        slot_b = mailbox.bind_scope("b", 0, 0)
        state = WorkerEpochState(
            mailbox=mailbox,
            device=None,
            table=None,
            counters=None,
            host_words=None,
            stream=None,
            stop_event=threading.Event(),
        )
        tags = {"a": (slot_a, pack_version(0, 0))}
        state.bind_step_authority(["a", "untagged"], tags)
        mailbox.write_slot(slot_a, pack_version(1, 1))
        assert not state.step_fully_tagged
        assert not state.step_is_locally_retired()

        tags["b"] = (slot_b, pack_version(0, 0))
        state.bind_step_authority(["a", "b"], tags)
        assert state.step_fully_tagged
        assert not state.step_is_locally_retired()
        mailbox.write_slot(slot_b, pack_version(1, 1))
        assert state.step_is_locally_retired()
    finally:
        mailbox.close()


def test_speculative_schedule_is_audited_without_authorizing_mixed_exit(
    tmp_path: Path,
) -> None:
    mailbox = EpochMailbox(tmp_path / "epoch.mailbox", capacity=2)
    try:
        slot = mailbox.bind_scope("a", 0, 0)
        state = WorkerEpochState(
            mailbox=mailbox,
            device=None,
            table=None,
            counters=None,
            host_words=None,
            stream=None,
            stop_event=threading.Event(),
        )
        state.bind_step_authority(
            ["a", "stock"],
            {"a": (slot, pack_version(0, 0))},
            {"a": [11, 12], "stock": [13]},
        )
        assert state.speculative_steps == 1
        assert state.speculative_draft_tokens == 3
        assert not state.step_fully_tagged
        assert not state.step_is_locally_retired()
    finally:
        mailbox.close()


def test_layer_horizon_exits_only_at_a_bounded_safe_point(
    tmp_path: Path, monkeypatch,
) -> None:
    class FakeStream:
        synchronizations = 0

        def synchronize(self) -> None:
            self.synchronizations += 1

    class FakeDecision:
        def __init__(self, value: int) -> None:
            self.value = value

        def item(self) -> int:
            return self.value

    stream = FakeStream()
    fake_torch = types.ModuleType("torch")
    fake_torch.int32 = object()
    fake_torch.cuda = types.SimpleNamespace(
        current_stream=lambda _device: stream,
    )
    fake_torch.tensor = lambda value, **_kwargs: FakeDecision(value)
    fake_torch.distributed = types.SimpleNamespace(
        ReduceOp=types.SimpleNamespace(MIN="min"),
        all_reduce=lambda *_args, **_kwargs: None,
    )
    fake_distributed = types.ModuleType("vllm.distributed")
    singleton = types.SimpleNamespace(world_size=1, device_group=None)
    fake_distributed.get_pp_group = lambda: singleton
    fake_distributed.get_tp_group = lambda: singleton
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__path__ = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(
        sys.modules, "vllm.distributed", fake_distributed
    )
    monkeypatch.setenv("RETIRE_LAYER_HORIZON", "4")

    mailbox = EpochMailbox(tmp_path / "epoch.mailbox", capacity=2)
    try:
        slot = mailbox.bind_scope("scope", 0, 0)
        state = WorkerEpochState(
            mailbox=mailbox,
            device="fake-device",
            table=None,
            counters=None,
            host_words=None,
            stream=None,
            stop_event=threading.Event(),
        )
        state.bind_step_authority(
            ["request"],
            {"request": (slot, pack_version(0, 0))},
            scheduled_token_count=128,
        )
        monkeypatch.setattr(epoch_mailbox, "_WORKER_STATE", state)

        assert not epoch_mailbox.worker_layer_safe_point(1)
        assert not epoch_mailbox.worker_layer_safe_point(4)
        mailbox.write_slot(slot, pack_version(1, 1))
        assert not epoch_mailbox.worker_layer_safe_point(5)
        assert epoch_mailbox.worker_layer_safe_point(8)
        assert stream.synchronizations == 1
        assert state.operator_safe_points == 2
        assert state.operator_early_exits == 1
    finally:
        mailbox.close()


def test_adaptive_boundary_path_disarms_layer_and_device_checks(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setenv("RETIRE_LAYER_HORIZON", "4")
    monkeypatch.setenv("RETIRE_LAYER_SAFE_POINT_MIN_TOKENS", "32")
    mailbox = EpochMailbox(tmp_path / "epoch.mailbox", capacity=2)
    try:
        slot = mailbox.bind_scope("scope", 0, 0)
        state = WorkerEpochState(
            mailbox=mailbox,
            device=None,
            table=None,
            counters=None,
            host_words=None,
            stream=None,
            stop_event=threading.Event(),
        )
        monkeypatch.setattr(epoch_mailbox, "_WORKER_STATE", state)

        state.bind_step_authority(
            ["decode"],
            {"decode": (slot, pack_version(0, 0))},
            scheduled_token_count=1,
        )
        assert state.step_fully_tagged
        assert not state.step_requires_device_predicates
        assert not state.relay_active_event.is_set()
        assert epoch_mailbox.worker_layer_safe_point_plan() is None
        assert state.boundary_retired_steps == 1
        assert state.device_guarded_steps == 0

        state.bind_step_authority(
            ["prefill"],
            {"prefill": (slot, pack_version(0, 0))},
            scheduled_token_count=4096,
        )
        plan = epoch_mailbox.worker_layer_safe_point_plan()
        assert state.step_requires_device_predicates
        assert state.relay_active_event.is_set()
        assert plan is not None and plan[0] == 4
        assert state.boundary_retired_steps == 1
        assert state.device_guarded_steps == 1
        state.finish_step()
        assert not state.relay_active_event.is_set()
    finally:
        mailbox.close()


def test_pipeline_stage_uses_independent_nonblocking_safe_point(
    tmp_path: Path, monkeypatch,
) -> None:
    class FakeStream:
        synchronizations = 0

        def synchronize(self) -> None:
            self.synchronizations += 1

    class FakeDecision:
        def __init__(self, value: int) -> None:
            self.value = value

        def item(self) -> int:
            return self.value

    stream = FakeStream()
    fake_torch = types.ModuleType("torch")
    fake_torch.int32 = object()
    fake_torch.cuda = types.SimpleNamespace(
        current_stream=lambda _device: stream,
    )
    fake_torch.tensor = lambda value, **_kwargs: FakeDecision(value)
    fake_torch.distributed = types.SimpleNamespace(
        ReduceOp=types.SimpleNamespace(MIN="min"),
        all_reduce=lambda *_args, **_kwargs: None,
    )
    fake_distributed = types.ModuleType("vllm.distributed")
    tp_group = types.SimpleNamespace(world_size=1, device_group=None)
    pp_group = types.SimpleNamespace(world_size=2, device_group=object())
    fake_distributed.get_tp_group = lambda: tp_group
    # A PP group is deliberately present, but the safe point must not enter a
    # cross-stage collective because a downstream stage may be waiting on P2P.
    fake_distributed.get_pp_group = lambda: pp_group
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__path__ = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.distributed", fake_distributed)
    monkeypatch.setenv("RETIRE_LAYER_HORIZON", "4")

    mailbox = EpochMailbox(tmp_path / "epoch.mailbox", capacity=2)
    try:
        slot = mailbox.bind_scope("scope", 0, 0)
        state = WorkerEpochState(
            mailbox=mailbox,
            device="fake-device",
            table=None,
            counters=None,
            host_words=None,
            stream=None,
            stop_event=threading.Event(),
        )
        state.bind_step_authority(
            ["request"],
            {"request": (slot, pack_version(0, 0))},
            scheduled_token_count=128,
        )
        monkeypatch.setattr(epoch_mailbox, "_WORKER_STATE", state)
        mailbox.write_slot(slot, pack_version(1, 1))

        assert epoch_mailbox.worker_layer_safe_point(4)
        assert stream.synchronizations == 1
        assert state.operator_safe_points == 1
        assert state.operator_early_exits == 1
    finally:
        mailbox.close()


def test_admission_rejects_a_successor_overtaken_by_a_newer_version(
    tmp_path: Path,
) -> None:
    class FakeStream:
        def synchronize(self) -> None:
            return None

    class FakeWord:
        def __init__(self, value: int) -> None:
            self.value = value

        def item(self) -> int:
            return self.value

    mailbox = EpochMailbox(tmp_path / "epoch.mailbox", capacity=2)
    try:
        expected = pack_version(1, 1)
        observed = pack_version(2, 2)
        state = WorkerEpochState(
            mailbox=mailbox,
            device=None,
            table=[FakeWord(observed)],
            counters=None,
            host_words=None,
            stream=FakeStream(),
            stop_event=threading.Event(),
        )
        receipt = state.wait_for_version(0, expected, timeout_s=0.01)
        assert state.wake_event.is_set()
        assert not state.relay_active_event.is_set()
        assert not receipt["admission_safe"]
        assert receipt["superseded_before_admission"]
        assert receipt["observed_version"] == observed
    finally:
        mailbox.close()
