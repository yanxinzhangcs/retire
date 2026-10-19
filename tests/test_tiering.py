from __future__ import annotations

import pytest

from retire_serving import (
    DRAMTierStore,
    ParticipantCoordinate,
    TierObjectKey,
    ZlibKVCodec,
)


def make_key(*, tenant_id: str = "tenant-a", epoch: int = 0) -> TierObjectKey:
    return TierObjectKey(
        tenant_id=tenant_id,
        scope_id="scope",
        epoch=epoch,
        request_id=f"request-{epoch}",
        coordinate=ParticipantCoordinate(0, "attention"),
    )


def test_lossless_codec_round_trip_is_digest_checked_and_tenant_bound() -> None:
    store = DRAMTierStore(ZlibKVCodec(level=1))
    key = make_key()
    payload = b"exact-kv-bytes" * 64
    certificate = store.put(key, payload, available_tokens=128)

    assert certificate.admissible
    assert certificate.encoded_sha256 != certificate.decoded_sha256
    assert store.restore(key, target_tenant_id="tenant-a") == payload
    with pytest.raises(PermissionError, match="cross-tenant"):
        store.restore(key, target_tenant_id="tenant-b")


def test_cancelled_codec_completion_cannot_install_after_retirement() -> None:
    store = DRAMTierStore(ZlibKVCodec())
    key = make_key()
    ticket = store.begin_offload(key, available_tokens=64)

    assert store.cancel_inflight("tenant-a", "scope", 0) == 1
    assert store.complete_offload(ticket, b"late-old-kv") is None
    assert store.get(key) is None
    assert store.inflight_count == 0


def test_reclaim_is_scoped_by_tenant_and_epoch() -> None:
    store = DRAMTierStore()
    old = make_key(epoch=0)
    current = make_key(epoch=1)
    other_tenant = make_key(tenant_id="tenant-b", epoch=0)
    store.put(old, b"old", available_tokens=4)
    store.put(current, b"current", available_tokens=4)
    store.put(other_tenant, b"other", available_tokens=4)

    assert store.reclaim("tenant-a", "scope", 0) == 1
    assert store.get(old) is None
    assert store.restore(current, target_tenant_id="tenant-a") == b"current"
    assert store.restore(other_tenant, target_tenant_id="tenant-b") == b"other"
