"""Exact KV codec and tenant-isolated DRAM tier reference mechanisms."""

from __future__ import annotations

import hashlib
import zlib
from dataclasses import dataclass, field
from itertools import count
from threading import RLock
from typing import Protocol

from .types import CodecCertificate, CodecFingerprint, ParticipantCoordinate


class KVCodec(Protocol):
    fingerprint: CodecFingerprint

    def encode(self, payload: bytes) -> bytes: ...

    def decode(self, payload: bytes) -> bytes: ...


class IdentityKVCodec:
    fingerprint = CodecFingerprint("identity", "1", lossless=True)

    def encode(self, payload: bytes) -> bytes:
        return payload

    def decode(self, payload: bytes) -> bytes:
        return payload


class ZlibKVCodec:
    """Portable lossless codec used by the executable reference path."""

    fingerprint = CodecFingerprint("zlib", zlib.ZLIB_VERSION, lossless=True)

    def __init__(self, level: int = 3) -> None:
        if level < -1 or level > 9:
            raise ValueError("zlib level must be between -1 and 9")
        self.level = level

    def encode(self, payload: bytes) -> bytes:
        return zlib.compress(payload, self.level)

    def decode(self, payload: bytes) -> bytes:
        return zlib.decompress(payload)


@dataclass(frozen=True, slots=True)
class TierObjectKey:
    tenant_id: str
    scope_id: str
    epoch: int
    request_id: str
    coordinate: ParticipantCoordinate


@dataclass(frozen=True, slots=True)
class OffloadTicket:
    key: TierObjectKey
    generation: int
    operation_id: int
    available_tokens: int


@dataclass(frozen=True, slots=True)
class EncodedKVObject:
    key: TierObjectKey
    generation: int
    available_tokens: int
    certificate: CodecCertificate
    encoded_payload: bytes = field(repr=False)


class DRAMTierStore:
    """Generation-checked DRAM store for exact encoded KV state.

    Offload is split into begin/complete so a test or backend can model a DMA
    or codec completion arriving after retirement. Cancelling an epoch advances
    its tickets' generations; a late completion is then dropped instead of
    installing bytes reachable by a successor.
    """

    def __init__(self, codec: KVCodec | None = None) -> None:
        self.codec = codec or IdentityKVCodec()
        self._lock = RLock()
        self._operations = count()
        self._generations: dict[TierObjectKey, int] = {}
        self._inflight: dict[int, OffloadTicket] = {}
        self._objects: dict[TierObjectKey, EncodedKVObject] = {}

    @staticmethod
    def _sha256(payload: bytes) -> str:
        return hashlib.sha256(payload).hexdigest()

    def begin_offload(
        self,
        key: TierObjectKey,
        *,
        available_tokens: int,
    ) -> OffloadTicket:
        if available_tokens < 0:
            raise ValueError("available_tokens must be non-negative")
        with self._lock:
            generation = self._generations.get(key, 0) + 1
            self._generations[key] = generation
            ticket = OffloadTicket(
                key=key,
                generation=generation,
                operation_id=next(self._operations),
                available_tokens=available_tokens,
            )
            self._inflight[ticket.operation_id] = ticket
            return ticket

    def complete_offload(
        self,
        ticket: OffloadTicket,
        decoded_payload: bytes,
    ) -> CodecCertificate | None:
        encoded = self.codec.encode(decoded_payload)
        restored = self.codec.decode(encoded)
        certificate = CodecCertificate(
            fingerprint=self.codec.fingerprint,
            encoded_sha256=self._sha256(encoded),
            decoded_sha256=self._sha256(decoded_payload),
            operation_drained=True,
            integrity_verified=restored == decoded_payload,
        )
        with self._lock:
            current = self._inflight.pop(ticket.operation_id, None)
            if (
                current != ticket
                or self._generations.get(ticket.key) != ticket.generation
            ):
                return None
            if not certificate.admissible:
                return certificate
            self._objects[ticket.key] = EncodedKVObject(
                key=ticket.key,
                generation=ticket.generation,
                available_tokens=ticket.available_tokens,
                certificate=certificate,
                encoded_payload=encoded,
            )
            return certificate

    def put(
        self,
        key: TierObjectKey,
        payload: bytes,
        *,
        available_tokens: int,
    ) -> CodecCertificate:
        ticket = self.begin_offload(key, available_tokens=available_tokens)
        certificate = self.complete_offload(ticket, payload)
        if certificate is None or not certificate.admissible:
            raise RuntimeError("codec could not produce an exact DRAM object")
        return certificate

    def cancel_inflight(
        self,
        tenant_id: str,
        scope_id: str,
        epoch: int,
    ) -> int:
        """Invalidate unfinished codec/DMA work without deleting stable objects."""

        with self._lock:
            matching = [
                ticket
                for ticket in self._inflight.values()
                if ticket.key.tenant_id == tenant_id
                and ticket.key.scope_id == scope_id
                and ticket.key.epoch == epoch
            ]
            for ticket in matching:
                self._generations[ticket.key] = ticket.generation + 1
                self._inflight.pop(ticket.operation_id, None)
            return len(matching)

    def get(self, key: TierObjectKey) -> EncodedKVObject | None:
        with self._lock:
            return self._objects.get(key)

    def restore(self, key: TierObjectKey, *, target_tenant_id: str) -> bytes:
        if key.tenant_id != target_tenant_id:
            raise PermissionError("cross-tenant DRAM restore is forbidden")
        with self._lock:
            item = self._objects.get(key)
        if item is None:
            raise KeyError(key)
        decoded = self.codec.decode(item.encoded_payload)
        if self._sha256(decoded) != item.certificate.decoded_sha256:
            raise RuntimeError("decoded KV digest mismatch")
        if self._sha256(item.encoded_payload) != item.certificate.encoded_sha256:
            raise RuntimeError("encoded KV digest mismatch")
        return decoded

    def reclaim(
        self,
        tenant_id: str,
        scope_id: str,
        epoch: int,
    ) -> int:
        with self._lock:
            keys = [
                key
                for key in self._objects
                if key.tenant_id == tenant_id
                and key.scope_id == scope_id
                and key.epoch == epoch
            ]
            for key in keys:
                self._objects.pop(key, None)
            self.cancel_inflight(tenant_id, scope_id, epoch)
            return len(keys)

    @property
    def inflight_count(self) -> int:
        with self._lock:
            return len(self._inflight)
