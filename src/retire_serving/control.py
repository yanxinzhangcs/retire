"""Thread-safe logical scope and epoch registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock

from .types import AdvanceReceipt, RequestSpec


class UnknownScopeError(KeyError):
    pass


class StaleEpochError(RuntimeError):
    pass


class TenantIsolationError(RuntimeError):
    pass


@dataclass(slots=True)
class _ScopeRecord:
    tenant_id: str = "default"
    epoch: int = 0
    contexts: dict[int, dict[str, tuple[int, ...]]] = field(default_factory=dict)
    active: dict[int, set[str]] = field(default_factory=dict)
    reclaimed_epochs: set[int] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class ScopeSnapshot:
    scope_id: str
    tenant_id: str
    current_epoch: int
    requests_by_epoch: tuple[tuple[int, tuple[str, ...]], ...]
    reclaimed_epochs: tuple[int, ...]


class ScopeRegistry:
    """Owns the logical linearization point for supersession.

    `advance` changes the current epoch under one lock. It intentionally does
    not wait for vLLM, CUDA, or transport work from the retired epoch.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._scopes: dict[str, _ScopeRecord] = {}

    def ensure_scope(self, scope_id: str, tenant_id: str = "default") -> int:
        if not scope_id:
            raise ValueError("scope_id must be non-empty")
        if not tenant_id:
            raise ValueError("tenant_id must be non-empty")
        with self._lock:
            record = self._scopes.setdefault(
                scope_id,
                _ScopeRecord(tenant_id=tenant_id),
            )
            self._check_tenant(scope_id, record, tenant_id)
            return record.epoch

    def current_epoch(
        self, scope_id: str, tenant_id: str | None = None
    ) -> int:
        with self._lock:
            try:
                record = self._scopes[scope_id]
            except KeyError as exc:
                raise UnknownScopeError(scope_id) from exc
            if tenant_id is not None:
                self._check_tenant(scope_id, record, tenant_id)
            return record.epoch

    def is_current(
        self,
        scope_id: str,
        epoch: int,
        tenant_id: str | None = None,
    ) -> bool:
        with self._lock:
            record = self._scopes.get(scope_id)
            return (
                record is not None
                and (tenant_id is None or record.tenant_id == tenant_id)
                and record.epoch == epoch
            )

    def register(self, spec: RequestSpec) -> None:
        with self._lock:
            record = self._scopes.setdefault(
                spec.scope_id,
                _ScopeRecord(tenant_id=spec.tenant_id),
            )
            self._check_tenant(spec.scope_id, record, spec.tenant_id)
            if spec.epoch != record.epoch:
                raise StaleEpochError(
                    f"cannot launch {spec.request_id}: scope {spec.scope_id!r} "
                    f"is at epoch {record.epoch}, not {spec.epoch}"
                )
            contexts = record.contexts.setdefault(spec.epoch, {})
            if spec.request_id in contexts:
                raise ValueError(f"duplicate request_id: {spec.request_id}")
            contexts[spec.request_id] = spec.prompt_token_ids
            record.active.setdefault(spec.epoch, set()).add(spec.request_id)

    def complete(self, scope_id: str, epoch: int, request_id: str) -> None:
        """Remove exactly one physical request; never touch a successor."""

        with self._lock:
            record = self._scopes.get(scope_id)
            if record is None:
                return
            active = record.active.get(epoch)
            if active is not None:
                active.discard(request_id)

    def advance(
        self,
        scope_id: str,
        *,
        expected_epoch: int | None = None,
        new_epoch: int | None = None,
        tenant_id: str | None = None,
    ) -> AdvanceReceipt:
        with self._lock:
            try:
                record = self._scopes[scope_id]
            except KeyError as exc:
                raise UnknownScopeError(scope_id) from exc
            if tenant_id is not None:
                self._check_tenant(scope_id, record, tenant_id)

            old_epoch = record.epoch
            expected = old_epoch if expected_epoch is None else expected_epoch
            target = old_epoch + 1 if new_epoch is None else new_epoch
            if expected != old_epoch:
                raise StaleEpochError(
                    f"scope {scope_id!r} is at epoch {old_epoch}, "
                    f"not descriptor epoch {expected}"
                )
            if target != old_epoch + 1:
                raise StaleEpochError(
                    f"scope {scope_id!r} must advance from {old_epoch} "
                    f"to {old_epoch + 1}, not {target}"
                )
            old_contexts = record.contexts.get(old_epoch, {})
            old_active = record.active.get(old_epoch, set())
            # This assignment is the logical linearization point.
            record.epoch = target
            return AdvanceReceipt(
                scope_id=scope_id,
                old_epoch=old_epoch,
                new_epoch=record.epoch,
                retired_request_ids=tuple(old_active),
                retired_contexts=tuple(old_contexts.items()),
                tenant_id=record.tenant_id,
            )

    def mark_reclaimed(
        self,
        scope_id: str,
        retired_epoch: int,
        tenant_id: str | None = None,
    ) -> None:
        with self._lock:
            record = self._scopes[scope_id]
            if tenant_id is not None:
                self._check_tenant(scope_id, record, tenant_id)
            if retired_epoch >= record.epoch:
                raise StaleEpochError("cannot reclaim the current or a future epoch")
            if record.active.get(retired_epoch):
                raise RuntimeError("logical requests are still draining")
            record.reclaimed_epochs.add(retired_epoch)

    def snapshot(self, scope_id: str) -> ScopeSnapshot:
        with self._lock:
            try:
                record = self._scopes[scope_id]
            except KeyError as exc:
                raise UnknownScopeError(scope_id) from exc
            return ScopeSnapshot(
                scope_id=scope_id,
                tenant_id=record.tenant_id,
                current_epoch=record.epoch,
                requests_by_epoch=tuple(
                    (epoch, tuple(sorted(requests)))
                    for epoch, requests in sorted(record.active.items())
                ),
                reclaimed_epochs=tuple(sorted(record.reclaimed_epochs)),
            )

    @staticmethod
    def _check_tenant(
        scope_id: str,
        record: _ScopeRecord,
        tenant_id: str,
    ) -> None:
        if record.tenant_id != tenant_id:
            raise TenantIsolationError(
                f"scope {scope_id!r} belongs to tenant {record.tenant_id!r}, "
                f"not {tenant_id!r}"
            )
