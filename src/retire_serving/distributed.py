"""Distributed PP x TP x KV-group retirement coordinator and participant."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .backend import InferenceBackend
from .tiering import DRAMTierStore, TierObjectKey, ZlibKVCodec
from .types import (
    EngineFingerprint,
    FenceReceipt,
    KVGroupReuseEvidence,
    ParticipantCoordinate,
    ParticipantFenceReceipt,
    ParticipantReclaimReceipt,
    ReclaimReceipt,
    RecoveryPlan,
    RequestSpec,
    ReuseProbe,
    ReuseTier,
)


class RetirementParticipant(Protocol):
    coordinate: ParticipantCoordinate

    async def bind_request(self, spec: RequestSpec) -> None: ...

    async def fence(
        self,
        scope_id: str,
        retired_epoch: int,
        request_ids: Sequence[str],
    ) -> ParticipantFenceReceipt: ...

    async def probe_reuse(
        self,
        scope_id: str,
        retired_epoch: int,
        source_request_id: str | None,
        block_aligned_tokens: int,
    ) -> KVGroupReuseEvidence: ...

    async def prepare_resume(
        self,
        spec: RequestSpec,
        retired_epoch: int,
        recovery: RecoveryPlan,
    ) -> None: ...

    async def reclaim(
        self,
        scope_id: str,
        retired_epoch: int,
    ) -> ParticipantReclaimReceipt: ...


class DistributedRetirementBackend:
    """One generation stream plus a distributed retirement control group.

    The data plane remains owned by the serving framework. Every pipeline
    stage, TP rank, and KV group independently fences its writers and reports
    reuse evidence. The runtime can resume only from the intersection of all
    those receipts.
    """

    def __init__(
        self,
        data_plane: InferenceBackend,
        participants: Sequence[RetirementParticipant],
        *,
        fingerprint: EngineFingerprint | None = None,
    ) -> None:
        self.data_plane = data_plane
        self.fingerprint = fingerprint or data_plane.fingerprint
        self.participants = tuple(participants)
        expected = set(self.fingerprint.participant_coordinates)
        observed = [participant.coordinate for participant in self.participants]
        if len(observed) != len(set(observed)) or set(observed) != expected:
            raise ValueError(
                "participants must cover every PP stage, TP rank, and KV-cache "
                "group exactly once"
            )

    async def generate(self, spec: RequestSpec) -> AsyncIterator[Any]:
        await asyncio.gather(
            *(participant.bind_request(spec) for participant in self.participants)
        )
        async for output in self.data_plane.generate(spec):
            yield output

    async def discard_unlaunched(self, spec: RequestSpec) -> None:
        discard = getattr(self.data_plane, "discard_unlaunched", None)
        if discard is not None:
            await discard(spec)

    async def fence(
        self,
        scope_id: str,
        retired_epoch: int,
        request_ids: Sequence[str],
    ) -> FenceReceipt:
        base_result, *participant_results = await asyncio.gather(
            self.data_plane.fence(scope_id, retired_epoch, request_ids),
            *(
                participant.fence(scope_id, retired_epoch, request_ids)
                for participant in self.participants
            ),
        )
        if not isinstance(base_result, FenceReceipt):
            raise TypeError("data-plane fence returned an invalid receipt")
        receipts = tuple(participant_results)
        if not all(isinstance(item, ParticipantFenceReceipt) for item in receipts):
            raise TypeError("retirement participant returned an invalid fence receipt")
        return FenceReceipt(
            scope_id=base_result.scope_id,
            retired_epoch=base_result.retired_epoch,
            requested_request_ids=base_result.requested_request_ids,
            acknowledged_request_ids=base_result.acknowledged_request_ids,
            launch_fence_installed=base_result.launch_fence_installed,
            participants=receipts,
        )

    async def probe_reuse(
        self,
        scope_id: str,
        retired_epoch: int,
        source_request_id: str | None,
        block_aligned_tokens: int,
    ) -> ReuseProbe:
        evidence = tuple(
            await asyncio.gather(
                *(
                    participant.probe_reuse(
                        scope_id,
                        retired_epoch,
                        source_request_id,
                        block_aligned_tokens,
                    )
                    for participant in self.participants
                )
            )
        )
        tiers = {item.tier for item in evidence}
        tier = next(iter(tiers)) if len(tiers) == 1 else ReuseTier.MIXED
        available = min((item.available_tokens for item in evidence), default=0)
        committed = min(
            (
                item.available_tokens
                if item.committed_tokens is None
                else item.committed_tokens
                for item in evidence
            ),
            default=0,
        )
        tenants = {item.source_tenant_id for item in evidence}
        attestations = {item.numerical_attestation for item in evidence}
        return ReuseProbe(
            source_request_id=source_request_id,
            tier=tier,
            available_tokens=available,
            no_pending_writer=all(item.no_pending_writer for item in evidence),
            source_fingerprint=(
                self.fingerprint
                if all(item.source_fingerprint == self.fingerprint for item in evidence)
                else None
            ),
            numerical_attestation=(
                next(iter(attestations)) if len(attestations) == 1 else None
            ),
            source_tenant_id=(
                next(iter(tenants)) if len(tenants) == 1 else "mixed"
            ),
            committed_tokens=committed,
            speculation_drained=all(item.speculation_drained for item in evidence),
            codec_drained=all(item.codec_drained for item in evidence),
            transfer_drained=all(item.transfer_drained for item in evidence),
            group_evidence=evidence,
        )

    async def prepare_resume(
        self,
        spec: RequestSpec,
        retired_epoch: int,
        recovery: RecoveryPlan,
    ) -> None:
        await asyncio.gather(
            *(
                participant.prepare_resume(spec, retired_epoch, recovery)
                for participant in self.participants
            )
        )
        await self.data_plane.prepare_resume(spec, retired_epoch, recovery)

    async def prepare_termination(self, scope_id: str, retired_epoch: int) -> None:
        prepare = getattr(self.data_plane, "prepare_termination", None)
        if prepare is not None:
            await prepare(scope_id, retired_epoch)

    async def reclaim(
        self,
        scope_id: str,
        retired_epoch: int,
    ) -> ReclaimReceipt:
        base_result, *participant_results = await asyncio.gather(
            self.data_plane.reclaim(scope_id, retired_epoch),
            *(
                participant.reclaim(scope_id, retired_epoch)
                for participant in self.participants
            ),
        )
        if not isinstance(base_result, ReclaimReceipt):
            raise TypeError("data-plane reclaim returned an invalid receipt")
        receipts = tuple(participant_results)
        safe = base_result.safe_to_reuse and all(
            item.safe_to_reuse for item in receipts
        )
        return ReclaimReceipt(
            scope_id=scope_id,
            retired_epoch=retired_epoch,
            safe_to_reuse=safe,
            reclaimed_blocks=(
                base_result.reclaimed_blocks
                + sum(item.reclaimed_blocks for item in receipts)
            ),
            detail=(
                f"distributed reclaim {'complete' if safe else 'incomplete'} "
                f"for {len(receipts)} participants"
            ),
            participants=receipts,
        )


@dataclass(slots=True)
class _ParticipantState:
    spec: RequestSpec
    available_tokens: int
    committed_tokens: int
    payload: bytes
    tier_key: TierObjectKey | None = None


class ScriptedRetirementParticipant:
    """Fault-injectable participant for distributed/codec/tiering experiments."""

    def __init__(
        self,
        coordinate: ParticipantCoordinate,
        *,
        fingerprint: EngineFingerprint,
        tier: ReuseTier = ReuseTier.HBM,
        numerical_attestation: str = "strict-clean-recomputation-v1",
        committed_token_limit: int | None = None,
        dram_store: DRAMTierStore | None = None,
    ) -> None:
        if coordinate not in fingerprint.participant_coordinates:
            raise ValueError("participant coordinate is absent from fingerprint")
        if tier not in {ReuseTier.HBM, ReuseTier.DRAM, ReuseTier.CPU}:
            raise ValueError("scripted participant requires a reusable storage tier")
        self.coordinate = coordinate
        self.fingerprint = fingerprint
        self.tier = tier
        self.numerical_attestation = numerical_attestation
        self.committed_token_limit = committed_token_limit
        self.dram_store = dram_store or (
            DRAMTierStore(ZlibKVCodec()) if tier is ReuseTier.DRAM else None
        )
        self.writer_drained = True
        self.speculation_drained = True
        self.codec_drained = True
        self.transfer_drained = True
        self.reclaim_safe = True
        self._safe_point_sequence = 0
        self._states: dict[str, _ParticipantState] = {}
        self.prepared_requests: list[str] = []
        self.restored_payloads: dict[str, bytes] = {}

    @property
    def _block_size(self) -> int:
        return next(
            group.block_size
            for group in self.fingerprint.normalized_kv_cache_groups
            if group.group_id == self.coordinate.kv_group_id
        )

    @staticmethod
    def _payload(
        spec: RequestSpec,
        coordinate: ParticipantCoordinate,
        through_token: int | None = None,
    ) -> bytes:
        prefix = f"{coordinate.participant_id}|".encode()
        selected = spec.prompt_token_ids[:through_token]
        tokens = b",".join(str(token).encode() for token in selected)
        return prefix + tokens

    async def bind_request(self, spec: RequestSpec) -> None:
        if spec.request_id in self._states:
            return
        available = len(spec.prompt_token_ids)
        available -= available % self._block_size
        committed = (
            available
            if self.committed_token_limit is None
            else min(available, self.committed_token_limit)
        )
        payload = self._payload(spec, self.coordinate)
        tier_key: TierObjectKey | None = None
        if self.tier is ReuseTier.DRAM:
            if self.dram_store is None:
                raise RuntimeError("DRAM participant has no tier store")
            tier_key = TierObjectKey(
                tenant_id=spec.tenant_id,
                scope_id=spec.scope_id,
                epoch=spec.epoch,
                request_id=spec.request_id,
                coordinate=self.coordinate,
            )
            self.dram_store.put(
                tier_key,
                payload,
                available_tokens=available,
            )
        self._states[spec.request_id] = _ParticipantState(
            spec=spec,
            available_tokens=available,
            committed_tokens=committed,
            payload=payload,
            tier_key=tier_key,
        )

    async def fence(
        self,
        scope_id: str,
        retired_epoch: int,
        request_ids: Sequence[str],
    ) -> ParticipantFenceReceipt:
        self._safe_point_sequence += 1
        states = [
            self._states[request_id]
            for request_id in request_ids
            if request_id in self._states
        ]
        cancelled = 0
        if self.dram_store is not None:
            tenants = {state.spec.tenant_id for state in states}
            if len(tenants) > 1:
                raise RuntimeError("one scope epoch crossed tenant domains")
            if tenants:
                cancelled = self.dram_store.cancel_inflight(
                    next(iter(tenants)),
                    scope_id,
                    retired_epoch,
                )
        return ParticipantFenceReceipt(
            coordinate=self.coordinate,
            acknowledged=True,
            writer_drained=self.writer_drained,
            speculation_drained=self.speculation_drained,
            codec_drained=self.codec_drained,
            transfer_drained=self.transfer_drained,
            safe_point_sequence=self._safe_point_sequence,
            detail=f"cancelled {cancelled} unfinished tier operations",
        )

    async def probe_reuse(
        self,
        scope_id: str,
        retired_epoch: int,
        source_request_id: str | None,
        block_aligned_tokens: int,
    ) -> KVGroupReuseEvidence:
        state = self._states.get(source_request_id or "")
        valid_source = (
            state is not None
            and state.spec.scope_id == scope_id
            and state.spec.epoch == retired_epoch
        )
        certificate = None
        if valid_source and state is not None and state.tier_key is not None:
            if self.dram_store is None:
                raise RuntimeError("DRAM state has no tier store")
            item = self.dram_store.get(state.tier_key)
            certificate = item.certificate if item is not None else None
            valid_source = item is not None
        available = (
            min(state.available_tokens, block_aligned_tokens)
            if valid_source and state is not None
            else 0
        )
        committed = (
            min(state.committed_tokens, available)
            if valid_source and state is not None
            else 0
        )
        return KVGroupReuseEvidence(
            coordinate=self.coordinate,
            source_request_id=source_request_id,
            source_tenant_id=(
                state.spec.tenant_id if valid_source and state is not None else "default"
            ),
            tier=self.tier if valid_source else ReuseTier.COLD,
            available_tokens=available,
            no_pending_writer=valid_source and self.writer_drained,
            source_fingerprint=self.fingerprint if valid_source else None,
            numerical_attestation=(
                self.numerical_attestation if valid_source else None
            ),
            committed_tokens=committed,
            speculation_drained=self.speculation_drained,
            codec_drained=self.codec_drained,
            transfer_drained=self.transfer_drained,
            codec_certificate=certificate,
        )

    async def prepare_resume(
        self,
        spec: RequestSpec,
        retired_epoch: int,
        recovery: RecoveryPlan,
    ) -> None:
        del retired_epoch
        plan = next(
            (
                item
                for item in recovery.group_plans
                if item.coordinate == self.coordinate
            ),
            None,
        )
        if plan is None:
            return
        source_id = recovery.certificate.source_request_id
        if source_id is None:
            raise RuntimeError("participant recovery has no source request")
        source = self._states[source_id]
        if source.spec.tenant_id != spec.tenant_id:
            raise PermissionError("cross-tenant participant restore is forbidden")
        if (
            source.spec.prompt_token_ids[: plan.reuse_tokens]
            != spec.prompt_token_ids[: plan.reuse_tokens]
        ):
            raise RuntimeError("participant source prefix differs from successor")
        if source.tier_key is not None:
            if self.dram_store is None:
                raise RuntimeError("DRAM state has no tier store")
            self.dram_store.restore(
                source.tier_key,
                target_tenant_id=spec.tenant_id,
            )
        payload = self._payload(spec, self.coordinate, plan.reuse_tokens)
        successor_tier_key: TierObjectKey | None = None
        if self.tier is ReuseTier.DRAM:
            if self.dram_store is None:
                raise RuntimeError("DRAM participant has no tier store")
            successor_tier_key = TierObjectKey(
                tenant_id=spec.tenant_id,
                scope_id=spec.scope_id,
                epoch=spec.epoch,
                request_id=spec.request_id,
                coordinate=self.coordinate,
            )
            self.dram_store.put(
                successor_tier_key,
                payload,
                available_tokens=plan.reuse_tokens,
            )
        self.restored_payloads[spec.request_id] = payload
        self.prepared_requests.append(spec.request_id)
        self._states[spec.request_id] = _ParticipantState(
            spec=spec,
            available_tokens=plan.reuse_tokens,
            committed_tokens=plan.reuse_tokens,
            payload=payload,
            tier_key=successor_tier_key,
        )

    async def reclaim(
        self,
        scope_id: str,
        retired_epoch: int,
    ) -> ParticipantReclaimReceipt:
        matching = [
            (request_id, state)
            for request_id, state in self._states.items()
            if state.spec.scope_id == scope_id and state.spec.epoch == retired_epoch
        ]
        reclaimed = 0
        for request_id, state in matching:
            reclaimed += state.available_tokens // self._block_size
            if self.dram_store is not None:
                self.dram_store.reclaim(
                    state.spec.tenant_id,
                    scope_id,
                    retired_epoch,
                )
            self._states.pop(request_id, None)
        return ParticipantReclaimReceipt(
            coordinate=self.coordinate,
            safe_to_reuse=self.reclaim_safe,
            reclaimed_blocks=reclaimed,
            detail=f"reclaimed {len(matching)} retired request states",
        )
