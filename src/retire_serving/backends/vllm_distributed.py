"""vLLM EngineCore RPC adapter for one distributed retirement participant."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..types import (
    CodecCertificate,
    CodecFingerprint,
    EngineFingerprint,
    KVGroupReuseEvidence,
    ParticipantCoordinate,
    ParticipantFenceReceipt,
    ParticipantReclaimReceipt,
    RecoveryPlan,
    RequestSpec,
    ReuseTier,
)


class VLLMDistributedParticipant:
    """Fail-closed translation of group-level vLLM utility receipts.

    The EngineCore side is expected to route each utility to the named TP rank
    and KV-cache group. This adapter deliberately treats absent booleans as
    false, so an older or partially patched core cannot accidentally certify a
    distributed handoff.
    """

    def __init__(
        self,
        engine: Any,
        coordinate: ParticipantCoordinate,
        *,
        fingerprint: EngineFingerprint,
    ) -> None:
        if coordinate not in fingerprint.participant_coordinates:
            raise ValueError("participant coordinate is absent from fingerprint")
        self.engine = engine
        self.coordinate = coordinate
        self.fingerprint = fingerprint
        self._request_specs: dict[str, RequestSpec] = {}

    async def _utility(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        core = getattr(self.engine, "engine_core", None)
        call = getattr(core, "call_utility_async", None)
        if call is None:
            raise RuntimeError("vLLM engine core lacks asynchronous utility RPC")
        result = await call(method, payload)
        if not isinstance(result, dict):
            raise TypeError(f"{method} returned a non-dictionary receipt")
        rank = int(result.get("tensor_parallel_rank", -1))
        pipeline_rank = int(result.get("pipeline_parallel_rank", -1))
        group = str(result.get("kv_group_id", ""))
        if (pipeline_rank, rank, group) != (
            self.coordinate.pipeline_parallel_rank,
            self.coordinate.tensor_parallel_rank,
            self.coordinate.kv_group_id,
        ):
            raise RuntimeError(f"{method} receipt coordinate mismatch")
        return result

    def _base_payload(self) -> dict[str, Any]:
        return {
            "pipeline_parallel_rank": self.coordinate.pipeline_parallel_rank,
            "tensor_parallel_rank": self.coordinate.tensor_parallel_rank,
            "kv_group_id": self.coordinate.kv_group_id,
        }

    async def bind_request(self, spec: RequestSpec) -> None:
        self._request_specs[spec.request_id] = spec

    async def fence(
        self,
        scope_id: str,
        retired_epoch: int,
        request_ids: Sequence[str],
    ) -> ParticipantFenceReceipt:
        payload = self._base_payload()
        payload.update(
            {
                "scope_id": scope_id,
                "retired_epoch": retired_epoch,
                "request_ids": list(request_ids),
            }
        )
        result = await self._utility("retire_participant_fence", payload)
        return ParticipantFenceReceipt(
            coordinate=self.coordinate,
            acknowledged=bool(result.get("acknowledged", False)),
            writer_drained=bool(result.get("writer_drained", False)),
            speculation_drained=bool(result.get("speculation_drained", False)),
            codec_drained=bool(result.get("codec_drained", False)),
            transfer_drained=bool(result.get("transfer_drained", False)),
            safe_point_sequence=int(result.get("safe_point_sequence", 0)),
            detail=str(result.get("detail", "")),
        )

    @staticmethod
    def _codec_certificate(payload: Any) -> CodecCertificate | None:
        if not isinstance(payload, dict):
            return None
        return CodecCertificate(
            fingerprint=CodecFingerprint(
                name=str(payload.get("name", "")),
                version=str(payload.get("version", "")),
                lossless=bool(payload.get("lossless", False)),
            ),
            encoded_sha256=str(payload.get("encoded_sha256", "")),
            decoded_sha256=str(payload.get("decoded_sha256", "")),
            operation_drained=bool(payload.get("operation_drained", False)),
            integrity_verified=bool(payload.get("integrity_verified", False)),
        )

    async def probe_reuse(
        self,
        scope_id: str,
        retired_epoch: int,
        source_request_id: str | None,
        block_aligned_tokens: int,
    ) -> KVGroupReuseEvidence:
        payload = self._base_payload()
        payload.update(
            {
                "scope_id": scope_id,
                "retired_epoch": retired_epoch,
                "source_request_id": source_request_id,
                "block_aligned_tokens": block_aligned_tokens,
            }
        )
        result = await self._utility("retire_participant_probe", payload)
        try:
            tier = ReuseTier(str(result.get("tier", "cold")))
        except ValueError:
            tier = ReuseTier.COLD
        committed = result.get("committed_tokens")
        return KVGroupReuseEvidence(
            coordinate=self.coordinate,
            source_request_id=(
                str(result["source_request_id"])
                if result.get("source_request_id") is not None
                else None
            ),
            source_tenant_id=str(result.get("source_tenant_id", "unattested")),
            tier=tier,
            available_tokens=int(result.get("available_tokens", 0)),
            no_pending_writer=bool(result.get("writer_drained", False)),
            source_fingerprint=(
                self.fingerprint if result.get("fingerprint_match") is True else None
            ),
            numerical_attestation=(
                str(result["numerical_attestation"])
                if result.get("numerical_attestation") is not None
                else None
            ),
            committed_tokens=int(committed) if committed is not None else None,
            speculation_drained=bool(result.get("speculation_drained", False)),
            codec_drained=bool(result.get("codec_drained", False)),
            transfer_drained=bool(result.get("transfer_drained", False)),
            codec_certificate=self._codec_certificate(result.get("codec")),
        )

    async def prepare_resume(
        self,
        spec: RequestSpec,
        retired_epoch: int,
        recovery: RecoveryPlan,
    ) -> None:
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
        payload = self._base_payload()
        payload.update(
            {
                "scope_id": spec.scope_id,
                "tenant_id": spec.tenant_id,
                "retired_epoch": retired_epoch,
                "successor_epoch": spec.epoch,
                "successor_request_id": spec.request_id,
                "source_request_id": recovery.certificate.source_request_id,
                "tier": plan.tier.value,
                "reuse_tokens": plan.reuse_tokens,
            }
        )
        result = await self._utility("retire_participant_prepare", payload)
        if result.get("prepared") is not True:
            raise RuntimeError("vLLM participant did not prepare successor state")

    async def reclaim(
        self,
        scope_id: str,
        retired_epoch: int,
    ) -> ParticipantReclaimReceipt:
        payload = self._base_payload()
        payload.update({"scope_id": scope_id, "retired_epoch": retired_epoch})
        result = await self._utility("retire_participant_reclaim", payload)
        return ParticipantReclaimReceipt(
            coordinate=self.coordinate,
            safe_to_reuse=bool(result.get("safe_to_reuse", False)),
            reclaimed_blocks=int(result.get("reclaimed_blocks", 0)),
            detail=str(result.get("detail", "")),
        )
