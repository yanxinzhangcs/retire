"""Lossless worker-local DRAM tiering for the pinned vLLM integration."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any

from ..types import (
    CodecCertificate,
    CodecFingerprint,
    ParticipantFenceReceipt,
    RecoveryPlan,
    RequestSpec,
    ReuseProbe,
    ReuseTier,
)
from .vllm_pinned import PinnedVLLMHooks


class TieredPinnedVLLMHooks(PinnedVLLMHooks):
    """Stage certified vLLM pages through host DRAM before successor reuse."""

    def __init__(
        self,
        *args: Any,
        codec: str = "identity",
        reclaim_hbm_capacity: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if codec not in {"identity", "zlib"}:
            raise ValueError("tiered vLLM hook supports identity or zlib")
        self.codec = codec
        self.reclaim_hbm_capacity = reclaim_hbm_capacity
        self.offload_receipts: list[dict[str, Any]] = []
        self.restore_receipts: list[dict[str, Any]] = []

    @staticmethod
    def _aggregate_digest(workers: list[dict[str, Any]], field: str) -> str:
        digest = hashlib.sha256()
        for item in sorted(
            workers,
            key=lambda row: (
                int(row.get("pipeline_parallel_rank", 0)),
                int(row.get("tensor_parallel_rank", 0)),
            ),
        ):
            value = str(item.get(field, ""))
            if not value:
                return ""
            digest.update(value.encode())
        return digest.hexdigest()

    def _codec_certificate(self, receipt: dict[str, Any]) -> CodecCertificate:
        workers = list(receipt.get("workers", ()))
        return CodecCertificate(
            fingerprint=CodecFingerprint(
                name=self.codec,
                version="worker-local-v1",
                lossless=True,
            ),
            encoded_sha256=self._aggregate_digest(workers, "encoded_sha256"),
            decoded_sha256=self._aggregate_digest(workers, "decoded_sha256"),
            operation_drained=bool(receipt.get("operation_drained")),
            integrity_verified=bool(workers)
            and all(
                bool(item.get("encoded_sha256"))
                and bool(item.get("decoded_sha256"))
                for item in workers
            ),
        )

    async def install_scope_fence(
        self, scope_id: str, retired_epoch: int, request_ids: tuple[str, ...]
    ) -> tuple[ParticipantFenceReceipt, ...]:
        participants = await super().install_scope_fence(
            scope_id, retired_epoch, request_ids
        )
        key = (scope_id, retired_epoch)
        receipt = self._receipts[key]
        # A numerical certificate can be tighter than the resident prefix.
        # Apply that bound before D2H/codec work when this fence has one
        # unambiguous source, so bytes that ALIGN can never admit do not cross
        # PCIe or enter the codec. Multi-source fences defer trimming until
        # ALIGN chooses the source.
        if self.max_reuse_tokens is not None and len(request_ids) == 1:
            source = request_ids[0]
            alignment = self.fingerprint.alignment_block_size
            transfer_tokens = (
                int(self.max_reuse_tokens) // alignment * alignment
            )
            trim = await self._utility(
                "retire_trim_pins",
                self._pin_key(*key),
                source,
                transfer_tokens,
            )
            receipt["pre_offload_trim"] = trim
            receipt["active_pinned_blocks"] = int(trim["kept_blocks"])
            receipt["sources"][source]["group_block_ids"] = list(
                trim["kept_group_block_ids"]
            )
        if self.reclaim_hbm_capacity:
            offload = await self._utility(
                "retire_offload_pins_capacity",
                self._pin_key(*key),
                self.codec,
            )
            receipt["active_pinned_blocks"] = 0
        else:
            offload = await self._utility(
                "retire_offload_pins",
                self._pin_key(*key),
                self.codec,
                True,
            )
        if not isinstance(offload, dict) or not offload.get("operation_drained"):
            raise RuntimeError("vLLM DRAM offload did not reach a safe point")
        certificate = self._codec_certificate(offload)
        if not certificate.admissible:
            raise RuntimeError("vLLM DRAM codec certificate failed closed")
        receipt["dram_offload"] = offload
        self.offload_receipts.append(offload)
        return participants

    async def probe_reuse(
        self,
        scope_id: str,
        retired_epoch: int,
        source_request_id: str | None,
        block_aligned_tokens: int,
    ) -> ReuseProbe:
        probe = await super().probe_reuse(
            scope_id,
            retired_epoch,
            source_request_id,
            block_aligned_tokens,
        )
        offload = self._receipts[(scope_id, retired_epoch)]["dram_offload"]
        certificate = self._codec_certificate(offload)
        evidence = tuple(
            replace(
                item,
                tier=(
                    ReuseTier.DRAM
                    if item.tier is not ReuseTier.COLD
                    else ReuseTier.COLD
                ),
                codec_drained=certificate.operation_drained,
                transfer_drained=certificate.operation_drained,
                codec_certificate=certificate,
            )
            for item in probe.group_evidence
        )
        return replace(
            probe,
            tier=(
                ReuseTier.DRAM
                if probe.tier is not ReuseTier.COLD
                else ReuseTier.COLD
            ),
            codec_drained=certificate.operation_drained,
            transfer_drained=certificate.operation_drained,
            codec_certificate=certificate,
            group_evidence=evidence,
        )

    async def prepare_resume(
        self,
        spec: RequestSpec,
        retired_epoch: int,
        recovery: RecoveryPlan,
    ) -> None:
        if not self.reclaim_hbm_capacity:
            await super().prepare_resume(spec, retired_epoch, recovery)
            if not recovery.certificate.admissible:
                return
            key = (spec.scope_id, retired_epoch)
            restore = await self._utility(
                "retire_restore_pins",
                self._pin_key(*key),
            )
            if (
                not isinstance(restore, dict)
                or not restore.get("operation_drained")
                or not restore.get("integrity_verified")
            ):
                raise RuntimeError("vLLM DRAM restore failed integrity verification")
            self._receipts[key]["dram_restore"] = restore
            self.restore_receipts.append(restore)
            return

        key = (spec.scope_id, retired_epoch)
        if not recovery.certificate.admissible:
            await self._release(key)
            return
        source = recovery.certificate.source_request_id
        if source is None:
            raise RuntimeError("admissible DRAM recovery has no source request")
        restore = await self._utility(
            "retire_restore_capacity_pins",
            self._pin_key(*key),
            source,
            recovery.reuse_tokens,
        )
        if (
            not isinstance(restore, dict)
            or not restore.get("operation_drained")
            or not restore.get("integrity_verified")
        ):
            raise RuntimeError("vLLM DRAM restore failed integrity verification")
        receipt = self._receipts[key]
        receipt["dram_restore"] = restore
        receipt["active_pinned_blocks"] = int(
            restore.get("allocated_hbm_blocks", 0)
        )
        restored_groups = [
            list(map(int, group))
            for group in restore.get("restored_group_block_ids", ())
        ]
        receipt["sources"][source]["group_block_ids"] = restored_groups
        expectation = await self._utility(
            "retire_expect_request_blocks",
            spec.request_id,
            restored_groups,
        )
        if not isinstance(expectation, dict) or not expectation.get("armed"):
            raise RuntimeError("vLLM core did not arm DRAM successor verification")
        self._successor_plans[spec.request_id] = (
            key,
            source,
            recovery.reuse_tokens,
        )
        self.restore_receipts.append(restore)
