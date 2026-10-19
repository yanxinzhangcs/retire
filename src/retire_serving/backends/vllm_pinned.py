"""Pinned paged-KV hooks for the pinned vLLM 0.13 research integration."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

from ..types import (
    EngineFingerprint,
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


@dataclass(frozen=True, slots=True)
class PhysicalReuseVerification:
    request_id: str
    kv_group_id: str
    expected_block_ids: tuple[int, ...]
    installed_block_ids: tuple[int, ...]
    retired_expected_block_ids: tuple[int, ...]
    retired_installed_block_ids: tuple[int, ...]
    live_expected_block_ids: tuple[int, ...]
    live_installed_block_ids: tuple[int, ...]
    verified_live_physical_block_ids: tuple[int, ...]
    retired_slots_cleared: bool
    same_physical_blocks: bool


class PinnedVLLMHooks:
    """Use core-side safe points and block refcounts as a reuse certificate.

    The companion vLLM patch exposes three utility RPCs.  The fence RPC runs
    only between engine steps, pins complete cached blocks before abort frees
    the old request, and returns only after the old GPU step has drained.  The
    successor then goes through vLLM's normal prefix lookup.  Its first output
    triggers an asynchronous audit that confirms the expected physical block
    IDs are installed.  Pins are released as soon as adoption is verified.
    """

    def __init__(
        self,
        engine: Any,
        *,
        fingerprint: EngineFingerprint,
        numerical_attestation: str,
        max_reuse_tokens: int | None = None,
        tenant_pin_quota_blocks: int | None = None,
        enable_epoch_fast_path: bool = True,
        cold_fast_path: bool = False,
    ) -> None:
        self.engine = engine
        self.fingerprint = fingerprint
        self.numerical_attestation = numerical_attestation
        self.max_reuse_tokens = max_reuse_tokens
        self.tenant_pin_quota_blocks = tenant_pin_quota_blocks
        self.enable_epoch_fast_path = enable_epoch_fast_path
        self.cold_fast_path = cold_fast_path
        if self.cold_fast_path and self.max_reuse_tokens != 0:
            raise ValueError("cold fast path requires max_reuse_tokens=0")
        self._receipts: dict[tuple[str, int], dict[str, Any]] = {}
        self._successor_plans: dict[str, tuple[tuple[str, int], str, int]] = {}
        self._release_events: dict[tuple[str, int], asyncio.Event] = {}
        self._verified_requests: set[str] = set()
        self._verification_tasks: dict[str, asyncio.Task[None]] = {}
        self._verification_errors: dict[tuple[str, int], BaseException] = {}
        self._request_specs: dict[str, RequestSpec] = {}
        self._request_authorities: dict[str, tuple[int, int]] = {}
        self.verifications: list[PhysicalReuseVerification] = []
        self.admission_rejections = 0

    @property
    def fence_receipts(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._receipts.values())

    @property
    def requires_participant_fence(self) -> bool:
        return not self.cold_fast_path

    async def _utility(self, method: str, *args: Any) -> Any:
        call = getattr(self.engine.engine_core, "call_utility_async", None)
        if call is None:
            raise RuntimeError("vLLM engine core lacks asynchronous utility RPC")
        return await call(method, *args)

    @staticmethod
    def _pin_key(scope_id: str, retired_epoch: int) -> str:
        return f"{scope_id}:e{retired_epoch}"

    async def bind_request(self, spec: RequestSpec) -> bool:
        self._request_specs[spec.request_id] = spec
        if self.enable_epoch_fast_path and os.environ.get(
            "RETIRE_EPOCH_MAILBOX_PATH"
        ):
            from ..epoch_mailbox import (
                get_frontend_mailbox,
                host_check_version,
                pack_version,
            )

            mailbox = get_frontend_mailbox()
            slot = mailbox.bind_scope(spec.scope_id, spec.epoch, spec.generation)
            expected_version = pack_version(spec.epoch, spec.generation)
            receipt = host_check_version(
                slot,
                expected_version,
            )
            if not bool(receipt.get("admission_safe")):
                self._request_specs.pop(spec.request_id, None)
                self.admission_rejections += 1
                return False
            self._request_authorities[spec.request_id] = (
                slot,
                expected_version,
            )
        return True

    def encode_request_cache_salt(self, spec: RequestSpec) -> str:
        from ..epoch_mailbox import encode_cache_salt_authority

        authority = self._request_authorities.get(spec.request_id)
        if authority is None:
            return spec.cache_salt
        return encode_cache_salt_authority(
            spec.cache_salt,
            authority[0],
            authority[1],
        )

    def advance_epoch(
        self, scope_id: str, new_epoch: int, generation: int
    ) -> None:
        """Publish ADVANCE through mmap, bypassing queued EngineCore RPCs."""
        if self.enable_epoch_fast_path and os.environ.get(
            "RETIRE_EPOCH_MAILBOX_PATH"
        ):
            from ..epoch_mailbox import get_frontend_mailbox

            get_frontend_mailbox().publish(scope_id, new_epoch, generation)

    async def epoch_audit(self) -> dict[str, Any]:
        if not os.environ.get("RETIRE_EPOCH_MAILBOX_PATH"):
            return {"workers": []}
        result = await self._utility("retire_epoch_audit")
        if not isinstance(result, dict):
            raise TypeError("vLLM epoch audit returned a non-dictionary receipt")
        return result

    async def install_scope_fence(
        self, scope_id: str, retired_epoch: int, request_ids: tuple[str, ...]
    ) -> tuple[ParticipantFenceReceipt, ...]:
        key = (scope_id, retired_epoch)
        if self.cold_fast_path:
            self._receipts[key] = {
                "safe_point": True,
                "logical_launch_fence": True,
                "cold_no_reuse": True,
                "pinned_blocks": 0,
                "sources": {},
            }
            event = self._release_events[key] = asyncio.Event()
            event.set()
            return ()
        tenant_ids = {
            self._request_specs[request_id].tenant_id
            for request_id in request_ids
            if request_id in self._request_specs
        }
        if len(tenant_ids) > 1:
            raise RuntimeError("one retirement fence cannot span tenant domains")
        tenant_id = next(iter(tenant_ids), "default")
        receipt = await self._utility(
            "retire_fence_and_pin",
            self._pin_key(*key),
            list(request_ids),
            tenant_id,
            self.tenant_pin_quota_blocks,
        )
        if not isinstance(receipt, dict) or not receipt.get("safe_point"):
            raise RuntimeError("vLLM core did not return a safe fence receipt")
        participants = tuple(
            ParticipantFenceReceipt(
                coordinate=ParticipantCoordinate(
                    tensor_parallel_rank=int(item["tensor_parallel_rank"]),
                    kv_group_id=str(item["kv_group_id"]),
                    pipeline_parallel_rank=int(
                        item.get("pipeline_parallel_rank", 0)
                    ),
                ),
                acknowledged=bool(item.get("acknowledged")),
                writer_drained=bool(item.get("writer_drained")),
                speculation_drained=bool(item.get("speculation_drained")),
                codec_drained=bool(item.get("codec_drained")),
                transfer_drained=bool(item.get("transfer_drained")),
                safe_point_sequence=int(item.get("safe_point_sequence", 0)),
                detail=str(item.get("detail", "")),
            )
            for item in receipt.get("participants", ())
        )
        expected = set(self.fingerprint.participant_coordinates)
        observed = {item.coordinate for item in participants}
        if observed != expected or len(observed) != len(participants):
            raise RuntimeError(
                "vLLM core fence receipt does not cover TP rank x KV group"
            )
        if not all(item.safe for item in participants):
            raise RuntimeError("vLLM core returned an unsafe participant fence")
        self._receipts[key] = receipt
        self._release_events[key] = asyncio.Event()
        return participants

    async def probe_reuse(
        self,
        scope_id: str,
        retired_epoch: int,
        source_request_id: str | None,
        block_aligned_tokens: int,
    ) -> ReuseProbe:
        if self.cold_fast_path:
            return ReuseProbe.unavailable(source_request_id)
        receipt = self._receipts.get((scope_id, retired_epoch), {})
        sources = receipt.get("sources", {})
        source = sources.get(source_request_id, {}) if source_request_id else {}
        available = int(source.get("pinned_tokens", 0))
        if self.max_reuse_tokens is not None:
            available = min(available, self.max_reuse_tokens)
        physically_safe = bool(source.get("writer_drained")) and bool(
            source.get("all_blocks_resident")
        )
        group_tokens = [int(item) for item in source.get("group_tokens", ())]
        bounded_group_tokens = [
            min(item, block_aligned_tokens) for item in group_tokens
        ]
        if self.max_reuse_tokens is not None:
            bounded_group_tokens = [
                min(item, self.max_reuse_tokens) for item in bounded_group_tokens
            ]
        group_evidence = tuple(
            KVGroupReuseEvidence(
                coordinate=coordinate,
                source_request_id=source_request_id,
                source_tenant_id=(
                    self._request_specs[source_request_id].tenant_id
                    if source_request_id in self._request_specs
                    else "default"
                ),
                tier=ReuseTier.HBM if physically_safe else ReuseTier.COLD,
                available_tokens=min(
                    bounded_group_tokens[group_index]
                    if group_index < len(bounded_group_tokens)
                    else 0,
                    block_aligned_tokens,
                ),
                no_pending_writer=physically_safe,
                source_fingerprint=(self.fingerprint if physically_safe else None),
                numerical_attestation=(
                    self.numerical_attestation if physically_safe else None
                ),
                committed_tokens=min(
                    int(source.get("committed_tokens", 0)), block_aligned_tokens
                ),
                speculation_drained=True,
            )
            for coordinate in self.fingerprint.participant_coordinates
            for group_index, group in enumerate(
                self.fingerprint.normalized_kv_cache_groups
            )
            if coordinate.kv_group_id == group.group_id
        )
        return ReuseProbe(
            source_request_id=source_request_id,
            tier=ReuseTier.HBM if physically_safe else ReuseTier.COLD,
            available_tokens=min(available, block_aligned_tokens),
            no_pending_writer=physically_safe,
            source_fingerprint=self.fingerprint if physically_safe else None,
            numerical_attestation=(
                self.numerical_attestation if physically_safe else None
            ),
            source_tenant_id=(
                self._request_specs[source_request_id].tenant_id
                if source_request_id in self._request_specs
                else "default"
            ),
            committed_tokens=min(
                int(source.get("committed_tokens", 0)), block_aligned_tokens
            ),
            speculation_drained=True,
            group_evidence=group_evidence,
        )

    async def prepare_resume(
        self,
        spec: RequestSpec,
        retired_epoch: int,
        recovery: RecoveryPlan,
    ) -> None:
        if self.cold_fast_path:
            if recovery.certificate.admissible or recovery.reuse_tokens:
                raise RuntimeError("cold fast path cannot install reused state")
            return
        key = (spec.scope_id, retired_epoch)
        if recovery.certificate.admissible:
            source = recovery.certificate.source_request_id
            if source is None:
                raise RuntimeError("admissible recovery has no source request")
            receipt = self._receipts[key]
            trim = await self._utility(
                "retire_trim_pins",
                self._pin_key(*key),
                source,
                recovery.reuse_tokens,
            )
            receipt["active_pinned_blocks"] = int(trim["kept_blocks"])
            receipt["sources"][source]["group_block_ids"] = list(
                trim["kept_group_block_ids"]
            )
            expected_groups = [
                list(blocks[: recovery.reuse_tokens // group.block_size])
                for blocks, group in zip(
                    receipt["sources"][source]["group_block_ids"],
                    self.fingerprint.normalized_kv_cache_groups,
                    strict=True,
                )
            ]
            expectation = await self._utility(
                "retire_expect_request_blocks",
                spec.request_id,
                expected_groups,
            )
            if not isinstance(expectation, dict) or not expectation.get("armed"):
                raise RuntimeError("vLLM core did not arm successor KV verification")
            self._successor_plans[spec.request_id] = (
                key,
                source,
                recovery.reuse_tokens,
            )
            return
        await self._release(key)

    async def prepare_termination(self, scope_id: str, retired_epoch: int) -> None:
        """A terminal retirement has no successor that could adopt its pins."""

        await self._release((scope_id, retired_epoch))

    async def observe_output(self, spec: RequestSpec, output: Any) -> None:
        plan = self._successor_plans.get(spec.request_id)
        if (
            plan is None
            or spec.request_id in self._verified_requests
            or spec.request_id in self._verification_tasks
        ):
            return
        candidates = getattr(output, "outputs", None)
        if candidates is not None and not any(
            getattr(candidate, "token_ids", ()) for candidate in candidates
        ):
            # vLLM emits a terminal empty frame when a successor is superseded
            # before producing a token.  No state was consumed by an externally
            # visible effect, so there is no adoption to certify.
            return
        task = asyncio.create_task(self._verify_and_release(spec, plan))
        self._verification_tasks[spec.request_id] = task

    def needs_output_observation(self, spec: RequestSpec) -> bool:
        """Only an adopted prefix needs a first-output ownership check."""
        return spec.request_id in self._successor_plans

    async def _verify_and_release(
        self,
        spec: RequestSpec,
        plan: tuple[tuple[str, int], str, int],
    ) -> None:
        key, source_id, reuse_tokens = plan
        try:
            receipt = self._receipts[key]
            expected_groups = tuple(
                tuple(blocks[: reuse_tokens // group.block_size])
                for blocks, group in zip(
                    receipt["sources"][source_id]["group_block_ids"],
                    self.fingerprint.normalized_kv_cache_groups,
                    strict=True,
                )
            )
            result = await self._utility(
                "retire_verify_request_blocks",
                spec.request_id,
                [list(group) for group in expected_groups],
            )
            installed_groups = tuple(
                tuple(int(item) for item in group)
                for group in result.get("installed_group_block_ids", ())
            )
            core_groups = result.get("group_verifications", ())
            verifications = []
            for group_index, (group, expected) in enumerate(
                zip(
                    self.fingerprint.normalized_kv_cache_groups,
                    expected_groups,
                    strict=True,
                )
            ):
                installed = (
                    installed_groups[group_index]
                    if group_index < len(installed_groups)
                    else ()
                )
                core_group = (
                    core_groups[group_index]
                    if group_index < len(core_groups)
                    else {}
                )
                live_expected = tuple(
                    int(item)
                    for item in core_group.get("live_expected_block_ids", expected)
                )
                live_installed = tuple(
                    int(item)
                    for item in core_group.get("live_installed_block_ids", installed)
                )
                retired_expected = tuple(
                    int(item)
                    for item in core_group.get("retired_expected_block_ids", ())
                )
                retired_installed = tuple(
                    int(item)
                    for item in core_group.get("retired_installed_block_ids", ())
                )
                exact = bool(
                    core_group.get(
                        "same_physical_blocks",
                        installed == expected,
                    )
                )
                verifications.append(
                    PhysicalReuseVerification(
                        request_id=spec.request_id,
                        kv_group_id=group.group_id,
                        expected_block_ids=expected,
                        installed_block_ids=installed,
                        retired_expected_block_ids=retired_expected,
                        retired_installed_block_ids=retired_installed,
                        live_expected_block_ids=live_expected,
                        live_installed_block_ids=live_installed,
                        verified_live_physical_block_ids=tuple(
                            int(item)
                            for item in core_group.get(
                                "verified_live_physical_block_ids",
                                live_expected,
                            )
                        ),
                        retired_slots_cleared=bool(
                            core_group.get("retired_slots_cleared", True)
                        ),
                        same_physical_blocks=exact,
                    )
                )
            self.verifications.extend(verifications)
            self._verified_requests.add(spec.request_id)
            if not bool(result.get("same_physical_blocks")) or not all(
                item.same_physical_blocks for item in verifications
            ):
                raise RuntimeError(
                    "successor did not preserve all live pinned physical KV "
                    f"blocks: {core_groups or result}"
                )
            # The successor's block table now owns the verified live references.
            # The transient retired hold is no longer needed; releasing it here
            # bounds memory retention by adoption, not by successor completion.
            await self._release(key)
            self._successor_plans.pop(spec.request_id, None)
        except BaseException as exc:
            self._verification_errors[key] = exc
            event = self._release_events.get(key)
            if event is not None:
                event.set()
            raise

    async def request_complete(self, spec: RequestSpec) -> None:
        self._request_authorities.pop(spec.request_id, None)
        task = self._verification_tasks.pop(spec.request_id, None)
        if task is not None:
            await task
        plan = self._successor_plans.pop(spec.request_id, None)
        if plan is not None:
            key, _, _ = plan
            await self._utility(
                "retire_cancel_request_expectation", spec.request_id
            )
            await self._release(key)
        self._verified_requests.discard(spec.request_id)

    async def _release(self, key: tuple[str, int]) -> None:
        event = self._release_events.get(key)
        if event is None or event.is_set():
            return
        release = await self._utility("retire_release_pins", self._pin_key(*key))
        if not isinstance(release, dict):
            raise TypeError("vLLM core returned an invalid release receipt")
        self._receipts[key]["release_receipt"] = release
        event.set()

    async def reclaim_scope(self, scope_id: str, retired_epoch: int) -> ReclaimReceipt:
        key = (scope_id, retired_epoch)
        event = self._release_events.get(key)
        if event is not None:
            await event.wait()
        failure = self._verification_errors.pop(key, None)
        if failure is not None:
            raise RuntimeError("successor KV adoption verification failed") from failure
        receipt = self._receipts.get(key, {})
        if self.cold_fast_path:
            return ReclaimReceipt(
                scope_id=scope_id,
                retired_epoch=retired_epoch,
                safe_to_reuse=True,
                reclaimed_blocks=0,
                detail=(
                    "cold successor published independently; vLLM reclaimed "
                    "the drained old request"
                ),
            )
        release = receipt.get("release_receipt", {})
        worker_receipts = tuple(release.get("worker_receipts", ()))
        fence_sequence = int(receipt.get("safe_point_sequence", 0))
        release_sequence = int(release.get("safe_point_sequence", 0))
        participants = tuple(
            ParticipantReclaimReceipt(
                coordinate=coordinate,
                safe_to_reuse=(
                    release_sequence > fence_sequence
                    and bool(worker_receipts)
                    and all(
                        bool(item.get("writer_drained"))
                        for item in worker_receipts
                        if int(item.get("tensor_parallel_rank", -1))
                        == coordinate.tensor_parallel_rank
                        and int(item.get("pipeline_parallel_rank", -1))
                        == coordinate.pipeline_parallel_rank
                    )
                    and any(
                        int(item.get("tensor_parallel_rank", -1))
                        == coordinate.tensor_parallel_rank
                        and int(item.get("pipeline_parallel_rank", -1))
                        == coordinate.pipeline_parallel_rank
                        for item in worker_receipts
                    )
                ),
                detail="logical block pin released after distributed writer fence",
            )
            for coordinate in self.fingerprint.participant_coordinates
        )
        safe = bool(participants) and all(item.safe_to_reuse for item in participants)
        return ReclaimReceipt(
            scope_id=scope_id,
            retired_epoch=retired_epoch,
            safe_to_reuse=safe,
            reclaimed_blocks=int(receipt.get("pinned_blocks", 0)),
            detail="vLLM core safe-point pins released after successor completion",
            participants=participants,
        )
