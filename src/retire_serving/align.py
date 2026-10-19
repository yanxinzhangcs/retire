"""Conservative structural, numerical, and physical reuse certification."""

from __future__ import annotations

from collections.abc import Iterable

from .types import (
    AlignmentCertificate,
    EngineFingerprint,
    GroupRecoveryPlan,
    KVGroupReuseEvidence,
    NumericalContract,
    RecoveryPlan,
    ReuseProbe,
    ReuseTier,
)


def common_prefix_length(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    length = 0
    for lhs, rhs in zip(left, right, strict=False):
        if lhs != rhs:
            break
        length += 1
    return length


class AlignmentPlanner:
    def __init__(self, block_size: int) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.block_size = block_size

    def choose_source(
        self,
        retired_contexts: Iterable[tuple[str, tuple[int, ...]]],
        revised_context: tuple[int, ...],
    ) -> tuple[str | None, int]:
        best_request: str | None = None
        best_prefix = 0
        for request_id, old_context in retired_contexts:
            prefix = common_prefix_length(old_context, revised_context)
            if prefix > best_prefix:
                best_request = request_id
                best_prefix = prefix
        return best_request, best_prefix

    def plan(
        self,
        *,
        source_request_id: str | None,
        common_prefix_tokens: int,
        revised_tokens: int,
        contract: NumericalContract,
        current_fingerprint: EngineFingerprint,
        current_tenant_id: str,
        probe: ReuseProbe,
    ) -> RecoveryPlan:
        aligned = common_prefix_tokens - (common_prefix_tokens % self.block_size)
        evidence = probe.group_evidence
        if evidence:
            result = self._plan_groups(
                evidence=evidence,
                source_request_id=source_request_id,
                aligned=aligned,
                contract=contract,
                current_fingerprint=current_fingerprint,
                current_tenant_id=current_tenant_id,
            )
            (
                selected,
                structural_valid,
                numerical_valid,
                physical_valid,
                group_valid,
                tenant_valid,
                speculative_valid,
                codec_valid,
                transfer_valid,
                tier,
                group_plans,
            ) = result
            no_pending_writer = all(item.no_pending_writer for item in evidence)
        else:
            committed = (
                probe.available_tokens
                if probe.committed_tokens is None
                else min(probe.available_tokens, probe.committed_tokens)
            )
            available_aligned = committed - (committed % self.block_size)
            selected = min(aligned, available_aligned)
            group_valid = len(current_fingerprint.participant_coordinates) == 1
            tenant_valid = probe.source_tenant_id == current_tenant_id
            speculative_valid = probe.speculation_drained and (
                current_fingerprint.speculative_mode == "disabled"
                or probe.committed_tokens is not None
            )
            codec_valid = (
                probe.codec_drained
                and not (
                    probe.tier is ReuseTier.DRAM
                    and probe.codec_certificate is None
                )
                and (
                    probe.codec_certificate is None
                    or probe.codec_certificate.admissible
                )
            )
            transfer_valid = probe.transfer_drained
            structural_valid = (
                source_request_id is not None
                and aligned > 0
                and probe.source_request_id == source_request_id
                and probe.source_fingerprint == current_fingerprint
            )
            numerical_valid = (
                contract.required_attestation is None
                or probe.numerical_attestation == contract.required_attestation
            )
            no_pending_writer = probe.no_pending_writer
            physical_valid = (
                probe.tier is not ReuseTier.COLD
                and no_pending_writer
                and selected > 0
                and speculative_valid
                and codec_valid
                and transfer_valid
            )
            tier = probe.tier
            group_plans = ()

        admissible = (
            structural_valid
            and numerical_valid
            and physical_valid
            and group_valid
            and tenant_valid
            and speculative_valid
            and codec_valid
            and transfer_valid
        )

        if source_request_id is None or aligned == 0:
            reason = "no block-aligned common prefix"
        elif not group_valid:
            reason = "PP stage, TP rank, or KV-cache group evidence is incomplete"
        elif not tenant_valid:
            reason = "source and successor belong to different tenant domains"
        elif not structural_valid:
            reason = "source identity or engine fingerprint was not certified"
        elif not numerical_valid:
            reason = "declared numerical contract lacks a matching attestation"
        elif not speculative_valid:
            reason = "draft or verifier state has not drained to a committed boundary"
        elif not codec_valid:
            reason = "codec completion or decoded-byte integrity was not certified"
        elif not transfer_valid:
            reason = "tier transfer has not reached a writer-free safe point"
        elif not physical_valid:
            reason = (
                "reusable bytes are absent or still have a pending writer"
                if not no_pending_writer or selected == 0
                else "physical reuse evidence was not certified"
            )
        else:
            reason = (
                f"certified {selected} of {aligned} eligible tokens "
                f"from {tier.value}"
            )

        reuse_tokens = selected if admissible else 0
        selected_tier = tier if admissible else ReuseTier.COLD
        certificate = AlignmentCertificate(
            source_request_id=source_request_id,
            common_prefix_tokens=common_prefix_tokens,
            block_aligned_tokens=aligned,
            structural_valid=structural_valid,
            numerical_valid=numerical_valid,
            physical_valid=physical_valid,
            admissible=admissible,
            reason=reason,
            group_valid=group_valid,
            tenant_valid=tenant_valid,
            speculative_valid=speculative_valid,
            codec_valid=codec_valid,
            transfer_valid=transfer_valid,
        )
        return RecoveryPlan(
            tier=selected_tier,
            reuse_tokens=reuse_tokens,
            recompute_tokens=max(0, revised_tokens - reuse_tokens),
            certificate=certificate,
            group_plans=group_plans if admissible else (),
        )

    def _plan_groups(
        self,
        *,
        evidence: tuple[KVGroupReuseEvidence, ...],
        source_request_id: str | None,
        aligned: int,
        contract: NumericalContract,
        current_fingerprint: EngineFingerprint,
        current_tenant_id: str,
    ) -> tuple[
        int,
        bool,
        bool,
        bool,
        bool,
        bool,
        bool,
        bool,
        bool,
        ReuseTier,
        tuple[GroupRecoveryPlan, ...],
    ]:
        expected = set(current_fingerprint.participant_coordinates)
        observed = [item.coordinate for item in evidence]
        group_valid = len(observed) == len(set(observed)) and set(observed) == expected
        limits = [
            min(
                item.available_tokens,
                item.available_tokens
                if item.committed_tokens is None
                else item.committed_tokens,
            )
            for item in evidence
        ]
        available = min(limits, default=0)
        available_aligned = available - (available % self.block_size)
        selected = min(aligned, available_aligned)
        structural_valid = bool(evidence) and all(
            item.source_request_id == source_request_id
            and item.source_fingerprint == current_fingerprint
            for item in evidence
        )
        tenant_valid = all(
            item.source_tenant_id == current_tenant_id for item in evidence
        )
        numerical_valid = all(
            contract.required_attestation is None
            or item.numerical_attestation == contract.required_attestation
            for item in evidence
        )
        speculative_valid = all(
            item.speculation_drained
            and (
                current_fingerprint.speculative_mode == "disabled"
                or item.committed_tokens is not None
            )
            for item in evidence
        )
        codec_valid = all(item.codec_valid for item in evidence)
        transfer_valid = all(item.transfer_drained for item in evidence)
        physical_valid = (
            bool(evidence)
            and selected > 0
            and all(
                item.tier is not ReuseTier.COLD and item.no_pending_writer
                for item in evidence
            )
            and speculative_valid
            and codec_valid
            and transfer_valid
        )
        tiers = {item.tier for item in evidence}
        tier = next(iter(tiers)) if len(tiers) == 1 else ReuseTier.MIXED
        plans = tuple(
            GroupRecoveryPlan(
                coordinate=item.coordinate,
                tier=item.tier,
                reuse_tokens=selected,
            )
            for item in evidence
        )
        return (
            selected,
            structural_valid,
            numerical_valid,
            physical_valid,
            group_valid,
            tenant_valid,
            speculative_valid,
            codec_valid,
            transfer_valid,
            tier,
            plans,
        )
