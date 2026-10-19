"""Public value types for scope-epoch supersession."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import lcm
from time import monotonic_ns
from typing import Any


class ProtocolStage(str, Enum):
    ADVANCE = "ADVANCE"
    FENCE = "FENCE"
    ALIGN = "ALIGN"
    RESUME = "RESUME"
    RECLAIM = "RECLAIM"


class ReuseTier(str, Enum):
    HBM = "hbm"
    DRAM = "dram"
    CPU = "cpu"
    MIXED = "mixed"
    COLD = "cold"


@dataclass(frozen=True, slots=True)
class CodecFingerprint:
    """Codec identity included in a reusable-state certificate."""

    name: str
    version: str
    lossless: bool = True


@dataclass(frozen=True, slots=True)
class CodecCertificate:
    """Integrity and completion proof for one encoded KV object."""

    fingerprint: CodecFingerprint
    encoded_sha256: str
    decoded_sha256: str
    operation_drained: bool
    integrity_verified: bool

    @property
    def admissible(self) -> bool:
        return (
            self.fingerprint.lossless
            and self.operation_drained
            and self.integrity_verified
            and bool(self.encoded_sha256)
            and bool(self.decoded_sha256)
        )


@dataclass(frozen=True, slots=True)
class KVCacheGroupFingerprint:
    """Layout identity for one independently allocated KV-cache group."""

    group_id: str
    block_size: int
    dtype: str = "unknown"
    layout: str = "paged"

    def __post_init__(self) -> None:
        if not self.group_id:
            raise ValueError("KV-cache group_id must be non-empty")
        if self.block_size <= 0:
            raise ValueError("KV-cache group block_size must be positive")


@dataclass(frozen=True, slots=True)
class ParticipantCoordinate:
    """One retirement participant in the PP x TP x KV-group product space."""

    tensor_parallel_rank: int
    kv_group_id: str
    pipeline_parallel_rank: int = 0

    def __post_init__(self) -> None:
        if self.tensor_parallel_rank < 0:
            raise ValueError("tensor_parallel_rank must be non-negative")
        if self.pipeline_parallel_rank < 0:
            raise ValueError("pipeline_parallel_rank must be non-negative")
        if not self.kv_group_id:
            raise ValueError("kv_group_id must be non-empty")

    @property
    def participant_id(self) -> str:
        return (
            f"pp{self.pipeline_parallel_rank}:"
            f"tp{self.tensor_parallel_rank}:{self.kv_group_id}"
        )


@dataclass(frozen=True, slots=True)
class ExecutionTag:
    """Lossless software form of the paper's cross-layer execution tag."""

    scope_id: str
    epoch: int
    operation_id: int
    object_type: str
    generation: int = 0
    tenant_id: str = "default"


@dataclass(frozen=True, slots=True)
class EngineFingerprint:
    """Fields that must agree before KV reuse is even considered."""

    model: str
    revision: str = "unknown"
    dtype: str = "unknown"
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    block_size: int = 16
    attention_backend: str = "unknown"
    kv_cache_groups: tuple[KVCacheGroupFingerprint, ...] = ()
    speculative_mode: str = "disabled"
    draft_model: str | None = None

    def __post_init__(self) -> None:
        if self.tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be positive")
        if self.pipeline_parallel_size <= 0:
            raise ValueError("pipeline_parallel_size must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        group_ids = [group.group_id for group in self.kv_cache_groups]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("KV-cache group IDs must be unique")

    @property
    def normalized_kv_cache_groups(self) -> tuple[KVCacheGroupFingerprint, ...]:
        if self.kv_cache_groups:
            return self.kv_cache_groups
        return (
            KVCacheGroupFingerprint(
                group_id="group-0",
                block_size=self.block_size,
                dtype=self.dtype,
            ),
        )

    @property
    def alignment_block_size(self) -> int:
        return lcm(
            *(group.block_size for group in self.normalized_kv_cache_groups)
        )

    @property
    def participant_coordinates(self) -> tuple[ParticipantCoordinate, ...]:
        return tuple(
            ParticipantCoordinate(
                tensor_parallel_rank=tp_rank,
                kv_group_id=group.group_id,
                pipeline_parallel_rank=pp_rank,
            )
            for pp_rank in range(self.pipeline_parallel_size)
            for tp_rank in range(self.tensor_parallel_size)
            for group in self.normalized_kv_cache_groups
        )


@dataclass(frozen=True, slots=True)
class NumericalContract:
    """The declared numerical contract for reused state.

    Reuse is conservative: a required attestation must be supplied by the
    serving backend. Matching configuration strings alone are not an
    attestation.
    """

    name: str
    required_attestation: str | None

    @classmethod
    def strict_clean_recomputation(cls) -> NumericalContract:
        return cls(
            name="strict-clean-recomputation-v1",
            required_attestation="strict-clean-recomputation-v1",
        )

    @classmethod
    def structural_only_for_testing(cls) -> NumericalContract:
        return cls(name="structural-only-test", required_attestation=None)


@dataclass(frozen=True, slots=True)
class RequestSpec:
    scope_id: str
    epoch: int
    request_id: str
    operation_id: int
    prompt_token_ids: tuple[int, ...]
    cache_salt: str
    sampling_params: Any = field(repr=False, compare=False)
    generation: int = 0
    tenant_id: str = "default"

    @property
    def tag(self) -> ExecutionTag:
        return ExecutionTag(
            scope_id=self.scope_id,
            epoch=self.epoch,
            operation_id=self.operation_id,
            object_type="request",
            generation=self.generation,
            tenant_id=self.tenant_id,
        )


@dataclass(frozen=True, slots=True)
class OutputEnvelope:
    scope_id: str
    epoch: int
    request_id: str
    operation_id: int
    generation: int
    payload: Any = field(repr=False, compare=False)
    tenant_id: str = "default"

    @property
    def tag(self) -> ExecutionTag:
        return ExecutionTag(
            scope_id=self.scope_id,
            epoch=self.epoch,
            operation_id=self.operation_id,
            object_type="output",
            generation=self.generation,
            tenant_id=self.tenant_id,
        )


@dataclass(frozen=True, slots=True)
class SpeculativeOutput:
    """A draft/verify result presented to the frontend publication gate."""

    payload: Any = field(repr=False, compare=False)
    generation: int
    verified: bool


@dataclass(frozen=True, slots=True)
class SupersessionDescriptor:
    """The paper's ``D=<sid,e,e',X',C,deadline>`` transition descriptor."""

    scope_id: str
    old_epoch: int
    new_epoch: int
    revised_prompt_token_ids: tuple[int, ...]
    contract: NumericalContract
    deadline_ns: int | None = None
    tenant_id: str = "default"

    def __post_init__(self) -> None:
        if not self.scope_id:
            raise ValueError("scope_id must be non-empty")
        if not self.tenant_id:
            raise ValueError("tenant_id must be non-empty")
        if self.old_epoch < 0 or self.new_epoch != self.old_epoch + 1:
            raise ValueError("supersession must advance exactly one epoch")


@dataclass(frozen=True, slots=True)
class AdvanceReceipt:
    scope_id: str
    old_epoch: int
    new_epoch: int
    retired_request_ids: tuple[str, ...]
    retired_contexts: tuple[tuple[str, tuple[int, ...]], ...]
    tenant_id: str = "default"


@dataclass(frozen=True, slots=True)
class FenceReceipt:
    scope_id: str
    retired_epoch: int
    requested_request_ids: tuple[str, ...]
    acknowledged_request_ids: tuple[str, ...]
    launch_fence_installed: bool
    participants: tuple[ParticipantFenceReceipt, ...] = ()

    @property
    def safe(self) -> bool:
        acknowledged = set(self.acknowledged_request_ids)
        return (
            self.launch_fence_installed
            and set(self.requested_request_ids).issubset(acknowledged)
            and all(participant.safe for participant in self.participants)
        )


@dataclass(frozen=True, slots=True)
class ParticipantFenceReceipt:
    coordinate: ParticipantCoordinate
    acknowledged: bool
    writer_drained: bool
    speculation_drained: bool = True
    codec_drained: bool = True
    transfer_drained: bool = True
    safe_point_sequence: int = 0
    detail: str = ""

    @property
    def safe(self) -> bool:
        return (
            self.acknowledged
            and self.writer_drained
            and self.speculation_drained
            and self.codec_drained
            and self.transfer_drained
        )


@dataclass(frozen=True, slots=True)
class KVGroupReuseEvidence:
    """Reuse evidence for one pipeline stage, TP rank, and KV-cache group."""

    coordinate: ParticipantCoordinate
    source_request_id: str | None
    source_tenant_id: str = "default"
    tier: ReuseTier = ReuseTier.COLD
    available_tokens: int = 0
    no_pending_writer: bool = False
    source_fingerprint: EngineFingerprint | None = None
    numerical_attestation: str | None = None
    committed_tokens: int | None = None
    speculation_drained: bool = True
    codec_drained: bool = True
    transfer_drained: bool = True
    codec_certificate: CodecCertificate | None = None

    @property
    def codec_valid(self) -> bool:
        certificate_present = self.codec_certificate is not None
        if self.tier is ReuseTier.DRAM and not certificate_present:
            return False
        return self.codec_drained and (
            self.codec_certificate is None or self.codec_certificate.admissible
        )


@dataclass(frozen=True, slots=True)
class ReuseProbe:
    """Physical and numerical evidence returned by an internal vLLM hook."""

    source_request_id: str | None
    tier: ReuseTier = ReuseTier.COLD
    available_tokens: int = 0
    no_pending_writer: bool = False
    source_fingerprint: EngineFingerprint | None = None
    numerical_attestation: str | None = None
    source_tenant_id: str = "default"
    committed_tokens: int | None = None
    speculation_drained: bool = True
    codec_drained: bool = True
    transfer_drained: bool = True
    codec_certificate: CodecCertificate | None = None
    group_evidence: tuple[KVGroupReuseEvidence, ...] = ()

    @classmethod
    def unavailable(cls, source_request_id: str | None) -> ReuseProbe:
        return cls(source_request_id=source_request_id)


@dataclass(frozen=True, slots=True)
class AlignmentCertificate:
    source_request_id: str | None
    common_prefix_tokens: int
    block_aligned_tokens: int
    structural_valid: bool
    numerical_valid: bool
    physical_valid: bool
    admissible: bool
    reason: str
    group_valid: bool = True
    tenant_valid: bool = True
    speculative_valid: bool = True
    codec_valid: bool = True
    transfer_valid: bool = True


@dataclass(frozen=True, slots=True)
class GroupRecoveryPlan:
    coordinate: ParticipantCoordinate
    tier: ReuseTier
    reuse_tokens: int


@dataclass(frozen=True, slots=True)
class RecoveryPlan:
    tier: ReuseTier
    reuse_tokens: int
    recompute_tokens: int
    certificate: AlignmentCertificate
    group_plans: tuple[GroupRecoveryPlan, ...] = ()


@dataclass(frozen=True, slots=True)
class ParticipantReclaimReceipt:
    coordinate: ParticipantCoordinate
    safe_to_reuse: bool
    reclaimed_blocks: int = 0
    detail: str = ""


@dataclass(frozen=True, slots=True)
class ReclaimReceipt:
    scope_id: str
    retired_epoch: int
    safe_to_reuse: bool
    reclaimed_blocks: int = 0
    detail: str = ""
    participants: tuple[ParticipantReclaimReceipt, ...] = ()


@dataclass(frozen=True, slots=True)
class ProtocolEvent:
    stage: ProtocolStage
    scope_id: str
    epoch: int
    timestamp_ns: int = field(default_factory=monotonic_ns)
    detail: str = ""
    tenant_id: str = "default"
