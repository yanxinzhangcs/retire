"""RETIRE Serving: exact scope-epoch supersession across serving engines."""

from .align import AlignmentPlanner, common_prefix_length
from .backend import InferenceBackend
from .backends.vllm import VLLMAsyncBackend, VLLMInternalHooks
from .backends.vllm_distributed import VLLMDistributedParticipant
from .backends.vllm_pinned import PhysicalReuseVerification, PinnedVLLMHooks
from .backends.vllm_tiered import TieredPinnedVLLMHooks
from .control import (
    ScopeRegistry,
    StaleEpochError,
    TenantIsolationError,
    UnknownScopeError,
)
from .distributed import (
    DistributedRetirementBackend,
    RetirementParticipant,
    ScriptedRetirementParticipant,
)
from .invariants import InvariantReport, InvariantViolation, check_invariants
from .kv import KVBlockPool, KVOwner, KVSlotState, WriterTicket
from .paged_scripted import PagedScriptedBackend
from .runtime import ExecutionHandle, ExecutionStats, RetireServing, Supersession
from .policy import PolicyDecision, RevisionAwarePolicy, RevisionPressureEstimator
from .tiering import (
    DRAMTierStore,
    EncodedKVObject,
    IdentityKVCodec,
    KVCodec,
    OffloadTicket,
    TierObjectKey,
    ZlibKVCodec,
)
from .trace import TraceRecord, TraceRecorder
from .workload import (
    RevisionCause,
    RevisionEvent,
    RevisionPattern,
    RevisionTrace,
    ServingPhase,
    TraceGeneratorConfig,
    characterize_revision_trace,
    generate_revision_trace,
)
from .types import (
    AlignmentCertificate,
    CodecCertificate,
    CodecFingerprint,
    EngineFingerprint,
    ExecutionTag,
    GroupRecoveryPlan,
    KVCacheGroupFingerprint,
    KVGroupReuseEvidence,
    NumericalContract,
    ParticipantCoordinate,
    ParticipantFenceReceipt,
    ParticipantReclaimReceipt,
    ProtocolEvent,
    ProtocolStage,
    RecoveryPlan,
    RequestSpec,
    ReuseProbe,
    ReuseTier,
    SpeculativeOutput,
    SupersessionDescriptor,
)

__all__ = [
    "AlignmentCertificate",
    "AlignmentPlanner",
    "CodecCertificate",
    "CodecFingerprint",
    "DRAMTierStore",
    "DistributedRetirementBackend",
    "EncodedKVObject",
    "EngineFingerprint",
    "ExecutionHandle",
    "ExecutionStats",
    "ExecutionTag",
    "GroupRecoveryPlan",
    "IdentityKVCodec",
    "InferenceBackend",
    "InvariantReport",
    "InvariantViolation",
    "KVBlockPool",
    "KVCacheGroupFingerprint",
    "KVCodec",
    "KVGroupReuseEvidence",
    "KVOwner",
    "KVSlotState",
    "NumericalContract",
    "OffloadTicket",
    "PagedScriptedBackend",
    "ParticipantCoordinate",
    "ParticipantFenceReceipt",
    "ParticipantReclaimReceipt",
    "PhysicalReuseVerification",
    "PolicyDecision",
    "PinnedVLLMHooks",
    "ProtocolEvent",
    "ProtocolStage",
    "RecoveryPlan",
    "RequestSpec",
    "RetireServing",
    "RetirementParticipant",
    "ReuseProbe",
    "ReuseTier",
    "RevisionAwarePolicy",
    "RevisionCause",
    "RevisionEvent",
    "RevisionPattern",
    "RevisionPressureEstimator",
    "RevisionTrace",
    "ScopeRegistry",
    "ScriptedRetirementParticipant",
    "SpeculativeOutput",
    "StaleEpochError",
    "Supersession",
    "SupersessionDescriptor",
    "TenantIsolationError",
    "TierObjectKey",
    "TieredPinnedVLLMHooks",
    "TraceGeneratorConfig",
    "TraceRecord",
    "TraceRecorder",
    "UnknownScopeError",
    "VLLMAsyncBackend",
    "VLLMDistributedParticipant",
    "VLLMInternalHooks",
    "WriterTicket",
    "ZlibKVCodec",
    "check_invariants",
    "characterize_revision_trace",
    "common_prefix_length",
    "generate_revision_trace",
    "ServingPhase",
]

__version__ = "0.1.0"
