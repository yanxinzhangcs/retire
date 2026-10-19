"""Trace model for revisable-execution serving workloads.

The trace deliberately records systems-visible revision events rather than
prompt text.  A real agent runtime and a synthetic generator can therefore
produce the same schema without exporting user content.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class RevisionPattern(str, Enum):
    POISSON = "poisson"
    PERIODIC = "periodic"
    BURST = "burst"
    RECORDED = "recorded"


class ServingPhase(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    SPECULATIVE = "speculative"
    TOOL_WAIT = "tool_wait"


class RevisionCause(str, Enum):
    USER = "user"
    TOOL = "tool"
    ENVIRONMENT = "environment"
    VALIDATION = "validation"
    PLANNING = "planning"
    RESOURCE = "resource"


@dataclass(frozen=True, slots=True)
class RevisionEvent:
    event_id: str
    scope_id: str
    old_epoch: int
    new_epoch: int
    scheduled_offset_s: float
    cause: RevisionCause
    phase: ServingPhase
    prefix_overlap_fraction: float
    scope_width: int = 1
    burst_id: str | None = None
    source: str = "synthetic"

    def __post_init__(self) -> None:
        if not self.event_id or not self.scope_id:
            raise ValueError("revision event and scope IDs must be non-empty")
        if self.old_epoch < 0 or self.new_epoch != self.old_epoch + 1:
            raise ValueError("a revision event must advance exactly one epoch")
        if not math.isfinite(self.scheduled_offset_s) or self.scheduled_offset_s < 0:
            raise ValueError("scheduled revision offset must be finite and non-negative")
        if not 0.0 <= self.prefix_overlap_fraction <= 1.0:
            raise ValueError("prefix overlap must lie in [0, 1]")
        if self.scope_width <= 0:
            raise ValueError("scope width must be positive")
        if not self.source:
            raise ValueError("revision event source must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "scope_id": self.scope_id,
            "old_epoch": self.old_epoch,
            "new_epoch": self.new_epoch,
            "scheduled_offset_s": self.scheduled_offset_s,
            "cause": self.cause.value,
            "phase": self.phase.value,
            "prefix_overlap_fraction": self.prefix_overlap_fraction,
            "scope_width": self.scope_width,
            "burst_id": self.burst_id,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RevisionEvent:
        return cls(
            event_id=str(payload["event_id"]),
            scope_id=str(payload["scope_id"]),
            old_epoch=int(payload["old_epoch"]),
            new_epoch=int(payload["new_epoch"]),
            scheduled_offset_s=float(payload["scheduled_offset_s"]),
            cause=RevisionCause(str(payload["cause"])),
            phase=ServingPhase(str(payload["phase"])),
            prefix_overlap_fraction=float(payload["prefix_overlap_fraction"]),
            scope_width=int(payload.get("scope_width", 1)),
            burst_id=(
                None if payload.get("burst_id") is None else str(payload["burst_id"])
            ),
            source=str(payload.get("source", "recorded")),
        )


@dataclass(frozen=True, slots=True)
class RevisionTrace:
    duration_s: float
    concurrency: int
    configured_rate_per_scope_s: float
    pattern: RevisionPattern
    seed: int
    events: tuple[RevisionEvent, ...]
    schema_version: int = 1
    trace_id: str = "revision-trace"
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported revision trace schema")
        if not self.trace_id:
            raise ValueError("trace ID must be non-empty")
        if not math.isfinite(self.duration_s) or self.duration_s <= 0:
            raise ValueError("trace duration must be finite and positive")
        if self.concurrency <= 0:
            raise ValueError("trace concurrency must be positive")
        if (
            not math.isfinite(self.configured_rate_per_scope_s)
            or self.configured_rate_per_scope_s < 0
        ):
            raise ValueError("configured revision rate must be finite and non-negative")
        ordered = tuple(
            sorted(self.events, key=lambda event: (event.scheduled_offset_s, event.scope_id))
        )
        if ordered != self.events:
            raise ValueError("revision events must be globally time ordered")
        if any(event.scheduled_offset_s > self.duration_s for event in self.events):
            raise ValueError("revision event lies outside trace duration")
        if len({event.event_id for event in self.events}) != len(self.events):
            raise ValueError("revision event IDs must be unique")
        epochs: dict[str, int] = {}
        for event in self.events:
            expected = epochs.get(event.scope_id, 0)
            if event.old_epoch != expected:
                raise ValueError(
                    f"scope {event.scope_id!r} expected old epoch {expected}, "
                    f"observed {event.old_epoch}"
                )
            epochs[event.scope_id] = event.new_epoch

    @property
    def scope_ids(self) -> tuple[str, ...]:
        observed = {event.scope_id for event in self.events}
        observed.update(f"scope-{index:04d}" for index in range(self.concurrency))
        return tuple(sorted(observed))

    def events_by_scope(self) -> dict[str, tuple[RevisionEvent, ...]]:
        return {
            scope_id: tuple(event for event in self.events if event.scope_id == scope_id)
            for scope_id in self.scope_ids
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "trace_id": self.trace_id,
            "duration_s": self.duration_s,
            "concurrency": self.concurrency,
            "configured_rate_per_scope_s": self.configured_rate_per_scope_s,
            "pattern": self.pattern.value,
            "seed": self.seed,
            "metadata": self.metadata,
            "events": [event.to_dict() for event in self.events],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RevisionTrace:
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            trace_id=str(payload.get("trace_id", "revision-trace")),
            duration_s=float(payload["duration_s"]),
            concurrency=int(payload["concurrency"]),
            configured_rate_per_scope_s=float(
                payload["configured_rate_per_scope_s"]
            ),
            pattern=RevisionPattern(str(payload["pattern"])),
            seed=int(payload["seed"]),
            metadata=dict(payload.get("metadata", {})),
            events=tuple(
                RevisionEvent.from_dict(item) for item in payload.get("events", [])
            ),
        )

    def write_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def read_json(cls, path: Path) -> RevisionTrace:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True, slots=True)
class TraceGeneratorConfig:
    duration_s: float
    concurrency: int
    rate_per_scope_s: float
    pattern: RevisionPattern = RevisionPattern.POISSON
    seed: int = 20260831
    burst_size: int = 4
    burst_gap_s: float = 0.025
    overlaps: tuple[float, ...] = (0.0, 0.5, 0.75, 0.9)
    overlap_weights: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40)
    phases: tuple[ServingPhase, ...] = (
        ServingPhase.PREFILL,
        ServingPhase.DECODE,
        ServingPhase.SPECULATIVE,
    )
    phase_weights: tuple[float, ...] = (0.55, 0.35, 0.10)
    causes: tuple[RevisionCause, ...] = tuple(RevisionCause)
    cause_weights: tuple[float, ...] = (0.30, 0.25, 0.10, 0.15, 0.15, 0.05)
    scope_widths: tuple[int, ...] = (1, 2, 4)
    scope_width_weights: tuple[float, ...] = (0.80, 0.15, 0.05)

    def __post_init__(self) -> None:
        if self.duration_s <= 0 or self.concurrency <= 0 or self.rate_per_scope_s < 0:
            raise ValueError("duration/concurrency must be positive and rate non-negative")
        if self.pattern is RevisionPattern.RECORDED:
            raise ValueError("recorded traces are not generated synthetically")
        if self.burst_size <= 0 or self.burst_gap_s < 0:
            raise ValueError("burst size must be positive and gap non-negative")
        for values, weights, name in (
            (self.overlaps, self.overlap_weights, "overlap"),
            (self.phases, self.phase_weights, "phase"),
            (self.causes, self.cause_weights, "cause"),
            (self.scope_widths, self.scope_width_weights, "scope width"),
        ):
            if not values or len(values) != len(weights) or sum(weights) <= 0:
                raise ValueError(f"invalid {name} choices or weights")
        if any(not 0.0 <= overlap <= 1.0 for overlap in self.overlaps):
            raise ValueError("overlap choices must lie in [0, 1]")
        if any(width <= 0 for width in self.scope_widths):
            raise ValueError("scope-width choices must be positive")


def generate_revision_trace(config: TraceGeneratorConfig) -> RevisionTrace:
    """Generate a deterministic systems-event trace from ``config``."""

    rng = random.Random(config.seed)
    staged: list[tuple[str, float, str | None]] = []
    if config.rate_per_scope_s > 0:
        for scope_index in range(config.concurrency):
            scope_id = f"scope-{scope_index:04d}"
            if config.pattern is RevisionPattern.POISSON:
                offset = rng.expovariate(config.rate_per_scope_s)
                while offset <= config.duration_s:
                    staged.append((scope_id, offset, None))
                    offset += rng.expovariate(config.rate_per_scope_s)
            elif config.pattern is RevisionPattern.PERIODIC:
                interval = 1.0 / config.rate_per_scope_s
                offset = interval * (0.5 + scope_index / config.concurrency)
                while offset <= config.duration_s:
                    staged.append((scope_id, offset, None))
                    offset += interval
            elif config.pattern is RevisionPattern.BURST:
                cluster_rate = config.rate_per_scope_s / config.burst_size
                offset = rng.expovariate(cluster_rate)
                burst_index = 0
                while offset <= config.duration_s:
                    burst_id = f"{scope_id}:burst-{burst_index:04d}"
                    for member in range(config.burst_size):
                        event_offset = offset + member * config.burst_gap_s
                        if event_offset <= config.duration_s:
                            staged.append((scope_id, event_offset, burst_id))
                    burst_index += 1
                    offset += rng.expovariate(cluster_rate)
            else:  # pragma: no cover - guarded by TraceGeneratorConfig
                raise AssertionError(config.pattern)

    staged.sort(key=lambda item: (item[1], item[0]))
    epochs: dict[str, int] = {}
    events = []
    for scope_id, offset, burst_id in staged:
        old_epoch = epochs.get(scope_id, 0)
        new_epoch = old_epoch + 1
        epochs[scope_id] = new_epoch
        events.append(
            RevisionEvent(
                event_id=f"{scope_id}:e{old_epoch}-e{new_epoch}",
                scope_id=scope_id,
                old_epoch=old_epoch,
                new_epoch=new_epoch,
                scheduled_offset_s=offset,
                cause=rng.choices(config.causes, weights=config.cause_weights, k=1)[0],
                phase=rng.choices(config.phases, weights=config.phase_weights, k=1)[0],
                prefix_overlap_fraction=float(
                    rng.choices(
                        config.overlaps, weights=config.overlap_weights, k=1
                    )[0]
                ),
                scope_width=int(
                    rng.choices(
                        config.scope_widths,
                        weights=config.scope_width_weights,
                        k=1,
                    )[0]
                ),
                burst_id=burst_id,
            )
        )
    return RevisionTrace(
        trace_id=(
            f"synthetic-{config.pattern.value}-r{config.rate_per_scope_s:g}-"
            f"c{config.concurrency}-s{config.seed}"
        ),
        duration_s=config.duration_s,
        concurrency=config.concurrency,
        configured_rate_per_scope_s=config.rate_per_scope_s,
        pattern=config.pattern,
        seed=config.seed,
        events=tuple(events),
        metadata={
            "content_free": True,
            "generator": "retire-serving",
            "burst_size": config.burst_size,
            "burst_gap_s": config.burst_gap_s,
        },
    )


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def characterize_revision_trace(trace: RevisionTrace) -> dict[str, Any]:
    """Return workload-level statistics without trusting generator metadata."""

    by_scope = trace.events_by_scope()
    interarrival_s: list[float] = []
    for events in by_scope.values():
        previous = 0.0
        for event in events:
            interarrival_s.append(event.scheduled_offset_s - previous)
            previous = event.scheduled_offset_s
    overlaps = [event.prefix_overlap_fraction for event in trace.events]
    widths = [event.scope_width for event in trace.events]
    phase_counts = {
        phase.value: sum(event.phase is phase for event in trace.events)
        for phase in ServingPhase
    }
    cause_counts = {
        cause.value: sum(event.cause is cause for event in trace.events)
        for cause in RevisionCause
    }
    burst_ids = {event.burst_id for event in trace.events if event.burst_id}
    mean_interarrival = statistics.fmean(interarrival_s) if interarrival_s else None
    interarrival_cv = (
        statistics.stdev(interarrival_s) / mean_interarrival
        if mean_interarrival and len(interarrival_s) > 1
        else None
    )
    return {
        "event_count": len(trace.events),
        "scope_count": trace.concurrency,
        "active_scope_seconds": trace.duration_s * trace.concurrency,
        "configured_rate_per_scope_s": trace.configured_rate_per_scope_s,
        "observed_rate_per_scope_s": (
            len(trace.events) / (trace.duration_s * trace.concurrency)
        ),
        "events_per_scope": {
            scope_id: len(events) for scope_id, events in by_scope.items()
        },
        "interarrival_s": {
            "mean": mean_interarrival,
            "p50": _percentile(interarrival_s, 0.50),
            "p95": _percentile(interarrival_s, 0.95),
            "coefficient_of_variation": interarrival_cv,
        },
        "prefix_overlap_fraction": {
            "mean": statistics.fmean(overlaps) if overlaps else None,
            "p50": _percentile(overlaps, 0.50),
            "p95": _percentile(overlaps, 0.95),
        },
        "scope_width": {
            "mean": statistics.fmean(widths) if widths else None,
            "maximum": max(widths, default=None),
        },
        "phase_counts": phase_counts,
        "cause_counts": cause_counts,
        "burst_count": len(burst_ids),
    }
