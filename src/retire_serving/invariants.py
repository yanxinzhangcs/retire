"""Machine-checkable forms of the paper's three epoch rules."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .trace import TraceRecord


@dataclass(frozen=True, slots=True)
class InvariantViolation:
    rule: str
    detail: str
    record: TraceRecord


@dataclass(frozen=True, slots=True)
class InvariantReport:
    records_checked: int
    violations: tuple[InvariantViolation, ...]

    @property
    def passed(self) -> bool:
        return not self.violations


def check_invariants(records: Iterable[TraceRecord]) -> InvariantReport:
    ordered = sorted(records, key=lambda record: record.timestamp_ns)
    current_epoch: dict[tuple[str, str], int] = {}
    block_generation: dict[int, tuple[str, int, int]] = {}
    aligned: set[tuple[str, str, int, int]] = set()
    stages: dict[tuple[str, str, int, int], list[str]] = {}
    violations: list[InvariantViolation] = []

    for record in ordered:
        tag = record.tag
        domain = (tag.tenant_id, tag.scope_id)
        current_epoch.setdefault(domain, tag.epoch)
        if tag.object_type == "control":
            key = (tag.tenant_id, tag.scope_id, tag.epoch, tag.operation_id)
            stages.setdefault(key, []).append(record.action)
            if record.action == "ADVANCE":
                previous = current_epoch[domain]
                if tag.epoch < previous or tag.epoch > previous + 1:
                    violations.append(
                        InvariantViolation(
                            "monotonic_epoch",
                            f"ADVANCE moved from {previous} to {tag.epoch}",
                            record,
                        )
                    )
                current_epoch[domain] = tag.epoch
            elif record.action == "ALIGN" and record.detail.startswith("certified "):
                aligned.add(key)

        if (
            record.action == "output_publication"
            and record.phase == "commit"
            and tag.epoch != current_epoch[domain]
        ):
            violations.append(
                InvariantViolation(
                    "current_epoch_commit",
                    f"published epoch {tag.epoch} while current is {current_epoch[domain]}",
                    record,
                )
            )

        if record.action == "kv_handoff" and record.phase == "commit":
            if tag.epoch != current_epoch[domain]:
                violations.append(
                    InvariantViolation(
                        "certified_reuse",
                        "KV handoff targeted a non-current epoch",
                        record,
                    )
                )
            if not any(
                tenant == tag.tenant_id
                and scope == tag.scope_id
                and epoch == tag.epoch
                for tenant, scope, epoch, _ in aligned
            ):
                violations.append(
                    InvariantViolation(
                        "certified_reuse",
                        "KV handoff had no preceding positive ALIGN certificate",
                        record,
                    )
                )

        if (
            record.action == "late_kv_install"
            and record.phase == "commit"
            and tag.epoch != current_epoch[domain]
        ):
            violations.append(
                InvariantViolation(
                    "retired_state_isolation",
                    "a retired writer installed into reachable KV state",
                    record,
                )
            )

        if record.physical_block is not None and record.phase in {"commit", "complete"}:
            previous = block_generation.get(record.physical_block)
            if previous is not None:
                previous_tenant, previous_epoch, previous_generation = previous
                if tag.generation < previous_generation or (
                    (tag.tenant_id != previous_tenant or tag.epoch != previous_epoch)
                    and tag.generation == previous_generation
                ):
                    violations.append(
                        InvariantViolation(
                            "retired_state_isolation",
                            "physical block ownership changed without advancing generation",
                            record,
                        )
                    )
            block_generation[record.physical_block] = (
                tag.tenant_id,
                tag.epoch,
                tag.generation,
            )

    required = ["ADVANCE", "FENCE", "ALIGN", "RESUME", "RECLAIM"]
    for key, observed in stages.items():
        if "ADVANCE" in observed and observed != required:
            first = next(
                record
                for record in ordered
                if record.tag.object_type == "control"
                and (
                    record.tag.tenant_id,
                    record.tag.scope_id,
                    record.tag.epoch,
                    record.tag.operation_id,
                )
                == key
            )
            violations.append(
                InvariantViolation(
                    "protocol_stage_order",
                    f"observed {observed}, expected {required}",
                    first,
                )
            )
    return InvariantReport(len(ordered), tuple(violations))
