"""Revision-pressure estimators and execution-granularity policies."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    policy_name: str
    revision_rate_per_scope_s: float
    prefill_quantum_tokens: int
    layer_horizon: int
    layer_safe_point_min_tokens: int
    speculation_depth: int
    predicted_revision_probability_per_quantum: float
    reason: str


class RevisionPressureEstimator:
    """EWMA arrival-rate estimator keyed by scope or workload class."""

    def __init__(self, *, alpha: float = 0.25, prior_rate_per_s: float = 0.0) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("EWMA alpha must lie in (0, 1]")
        if prior_rate_per_s < 0 or not math.isfinite(prior_rate_per_s):
            raise ValueError("prior revision rate must be finite and non-negative")
        self.alpha = alpha
        self.prior_rate_per_s = prior_rate_per_s
        self._last_timestamp_s: dict[str, float] = {}
        self._rates: dict[str, float] = {}

    def rate(self, scope_class: str) -> float:
        return self._rates.get(scope_class, self.prior_rate_per_s)

    def observe(self, scope_class: str, timestamp_s: float) -> float:
        if not scope_class:
            raise ValueError("scope class must be non-empty")
        if timestamp_s < 0 or not math.isfinite(timestamp_s):
            raise ValueError("revision timestamp must be finite and non-negative")
        previous = self._last_timestamp_s.get(scope_class)
        if previous is not None:
            interval = timestamp_s - previous
            if interval <= 0:
                raise ValueError("revision timestamps must increase per scope class")
            instantaneous = 1.0 / interval
            old = self._rates.get(scope_class, self.prior_rate_per_s)
            self._rates[scope_class] = (
                self.alpha * instantaneous + (1.0 - self.alpha) * old
            )
        else:
            self._rates.setdefault(scope_class, self.prior_rate_per_s)
        self._last_timestamp_s[scope_class] = timestamp_s
        return self.rate(scope_class)


class RevisionAwarePolicy:
    """Choose the coarsest quantum within a revision-risk budget.

    ``prefill_tokens_per_second`` is measured by a zero-revision calibration
    run.  For a Poisson revision process with rate ``lambda``, the probability
    that a quantum overlaps a revision is ``1-exp(-lambda * duration)``.
    """

    def __init__(
        self,
        *,
        candidate_prefill_quanta: tuple[int, ...] = (
            256,
            512,
            1024,
            2048,
            4096,
            8192,
            16384,
        ),
        prefill_tokens_per_second: float,
        max_revision_probability_per_quantum: float = 0.10,
        maximum_speculation_depth: int = 5,
    ) -> None:
        if not candidate_prefill_quanta or any(
            quantum <= 0 for quantum in candidate_prefill_quanta
        ):
            raise ValueError("candidate prefill quanta must be positive")
        if prefill_tokens_per_second <= 0 or not math.isfinite(
            prefill_tokens_per_second
        ):
            raise ValueError("prefill calibration must be finite and positive")
        if not 0.0 < max_revision_probability_per_quantum < 1.0:
            raise ValueError("revision probability budget must lie in (0, 1)")
        if maximum_speculation_depth < 0:
            raise ValueError("maximum speculation depth must be non-negative")
        self.quanta = tuple(sorted(set(candidate_prefill_quanta)))
        self.prefill_tokens_per_second = prefill_tokens_per_second
        self.risk_budget = max_revision_probability_per_quantum
        self.maximum_speculation_depth = maximum_speculation_depth

    def _risk(self, rate: float, quantum: int) -> float:
        duration_s = quantum / self.prefill_tokens_per_second
        return 1.0 - math.exp(-rate * duration_s)

    def decide(self, revision_rate_per_scope_s: float) -> PolicyDecision:
        if revision_rate_per_scope_s < 0 or not math.isfinite(
            revision_rate_per_scope_s
        ):
            raise ValueError("revision rate must be finite and non-negative")
        admissible = [
            quantum
            for quantum in self.quanta
            if self._risk(revision_rate_per_scope_s, quantum) <= self.risk_budget
        ]
        quantum = max(admissible, default=min(self.quanta))
        risk = self._risk(revision_rate_per_scope_s, quantum)
        if revision_rate_per_scope_s <= 0.05:
            layer_horizon = 0
        elif revision_rate_per_scope_s <= 0.20:
            layer_horizon = 4
        else:
            layer_horizon = 2
        speculation_depth = max(
            0,
            round(
                self.maximum_speculation_depth
                * math.exp(-revision_rate_per_scope_s)
            ),
        )
        within_budget = risk <= self.risk_budget
        return PolicyDecision(
            policy_name="revision-aware-risk-budget-v1",
            revision_rate_per_scope_s=revision_rate_per_scope_s,
            prefill_quantum_tokens=quantum,
            layer_horizon=layer_horizon,
            layer_safe_point_min_tokens=max(32, min(quantum, 1024)),
            speculation_depth=speculation_depth,
            predicted_revision_probability_per_quantum=risk,
            reason=(
                f"largest calibrated quantum with revision probability <= "
                f"{self.risk_budget:.3f}"
                if within_budget
                else "minimum quantum selected; calibrated revision risk exceeds budget"
            ),
        )
