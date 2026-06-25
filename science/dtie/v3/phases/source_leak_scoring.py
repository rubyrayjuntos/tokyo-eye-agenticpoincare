from __future__ import annotations

from dataclasses import dataclass


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class SourceLeakWeights:
    genotype_persistence: float = 0.35
    state_stability: float = 0.20
    depth_persistence: float = 0.20
    uncertainty_robustness: float = 0.15
    leak_intensity: float = 0.10


@dataclass(frozen=True)
class PropagationWeights:
    conductance: float = 0.45
    source_support: float = 0.25
    path_coverage: float = 0.20
    uncertainty_robustness: float = 0.10


@dataclass(frozen=True)
class PriorityWeights:
    source: float = 0.70
    propagation: float = 0.30


def source_leak_score(
    genotype_persistence: float,
    state_stability: float,
    depth_persistence: float,
    uncertainty_robustness: float,
    leak_intensity: float,
    weights: SourceLeakWeights = SourceLeakWeights(),
) -> float:
    """Primary score for mutation-persistent leak sources."""
    return (
        weights.genotype_persistence * _clamp01(genotype_persistence)
        + weights.state_stability * _clamp01(state_stability)
        + weights.depth_persistence * _clamp01(depth_persistence)
        + weights.uncertainty_robustness * _clamp01(uncertainty_robustness)
        + weights.leak_intensity * _clamp01(leak_intensity)
    )


def propagation_score(
    conductance: float,
    source_support: float,
    path_coverage: float,
    uncertainty_robustness: float,
    weights: PropagationWeights = PropagationWeights(),
) -> float:
    """Secondary score for downstream communication potential."""
    return (
        weights.conductance * _clamp01(conductance)
        + weights.source_support * _clamp01(source_support)
        + weights.path_coverage * _clamp01(path_coverage)
        + weights.uncertainty_robustness * _clamp01(uncertainty_robustness)
    )


def candidate_priority(
    source_score: float,
    propagation: float,
    weights: PriorityWeights = PriorityWeights(),
) -> float:
    """Final candidate ranking with source score dominant by design."""
    return weights.source * _clamp01(source_score) + weights.propagation * _clamp01(
        propagation
    )
