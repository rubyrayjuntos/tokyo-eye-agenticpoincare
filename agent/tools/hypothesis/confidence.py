"""Confidence calculation and hypothesis lifecycle logic.

Pure functions for:
- Calculating confidence from evidence (weighted ratio)
- Applying time-based decay toward 0.5
- Determining status transitions based on confidence thresholds
"""

from __future__ import annotations

from agent.tools.hypothesis.models import Evidence, HypothesisStatus


def calculate_confidence(evidence: list[Evidence]) -> float:
    """Calculate confidence from evidence list.

    Returns supporting_strength / (supporting + contradicting).
    Returns 0.5 when no evidence or zero total strength.
    """
    if not evidence:
        return 0.5

    supporting = sum(e.strength for e in evidence if e.supports)
    contradicting = sum(e.strength for e in evidence if not e.supports)
    total = supporting + contradicting

    if total == 0.0:
        return 0.5

    return supporting / total


def apply_decay(confidence: float, days_stale: int) -> float:
    """Decay confidence toward 0.5 by 10% per day.

    Formula: confidence + (0.5 - confidence) * (1 - 0.9^days_stale)
    For days_stale <= 0, returns confidence unchanged.
    """
    if days_stale <= 0:
        return confidence
    decay_factor = 1.0 - (0.9 ** days_stale)
    return confidence + (0.5 - confidence) * decay_factor


def determine_status(
    confidence: float, evidence_count: int, current_status: HypothesisStatus
) -> HypothesisStatus:
    """Determine hypothesis status from confidence and evidence count.

    Transitions:
    - confidence > 0.7 and evidence >= 2 → supported
    - confidence < 0.3 and evidence >= 2 → contradicted
    - otherwise keep current status
    """
    if evidence_count < 2:
        return current_status
    if confidence > 0.7:
        return HypothesisStatus.SUPPORTED
    if confidence < 0.3:
        return HypothesisStatus.CONTRADICTED
    return current_status
