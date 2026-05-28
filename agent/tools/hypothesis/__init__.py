"""Hypothesis engine tools for the agent coordinator.

These tools provide the agent with structured scientific reasoning:
- Propose testable hypotheses with falsifiability guardrails
- Test predictions using existing pipeline tools
- Track evidence and confidence lifecycle
- Detect contradictions from new pipeline results
"""

from agent.tools.hypothesis.contradiction import check_contradictions
from agent.tools.hypothesis.models import (
    Evidence,
    Hypothesis,
    HypothesisStatus,
    Prediction,
)

__all__ = [
    "Evidence",
    "Hypothesis",
    "HypothesisStatus",
    "Prediction",
    "check_contradictions",
]
