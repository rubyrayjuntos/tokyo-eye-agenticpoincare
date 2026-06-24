# Backend Orchestration Layer — Data Models
"""Pydantic models for discovery phase, hypothesis lifecycle, planner policy,
and residue selection. These are the canonical representations that the
backend pushes to all connected clients.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Discovery Phase
# ---------------------------------------------------------------------------


class DiscoveryPhase(str, Enum):
    """Scientific workflow stages, ordered by maturity."""

    RESIDUE = "residue"
    TOPOLOGY = "topology"
    STRUCTURE = "structure"
    POCKET = "pocket"
    SCREENING = "screening"
    REPORT = "report"


PHASE_RANK: dict[DiscoveryPhase, int] = {
    DiscoveryPhase.RESIDUE: 0,
    DiscoveryPhase.TOPOLOGY: 1,
    DiscoveryPhase.STRUCTURE: 2,
    DiscoveryPhase.POCKET: 3,
    DiscoveryPhase.SCREENING: 4,
    DiscoveryPhase.REPORT: 5,
}


class DiscoveryPhaseState(BaseModel):
    """Full state of the discovery phase, including progression flags."""

    phase: DiscoveryPhase = DiscoveryPhase.RESIDUE
    phase_source: Literal["default", "user", "inferred", "system"] = "default"
    has_residue_selection: bool = False
    has_topology_selection: bool = False
    has_structure_mapping: bool = False
    has_pocket_extraction: bool = False
    has_screening_results: bool = False
    allowed_tools: list[str] = Field(default_factory=list)
    blocked_tools: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Hypothesis Lifecycle
# ---------------------------------------------------------------------------


class HypothesisLifecycleState(str, Enum):
    """Epistemic status of the current scientific claim."""

    EMERGENT = "emergent"
    FRAMED = "framed"
    TESTING = "testing"
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    REVISED = "revised"
    SYNTHESIZED = "synthesized"


class HypothesisState(BaseModel):
    """Full state of the hypothesis lifecycle."""

    lifecycle: HypothesisLifecycleState = HypothesisLifecycleState.EMERGENT
    hypothesis_text: str | None = None
    supporting_evidence_count: int = 0
    unresolved_contradictions: list[str] = Field(default_factory=list)
    confidence_band: Literal["low", "medium", "high"] = "low"
    reasoning_mode: str = "explore"
    allowed_speech_acts: list[str] = Field(default_factory=list)
    blocked_speech_acts: list[str] = Field(default_factory=list)
    can_escalate_tools: bool = False


# ---------------------------------------------------------------------------
# Planner Policy (derived output)
# ---------------------------------------------------------------------------


class PlannerPolicy(BaseModel):
    """Derived policy combining phase, lifecycle, structure scope, and viewport."""

    discovery_phase: DiscoveryPhase
    hypothesis_state: HypothesisLifecycleState
    reasoning_mode: str
    allowed_tools: list[str] = Field(default_factory=list)
    blocked_tools: list[str] = Field(default_factory=list)
    preferred_tools: list[str] = Field(default_factory=list)
    speech_style: str = "exploratory"
    allowed_speech_acts: list[str] = Field(default_factory=list)
    blocked_speech_acts: list[str] = Field(default_factory=list)
    should_ask_clarifying_question: bool = False
    target_residues: list[str] = Field(default_factory=list)
    report_ready: bool = False
    rationale: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Residue Selection
# ---------------------------------------------------------------------------


class ResidueSelection(BaseModel):
    """Canonical residue identity for cross-viewport sync."""

    model_config = {"frozen": True}

    structure_id: str
    chain_id: str
    residue_number: int
    source: Literal["poincare", "3d", "table", "agent", "api"] = "api"


class SelectionResult(BaseModel):
    """Result of a selection validation attempt."""

    valid: bool
    selection: ResidueSelection | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Viewport Directive Result
# ---------------------------------------------------------------------------


class DirectiveResult(BaseModel):
    """Result of a directive emission attempt."""

    valid: bool
    directive: dict | None = None
    error: str | None = None
