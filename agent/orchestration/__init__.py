# Backend Orchestration Layer
"""Session-scoped orchestration: discovery phase, hypothesis lifecycle,
tool gating, context enrichment, and cross-viewport residue selection sync.

The orchestrator is the single source of truth for scientific workflow state.
Frontend state machines derive their displayed state from WebSocket-pushed snapshots.
"""

from agent.orchestration.models import (
    DirectiveResult,
    DiscoveryPhase,
    DiscoveryPhaseState,
    HypothesisLifecycleState,
    HypothesisState,
    PlannerPolicy,
    ResidueSelection,
    SelectionResult,
)
from agent.orchestration.tool_policy import TOOL_POLICY, tool_policy_for_phase
from agent.orchestration.reasoning_policy import reasoning_policy_for_lifecycle_state
from agent.orchestration.context_builder import build_context_block
from agent.orchestration.gates import gated_tool_invoke
from agent.orchestration.orchestrator import (
    SessionOrchestrator,
    StructureScopeContext,
    ViewportState,
)

__all__ = [
    "DirectiveResult",
    "DiscoveryPhase",
    "DiscoveryPhaseState",
    "HypothesisLifecycleState",
    "HypothesisState",
    "PlannerPolicy",
    "ResidueSelection",
    "SelectionResult",
    "SessionOrchestrator",
    "StructureScopeContext",
    "ViewportState",
    "TOOL_POLICY",
    "tool_policy_for_phase",
    "reasoning_policy_for_lifecycle_state",
    "build_context_block",
    "gated_tool_invoke",
]
