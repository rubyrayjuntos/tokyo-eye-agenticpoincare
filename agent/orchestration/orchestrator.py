# Backend Orchestration Layer — SessionOrchestrator
"""Central coordinator that maintains canonical state for a session and derives
planner policy. One instance per active session.

The orchestrator is the single source of truth. Frontend state machines derive
their displayed state from WebSocket-pushed snapshots.
"""

from __future__ import annotations

from typing import Any

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
from agent.orchestration.reasoning_policy import reasoning_policy_for_lifecycle_state
from agent.orchestration.tool_policy import TOOL_POLICY, tool_policy_for_phase


# ---------------------------------------------------------------------------
# Discovery phase transition events
# ---------------------------------------------------------------------------

VALID_DISCOVERY_EVENTS: dict[str, dict[str, DiscoveryPhase | None]] = {
    "ADVANCE": {},  # handled dynamically based on current phase
    "USER_SET_PHASE": {},  # requires "phase" key in event payload
    "TOOL_COMPLETION": {},  # may trigger auto-advance
}

# Ordered phases for advancement
_PHASE_ORDER: list[DiscoveryPhase] = [
    DiscoveryPhase.RESIDUE,
    DiscoveryPhase.TOPOLOGY,
    DiscoveryPhase.STRUCTURE,
    DiscoveryPhase.POCKET,
    DiscoveryPhase.SCREENING,
    DiscoveryPhase.REPORT,
]


# ---------------------------------------------------------------------------
# Hypothesis lifecycle transition table
# ---------------------------------------------------------------------------

_HYPOTHESIS_TRANSITIONS: dict[
    HypothesisLifecycleState, dict[str, HypothesisLifecycleState]
] = {
    HypothesisLifecycleState.EMERGENT: {
        "FRAME": HypothesisLifecycleState.FRAMED,
    },
    HypothesisLifecycleState.FRAMED: {
        "BEGIN_TESTING": HypothesisLifecycleState.TESTING,
        "REVISE": HypothesisLifecycleState.REVISED,
    },
    HypothesisLifecycleState.TESTING: {
        "SUPPORT": HypothesisLifecycleState.SUPPORTED,
        "CONTRADICT": HypothesisLifecycleState.CONTRADICTED,
    },
    HypothesisLifecycleState.SUPPORTED: {
        "SYNTHESIZE": HypothesisLifecycleState.SYNTHESIZED,
        "CONTRADICT": HypothesisLifecycleState.CONTRADICTED,
    },
    HypothesisLifecycleState.CONTRADICTED: {
        "REVISE": HypothesisLifecycleState.REVISED,
        "RESET": HypothesisLifecycleState.EMERGENT,
    },
    HypothesisLifecycleState.REVISED: {
        "FRAME": HypothesisLifecycleState.FRAMED,
        "BEGIN_TESTING": HypothesisLifecycleState.TESTING,
    },
    HypothesisLifecycleState.SYNTHESIZED: {
        "RESET": HypothesisLifecycleState.EMERGENT,
    },
}



# ---------------------------------------------------------------------------
# Structure scope context (for residue validation)
# ---------------------------------------------------------------------------


class StructureScopeContext:
    """Tracks which structure is loaded and its known residues for validation."""

    def __init__(
        self,
        structure_id: str | None = None,
        known_residues: set[tuple[str, str, int]] | None = None,
    ):
        self.structure_id = structure_id
        # Set of (structure_id, chain_id, residue_number) tuples
        self.known_residues: set[tuple[str, str, int]] = known_residues or set()

    def has_residue(self, structure_id: str, chain_id: str, residue_number: int) -> bool:
        """Check if a residue exists in the current structure context."""
        if not self.known_residues:
            # If no residues loaded, allow all (permissive mode)
            return True
        return (structure_id, chain_id, residue_number) in self.known_residues

    def to_dict(self) -> dict[str, Any]:
        return {
            "structure_id": self.structure_id,
            "residue_count": len(self.known_residues),
        }


def default_scope() -> StructureScopeContext:
    """Create a default empty structure scope."""
    return StructureScopeContext()


# ---------------------------------------------------------------------------
# Viewport state (cached from frontend)
# ---------------------------------------------------------------------------


class ViewportState:
    """Cached viewport state received from the frontend."""

    def __init__(
        self,
        *,
        structure_id: str | None = None,
        selected_residue: str | None = None,
        poincare_color_mode: str | None = None,
        viewer_3d_color_mode: str | None = None,
        highlighted_residues: list[str] | None = None,
        active_panel: str | None = None,
        pipeline_flags: dict[str, bool] | None = None,
        risk_threshold: float | None = None,
        brush_selection: list[str] | None = None,
    ):
        self.structure_id = structure_id
        self.selected_residue = selected_residue
        self.poincare_color_mode = poincare_color_mode
        self.viewer_3d_color_mode = viewer_3d_color_mode
        self.highlighted_residues = highlighted_residues or []
        self.active_panel = active_panel
        self.pipeline_flags = pipeline_flags or {}
        self.risk_threshold = risk_threshold
        self.brush_selection = brush_selection or []

    def to_dict(self) -> dict[str, Any]:
        return {
            "structure_id": self.structure_id,
            "selected_residue": self.selected_residue,
            "poincare_color_mode": self.poincare_color_mode,
            "viewer_3d_color_mode": self.viewer_3d_color_mode,
            "highlighted_residues": self.highlighted_residues,
            "active_panel": self.active_panel,
            "pipeline_flags": self.pipeline_flags,
            "risk_threshold": self.risk_threshold,
            "brush_selection": self.brush_selection,
        }



# ---------------------------------------------------------------------------
# SessionOrchestrator
# ---------------------------------------------------------------------------


class SessionOrchestrator:
    """Maintains canonical state for a session and derives planner policy.

    Responsibilities:
    - Initialize with discovery_phase=residue, hypothesis_lifecycle=emergent
    - Derive PlannerPolicy from current state
    - Enforce tool gating via can_invoke_tool()
    - Apply phase/lifecycle transitions with guards
    - Validate and propagate residue selections
    - Cache viewport state from frontend
    - Produce serializable state snapshots for WebSocket push
    """

    def __init__(self, session_id: str):
        self.session_id = session_id

        # Callback for pushing state changes (set by viewport integration)
        self._on_state_change: Any | None = None

        # Core state — Requirements 1.1
        self.discovery_phase_state = DiscoveryPhaseState(
            phase=DiscoveryPhase.RESIDUE,
            phase_source="default",
        )
        self.hypothesis_state = HypothesisState(
            lifecycle=HypothesisLifecycleState.EMERGENT,
            reasoning_mode="explore",
            allowed_speech_acts=list(
                reasoning_policy_for_lifecycle_state(
                    HypothesisLifecycleState.EMERGENT
                ).allowed_speech_acts
            ),
            blocked_speech_acts=list(
                reasoning_policy_for_lifecycle_state(
                    HypothesisLifecycleState.EMERGENT
                ).blocked_speech_acts
            ),
        )

        # Structure scope and viewport
        self.structure_scope = default_scope()
        self.viewport_state: ViewportState | None = None
        self.selected_residue: ResidueSelection | None = None

        # Apply initial tool policy
        policy = tool_policy_for_phase(self.discovery_phase)
        self.discovery_phase_state.allowed_tools = policy["allowed"]
        self.discovery_phase_state.blocked_tools = policy["blocked"]

    def set_state_change_callback(self, callback: Any) -> None:
        """Set a callback invoked after any state transition.

        The callback receives (session_id, event_type, event_data) where:
        - event_type is "phase_transition", "hypothesis_transition", or "state_snapshot"
        - event_data is a dict with the relevant state info
        """
        self._on_state_change = callback

    def _notify_state_change(self, event_type: str, event_data: dict[str, Any]) -> None:
        """Notify listeners of a state change (non-async, queues for push)."""
        if self._on_state_change is not None:
            self._on_state_change(self.session_id, event_type, event_data)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def discovery_phase(self) -> DiscoveryPhase:
        return self.discovery_phase_state.phase

    @property
    def hypothesis_lifecycle(self) -> HypothesisLifecycleState:
        return self.hypothesis_state.lifecycle

    # ------------------------------------------------------------------
    # Policy derivation — Requirements 4.1, 4.5
    # ------------------------------------------------------------------

    def derive_policy(self) -> PlannerPolicy:
        """Compute the current planner policy from all state."""
        tool_pol = tool_policy_for_phase(self.discovery_phase)
        reasoning_pol = reasoning_policy_for_lifecycle_state(self.hypothesis_lifecycle)

        # Preferred tools: currently allowed tools that relate to the phase
        preferred = tool_pol["allowed"][:3]  # Top 3 as preferred

        # Target residues from selection
        target_residues: list[str] = []
        if self.selected_residue:
            target_residues = [
                f"{self.selected_residue.chain_id}:{self.selected_residue.residue_number}"
            ]

        return PlannerPolicy(
            discovery_phase=self.discovery_phase,
            hypothesis_state=self.hypothesis_lifecycle,
            reasoning_mode=reasoning_pol.reasoning_mode,
            allowed_tools=tool_pol["allowed"],
            blocked_tools=tool_pol["blocked"],
            preferred_tools=preferred,
            speech_style=reasoning_pol.speech_style,
            allowed_speech_acts=list(reasoning_pol.allowed_speech_acts),
            blocked_speech_acts=list(reasoning_pol.blocked_speech_acts),
            should_ask_clarifying_question=(
                self.hypothesis_lifecycle == HypothesisLifecycleState.EMERGENT
            ),
            target_residues=target_residues,
            report_ready=(self.discovery_phase == DiscoveryPhase.REPORT),
            rationale=self._build_rationale(),
        )

    def _build_rationale(self) -> list[str]:
        """Build human-readable rationale for current policy."""
        rationale = [
            f"Discovery phase: {self.discovery_phase.value}",
            f"Hypothesis lifecycle: {self.hypothesis_lifecycle.value}",
        ]
        if self.selected_residue:
            rationale.append(
                f"Selected residue: {self.selected_residue.chain_id}"
                f":{self.selected_residue.residue_number}"
            )
        if self.structure_scope.structure_id:
            rationale.append(f"Structure: {self.structure_scope.structure_id}")
        return rationale

    # ------------------------------------------------------------------
    # Tool gating — Requirements 4.1, 4.2
    # ------------------------------------------------------------------

    def can_invoke_tool(self, tool_name: str) -> tuple[bool, str | None]:
        """Check if a tool is allowed in the current state.

        Returns (allowed, reason). If blocked, reason explains why and
        what is needed to unlock.
        """
        policy = tool_policy_for_phase(self.discovery_phase)
        blocked = policy["blocked"]
        allowed = policy["allowed"]

        if tool_name in blocked:
            # Find which phase unlocks this tool
            unlock_phase = self._find_unlock_phase(tool_name)
            hint = ""
            if unlock_phase:
                hint = f" Advance to '{unlock_phase.value}' phase to unlock this tool."
            return (
                False,
                f"Tool '{tool_name}' is blocked in the '{self.discovery_phase.value}' "
                f"phase.{hint}",
            )

        # If tool is in allowed list, explicitly permit
        if tool_name in allowed:
            return (True, None)

        # Tool not in either list — allow by default (not explicitly gated)
        return (True, None)

    def _find_unlock_phase(self, tool_name: str) -> DiscoveryPhase | None:
        """Find the earliest phase where tool_name appears in allowed list."""
        for phase in _PHASE_ORDER:
            if tool_name in TOOL_POLICY[phase]["allowed"]:
                return phase
        return None

    # ------------------------------------------------------------------
    # Discovery phase transitions — Requirements 1.2
    # ------------------------------------------------------------------

    def transition_discovery(self, event: dict[str, Any]) -> bool:
        """Apply a discovery phase transition event.

        Supported event types:
        - {"type": "ADVANCE"}: Move to next phase in sequence
        - {"type": "USER_SET_PHASE", "phase": "<phase_value>"}: Jump to specific phase
        - {"type": "TOOL_COMPLETION", "tool": "<name>"}: May trigger auto-advance

        Returns True if transition occurred, False if guarded/invalid.
        After a successful transition, pushes state to all connected clients.
        """
        event_type = event.get("type", "")
        old_phase = self.discovery_phase

        if event_type == "ADVANCE":
            result = self._advance_phase(source="inferred")

        elif event_type == "USER_SET_PHASE":
            phase_value = event.get("phase")
            if phase_value is None:
                return False
            try:
                if isinstance(phase_value, DiscoveryPhase):
                    target = phase_value
                else:
                    target = DiscoveryPhase(phase_value)
            except ValueError:
                return False
            result = self._set_phase(target, source="user")

        elif event_type == "TOOL_COMPLETION":
            tool = event.get("tool", "")
            result = self._handle_tool_completion_transition(tool)

        else:
            return False

        # Notify listeners if phase actually changed
        if result and self.discovery_phase != old_phase:
            self._notify_state_change("phase_transition", {
                "phase": self.discovery_phase.value,
                "source": self.discovery_phase_state.phase_source,
                "snapshot": self.get_state_snapshot(),
            })

        return result

    def _advance_phase(self, source: str = "inferred") -> bool:
        """Advance to the next phase in sequence."""
        current_idx = _PHASE_ORDER.index(self.discovery_phase)
        if current_idx >= len(_PHASE_ORDER) - 1:
            return False  # Already at final phase
        next_phase = _PHASE_ORDER[current_idx + 1]
        return self._set_phase(next_phase, source=source)

    def _set_phase(self, target: DiscoveryPhase, source: str = "user") -> bool:
        """Set phase directly, updating tool policy."""
        self.discovery_phase_state.phase = target
        self.discovery_phase_state.phase_source = source  # type: ignore[assignment]

        # Update tool policy
        policy = tool_policy_for_phase(target)
        self.discovery_phase_state.allowed_tools = policy["allowed"]
        self.discovery_phase_state.blocked_tools = policy["blocked"]
        return True

    def _handle_tool_completion_transition(self, tool_name: str) -> bool:
        """Check if tool completion should trigger phase advancement."""
        # Auto-advance rules based on tool completions
        phase = self.discovery_phase

        if phase == DiscoveryPhase.RESIDUE and tool_name in (
            "get_residue_state",
            "get_high_uncertainty_residues",
        ):
            self.discovery_phase_state.has_residue_selection = True
            # Don't auto-advance, just mark progress

        elif phase == DiscoveryPhase.TOPOLOGY and tool_name in (
            "get_graph_metrics",
            "find_graph_bridges",
        ):
            self.discovery_phase_state.has_topology_selection = True

        elif phase == DiscoveryPhase.STRUCTURE and tool_name in (
            "get_allosteric_sites",
            "compare_wt_mutant",
        ):
            self.discovery_phase_state.has_structure_mapping = True

        elif phase == DiscoveryPhase.POCKET and tool_name == "get_allosteric_sites":
            self.discovery_phase_state.has_pocket_extraction = True

        elif phase == DiscoveryPhase.SCREENING and tool_name in (
            "screen_fragments",
            "run_docking_surrogate",
        ):
            self.discovery_phase_state.has_screening_results = True

        return False  # Tool completions mark progress but don't auto-advance

    # ------------------------------------------------------------------
    # Hypothesis lifecycle transitions — Requirements 1.2
    # ------------------------------------------------------------------

    def transition_hypothesis(self, event: dict[str, Any]) -> bool:
        """Apply a hypothesis lifecycle transition event.

        Event format: {"type": "<EVENT_NAME>", ...optional fields}

        Returns True if transition occurred, False if guarded/invalid.
        After a successful transition, pushes state to all connected clients.
        """
        event_type = event.get("type", "")
        current = self.hypothesis_lifecycle

        transitions = _HYPOTHESIS_TRANSITIONS.get(current, {})
        target = transitions.get(event_type)

        if target is None:
            return False  # Invalid transition from current state

        # Apply transition
        self.hypothesis_state.lifecycle = target

        # Update reasoning policy
        reasoning_pol = reasoning_policy_for_lifecycle_state(target)
        self.hypothesis_state.reasoning_mode = reasoning_pol.reasoning_mode
        self.hypothesis_state.allowed_speech_acts = list(reasoning_pol.allowed_speech_acts)
        self.hypothesis_state.blocked_speech_acts = list(reasoning_pol.blocked_speech_acts)

        # Update confidence band based on state
        if target == HypothesisLifecycleState.SUPPORTED:
            self.hypothesis_state.confidence_band = "high"
        elif target in (
            HypothesisLifecycleState.TESTING,
            HypothesisLifecycleState.FRAMED,
        ):
            self.hypothesis_state.confidence_band = "medium"
        elif target == HypothesisLifecycleState.CONTRADICTED:
            self.hypothesis_state.confidence_band = "low"
        else:
            self.hypothesis_state.confidence_band = "low"

        # Update hypothesis text if provided
        if "hypothesis_text" in event:
            self.hypothesis_state.hypothesis_text = event["hypothesis_text"]

        # Notify listeners of hypothesis transition
        self._notify_state_change("hypothesis_transition", {
            "state": target.value,
            "snapshot": self.get_state_snapshot(),
        })

        return True

    # ------------------------------------------------------------------
    # Viewport state — Requirements 2.1
    # ------------------------------------------------------------------

    def update_viewport(self, state: ViewportState) -> None:
        """Cache the latest viewport state from the frontend."""
        self.viewport_state = state

    # ------------------------------------------------------------------
    # Residue selection — Requirements 5.4, 5.5
    # ------------------------------------------------------------------

    def select_residue(self, selection: ResidueSelection) -> SelectionResult:
        """Validate and process a residue selection.

        Validates the selection against the current structure scope.
        If valid, clears previous selection and sets the new one.
        If invalid, returns error without modifying state.
        """
        # Validate against structure scope
        if not self.structure_scope.has_residue(
            selection.structure_id,
            selection.chain_id,
            selection.residue_number,
        ):
            return SelectionResult(
                valid=False,
                selection=None,
                error=(
                    f"Residue {selection.chain_id}:{selection.residue_number} "
                    f"not found in structure '{selection.structure_id}'"
                ),
            )

        # Clear previous selection and set new one
        self.selected_residue = selection

        return SelectionResult(
            valid=True,
            selection=selection,
            error=None,
        )

    # ------------------------------------------------------------------
    # Tool completion handler (for gate integration)
    # ------------------------------------------------------------------

    def handle_tool_completion(self, tool_name: str, result: Any = None) -> None:
        """Handle a successful tool completion — may update state."""
        self.transition_discovery({"type": "TOOL_COMPLETION", "tool": tool_name})

    # ------------------------------------------------------------------
    # Viewport directive emission — Requirements 3.1, 3.2, 3.3, 3.5
    # ------------------------------------------------------------------

    def emit_directive(self, directive: dict[str, Any]) -> "DirectiveResult":
        """Validate and prepare a viewport directive for emission.

        Validates the directive against the current structure context:
        - action must be a recognized directive action
        - residue IDs (in highlight/focus) must exist in the loaded structure
          (when structure scope has known residues)
        - metric names (in set_metric) are not validated here (frontend handles)

        Returns DirectiveResult with valid=True and the validated directive,
        or valid=False with an error description.

        The caller is responsible for pushing the directive to clients via
        ViewportConnectionManager after receiving a valid result.
        """
        action = directive.get("action")

        # Validate action is recognized
        valid_actions = {
            "highlight", "focus", "clear", "set_metric",
            "set_curvature", "toggle_labels", "filter_site",
            "show_uncertainty", "compare_runs", "annotate",
        }
        if not action or action not in valid_actions:
            return DirectiveResult(
                valid=False,
                directive=None,
                error=f"Unknown directive action: '{action}'. "
                      f"Valid actions: {sorted(valid_actions)}",
            )

        # For 'clear' action, no further validation needed
        if action == "clear":
            return DirectiveResult(valid=True, directive=directive, error=None)

        # For 'set_metric', just validate that metric field is present
        if action == "set_metric":
            metric = directive.get("metric")
            if not metric:
                return DirectiveResult(
                    valid=False,
                    directive=None,
                    error="'set_metric' directive requires a 'metric' field.",
                )
            return DirectiveResult(valid=True, directive=directive, error=None)

        # For 'highlight' and 'focus', validate residue IDs against structure context
        if action in ("highlight", "focus"):
            residue_ids = self._extract_residue_ids_from_directive(directive)
            if not residue_ids:
                return DirectiveResult(
                    valid=False,
                    directive=None,
                    error=f"'{action}' directive requires residue IDs "
                          f"(via 'residue_ids', 'focus_residues', or 'highlight_groups').",
                )

            # Validate against structure scope if residues are loaded
            if self.structure_scope.known_residues:
                invalid_residues = []
                for rid in residue_ids:
                    # Parse residue ID format: "chain_id:residue_number" or
                    # "structure_id:chain_id:residue_number"
                    parsed = self._parse_residue_id(rid)
                    if parsed and not self.structure_scope.has_residue(*parsed):
                        invalid_residues.append(rid)

                if invalid_residues:
                    return DirectiveResult(
                        valid=False,
                        directive=None,
                        error=f"Residues not found in current structure: "
                              f"{invalid_residues[:5]}",  # Cap error message
                    )

            return DirectiveResult(valid=True, directive=directive, error=None)

        # All other actions pass through without residue validation
        return DirectiveResult(valid=True, directive=directive, error=None)

    def _extract_residue_ids_from_directive(self, directive: dict[str, Any]) -> list[str]:
        """Extract all residue IDs from a directive payload."""
        residue_ids: list[str] = []

        # Direct residue_ids field
        if "residue_ids" in directive:
            residue_ids.extend(directive["residue_ids"])

        # focus_residues field (for focus action)
        if "focus_residues" in directive:
            residue_ids.extend(directive["focus_residues"])

        # highlight_groups (list of groups with residue_ids)
        highlight_groups = directive.get("highlight_groups", [])
        for group in highlight_groups:
            if isinstance(group, dict) and "residue_ids" in group:
                residue_ids.extend(group["residue_ids"])

        return residue_ids

    def _parse_residue_id(self, rid: str) -> tuple[str, str, int] | None:
        """Parse a residue ID string into (structure_id, chain_id, residue_number).

        Supports formats:
        - "chain_id:residue_number" (uses current structure_id)
        - "structure_id:chain_id:residue_number"
        """
        parts = rid.split(":")
        if len(parts) == 2:
            chain_id = parts[0]
            try:
                residue_number = int(parts[1])
            except ValueError:
                return None
            structure_id = self.structure_scope.structure_id or ""
            return (structure_id, chain_id, residue_number)
        elif len(parts) == 3:
            structure_id = parts[0]
            chain_id = parts[1]
            try:
                residue_number = int(parts[2])
            except ValueError:
                return None
            return (structure_id, chain_id, residue_number)
        return None

    # ------------------------------------------------------------------
    # State snapshot — Requirements 1.2, 1.4
    # ------------------------------------------------------------------

    def get_state_snapshot(self) -> dict[str, Any]:
        """Return the full serializable state for WebSocket push."""
        policy = self.derive_policy()
        return {
            "session_id": self.session_id,
            "discovery_phase": self.discovery_phase.value,
            "discovery_phase_state": self.discovery_phase_state.model_dump(),
            "hypothesis_lifecycle": self.hypothesis_lifecycle.value,
            "hypothesis_state": self.hypothesis_state.model_dump(),
            "structure_scope": self.structure_scope.to_dict(),
            "selected_residue": (
                self.selected_residue.model_dump() if self.selected_residue else None
            ),
            "viewport_state": (
                self.viewport_state.to_dict() if self.viewport_state else None
            ),
            "policy": policy.model_dump(),
        }
