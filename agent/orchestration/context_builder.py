# Backend Orchestration Layer — Context Block Builder
"""Formats orchestrator state + viewport state into a compact text block
for injection into LLM context.

The context block gives the LLM awareness of:
- What structure the user is viewing
- Which residue is selected and its metrics
- Current color modes and highlighted residues
- Discovery phase and hypothesis lifecycle
- Reasoning mode and allowed/blocked speech acts
- Pipeline completion flags

The block is bounded to a configurable token budget (~500 tokens max)
to avoid consuming excessive LLM context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent.orchestration.disc_summary import format_disc_summary
from agent.orchestration.reasoning_policy import reasoning_policy_for_lifecycle_state

if TYPE_CHECKING:
    from agent.orchestration.orchestrator import SessionOrchestrator, ViewportState
    from agent.tools.disc_topology import DiscTopologyResult


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Maximum approximate token budget for the context block.
# We use a conservative estimate of ~4 chars per token.
MAX_TOKEN_BUDGET = 500
_CHARS_PER_TOKEN = 4
MAX_CHAR_BUDGET = MAX_TOKEN_BUDGET * _CHARS_PER_TOKEN  # 2000 chars

# Maximum number of highlighted residues to include before truncating
MAX_HIGHLIGHTED_RESIDUES = 20


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_context_block(
    orchestrator: SessionOrchestrator,
    viewport_state: ViewportState | None,
    disc_topology: DiscTopologyResult | None = None,
) -> str:
    """Build a compact text block for LLM context injection.

    Formats the current orchestrator state and viewport state into a structured
    text block suitable for prepending to the user's message before sending to
    the LLM.

    Args:
        orchestrator: The session orchestrator with canonical state.
        viewport_state: The current viewport state from the frontend, or None.
        disc_topology: Optional pre-computed disc topology for the active structure.

    Returns:
        A formatted context string. Returns empty string if viewport_state is None
        and no meaningful orchestrator state exists beyond defaults.
    """
    sections: list[str] = []

    # --- Section 1: Structure identity ---
    structure_id = _get_structure_id(orchestrator, viewport_state)
    if structure_id:
        sections.append(f"[Structure] {structure_id}")

    # --- Section 2: Discovery phase + hypothesis lifecycle ---
    sections.append(
        f"[Phase] {orchestrator.discovery_phase.value} | "
        f"[Hypothesis] {orchestrator.hypothesis_lifecycle.value}"
    )

    # --- Section 3: Selected residue ---
    selected = _get_selected_residue(orchestrator, viewport_state)
    if selected:
        sections.append(f"[Selected Residue] {selected}")

    # --- Section 4: Viewport configuration (color modes) ---
    if viewport_state is not None:
        color_info = _format_color_modes(viewport_state)
        if color_info:
            sections.append(color_info)

    # --- Section 4b: Disc topology summary ---
    if disc_topology is not None:
        disc_summary = format_disc_summary(disc_topology)
        if disc_summary:
            sections.append(disc_summary)

    # --- Section 5: Highlighted residues ---
    if viewport_state is not None:
        highlights = _format_highlights(viewport_state)
        if highlights:
            sections.append(highlights)

    # --- Section 6: Active panel ---
    if viewport_state is not None and viewport_state.active_panel:
        sections.append(f"[Active Panel] {viewport_state.active_panel}")

    # --- Section 7: Pipeline flags ---
    if viewport_state is not None:
        pipeline = _format_pipeline_flags(viewport_state)
        if pipeline:
            sections.append(pipeline)

    # --- Section 8: Reasoning mode + speech acts ---
    reasoning = _format_reasoning_policy(orchestrator)
    sections.append(reasoning)

    # --- Assemble and enforce token budget ---
    block = "\n".join(sections)
    block = _enforce_token_budget(block)

    return block


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_structure_id(
    orchestrator: SessionOrchestrator,
    viewport_state: ViewportState | None,
) -> str | None:
    """Extract structure ID from viewport state or orchestrator scope."""
    if viewport_state is not None and viewport_state.structure_id:
        return viewport_state.structure_id
    if orchestrator.structure_scope.structure_id:
        return orchestrator.structure_scope.structure_id
    return None


def _get_selected_residue(
    orchestrator: SessionOrchestrator,
    viewport_state: ViewportState | None,
) -> str | None:
    """Format the selected residue for display."""
    if orchestrator.selected_residue:
        sel = orchestrator.selected_residue
        return f"{sel.structure_id}:{sel.chain_id}:{sel.residue_number}"
    if viewport_state is not None and viewport_state.selected_residue:
        return viewport_state.selected_residue
    return None


def _format_color_modes(viewport_state: ViewportState) -> str | None:
    """Format Poincaré and 3D viewer color modes."""
    parts: list[str] = []
    if viewport_state.poincare_color_mode:
        parts.append(f"Poincaré={viewport_state.poincare_color_mode}")
    if viewport_state.viewer_3d_color_mode:
        parts.append(f"3D={viewport_state.viewer_3d_color_mode}")
    if parts:
        return f"[Color Modes] {', '.join(parts)}"
    return None


def _format_highlights(viewport_state: ViewportState) -> str | None:
    """Format highlighted residues with truncation for large lists."""
    residues = viewport_state.highlighted_residues
    if not residues:
        return None

    count = len(residues)
    if count <= MAX_HIGHLIGHTED_RESIDUES:
        return f"[Highlighted] {', '.join(residues)}"
    else:
        # Truncate: show first N and a count
        shown = residues[:MAX_HIGHLIGHTED_RESIDUES]
        return (
            f"[Highlighted] {', '.join(shown)} "
            f"(+{count - MAX_HIGHLIGHTED_RESIDUES} more, {count} total)"
        )


def _format_pipeline_flags(viewport_state: ViewportState) -> str | None:
    """Format pipeline completion flags."""
    flags = viewport_state.pipeline_flags
    if not flags:
        return None
    completed = [k for k, v in flags.items() if v]
    pending = [k for k, v in flags.items() if not v]
    parts: list[str] = []
    if completed:
        parts.append(f"done={','.join(completed)}")
    if pending:
        parts.append(f"pending={','.join(pending)}")
    if parts:
        return f"[Pipeline] {'; '.join(parts)}"
    return None


def _format_reasoning_policy(orchestrator: SessionOrchestrator) -> str:
    """Format reasoning mode and speech act constraints."""
    policy = reasoning_policy_for_lifecycle_state(orchestrator.hypothesis_lifecycle)

    parts = [f"[Reasoning] mode={policy.reasoning_mode}"]

    if policy.allowed_speech_acts:
        parts.append(f"[Allowed Acts] {', '.join(policy.allowed_speech_acts)}")
    if policy.blocked_speech_acts:
        parts.append(f"[Blocked Acts] {', '.join(policy.blocked_speech_acts)}")
    if policy.language_guidance:
        parts.append(f"[Guidance] {policy.language_guidance}")

    return "\n".join(parts)


def _enforce_token_budget(block: str) -> str:
    """Truncate block if it exceeds the maximum character budget.

    Truncation strategy:
    1. If within budget, return as-is.
    2. If over budget, truncate from the end of the block, preserving
       complete lines where possible, and append a truncation indicator.
    """
    if len(block) <= MAX_CHAR_BUDGET:
        return block

    # Truncate preserving complete lines
    truncation_marker = "\n[...truncated]"
    budget = MAX_CHAR_BUDGET - len(truncation_marker)

    # Find the last newline within budget
    truncated = block[:budget]
    last_newline = truncated.rfind("\n")
    if last_newline > 0:
        truncated = truncated[:last_newline]

    return truncated + truncation_marker
