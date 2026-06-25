# Backend Orchestration Layer — Tool Policy
"""Phase-aware tool gating policy. Maps each discovery phase to the set of
tools that are allowed and blocked at that stage.

This mirrors the frontend `toolPolicyForPhase` logic so that backend
enforcement is consistent with what the UI displays.
"""

from __future__ import annotations

from agent.orchestration.models import DiscoveryPhase


# ---------------------------------------------------------------------------
# TOOL_POLICY: canonical mapping from phase → allowed/blocked tool lists
# ---------------------------------------------------------------------------

TOOL_POLICY: dict[DiscoveryPhase, dict[str, list[str]]] = {
    DiscoveryPhase.RESIDUE: {
        "allowed": [
            "get_residue_state",
            "get_high_uncertainty_residues",
            "search_residues",
            "highlight_residues",
            "set_metric",
        ],
        "blocked": [
            "extract_pockets",
            "screen_fragments",
            "run_docking_surrogate",
            "generate_report",
        ],
    },
    DiscoveryPhase.TOPOLOGY: {
        "allowed": [
            "get_graph_metrics",
            "find_graph_bridges",
            "get_shortest_paths",
            "highlight_residues",
            "search_residues",
        ],
        "blocked": [
            "screen_fragments",
            "run_docking_surrogate",
        ],
    },
    DiscoveryPhase.STRUCTURE: {
        "allowed": [
            "get_residue_state",
            "get_allosteric_sites",
            "compare_wt_mutant",
            "highlight_residues",
        ],
        "blocked": [
            "run_docking_surrogate",
        ],
    },
    DiscoveryPhase.POCKET: {
        "allowed": [
            "get_allosteric_sites",
            "search_residues",
            "highlight_residues",
            "annotate_structure",
        ],
        "blocked": [
            "generate_report",
        ],
    },
    DiscoveryPhase.SCREENING: {
        "allowed": [
            "screen_fragments",
            "run_docking_surrogate",
            "highlight_residues",
            "annotate_structure",
        ],
        "blocked": [],
    },
    DiscoveryPhase.REPORT: {
        "allowed": [
            "generate_report",
            "export_structure_data",
            "get_provenance_lineage",
        ],
        "blocked": [
            "screen_fragments",
            "run_docking_surrogate",
        ],
    },
}


def tool_policy_for_phase(phase: DiscoveryPhase) -> dict[str, list[str]]:
    """Return the tool policy (allowed + blocked lists) for a given phase.

    Returns a dict with keys "allowed" and "blocked", each containing a list
    of tool name strings.

    Raises KeyError if the phase is not recognized (should not happen with
    the enum, but defensive).
    """
    return TOOL_POLICY[phase]
