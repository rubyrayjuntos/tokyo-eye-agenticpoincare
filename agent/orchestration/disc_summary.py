# Backend Orchestration Layer — Disc Summary Formatter
"""Formats DiscTopologyResult into a compact text block for LLM context injection.

The disc summary provides the agent with structured awareness of the Poincaré disc's
spatial organization: clusters, angular sectors, hub residues, bridge/peripheral residues,
and radial density — all within a 150-token (~600 char) budget.

Feature: poincare-visual-context
Requirements: 1.3, 1.4, 1.5
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.tools.disc_topology import DiscTopologyResult


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Maximum token budget for the disc summary sub-block
MAX_DISC_SUMMARY_TOKENS = 150
_CHARS_PER_TOKEN = 4
MAX_DISC_SUMMARY_CHARS = MAX_DISC_SUMMARY_TOKENS * _CHARS_PER_TOKEN  # 600 chars


# ---------------------------------------------------------------------------
# Radial zone label
# ---------------------------------------------------------------------------

def _radial_zone(radius: float, disc_boundary: float) -> str:
    """Classify a centroid radius into a radial zone label."""
    if disc_boundary <= 0:
        return "core"
    normalized = radius / disc_boundary
    if normalized < 0.3:
        return "core"
    elif normalized < 0.7:
        return "mid"
    else:
        return "periphery"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def format_disc_summary(topology: "DiscTopologyResult") -> str:
    """Format disc topology into a compact context block.

    Output format (bounded to ~150 tokens / 600 chars):
        [Disc Topology] 4 clusters, 187 residues
        C0 (45° NE, core, 52 res, hub: A:G12)
        C1 (180° W, mid, 41 res, hub: A:T35)
        C2 (270° S, periphery, 38 res, hub: A:D57)
        C3 (315° NW, mid, 29 res, hub: A:Y71)
        Bridges: A:E31, A:D33 | Peripheral: A:K147, A:R149

    Args:
        topology: A DiscTopologyResult from the disc topology computation.

    Returns:
        Formatted string bounded to MAX_DISC_SUMMARY_CHARS.
        Returns empty string if topology has no clusters.
    """
    import math

    if topology.cluster_count == 0:
        return ""

    disc_boundary = 1.0 / math.sqrt(topology.curvature_c) if topology.curvature_c > 0 else 1.0

    # Header line
    header = f"[Disc Topology] {topology.cluster_count} clusters, {topology.total_residues} residues"
    lines: list[str] = [header]

    # Cluster lines — sorted by cluster_id for determinism
    clusters_sorted = sorted(topology.clusters, key=lambda c: c.cluster_id)
    for cluster in clusters_sorted:
        zone = _radial_zone(cluster.centroid_radius, disc_boundary)
        angle_int = int(round(cluster.centroid_angle_deg))
        sector = cluster.angular_sector
        line = (
            f"C{cluster.cluster_id} ({angle_int}\u00b0 {sector}, {zone}, "
            f"{cluster.size} res, hub: {cluster.hub_residue_id})"
        )
        lines.append(line)

    # Bridge and peripheral residues
    footer_parts: list[str] = []
    if topology.bridge_residues:
        bridges_str = ", ".join(topology.bridge_residues[:6])
        if len(topology.bridge_residues) > 6:
            bridges_str += f" (+{len(topology.bridge_residues) - 6})"
        footer_parts.append(f"Bridges: {bridges_str}")
    if topology.peripheral_residues:
        periph_str = ", ".join(topology.peripheral_residues[:4])
        if len(topology.peripheral_residues) > 4:
            periph_str += f" (+{len(topology.peripheral_residues) - 4})"
        footer_parts.append(f"Peripheral: {periph_str}")
    if footer_parts:
        lines.append(" | ".join(footer_parts))

    # Assemble and enforce character budget
    result = "\n".join(lines)
    result = _enforce_disc_summary_budget(result)
    return result


# ---------------------------------------------------------------------------
# Budget enforcement
# ---------------------------------------------------------------------------


def _enforce_disc_summary_budget(text: str) -> str:
    """Truncate disc summary if it exceeds the character budget.

    Strategy: remove cluster lines from the bottom (least important clusters)
    until within budget, then truncate footer if still over.
    """
    if len(text) <= MAX_DISC_SUMMARY_CHARS:
        return text

    lines = text.split("\n")

    # Keep header (line 0) always
    # Try progressively removing lines from the end
    while len(lines) > 1 and len("\n".join(lines)) > MAX_DISC_SUMMARY_CHARS:
        lines.pop()

    result = "\n".join(lines)

    # Final hard truncation if single header still too long (unlikely)
    if len(result) > MAX_DISC_SUMMARY_CHARS:
        result = result[: MAX_DISC_SUMMARY_CHARS - 3] + "..."

    return result
