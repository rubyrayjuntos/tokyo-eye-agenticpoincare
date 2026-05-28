# Migrated from: SRC_AGENT/main.py (viewport models) on 2026-05-27
"""Viewport directive models for agent → visualizer communication.

The agent controls the Poincaré disc visualizer by emitting structured
ViewportDirective messages over WebSocket. These models define the
protocol.

Key concepts:
- ViewportRegistration: Frontend registers its capabilities on connect
- ViewportDirective: Agent sends commands to the visualizer
- ViewportState: Current state of the viewport (what's visible, selected)
- HighlightGroup: A set of residues to highlight with a specific color/style
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DirectiveAction(str, Enum):
    """Actions the agent can request of the visualizer."""

    HIGHLIGHT = "highlight"          # Highlight specific residues
    FOCUS = "focus"                  # Animate camera to focus on residues
    CLEAR_HIGHLIGHTS = "clear"       # Remove all highlights
    SET_METRIC = "set_metric"        # Change the displayed metric
    SET_CURVATURE = "set_curvature"  # Adjust curvature parameter
    TOGGLE_LABELS = "toggle_labels"  # Show/hide residue labels
    FILTER_BY_SITE = "filter_site"   # Show only residues in a site
    SHOW_UNCERTAINTY = "show_uncertainty"  # Color by uncertainty
    COMPARE_RUNS = "compare_runs"    # Show diff between two runs
    ANNOTATE = "annotate"            # Add text annotation to a residue


class HighlightStyle(str, Enum):
    """Visual styles for highlighted residues."""

    GLOW = "glow"
    OUTLINE = "outline"
    PULSE = "pulse"
    COLOR = "color"


class HighlightGroup(BaseModel):
    """A group of residues to highlight together."""

    residue_ids: list[str] = Field(..., description="Canonical residue_ids to highlight")
    color: str = Field(default="#ff6b6b", description="CSS color for the highlight")
    style: HighlightStyle = Field(default=HighlightStyle.GLOW)
    label: str | None = Field(default=None, description="Group label (e.g., 'Source Leak')")
    opacity: float = Field(default=0.8, ge=0.0, le=1.0)


class ViewportDirective(BaseModel):
    """A structured command from the agent to the visualizer.

    The agent emits these to control what the user sees in the
    Poincaré disc viewer.
    """

    action: DirectiveAction
    structure_id: str | None = Field(default=None)
    highlight_groups: list[HighlightGroup] | None = Field(default=None)
    focus_residues: list[str] | None = Field(default=None)
    metric: str | None = Field(default=None, description="e.g., 'cone_depth', 'uncertainty'")
    parameters: dict[str, Any] | None = Field(default=None)
    message: str | None = Field(default=None, description="Human-readable explanation")
    timestamp: datetime = Field(default_factory=lambda: datetime.now())


class ViewportCapabilities(BaseModel):
    """What the connected visualizer can do."""

    supports_3d: bool = True
    supports_2d: bool = True
    supports_animation: bool = True
    max_nodes: int = 5000
    supported_metrics: list[str] = Field(
        default_factory=lambda: ["cone_depth", "uncertainty", "dehydron", "site_type"]
    )


class ViewportRegistration(BaseModel):
    """Sent by the frontend when it connects to register its viewport."""

    viewport_id: str
    capabilities: ViewportCapabilities
    current_structure: str | None = None


class ViewportState(BaseModel):
    """Current state of the visualizer viewport."""

    viewport_id: str | None = None
    structure_id: str | None = None
    visible_residues: list[str] | None = None
    selected_residues: list[str] | None = None
    active_highlights: list[HighlightGroup] | None = None
    current_metric: str = "cone_depth"
    curvature: float = 1.0
    zoom_level: float = 1.0
    camera_target: list[float] | None = None
