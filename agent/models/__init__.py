# Migrated from: SRC_AGENT/main.py (viewport models) on 2026-05-27
"""Viewport directive and session state models.

These Pydantic models define the structured communication protocol
between the agent coordinator and the Poincaré visualizer frontend.
"""

from agent.models.viewport import (
    HighlightGroup,
    ViewportCapabilities,
    ViewportDirective,
    ViewportRegistration,
    ViewportState,
)

__all__ = [
    "HighlightGroup",
    "ViewportCapabilities",
    "ViewportDirective",
    "ViewportRegistration",
    "ViewportState",
]
