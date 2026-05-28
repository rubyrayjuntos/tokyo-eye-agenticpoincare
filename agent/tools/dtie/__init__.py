# Migrated from: new (written fresh per MIGRATION_MAP.md) on 2026-05-27
"""DTIE tool wrappers for the agent coordinator.

These tools provide the agent with structured access to:
- GNN inference (v3 and v4)
- DTIE phase execution
- Source-leak detection
- Residue-level queries against the governed data layer
- Viewport directive generation

Per MIGRATION_MAP.md: "Must be written fresh using the v3/v4 orchestrators"
"""

from agent.tools.dtie.tools import (
    compare_wt_mutant,
    get_high_uncertainty_residues,
    get_residue_state,
    get_source_leaks,
    run_gnn_inference,
    run_phase,
)

__all__ = [
    "compare_wt_mutant",
    "get_high_uncertainty_residues",
    "get_residue_state",
    "get_source_leaks",
    "run_gnn_inference",
    "run_phase",
]
