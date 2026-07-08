"""Topology-depth lineage — hyperbolic depth supervised by τ/ρ, not SASA."""

from __future__ import annotations

from science.dtie.common.residue_features import GnnInputMode, resolve_gnn_input_mode

# MLflow §3.2 — replace shell SASA probes on topology lineage
TOPOLOGY_MANDATORY_METRICS = frozenset(
    {
        "log_c",
        "effective_experts",
        "effective_experts_min",
        "min_routing_fraction",
        "sigma2_sigma1",
        "disc_thick",
        "r_d_tau",
        "r_d_rho",
        "stage_gate_passed",
    }
)


def topology_depth_lineage(*, master_cold: bool = False) -> bool:
    """True when depth gates/losses must not use SASA (MASTER cold or GNN_INPUT_MODE)."""
    if master_cold:
        return True
    return resolve_gnn_input_mode() == GnnInputMode.TOPOLOGY_THREE_VECTOR
