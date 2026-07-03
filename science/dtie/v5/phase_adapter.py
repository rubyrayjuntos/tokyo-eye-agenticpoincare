"""Phase adapter — converts V5 GNN output to the dict format phases expect.

The v3 phase implementations consume a `gnn_output` dict with keys like:
  - curvature_c: scalar
  - gdp_residue_ids: [N] residue identifiers
  - gdp_cone_depth: [N] hyperbolic depth
  - gdp_aleatoric: [N] aleatoric uncertainty
  - gdp_epistemic: [N] epistemic uncertainty
  - gdp_ca_coords: [N, 3] Cα coordinates
  - gdp_no_midpoints: [N, 3] dehydron midpoint positions
  - gdp_projections: [N, D] embedding projections

This adapter translates the V5 GNNInferenceResult (which uses the
modern dataclass format) into that dict so the phase code runs unchanged.

The phases don't care which GNN version produced the data — they just
need the right keys in the dict. This is the bridge.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from science.dtie.common.curvature_values import require_learned_curvature
from science.dtie.common.interfaces import GNNInferenceResult


def v5_result_to_phase_dict(
    result: GNNInferenceResult,
    ca_coords: np.ndarray | None = None,
    no_midpoints: np.ndarray | None = None,
    prefix: str = "gdp",
) -> dict[str, Any]:
    """Convert V5 GNNInferenceResult to the dict format phases expect.

    Args:
        result: V5 GNN inference output.
        ca_coords: [N, 3] Cα coordinates (from graph builder or dim_atom).
        no_midpoints: [N, 3] dehydron midpoint positions (from fact_dehydron).
        prefix: Key prefix ('gdp' for GDP state, 'gtp' for GTP state).

    Returns:
        Dict compatible with v3 phase function signatures.
    """
    n = len(result.nodes)

    # Extract per-node arrays
    residue_ids = np.array([node.residue_index for node in result.nodes])
    residue_id_strings = [
        f"{result.structure_id}:{node.chain_label}:{node.residue_index}"
        for node in result.nodes
    ]

    cone_depth = np.array([node.cone_depth for node in result.nodes], dtype=np.float64)
    cone_width = np.exp(-cone_depth)

    epistemic = np.array(
        [node.epistemic_uncertainty or 0.0 for node in result.nodes], dtype=np.float64
    )
    aleatoric = np.array(
        [node.aleatoric_uncertainty or 0.0 for node in result.nodes], dtype=np.float64
    )
    total_uncertainty = np.array(
        [node.total_uncertainty or (node.epistemic_uncertainty or 0.0) for node in result.nodes],
        dtype=np.float64,
    )

    projections = np.array([node.projections.tolist() if hasattr(node.projections, 'tolist') else node.projections for node in result.nodes], dtype=np.float32)

    # Hyperbolic embeddings (v5 specific)
    x_hyp = np.array(
        [node.x_hyp.tolist() if node.x_hyp is not None and hasattr(node.x_hyp, 'tolist') else (node.x_hyp if node.x_hyp is not None else [0.0] * result.embedding_dim) for node in result.nodes],
        dtype=np.float32,
    )

    hyp_projections = np.array(
        [node.hyp_projections.tolist() if node.hyp_projections is not None and hasattr(node.hyp_projections, 'tolist') else (node.hyp_projections if node.hyp_projections is not None else [0.0, 0.0]) for node in result.nodes],
        dtype=np.float32,
    )

    expert_weights = np.array(
        [node.expert_weights.tolist() if node.expert_weights is not None and hasattr(node.expert_weights, 'tolist') else (node.expert_weights if node.expert_weights is not None else []) for node in result.nodes],
        dtype=np.float32,
    ) if result.nodes[0].expert_weights is not None else None

    # Build the dict
    gnn_output: dict[str, Any] = {
        # Curvature (scalar)
        "curvature_c": require_learned_curvature(result.curvature, context="v5 phase adapter"),

        # Per-residue arrays with prefix
        f"{prefix}_residue_ids": np.array(residue_id_strings),
        f"{prefix}_cone_depth": cone_depth,
        f"{prefix}_cone_width": cone_width,
        f"{prefix}_aleatoric": aleatoric,
        f"{prefix}_epistemic": epistemic,
        f"{prefix}_total_uncertainty": total_uncertainty,
        f"{prefix}_projections": projections,
        f"{prefix}_x_hyp": x_hyp,
        f"{prefix}_hyp_projections": hyp_projections,

        # Coordinates (if provided)
        f"{prefix}_ca_coords": ca_coords if ca_coords is not None else np.zeros((n, 3)),
        f"{prefix}_no_midpoints": no_midpoints if no_midpoints is not None else np.zeros((n, 3)),

        # Metadata
        "model_version": result.model_version,
        "embedding_dim": result.embedding_dim,
        "space_type": result.space_type,
        "n_residues": n,
    }

    if expert_weights is not None:
        gnn_output[f"{prefix}_expert_weights"] = expert_weights

    return gnn_output


def merge_dual_state(
    gdp_result: GNNInferenceResult,
    gtp_result: GNNInferenceResult,
    gdp_ca_coords: np.ndarray | None = None,
    gtp_ca_coords: np.ndarray | None = None,
    gdp_no_midpoints: np.ndarray | None = None,
    gtp_no_midpoints: np.ndarray | None = None,
) -> dict[str, Any]:
    """Merge GDP and GTP state results into a single dict for dual-state phases.

    Some phases (2, 4, 6d) compare GDP vs GTP conformational states.
    This merges both into one dict with gdp_ and gtp_ prefixed keys.
    """
    gdp_dict = v5_result_to_phase_dict(
        gdp_result, ca_coords=gdp_ca_coords, no_midpoints=gdp_no_midpoints, prefix="gdp"
    )
    gtp_dict = v5_result_to_phase_dict(
        gtp_result, ca_coords=gtp_ca_coords, no_midpoints=gtp_no_midpoints, prefix="gtp"
    )

    # Merge — gdp keys + gtp keys + shared metadata
    merged = {**gdp_dict, **gtp_dict}
    merged["curvature_c"] = require_learned_curvature(
        gdp_result.curvature,
        context="v5 dual-state phase adapter",
    )
    merged["model_version"] = gdp_result.model_version

    return merged
