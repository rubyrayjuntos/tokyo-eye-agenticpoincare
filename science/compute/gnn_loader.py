"""Load GNNInferenceResult from governed DB embeddings for post-peel facade phases."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from science.dtie.common.interfaces import GNNInferenceResult, GNNNodeOutput

logger = logging.getLogger(__name__)

_EMPTY_FEATURES = np.zeros(4, dtype=np.float64)
_EMPTY_PROJECTIONS = np.zeros(64, dtype=np.float64)


def _parse_residue_id(residue_id: str) -> tuple[str, int]:
    parts = residue_id.split(":")
    if len(parts) >= 3:
        try:
            return parts[-2], int(parts[-1])
        except ValueError:
            pass
    return "A", 0


async def load_gnn_inference_result(
    db: Any,
    structure_id: str,
    *,
    gnn_run_id: str | None = None,
) -> GNNInferenceResult | None:
    """Reconstruct a minimal GNNInferenceResult from fact_gnn_node_embedding."""
    params: dict[str, Any] = {"structure_id": structure_id}
    run_filter = ""
    if gnn_run_id:
        run_filter = "AND e.run_id = :gnn_run_id"
        params["gnn_run_id"] = gnn_run_id

    rows = await db.fetch_all(
        f"""
        SELECT e.residue_id, e.epistemic_uncertainty, e.cone_depth, e.cone_width,
               e.run_id, e.model_version, p.started_at
        FROM fact_gnn_node_embedding e
        JOIN provenance_run p ON p.run_id = e.run_id
        JOIN embedding_space es ON es.space_id = e.space_id
        WHERE e.structure_id = :structure_id
          AND es.space_type = 'hyperbolic'
          AND COALESCE(p.parameters->>'audit_only', 'false') != 'true'
          {run_filter}
        ORDER BY p.started_at DESC, e.computed_at DESC
        """,
        params,
    )

    if not rows:
        logger.warning("No hyperbolic GNN embeddings for structure %s", structure_id)
        return None

    seen: set[str] = set()
    nodes: list[GNNNodeOutput] = []
    resolved_run_id = gnn_run_id
    model_version = "unknown"

    for row in rows:
        residue_id = str(row["residue_id"])
        if residue_id in seen:
            continue
        seen.add(residue_id)
        chain_label, residue_index = _parse_residue_id(residue_id)
        if resolved_run_id is None:
            resolved_run_id = str(row["run_id"])
        model_version = str(row.get("model_version") or model_version)

        nodes.append(
            GNNNodeOutput(
                residue_index=residue_index,
                chain_label=chain_label,
                input_features=_EMPTY_FEATURES.copy(),
                projections=_EMPTY_PROJECTIONS.copy(),
                cone_depth=float(row.get("cone_depth") or 0.0),
                cone_width=float(row.get("cone_width") or 0.0),
                epistemic_uncertainty=float(row.get("epistemic_uncertainty") or 0.0),
            )
        )

    return GNNInferenceResult(
        structure_id=structure_id,
        model_version=model_version,
        checkpoint_path=None,
        nodes=nodes,
        space_type="hyperbolic",
        metadata={"run_id": resolved_run_id, "loaded_from_db": True},
    )
