"""Load GNN + structure inputs for hyperbolic witness embedding (Phase 1 v4)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from science.dtie.common.curvature_values import require_learned_curvature

logger = logging.getLogger(__name__)

DEFAULT_CONDITION_PREFIX = "gdp_"


@dataclass
class WitnessEmbeddingInputs:
    ingestion_data: dict[str, Any]
    gnn_output: dict[str, Any]
    gnn_run_id: str | None
    curvature_c: float


def _parse_vector(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.astype(np.float64)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=np.float64)
    return None


async def load_witness_embedding_inputs(
    db: Any,
    structure_id: str,
    *,
    gnn_run_id: str | None = None,
    condition_prefix: str = DEFAULT_CONDITION_PREFIX,
) -> WitnessEmbeddingInputs | None:
    """Reconstruct Phase 1 v4 inputs from governed GNN embeddings and structure dims."""
    params: dict[str, Any] = {"structure_id": structure_id}
    run_filter = ""
    if gnn_run_id:
        run_filter = "AND e.run_id = :gnn_run_id"
        params["gnn_run_id"] = gnn_run_id

    rows = await db.fetch_all(
        f"""
        SELECT
            e.residue_id,
            e.embedding_double,
            e.embedding,
            e.epistemic_uncertainty,
            e.aleatoric_uncertainty,
            e.run_id,
            es.curvature AS space_curvature,
            a.x AS ca_x,
            a.y AS ca_y,
            a.z AS ca_z,
            c.chain_label,
            r.residue_index
        FROM fact_gnn_node_embedding e
        JOIN embedding_space es ON es.space_id = e.space_id
        JOIN provenance_run p ON p.run_id = e.run_id
        JOIN dim_residue r ON r.residue_id = e.residue_id
        JOIN dim_chain c ON c.chain_id = r.chain_id
        LEFT JOIN dim_atom a ON a.residue_id = r.residue_id AND a.atom_name = 'CA'
        WHERE e.structure_id = :structure_id
          AND es.space_type = 'hyperbolic'
          AND COALESCE(p.parameters->>'audit_only', 'false') != 'true'
          {run_filter}
        ORDER BY p.started_at DESC, c.chain_label, r.residue_index
        """,
        params,
    )
    if not rows:
        return None

    seen: set[str] = set()
    residue_ids: list[str] = []
    x_routed: list[np.ndarray] = []
    epistemic: list[float] = []
    aleatoric: list[float] = []
    no_midpoints: list[list[float]] = []
    resolved_run_id = gnn_run_id
    curvature_c: float | None = None

    for row in rows:
        residue_id = str(row["residue_id"])
        if residue_id in seen:
            continue
        seen.add(residue_id)

        vec = _parse_vector(row.get("embedding_double"))
        if vec is None:
            vec = _parse_vector(row.get("embedding"))
        if vec is None or vec.size == 0:
            logger.warning("Skipping residue %s — no hyperbolic embedding vector", residue_id)
            continue

        if row.get("ca_x") is None:
            logger.warning("Skipping residue %s — missing CA coordinates", residue_id)
            continue

        residue_ids.append(residue_id)
        x_routed.append(vec)
        epistemic.append(float(row.get("epistemic_uncertainty") or 0.0))
        aleatoric.append(float(row.get("aleatoric_uncertainty") or 0.0))
        no_midpoints.append([float(row["ca_x"]), float(row["ca_y"]), float(row["ca_z"])])

        if resolved_run_id is None:
            resolved_run_id = str(row["run_id"])
        if row.get("space_curvature") is not None:
            curvature_c = float(row["space_curvature"])

    if not residue_ids:
        return None

    resolved_curvature = require_learned_curvature(
        curvature_c,
        context="witness embedding inputs",
    )

    prefix = condition_prefix
    gnn_output = {
        f"{prefix}x_routed_hyp": np.stack(x_routed, axis=0),
        f"{prefix}epistemic": np.asarray(epistemic, dtype=np.float64),
        f"{prefix}aleatoric": np.asarray(aleatoric, dtype=np.float64),
        f"{prefix}residue_ids": residue_ids,
        "curvature_c": resolved_curvature,
    }
    ingestion_data = {
        "no_midpoints": np.asarray(no_midpoints, dtype=np.float64),
        "residue_ids": residue_ids,
    }
    return WitnessEmbeddingInputs(
        ingestion_data=ingestion_data,
        gnn_output=gnn_output,
        gnn_run_id=resolved_run_id,
        curvature_c=resolved_curvature,
    )
