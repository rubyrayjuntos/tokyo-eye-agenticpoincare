"""Source-leak detection — Act 02 Persistent Leak."""

from __future__ import annotations

from typing import Any, Protocol

from science.dtie.common.interfaces import PhaseResult
from science.dtie.v5.orchestrator.pipeline import PipelineConfig


class LeakDetectionDB(Protocol):
    async def fetch_all(self, query: str, params: dict[str, Any]) -> list[dict[str, Any]]: ...


async def detect_source_leaks(
    db: LeakDetectionDB,
    config: PipelineConfig,
    gnn_result: Any | None = None,
) -> PhaseResult:
    """Identify source-leak candidates from in-memory GNN nodes or governed DB rows."""
    use_db_path = gnn_result is None or not hasattr(gnn_result, "nodes")
    if use_db_path:
        rows = await db.fetch_all(
            """
            SELECT e.residue_id, e.cone_depth, e.epistemic_uncertainty
            FROM fact_gnn_node_embedding e
            JOIN embedding_space es ON es.space_id = e.space_id
            JOIN provenance_run p ON p.run_id = e.run_id
            WHERE e.structure_id = :structure_id
              AND es.space_type = 'hyperbolic'
              AND COALESCE(p.parameters->>'audit_only', 'false') != 'true'
              AND e.epistemic_uncertainty >= :uncertainty_threshold
              AND e.cone_depth >= :depth_threshold
            """,
            {
                "structure_id": config.structure_id,
                "uncertainty_threshold": config.uncertainty_threshold,
                "depth_threshold": config.depth_threshold,
            },
        )

        leak_residues = [str(row["residue_id"]) for row in rows]
        cone_depths = {
            str(row["residue_id"]): float(row["cone_depth"])
            for row in rows
            if row.get("cone_depth") is not None
        }
        leak_scores = {
            str(row["residue_id"]): float(row["epistemic_uncertainty"]) * float(row["cone_depth"])
            for row in rows
            if row.get("epistemic_uncertainty") is not None and row.get("cone_depth") is not None
        }
        residue_contributions = {
            str(row["residue_id"]): float(row["epistemic_uncertainty"])
            for row in rows
            if row.get("epistemic_uncertainty") is not None
        }

        return PhaseResult(
            phase_name="source_leak_detection",
            structure_id=config.structure_id,
            model_version="discovery-source-leak-v1",
            success=True,
            outputs={
                "source_leak_count": len(leak_residues),
                "source_leak_residues": leak_residues,
                "cone_depths": cone_depths,
                "leak_scores": leak_scores,
            },
            residue_contributions=residue_contributions,
        )

    nodes = gnn_result.nodes if gnn_result else []
    if not nodes:
        return PhaseResult(
            phase_name="source_leak_detection",
            structure_id=config.structure_id,
            model_version="discovery-source-leak-v1",
            success=False,
            outputs={"error": "No GNN nodes available"},
        )

    import numpy as np

    epistemics = np.array([n.epistemic_uncertainty or 0.0 for n in nodes])
    relative_unc = float(np.percentile(epistemics, 90))
    effective_unc_threshold = max(config.uncertainty_threshold, relative_unc)

    leak_nodes = [
        n
        for n in nodes
        if (n.epistemic_uncertainty or 0.0) >= effective_unc_threshold
        and n.cone_depth >= config.depth_threshold
    ]

    leak_residues: list[str] = []
    cone_depths: dict[str, float] = {}
    leak_scores: dict[str, float] = {}
    residue_contributions: dict[str, float] = {}

    for node in leak_nodes:
        rid = f"{config.structure_id}:{node.chain_label}:{node.residue_index}"
        leak_residues.append(rid)
        cone_depths[rid] = float(node.cone_depth)
        unc = float(node.epistemic_uncertainty or 0.0)
        depth = float(node.cone_depth)
        leak_scores[rid] = unc * depth
        residue_contributions[rid] = unc

    return PhaseResult(
        phase_name="source_leak_detection",
        structure_id=config.structure_id,
        model_version="discovery-source-leak-v1",
        success=True,
        outputs={
            "source_leak_count": len(leak_residues),
            "source_leak_residues": leak_residues,
            "cone_depths": cone_depths,
            "leak_scores": leak_scores,
            "threshold": effective_unc_threshold,
        },
        residue_contributions=residue_contributions,
    )
