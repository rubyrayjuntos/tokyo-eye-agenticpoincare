from __future__ import annotations

import json
import math
import uuid
from typing import Any, Dict, Iterable, List

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from gosp.db.connection import get_engine
from gosp.normalizer.jobs import write_job_status


def _jsonable_audit_trail(audit_trail: Dict[str, Any] | None) -> str | None:
    if not audit_trail:
        return None

    normalized: Dict[str, Any] = {}
    for key, value in audit_trail.items():
        if hasattr(value, "item"):
            normalized[key] = value.item()
        else:
            normalized[key] = value
    return json.dumps(normalized)


def _routing_entropy(weights: Iterable[float]) -> float:
    entropy = 0.0
    for weight in weights:
        weight = max(float(weight), 1e-8)
        entropy -= weight * math.log(weight)
    return entropy


def _vector_literal(values: List[float]) -> str:
    return "[" + ",".join(f"{float(value):.10f}" for value in values) + "]"


async def _ensure_embedding_schema(session: AsyncSession) -> None:
    await session.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
    await session.execute(
        sa.text(
            """
            CREATE TABLE IF NOT EXISTS fact_gnn_node_embedding (
                embedding_id         TEXT PRIMARY KEY,
                node_id              TEXT NOT NULL UNIQUE REFERENCES fact_gnn_node_output(node_id) ON DELETE CASCADE,
                gnn_run_id           TEXT NOT NULL REFERENCES fact_gnn_inference(gnn_run_id) ON DELETE CASCADE,
                structure_id         TEXT NOT NULL REFERENCES dim_structure(structure_id),
                residue_id           TEXT NOT NULL REFERENCES dim_residue(residue_id),
                model_version        TEXT NOT NULL,
                projection_dim       INTEGER NOT NULL,
                projection_embedding VECTOR NOT NULL,
                cone_depth           DOUBLE PRECISION,
                cone_width           DOUBLE PRECISION,
                routing_entropy      DOUBLE PRECISION,
                expert_winner        INTEGER,
                source_type          TEXT NOT NULL DEFAULT 'probabilistic',
                computed_at          TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )
    )
    await session.execute(
        sa.text("CREATE INDEX IF NOT EXISTS idx_gnn_embedding_run ON fact_gnn_node_embedding(gnn_run_id)")
    )
    await session.execute(
        sa.text("CREATE INDEX IF NOT EXISTS idx_gnn_embedding_structure ON fact_gnn_node_embedding(structure_id)")
    )
    await session.execute(
        sa.text("CREATE INDEX IF NOT EXISTS idx_gnn_embedding_residue ON fact_gnn_node_embedding(residue_id)")
    )


async def write_results_to_db(
    structure_id: str,
    payload: Dict[str, Any],
    model_output: Dict[str, Any],
    source_type: str = "probabilistic",
) -> str:
    residue_ids = payload.get("residue_ids") or []
    node_features = payload.get("node_features") or []
    per_node = model_output.get("per_node") or []

    if not structure_id:
        raise ValueError("structure_id is required for DB writeback")
    if not residue_ids:
        raise ValueError("payload.residue_ids is required for DB writeback")
    if len(residue_ids) != len(per_node):
        raise ValueError("residue_ids count does not match per-node output count")

    gnn_run_id = str(uuid.uuid4())
    engine = get_engine()

    async with AsyncSession(engine) as session:
        async with session.begin():
            await _ensure_embedding_schema(session)

            await write_job_status(
                session,
                structure_id=structure_id,
                job_type="gnn_inference",
                status="complete",
            )

            await session.execute(
                sa.text(
                    """
                    INSERT INTO fact_gnn_inference (
                        gnn_run_id, structure_id, model_version,
                        projection_dim, num_nodes, num_edges,
                        curvature_value, depth_conditioning_enabled,
                        balance_loss, audit_trail, source_type
                    ) VALUES (
                        :gnn_run_id, :structure_id, :model_version,
                        :projection_dim, :num_nodes, :num_edges,
                        :curvature_value, :depth_conditioning,
                        :balance_loss, :audit_trail, :source_type
                    )
                    """
                ),
                {
                    "gnn_run_id": gnn_run_id,
                    "structure_id": structure_id,
                    "model_version": model_output["model_version"],
                    "projection_dim": len(per_node[0]["projections"]),
                    "num_nodes": model_output["num_nodes"],
                    "num_edges": model_output["num_edges"],
                    "curvature_value": model_output["curvature"],
                    "depth_conditioning": model_output["depth_conditioning"],
                    "balance_loss": model_output["balance_loss"],
                    "audit_trail": _jsonable_audit_trail(model_output.get("audit_trail")),
                    "source_type": source_type,
                },
            )

            for index, node in enumerate(per_node):
                node_id = str(uuid.uuid4())
                embedding_id = str(uuid.uuid4())
                features = node_features[index]
                projection = node["projections"]
                expert_weights = node["expert_weights"]

                await session.execute(
                    sa.text(
                        """
                        INSERT INTO fact_gnn_node_output (
                            node_id, gnn_run_id, residue_id,
                            input_rho, input_tau_flag, input_ss_type, input_sasa,
                            projections, cone_depth, cone_width,
                            expert_weights,
                            epistemic_uncertainty, aleatoric_uncertainty, total_uncertainty,
                            evidence_mu, evidence_nu, evidence_alpha, evidence_beta,
                            source_type
                        ) VALUES (
                            :node_id, :gnn_run_id, :residue_id,
                            :input_rho, :input_tau_flag, :input_ss_type, :input_sasa,
                            :projections, :cone_depth, :cone_width,
                            :expert_weights,
                            :epistemic_uncertainty, :aleatoric_uncertainty, :total_uncertainty,
                            :evidence_mu, :evidence_nu, :evidence_alpha, :evidence_beta,
                            :source_type
                        )
                        """
                    ),
                    {
                        "node_id": node_id,
                        "gnn_run_id": gnn_run_id,
                        "residue_id": residue_ids[index],
                        "input_rho": features[0],
                        "input_tau_flag": features[1],
                        "input_ss_type": features[2],
                        "input_sasa": features[3],
                        "projections": json.dumps(projection),
                        "cone_depth": node["cone_depth"],
                        "cone_width": node["cone_width"],
                        "expert_weights": json.dumps(expert_weights),
                        "epistemic_uncertainty": node["epistemic"],
                        "aleatoric_uncertainty": node["aleatoric"],
                        "total_uncertainty": node["total_uncertainty"],
                        "evidence_mu": node["mu"],
                        "evidence_nu": node["nu"],
                        "evidence_alpha": node["alpha"],
                        "evidence_beta": node["beta"],
                        "source_type": source_type,
                    },
                )

                await session.execute(
                    sa.text(
                        """
                        INSERT INTO fact_gnn_node_embedding (
                            embedding_id, node_id, gnn_run_id,
                            structure_id, residue_id, model_version,
                            projection_dim, projection_embedding,
                            cone_depth, cone_width, routing_entropy, expert_winner,
                            source_type
                        ) VALUES (
                            :embedding_id, :node_id, :gnn_run_id,
                            :structure_id, :residue_id, :model_version,
                            :projection_dim, CAST(:projection_embedding AS vector),
                            :cone_depth, :cone_width, :routing_entropy, :expert_winner,
                            :source_type
                        )
                        """
                    ),
                    {
                        "embedding_id": embedding_id,
                        "node_id": node_id,
                        "gnn_run_id": gnn_run_id,
                        "structure_id": structure_id,
                        "residue_id": residue_ids[index],
                        "model_version": model_output["model_version"],
                        "projection_dim": len(projection),
                        "projection_embedding": _vector_literal(projection),
                        "cone_depth": node["cone_depth"],
                        "cone_width": node["cone_width"],
                        "routing_entropy": _routing_entropy(expert_weights),
                        "expert_winner": max(range(len(expert_weights)), key=lambda idx: float(expert_weights[idx])),
                        "source_type": source_type,
                    },
                )

    return gnn_run_id