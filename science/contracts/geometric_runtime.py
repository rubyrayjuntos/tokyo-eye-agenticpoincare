"""Runtime geometric enforcement and learned-curvature passthrough."""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

from science.contracts.onboard_contract import (
    geometric_enforcement_level,
    get_hyperbolic_jobs,
    job_requires_hyperbolic,
)

logger = logging.getLogger(__name__)

EnforcementLevel = Literal["warning", "error"]

LEARNED_CURVATURE_SQL = """
SELECT es.curvature
FROM fact_gnn_node_embedding e
JOIN embedding_space es ON es.space_id = e.space_id
JOIN provenance_run p ON p.run_id = e.run_id
WHERE e.structure_id = :structure_id
  AND es.space_type = 'hyperbolic'
  AND COALESCE(p.parameters->>'audit_only', 'false') != 'true'
ORDER BY COALESCE(p.completed_at, p.started_at) DESC, e.computed_at DESC
LIMIT 1
"""


def effective_enforcement_level(job_id: str) -> EnforcementLevel:
    """Resolve enforcement level from env overrides and contract defaults."""
    error_jobs = {
        j.strip()
        for j in os.getenv("GEOMETRIC_ENFORCEMENT_ERROR_JOBS", "").split(",")
        if j.strip()
    }
    if job_id in error_jobs:
        return "error"

    override = os.getenv("GEOMETRIC_ENFORCEMENT_LEVEL", "").strip().lower()
    if override in ("warning", "error"):
        return override  # type: ignore[return-value]

    level = geometric_enforcement_level()
    if level == "error":
        return "error"
    return "warning"


async def load_structure_learned_curvature(db: Any, structure_id: str) -> float | None:
    """Load model-learned curvature persisted at gnn_inference from embedding_space."""
    structure_id = structure_id.strip().lower()
    row = await db.fetch_one(LEARNED_CURVATURE_SQL, {"structure_id": structure_id})
    if row is None or row.get("curvature") is None:
        return None
    value = float(row["curvature"])
    return value if value > 0 else None


async def check_onboard_hyperbolic_prerequisites(
    db: Any,
    structure_id: str,
    *,
    residue_count: int,
    primary_chain_ids: list[str],
) -> list[str]:
    """Lightweight ingest-time notes before hyperbolic compute is scheduled."""
    notes: list[str] = []
    structure_id = structure_id.strip().lower()

    notes.append(
        "Onboard pathway runs hyperbolic GNN inference; curvature is learned at "
        "inference and persisted to embedding_space for downstream jobs."
    )

    if residue_count <= 0:
        notes.append(
            "No residues ingested — gnn_inference (hyperbolic) will fail until "
            "dim_residue is populated."
        )

    if not primary_chain_ids:
        notes.append(
            "Computation scope has no primary chains — assign_computation_scope "
            "may block hyperbolic GNN preconditions."
        )

    scope_row = await db.fetch_one(
        """
        SELECT 1 FROM structure_computation_scope
        WHERE structure_id = :structure_id
        LIMIT 1
        """,
        {"structure_id": structure_id},
    )
    if scope_row is None:
        notes.append("structure_computation_scope not yet persisted after ingest.")

    return notes


def validate_learned_curvature_in_result(
    job_id: str,
    *,
    success: bool,
    outputs: dict[str, Any],
    learned_curvature: float | None,
) -> list[str]:
    """Validate curvature production (GNN) or passthrough (downstream)."""
    messages: list[str] = []
    if not success:
        return messages

    if job_id == "gnn_inference":
        curvature = outputs.get("curvature")
        if curvature is None:
            messages.append(
                "gnn_inference succeeded but JobRunResult.outputs.curvature is missing; "
                "downstream hyperbolic jobs cannot inherit learned curvature."
            )
        elif float(curvature) <= 0:
            messages.append(
                f"gnn_inference reported non-positive learned curvature {curvature!r}"
            )
        return messages

    if job_requires_hyperbolic(job_id):
        if learned_curvature is None:
            messages.append(
                f"hyperbolic job {job_id!r} ran without learned curvature passthrough "
                "(expected from embedding_space after gnn_inference)"
            )
        elif outputs.get("curvature") is not None:
            out_c = float(outputs["curvature"])
            if abs(out_c - learned_curvature) > 1e-4:
                messages.append(
                    f"job {job_id!r} outputs.curvature {out_c} != "
                    f"embedding_space learned curvature {learned_curvature}"
                )
    return messages


def apply_enforcement(job_id: str, messages: list[str]) -> tuple[list[str], bool]:
    """Return (warnings, fail_job) according to effective enforcement level."""
    if not messages:
        return [], False
    level = effective_enforcement_level(job_id)
    if level == "error":
        return messages, True
    return messages, False
