"""Load persisted phase outputs for peeled atomic jobs."""

from __future__ import annotations

import json
import logging
from typing import Any

from science.dtie.common.interfaces import PhaseResult

logger = logging.getLogger(__name__)


async def load_phase_output(
    db: Any,
    structure_id: str,
    *phase_names: str,
) -> PhaseResult | None:
    """Load the latest phase output blob from fact_phase_output."""
    if not phase_names:
        return None

    row = await db.fetch_one(
        """
        SELECT phase_name, phase, output_data, model_version
        FROM fact_phase_output
        WHERE structure_id = :structure_id
          AND (
            phase_name = ANY(:names)
            OR phase = ANY(:names)
          )
        ORDER BY computed_at DESC
        LIMIT 1
        """,
        {"structure_id": structure_id, "names": list(phase_names)},
    )
    if row is None:
        return None

    outputs = row.get("output_data") or {}
    if isinstance(outputs, str):
        try:
            outputs = json.loads(outputs)
        except json.JSONDecodeError:
            outputs = {}

    phase_name = str(row.get("phase_name") or row.get("phase") or phase_names[0])
    return PhaseResult(
        phase_name=phase_name,
        structure_id=structure_id,
        model_version=str(row.get("model_version") or "loaded-from-db"),
        success=True,
        outputs=outputs if isinstance(outputs, dict) else {"data": outputs},
    )


async def load_pharmacophore_phase_result(
    db: Any,
    structure_id: str,
) -> PhaseResult | None:
    """Reconstruct a phase5-like result from fact_pharmacophore rows."""
    rows = await db.fetch_all(
        """
        SELECT pocket_index, center_x, center_y, center_z,
               druggability_score, residue_count, residue_indices,
               allosteric_coupling, volume_estimate
        FROM fact_pharmacophore
        WHERE structure_id = :structure_id
        ORDER BY druggability_score DESC
        """,
        {"structure_id": structure_id},
    )
    if not rows:
        return None

    pharmacophores: list[dict[str, Any]] = []
    for row in rows:
        indices = row.get("residue_indices") or []
        if isinstance(indices, str):
            try:
                indices = json.loads(indices)
            except json.JSONDecodeError:
                indices = []
        pharmacophores.append(
            {
                "pocket_index": row.get("pocket_index"),
                "center": [row.get("center_x"), row.get("center_y"), row.get("center_z")],
                "druggability_score": float(row.get("druggability_score") or 0.0),
                "residue_count": row.get("residue_count"),
                "residue_indices": indices,
                "allosteric_coupling": row.get("allosteric_coupling"),
                "volume_estimate": row.get("volume_estimate"),
            }
        )

    return PhaseResult(
        phase_name="phase5_pharmacophore",
        structure_id=structure_id,
        model_version="loaded-from-db",
        success=True,
        outputs={
            "pharmacophore_count": len(pharmacophores),
            "pharmacophores": pharmacophores,
        },
    )


async def load_source_leak_phase_result(
    db: Any,
    structure_id: str,
) -> PhaseResult | None:
    """Reconstruct source-leak phase outputs from fact_source_leak."""
    rows = await db.fetch_all(
        """
        SELECT residue_id, epistemic_uncertainty, cone_depth, leak_score
        FROM fact_source_leak
        WHERE structure_id = :structure_id
        ORDER BY leak_score DESC NULLS LAST
        """,
        {"structure_id": structure_id},
    )
    if not rows:
        return None

    leak_residues = [str(row["residue_id"]) for row in rows]
    return PhaseResult(
        phase_name="source_leak_detection",
        structure_id=structure_id,
        model_version="loaded-from-db",
        success=True,
        outputs={
            "source_leak_count": len(leak_residues),
            "source_leak_residues": leak_residues,
            "leak_scores": {
                str(row["residue_id"]): float(row.get("leak_score") or 0.0) for row in rows
            },
        },
    )
