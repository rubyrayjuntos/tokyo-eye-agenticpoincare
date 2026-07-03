"""Atomic job: pocket_pharmacophore_map (Act 03 — pocket-scoped phase5 subset)."""

from __future__ import annotations

from typing import Any

from science.compute.gnn_loader import load_gnn_inference_result
from science.dtie.common.interfaces import PhaseResult
from science.dtie.v5.orchestrator.pipeline import PipelineConfig


async def run_pocket_pharmacophore_map(
    db: Any,
    config: PipelineConfig,
    *,
    gnn_run_id: str | None = None,
) -> PhaseResult:
    """Map pharmacophore features for top-ranked binding pockets (subset of phase5)."""
    from science.dtie.v5.phases.phase5_pharmacophore import run_phase5_pharmacophore

    gnn_result = await load_gnn_inference_result(
        db, config.structure_id, gnn_run_id=gnn_run_id
    )
    if gnn_result is None:
        return PhaseResult(
            phase_name="pocket_pharmacophore_map",
            structure_id=config.structure_id,
            model_version="pocket-pharmacophore-v1",
            success=False,
            outputs={"error": "No GNN embeddings found"},
        )

    top_site = await db.fetch_one(
        """
        SELECT site_id, site_rank
        FROM fact_cryptic_site
        WHERE structure_id = :structure_id
        ORDER BY site_rank ASC NULLS LAST
        LIMIT 1
        """,
        {"structure_id": config.structure_id},
    )
    max_pockets = 1 if top_site else 3

    phase5 = await run_phase5_pharmacophore(
        db=db,
        gnn_result=gnn_result,
        structure_id=config.structure_id,
        max_pockets=max_pockets,
    )

    return PhaseResult(
        phase_name="pocket_pharmacophore_map",
        structure_id=config.structure_id,
        model_version=phase5.model_version,
        success=phase5.success,
        outputs={
            **(phase5.outputs or {}),
            "pocket_scope": "top_ranked_site" if top_site else "global_top3",
            "anchor_site_id": top_site["site_id"] if top_site else None,
            "pharmacophore_count": phase5.outputs.get("pharmacophore_count", 0),
        },
        warnings=phase5.warnings,
    )
