"""Atomic job: drug_candidate_ranking (Act 04 — Fragment)."""

from __future__ import annotations

from typing import Any

from science.compute.gnn_loader import load_gnn_inference_result
from science.compute.phase_loader import load_pharmacophore_phase_result
from science.dtie.common.interfaces import PhaseResult
from science.dtie.v5.orchestrator.pipeline import PipelineConfig


async def run_drug_candidate_ranking(
    db: Any,
    config: PipelineConfig,
    *,
    gnn_run_id: str | None = None,
) -> PhaseResult:
    """Rank drug-discovery candidates from pharmacophore pockets (phase6)."""
    from science.dtie.v5.phases.phase6_drug_discovery import run_phase6_drug_discovery

    gnn_result = await load_gnn_inference_result(
        db, config.structure_id, gnn_run_id=gnn_run_id
    )
    if gnn_result is None:
        return PhaseResult(
            phase_name="phase6_drug_discovery",
            structure_id=config.structure_id,
            model_version="drug-candidate-v1",
            success=False,
            outputs={"error": "No GNN embeddings found"},
        )

    phase5 = await load_pharmacophore_phase_result(db, config.structure_id)
    if phase5 is None:
        return PhaseResult(
            phase_name="phase6_drug_discovery",
            structure_id=config.structure_id,
            model_version="drug-candidate-v1",
            success=False,
            outputs={"error": "No pharmacophore data found — run pharmacophore_identification first"},
        )

    return await run_phase6_drug_discovery(
        db=db,
        gnn_result=gnn_result,
        structure_id=config.structure_id,
        phase5_result=phase5,
    )
