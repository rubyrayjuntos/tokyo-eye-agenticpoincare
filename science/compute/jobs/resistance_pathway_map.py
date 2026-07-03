"""Atomic job: resistance_pathway_map (Act 05 — Verdict)."""

from __future__ import annotations

from typing import Any

from science.compute.gnn_loader import load_gnn_inference_result
from science.compute.phase_loader import load_phase_output
from science.dtie.common.interfaces import PhaseResult
from science.dtie.v5.orchestrator.pipeline import PipelineConfig


async def run_resistance_pathway_map(
    db: Any,
    config: PipelineConfig,
    *,
    gnn_run_id: str | None = None,
) -> PhaseResult:
    from science.dtie.v5.phases.phase4_resistance import run_phase4_resistance

    gnn_result = await load_gnn_inference_result(
        db, config.structure_id, gnn_run_id=gnn_run_id
    )
    if gnn_result is None:
        return PhaseResult(
            phase_name="phase4_resistance_mapping",
            structure_id=config.structure_id,
            model_version="resistance-pathway-v1",
            success=False,
            outputs={"error": "No GNN embeddings found"},
        )

    phase35 = await load_phase_output(
        db, config.structure_id, "phase35", "phase35_topological_lift"
    )

    return await run_phase4_resistance(
        db=db,
        gnn_result=gnn_result,
        structure_id=config.structure_id,
        phase35_result=phase35,
        spatial_cutoff=config.spatial_cutoff,
    )
