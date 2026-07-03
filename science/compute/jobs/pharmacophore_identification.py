"""Atomic job: pharmacophore_identification (Act 04 — Fragment)."""

from __future__ import annotations

from typing import Any

from science.compute.gnn_loader import load_gnn_inference_result
from science.compute.phase_loader import load_phase_output
from science.dtie.common.interfaces import PhaseResult
from science.dtie.v5.orchestrator.pipeline import PipelineConfig


async def run_pharmacophore_identification(
    db: Any,
    config: PipelineConfig,
    *,
    gnn_run_id: str | None = None,
) -> PhaseResult:
    """Full-structure pharmacophore identification (phase5)."""
    from science.dtie.v5.phases.phase5_pharmacophore import run_phase5_pharmacophore

    gnn_result = await load_gnn_inference_result(
        db, config.structure_id, gnn_run_id=gnn_run_id
    )
    if gnn_result is None:
        return PhaseResult(
            phase_name="phase5_pharmacophore",
            structure_id=config.structure_id,
            model_version="pharmacophore-v1",
            success=False,
            outputs={"error": "No GNN embeddings found"},
        )

    phase35 = await load_phase_output(
        db, config.structure_id, "phase35", "phase35_topological_lift"
    )
    phase4 = await load_phase_output(
        db, config.structure_id, "phase4", "phase4_resistance_mapping"
    )

    return await run_phase5_pharmacophore(
        db=db,
        gnn_result=gnn_result,
        structure_id=config.structure_id,
        phase4_result=phase4,
        phase35_result=phase35,
    )
