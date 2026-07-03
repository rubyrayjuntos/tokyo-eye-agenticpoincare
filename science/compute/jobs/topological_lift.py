"""Atomic job: topological_lift (Act 05 — Verdict)."""

from __future__ import annotations

from typing import Any

from science.compute.gnn_loader import load_gnn_inference_result
from science.dtie.common.interfaces import PhaseResult
from science.dtie.v5.orchestrator.pipeline import PipelineConfig


async def run_topological_lift(
    db: Any,
    config: PipelineConfig,
    *,
    gnn_run_id: str | None = None,
) -> PhaseResult:
    from science.dtie.v5.phases.phase35_topological_lift import run_phase35_topological_lift

    gnn_result = await load_gnn_inference_result(
        db, config.structure_id, gnn_run_id=gnn_run_id
    )
    if gnn_result is None:
        return PhaseResult(
            phase_name="phase35_topological_lift",
            structure_id=config.structure_id,
            model_version="topological-lift-v1",
            success=False,
            outputs={"error": "No GNN embeddings found"},
        )

    return await run_phase35_topological_lift(
        db=db,
        gnn_result=gnn_result,
        structure_id=config.structure_id,
        phase3_result=None,
    )
