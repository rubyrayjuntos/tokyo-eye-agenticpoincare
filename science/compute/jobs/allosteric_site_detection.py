"""Atomic job: allosteric_site_detection (Act 05 — Verdict)."""

from __future__ import annotations

from typing import Any

from science.compute.phase_loader import load_phase_output, load_source_leak_phase_result
from science.dtie.common.interfaces import PhaseResult
from science.dtie.v5.orchestrator.pipeline import PipelineConfig, DTIEOrchestrator


async def run_allosteric_site_detection(
    db: Any,
    config: PipelineConfig,
    *,
    run_id: str,
) -> PhaseResult:
    orchestrator = DTIEOrchestrator(db=db)
    source_leak = await load_source_leak_phase_result(db, config.structure_id)
    phase35 = await load_phase_output(
        db, config.structure_id, "phase35", "phase35_topological_lift"
    )
    phase4 = await load_phase_output(
        db, config.structure_id, "phase4", "phase4_resistance_mapping"
    )

    return await orchestrator._identify_allosteric_sites(
        config,
        run_id,
        phase35_result=phase35,
        phase4_result=phase4,
        source_leak_result=source_leak,
    )
