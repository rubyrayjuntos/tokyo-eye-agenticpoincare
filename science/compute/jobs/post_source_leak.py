"""Post-source-leak tail phases (buffering atlas) via peeled job modules."""

from __future__ import annotations

from typing import Any

from science.dtie.common.interfaces import PhaseResult
from science.dtie.v5.orchestrator.pipeline import PipelineConfig


async def run_post_source_leak_tail(
    db: Any,
    *,
    structure_id: str,
    parent_run_id: str,
    gnn_run_id: str | None,
    identify_allosteric_sites: bool = False,
    run_buffering_atlas: bool = True,
) -> dict[str, PhaseResult]:
    """Run tail phases after peeled source_leak_detection and allosteric job."""
    from science.compute.gnn_loader import load_gnn_inference_result
    from science.compute.jobs.allosteric_site_detection import run_allosteric_site_detection
    from science.compute.jobs.topological_lift import run_topological_lift
    from science.compute.runners.common import resolve_gnn_parent_run_id
    from science.dtie.common.provenance_runtime import resolve_code_version

    structure_id = structure_id.strip().lower()
    code_version = resolve_code_version(None)
    resolved_gnn = gnn_run_id or await resolve_gnn_parent_run_id(db, structure_id)
    gnn_result = await load_gnn_inference_result(
        db, structure_id, gnn_run_id=resolved_gnn
    )

    config = PipelineConfig(
        structure_id=structure_id,
        parent_run_id=parent_run_id,
        code_version=code_version,
        run_gnn=False,
        detect_source_leaks=False,
        identify_allosteric_sites=identify_allosteric_sites,
        run_buffering_atlas=run_buffering_atlas,
    )

    phase_results: dict[str, PhaseResult] = {}

    if identify_allosteric_sites:
        phase_results["allosteric_site_detection"] = await run_allosteric_site_detection(
            db,
            config,
            run_id=f"job_allosteric_{parent_run_id}",
        )

    if run_buffering_atlas and gnn_result is not None:
        phase_results["topological_lift"] = await run_topological_lift(
            db,
            config,
            gnn_run_id=resolved_gnn,
        )

    return phase_results
