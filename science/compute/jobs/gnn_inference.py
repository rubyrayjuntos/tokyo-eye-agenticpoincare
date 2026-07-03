"""Atomic job: gnn_inference (foundation)."""

from __future__ import annotations

import logging
from typing import Any

from science.dtie.common.interfaces import GNNInferenceResult, PhaseResult
from science.dtie.v5.orchestrator.pipeline import PipelineConfig

logger = logging.getLogger(__name__)


async def run_gnn_inference(
    db: Any,
    config: PipelineConfig,
    *,
    run_id: str,
    pipeline_name: str,
    caller_identity: str = "compute_job_gnn_inference",
    device: str = "cpu",
) -> tuple[PhaseResult, GNNInferenceResult | None]:
    """Run V6 GNN inference and persist hyperbolic + Euclidean embeddings via Normalizer."""
    from data.normalizer.core import Normalizer
    from science.dtie.common.adapters import GNNOutputAdapter
    from science.dtie.common.graph_builder import GraphBuilder
    from science.dtie.v6.gnn.runner import V6GNNRunner

    structure_id = config.structure_id
    builder = GraphBuilder(db=db)
    try:
        graph = await builder.build_graph(structure_id)
    except ValueError as exc:
        return (
            PhaseResult(
                phase_name="gnn_inference",
                structure_id=structure_id,
                model_version="GOSPConeMapper-v6",
                success=False,
                outputs={"error": str(exc)},
            ),
            None,
        )

    pyg_data = builder.to_pyg(graph)
    runner = V6GNNRunner(
        checkpoint_path=config.checkpoint_path,
        device=device,
        curvature_override=config.curvature_override,
    )
    result = await runner.run_inference(structure_id, pyg_data)

    normalizer = Normalizer(db=db, caller_identity=caller_identity)
    adapter = GNNOutputAdapter(normalizer=normalizer)
    await adapter.normalize(
        result,
        run_id=run_id,
        code_version=config.code_version,
        parent_run_id=config.parent_run_id,
    )

    warnings: list[str] = []
    viewer_outputs: dict[str, Any] | None = None
    try:
        from science.dtie.v6.visualization.interactive_viewer import (
            generate_gnn_interactive_viewer,
            interactive_viewer_enabled,
        )

        if interactive_viewer_enabled():
            viewer_outputs = await generate_gnn_interactive_viewer(
                db,
                gnn_result=result,
                run_id=run_id,
                caller_identity=caller_identity,
                pipeline_name=pipeline_name,
                code_version=config.code_version,
            )
    except Exception as exc:
        logger.warning("GNN interactive viewer failed (non-fatal): %s", exc)
        warnings.append(f"interactive_viewer: {exc}")

    hyp_dist_outputs: dict[str, Any] | None = None
    if getattr(config, "populate_hyperbolic_distances", True):
        try:
            from science.dtie.v5.workers.hyperbolic_distance_populator import (
                run_post_gnn_hyperbolic_population,
            )

            hyp_dist_outputs = await run_post_gnn_hyperbolic_population(
                gnn_result=result,
                config=config,
                db=db,
                insert_batch_size=2000,
            )
        except Exception as exc:
            logger.warning("Hyperbolic distance population failed (non-fatal): %s", exc)
            warnings.append(str(exc))

    outputs: dict[str, Any] = {
        "node_count": len(result.nodes),
        "architecture": result.metadata.get("architecture", "hyperbolic_prototype_moe"),
        "gnn_model_version": result.model_version,
        "gnn_run_id": f"{run_id}_hyp",
        "embedding_run_id": f"{run_id}_hyp",
        "curvature": result.curvature,
        "deep_hyperbolic_gate": result.metadata.get("deep_hyperbolic_gate"),
        "gate_disc_scale": result.metadata.get("gate_disc_scale"),
    }
    if hyp_dist_outputs:
        outputs["hyperbolic_distances"] = hyp_dist_outputs
    if viewer_outputs:
        outputs["interactive_viewer"] = viewer_outputs

    return (
        PhaseResult(
            phase_name="gnn_inference",
            structure_id=structure_id,
            model_version=result.model_version,
            success=True,
            outputs=outputs,
            warnings=warnings or None,
        ),
        result,
    )
