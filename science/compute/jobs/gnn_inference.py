"""Atomic job: gnn_inference (foundation)."""

from __future__ import annotations

import json
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
    """Run GNN inference via contract-registered runner and persist via Normalizer."""
    from data.normalizer.core import Normalizer
    from science.contracts.model_registry import get_production_model
    from science.dtie.common.adapters import GNNOutputAdapter
    from science.dtie.common.graph_builder import GraphBuilder

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

    prod = get_production_model()
    import importlib

    runner_mod = importlib.import_module(prod.runner_module)
    runner_cls = getattr(runner_mod, prod.runner_class)

    pyg_data = builder.to_pyg(graph)
    runner = runner_cls(
        checkpoint_path=config.checkpoint_path,
        device=device,
        curvature_override=config.curvature_override,
    )

    structural_frozen = getattr(config, "structural_disc_frozen", True)
    structural_artifact = None
    if structural_frozen:
        from science.dtie.common.structural_disc_compose import (
            attach_structural_disc_to_pyg,
            compose_from_protein_graph,
        )

        await runner._ensure_model_loaded()
        curvature_c = config.curvature_override
        if curvature_c is None:
            curvature_c = float(runner._model.curvature.detach().cpu().item())
        structural_artifact = compose_from_protein_graph(graph, curvature_c)
        attach_structural_disc_to_pyg(pyg_data, structural_artifact, residue_ids=graph.residue_ids)
        logger.info(
            "Structural disc SSOT attached (%s nodes, c=%.6f)",
            len(graph.residue_ids),
            curvature_c,
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
        "structural_disc_frozen": result.metadata.get("structural_disc_frozen"),
        "structural_disc_layout": result.metadata.get("structural_disc_layout"),
        "hyp_projections_2d_source": result.metadata.get("hyp_projections_2d_source"),
    }
    if hyp_dist_outputs:
        outputs["hyperbolic_distances"] = hyp_dist_outputs
    if viewer_outputs:
        outputs["interactive_viewer"] = viewer_outputs
        if viewer_outputs.get("shell_signal_gate") is not None:
            outputs["shell_signal_gate"] = viewer_outputs["shell_signal_gate"]

    if structural_artifact is not None:
        from science.dtie.common.structural_disc_compose import export_viewer_json
        from shared.gnn_viewer_paths import viewer_output_dir

        ssot_path = (
            viewer_output_dir() / structure_id.strip().lower() / "structural_disc_ssot.json"
        )
        ssot_path.parent.mkdir(parents=True, exist_ok=True)
        ssot_path.write_text(
            json.dumps(export_viewer_json(structural_artifact), indent=2),
            encoding="utf-8",
        )
        outputs["structural_disc_ssot_path"] = str(ssot_path)

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
