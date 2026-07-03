"""Atomic job: witness_embedding (Act 01 — Signal)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from science.compute.gnn_witness_inputs import load_witness_embedding_inputs
from science.compute.persist import json_safe
from data.normalizer.core import Normalizer
from science.dtie.common.normalizer_payloads import (
    PhaseOutputPayload,
    PhaseOutputRecord,
    ProvenanceContext,
    RunType,
    SourceType,
)
from science.dtie.common.interfaces import PhaseResult
from science.dtie.v5.orchestrator.pipeline import PipelineConfig

logger = logging.getLogger(__name__)


async def _persist_witness_phase_output(
    db: Any,
    *,
    run_id: str,
    structure_id: str,
    phase1_output: Any,
    gnn_run_id: str | None,
    model_version: str,
) -> None:
    outputs = json_safe(
        {
            "phase_name": "phase1_witness_embedding",
            "n_witnesses": int(len(phase1_output.witnesses)),
            "n_landmarks": int(len(phase1_output.landmarks)),
            "curvature_c": float(phase1_output.curvature_c),
            "mapping_function": phase1_output.mapping_function,
            "landmark_to_residue_map": {
                str(k): v for k, v in phase1_output.landmark_to_residue_map.items()
            },
            "residue_count": len(phase1_output.residue_ids),
            "gnn_run_id": gnn_run_id,
        }
    )
    prov = ProvenanceContext(
        run_id=run_id,
        structure_id=structure_id,
        model_version=model_version,
        pipeline_name="witness_embedding",
        run_type=RunType.ANALYSIS,
        source_type=SourceType.DERIVED,
        parent_run_id=gnn_run_id,
        parameters={"job_id": "witness_embedding"},
    )
    normalizer = Normalizer(db=db, caller_identity="witness_embedding_job")
    await normalizer.normalize_phase_output(
        PhaseOutputPayload(
            provenance=prov,
            structure_id=structure_id,
            output=PhaseOutputRecord(
                phase="1",
                phase_name="phase1_witness_embedding",
                output_data=outputs,
                source_type="derived",
                model_version=model_version,
            ),
        )
    )


async def run_witness_embedding(
    db: Any,
    config: PipelineConfig,
    *,
    run_id: str,
    gnn_run_id: str | None = None,
    n_landmarks: int | None = None,
) -> PhaseResult:
    """Run hyperbolic witness selection (Phase 1 v4) from governed GNN embeddings."""
    inputs = await load_witness_embedding_inputs(
        db, config.structure_id, gnn_run_id=gnn_run_id
    )
    if inputs is None:
        return PhaseResult(
            phase_name="phase1_witness_embedding",
            structure_id=config.structure_id,
            model_version="witness-embedding-v4",
            success=False,
            outputs={"error": "No hyperbolic GNN embeddings with coordinates found"},
        )

    landmarks = n_landmarks if n_landmarks is not None else config.n_landmarks

    def _execute() -> Any:
        from science.dtie.v4.phases.phase1_witness_embedding_v4 import (
            execute_phase_1_witness_embedding,
        )

        return execute_phase_1_witness_embedding(
            inputs.ingestion_data,
            inputs.gnn_output,
            n_landmarks=landmarks,
        )

    try:
        phase1_output = await asyncio.to_thread(_execute)
    except Exception as exc:
        logger.exception("Witness embedding failed for %s", config.structure_id)
        return PhaseResult(
            phase_name="phase1_witness_embedding",
            structure_id=config.structure_id,
            model_version="witness-embedding-v4",
            success=False,
            outputs={"error": str(exc)},
        )

    await _persist_witness_phase_output(
        db,
        run_id=run_id,
        structure_id=config.structure_id,
        phase1_output=phase1_output,
        gnn_run_id=inputs.gnn_run_id or gnn_run_id,
        model_version="witness-embedding-v4",
    )

    return PhaseResult(
        phase_name="phase1_witness_embedding",
        structure_id=config.structure_id,
        model_version="witness-embedding-v4",
        success=True,
        outputs={
            "n_witnesses": len(phase1_output.witnesses),
            "n_landmarks": len(phase1_output.landmarks),
            "curvature_c": float(phase1_output.curvature_c),
            "mapping_function": phase1_output.mapping_function,
            "run_id": run_id,
        },
    )
