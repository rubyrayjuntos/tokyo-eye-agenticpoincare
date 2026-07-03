"""Shared persistence helpers for atomic compute jobs."""

from __future__ import annotations

import json
from typing import Any

from science.dtie.common.interfaces import PhaseResult
from science.dtie.common.normalizer_payloads import ProvenanceContext
from science.dtie.v5.orchestrator.pipeline import PipelineConfig, _canonical_phase_name


def json_safe(obj: Any) -> Any:
    import numpy as np

    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "model_dump"):
        return json_safe(obj.model_dump())
    return obj


async def persist_phase_result(
    db: Any,
    phase_result: PhaseResult,
    *,
    run_id: str,
    gnn_run_id: str | None,
    config: PipelineConfig,
    caller_identity: str = "compute_runner",
    pipeline_name: str = "discovery_story",
) -> None:
    """Persist a phase result through its registered adapter."""
    from data.normalizer.core import Normalizer
    from science.dtie.common.phase_persistence import PERSISTENCE_ADAPTERS, ensure_adapters_registered

    if not phase_result.success:
        return

    canonical_phase_name = _canonical_phase_name(phase_result.phase_name)
    if canonical_phase_name not in PERSISTENCE_ADAPTERS:
        ensure_adapters_registered()
    adapter = PERSISTENCE_ADAPTERS.get(canonical_phase_name)

    if adapter is None:
        if canonical_phase_name in ("buffering_atlas", "phase7", "buffering"):
            try:
                outs = json_safe(phase_result.outputs or {})
                await db.execute(
                    """
                    INSERT INTO fact_phase_output (
                        run_id, structure_id, phase, phase_name, output_data,
                        source_type, model_version, computed_at
                    )
                    VALUES (
                        :run_id, :sid, :phase, :phase_name, :output_data,
                        :source_type, :model_version, NOW()
                    )
                    ON CONFLICT (phase_output_id) DO NOTHING
                    """,
                    {
                        "run_id": run_id,
                        "sid": phase_result.structure_id,
                        "phase": "7",
                        "phase_name": "buffering_atlas",
                        "output_data": outs if isinstance(outs, dict) else json.dumps(outs),
                        "source_type": "derived",
                        "model_version": phase_result.model_version or "discovery-buffering-v1",
                    },
                )
                phase_result.metadata["persisted"] = True
                return
            except Exception as exc:
                phase_result.metadata["persisted"] = False
                phase_result.metadata["persistence_error"] = str(exc)
        phase_result.metadata.setdefault("persisted", False)
        return

    if phase_result.outputs:
        phase_result.outputs = json_safe(phase_result.outputs)

    provenance = ProvenanceContext(
        run_id=f"{run_id}_{canonical_phase_name}",
        structure_id=phase_result.structure_id,
        model_version=phase_result.model_version,
        pipeline_name=pipeline_name,
        parent_run_id=gnn_run_id or config.parent_run_id,
        code_version=config.code_version,
    )

    normalizer = Normalizer(db=db, caller_identity=caller_identity)
    result = await adapter.persist(phase_result, provenance, normalizer)
    phase_result.metadata["persisted"] = True
    if result.asset_metadata:
        phase_result.metadata.update(result.asset_metadata)
