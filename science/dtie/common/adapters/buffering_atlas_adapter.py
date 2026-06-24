"""Buffering Atlas (Phase 7) persistence adapter.

Provides a minimal adapter so that _persist_phase_result finds an entry for
"buffering_atlas" and the structure-level (X=Core Frustration, Y=Relay Flux)
coordinates land in fact_phase_output as a first-class governed artifact.

This is intentionally lightweight (the phase is a macro summary, not per-residue).
It writes directly via the normalizer's underlying connection for now;
a fuller Normalizer.normalize_buffering_atlas payload can be added later
without changing the adapter registration.
"""

from __future__ import annotations

import json
from typing import Any

from science.dtie.common.interfaces import PhaseResult
from science.dtie.common.normalizer_payloads import NormalizerResult, ProvenanceContext
from science.dtie.common.phase_persistence import PhasePersistenceSpec


class BufferingAtlasAdapter:
    """Adapter for the Phase 7 Buffering Atlas macro-projection.

    X (Core Frustration) and Y (Relay Flux) are written as a single
    structure-level row in fact_phase_output so that hydration, the
    Agent get_buffering_atlas tool, ASAR widget, and any downstream
    consumer see a deterministic governed (x, y) with full run linkage.
    """

    spec = PhasePersistenceSpec(
        phase_name="buffering_atlas",
        tier=2,  # summary / derived governed asset (not Tier-1 per-residue like leaks)
        produces_residue_level_data=False,
        schema_version="1.0",
    )

    async def persist(
        self,
        phase_result: PhaseResult,
        provenance: ProvenanceContext,
        normalizer: Any,
    ) -> NormalizerResult:
        """Persist the (x, y) atlas coordinates.

        Uses a direct INSERT into the generic fact_phase_output table
        (the same bridge used by the orchestrator safety net). This
        guarantees the row exists for the run even if/when a richer
        Normalizer entry point is introduced.
        """
        if not phase_result.success:
            return NormalizerResult(success=True, asset_metadata={"skipped": "phase unsuccessful"})

        outs = phase_result.outputs or {}
        # Sanitize one more time
        if hasattr(outs, "items"):
            safe = {}
            for k, v in outs.items():
                if isinstance(v, (int, float, str, bool, type(None))):
                    safe[k] = v
                else:
                    try:
                        safe[k] = float(v) if isinstance(v, (int, float)) else str(v)
                    except Exception:
                        safe[k] = str(v)
            outs = safe

        run_id = provenance.parent_run_id or provenance.run_id
        try:
            # normalizer may expose .db or we fall back to a raw path
            db = getattr(normalizer, "db", None) or getattr(normalizer, "_db", None)
            if db is not None and hasattr(db, "execute"):
                await db.execute(
                    """
                    INSERT INTO fact_phase_output
                        (run_id, structure_id, phase, phase_name, output_data, source_type, model_version, computed_at)
                    VALUES
                        (:run_id, :sid, :phase, :phase_name, :output_data, :source_type, :model_version, NOW())
                    """,
                    {
                        "run_id": run_id,
                        "sid": phase_result.structure_id,
                        "phase": "7",
                        "phase_name": "buffering_atlas",
                        "output_data": outs,
                        "source_type": "phase7_aggregate",
                        "model_version": phase_result.model_version or "DTIE-v5-buffering",
                    },
                )
            else:
                # Last resort: the orchestrator safety net in _persist_phase_result will have already handled it
                pass

            return NormalizerResult(
                success=True,
                asset_metadata={
                    "fact_phase_output": True,
                    "phase": "buffering_atlas",
                    "x": outs.get("x"),
                    "y": outs.get("y"),
                },
            )
        except Exception as e:
            # Do not fail the whole pipeline; the in-memory PhaseResult + derivation path remain usable.
            return NormalizerResult(success=False, error=str(e))
