"""Source-leak detection persistence adapter.

Converts source-leak PhaseResult outputs into a SourceLeakPayload
and writes them through the Normalizer's governed path.

Requirements: 2.1, 2.3, 8.1
"""

from __future__ import annotations

from typing import Any

from science.dtie.common.interfaces import PhaseResult
from science.dtie.common.normalizer_payloads import (
    NormalizerResult,
    ProvenanceContext,
    SourceLeakPayload,
    SourceLeakResidue,
)
from science.dtie.common.phase_persistence import PhasePersistenceSpec


class SourceLeakAdapter:
    """Adapts source-leak detection PhaseResult into Normalizer payloads.

    Extracts per-residue leak candidates from the PhaseResult outputs
    and constructs a SourceLeakPayload for governed persistence.
    """

    spec = PhasePersistenceSpec(
        phase_name="source_leak_detection",
        tier=1,
        produces_residue_level_data=True,
    )

    async def persist(
        self,
        phase_result: PhaseResult,
        provenance: ProvenanceContext,
        normalizer: Any,
    ) -> NormalizerResult:
        """Persist source-leak detection results through the Normalizer.

        Args:
            phase_result: Completed source-leak detection phase output.
            provenance: Provenance context with parent_run_id linking
                        to the GNN inference run.
            normalizer: Normalizer instance for governed writes.

        Returns:
            NormalizerResult indicating success and created assets.

        Raises:
            ValueError: If parent_run_id is not set in provenance context.
                        Tier 1 phases require provenance linkage to the
                        GNN inference run (Requirements 8.1, 8.2).
        """
        if not provenance.parent_run_id:
            raise ValueError(
                "parent_run_id is required for Tier 1 phase persistence. "
                "Source-leak results must be linked to the GNN inference run."
            )

        outputs = phase_result.outputs

        # Build per-residue leak records from phase outputs.
        # The orchestrator provides epistemic values in residue_contributions,
        # and may provide cone_depths/leak_scores in outputs. If not present
        # in outputs, we derive leak_score from epistemic * cone_depth.
        leak_residues: list[SourceLeakResidue] = []
        residue_contributions = phase_result.residue_contributions or {}
        cone_depths = outputs.get("cone_depths", {})
        leak_scores = outputs.get("leak_scores", {})
        confirmed_set = set(outputs.get("confirmed_leaks", []))

        # If the orchestrator didn't provide cone_depths/leak_scores in outputs,
        # attempt to reconstruct from the source_leak_rows if available.
        source_leak_rows = outputs.get("source_leak_rows", [])
        if source_leak_rows and not cone_depths:
            for row in source_leak_rows:
                rid = row.get("residue_id", "")
                cone_depths[rid] = row.get("cone_depth", 0.0)
                leak_scores[rid] = row.get("leak_score", 0.0)

        for rid, epistemic_val in residue_contributions.items():
            # Derive leak_score if not explicitly provided
            cone_depth = cone_depths.get(rid, 0.0)
            leak_score = leak_scores.get(rid, 0.0)
            if leak_score == 0.0 and epistemic_val > 0.0 and cone_depth > 0.0:
                leak_score = epistemic_val * cone_depth

            leak_residues.append(
                SourceLeakResidue(
                    residue_id=rid,
                    epistemic_uncertainty=epistemic_val,
                    cone_depth=cone_depth,
                    leak_score=leak_score,
                    is_confirmed=rid in confirmed_set,
                )
            )

        payload = SourceLeakPayload(
            provenance=provenance,
            leak_residues=leak_residues,
            total_leaks=outputs.get("source_leak_count", len(leak_residues)),
            threshold_used=outputs.get("threshold", 0.3),
        )

        return await normalizer.normalize_source_leaks(payload)
