"""Phase 6 drug discovery persistence adapter.

Converts Phase 6 drug discovery PhaseResult outputs into a
DrugCandidatePayload and writes them through the Normalizer's
governed path.

Requirements: 5.1, 5.2
"""

from __future__ import annotations

from typing import Any

from science.dtie.common.interfaces import PhaseResult
from science.dtie.common.normalizer_payloads import (
    DrugCandidate,
    DrugCandidatePayload,
    NormalizerResult,
    ProvenanceContext,
)
from science.dtie.common.phase_persistence import PhasePersistenceSpec


class Phase6DrugDiscoveryAdapter:
    """Adapts Phase 6 drug discovery PhaseResult into Normalizer payloads.

    Extracts scored pockets from the PhaseResult outputs and constructs
    a DrugCandidatePayload for governed persistence.
    """

    spec = PhasePersistenceSpec(
        phase_name="phase6_drug_discovery",
        tier=1,
        produces_residue_level_data=True,
    )

    async def persist(
        self,
        phase_result: PhaseResult,
        provenance: ProvenanceContext,
        normalizer: Any,
    ) -> NormalizerResult:
        """Persist Phase 6 drug discovery results through the Normalizer.

        Args:
            phase_result: Completed Phase 6 drug discovery output.
            provenance: Provenance context with parent_run_id linking
                        to the GNN inference run.
            normalizer: Normalizer instance for governed writes.

        Returns:
            NormalizerResult indicating success and created assets.

        Raises:
            ValueError: If parent_run_id is not set in provenance context.
        """
        if not provenance.parent_run_id:
            raise ValueError(
                "parent_run_id is required for Tier 1 phase persistence. "
                "Phase 6 drug discovery results must be linked to the GNN inference run."
            )

        outputs = phase_result.outputs
        raw_pockets = outputs.get("scored_pockets", [])

        candidates: list[DrugCandidate] = []
        for p in raw_pockets:
            center_xyz = p.get("center_xyz", [0.0, 0.0, 0.0])
            candidates.append(
                DrugCandidate(
                    pocket_index=p["pocket_index"],
                    center_x=center_xyz[0],
                    center_y=center_xyz[1],
                    center_z=center_xyz[2],
                    accessibility_score=p["accessibility_score"],
                    binding_potential=p["binding_potential"],
                    admet_pass=p["admet_pass"],
                    selectivity_ratio=p["selectivity_ratio"],
                    is_state_selective=p["is_state_selective"],
                    combined_druggability=p["combined_druggability"],
                )
            )

        payload = DrugCandidatePayload(
            provenance=provenance,
            candidates=candidates,
            admet_passed_count=outputs.get("admet_passed_count", 0),
            state_selective_count=outputs.get("state_selective_count", 0),
        )

        return await normalizer.normalize_drug_candidates(payload)
