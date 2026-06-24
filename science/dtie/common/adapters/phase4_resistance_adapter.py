"""Phase 4 resistance pathway persistence adapter.

Converts Phase 4 resistance mapping PhaseResult outputs into a
ResistancePathwayPayload and writes them through the Normalizer's
governed path.

Requirements: 3.1, 3.2
"""

from __future__ import annotations

from typing import Any

from science.dtie.common.interfaces import PhaseResult
from science.dtie.common.normalizer_payloads import (
    NormalizerResult,
    ProvenanceContext,
    ResistancePathway,
    ResistancePathwayPayload,
)
from science.dtie.common.phase_persistence import PhasePersistenceSpec


class Phase4ResistanceAdapter:
    """Adapts Phase 4 resistance mapping PhaseResult into Normalizer payloads.

    Extracts resistance pathways and spectral analysis data from the
    PhaseResult outputs and constructs a ResistancePathwayPayload for
    governed persistence.
    """

    spec = PhasePersistenceSpec(
        phase_name="phase4_resistance_mapping",
        tier=1,
        produces_residue_level_data=True,
    )

    async def persist(
        self,
        phase_result: PhaseResult,
        provenance: ProvenanceContext,
        normalizer: Any,
    ) -> NormalizerResult:
        """Persist Phase 4 resistance pathway results through the Normalizer.

        Args:
            phase_result: Completed Phase 4 resistance mapping output.
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
                "Phase 4 resistance results must be linked to the GNN inference run."
            )

        outputs = phase_result.outputs
        raw_pathways = outputs.get("pathways", [])
        spectral = outputs.get("spectral", {})

        pathways: list[ResistancePathway] = []
        for p in raw_pathways:
            pathways.append(
                ResistancePathway(
                    source_node=p["source_node"],
                    target_node=p["target_node"],
                    source_residue=p["source_residue"],
                    target_residue=p["target_residue"],
                    effective_resistance=p["r_eff"],
                    coupling_strength=p["coupling_strength"],
                )
            )

        payload = ResistancePathwayPayload(
            provenance=provenance,
            pathways=pathways,
            lambda_2=spectral.get("lambda_2", 0.0),
            hinge_residues=spectral.get("hinge_residues", []),
            graph_nodes=outputs.get("graph_nodes", 0),
            graph_edges=outputs.get("graph_edges", 0),
        )

        return await normalizer.normalize_resistance_pathways(payload)
