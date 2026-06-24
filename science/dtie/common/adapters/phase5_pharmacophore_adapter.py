"""Phase 5 pharmacophore persistence adapter.

Converts Phase 5 pharmacophore identification PhaseResult outputs into a
PharmacophorePayload and writes them through the Normalizer's governed path.

Requirements: 4.1, 4.2
"""

from __future__ import annotations

from typing import Any

from science.dtie.common.interfaces import PhaseResult
from science.dtie.common.normalizer_payloads import (
    NormalizerResult,
    PharmacophorePayload,
    PharmacophoreRecord,
    ProvenanceContext,
)
from science.dtie.common.phase_persistence import PhasePersistenceSpec


class Phase5PharmacophoreAdapter:
    """Adapts Phase 5 pharmacophore PhaseResult into Normalizer payloads.

    Extracts pharmacophore features from the PhaseResult outputs and
    constructs a PharmacophorePayload for governed persistence.
    """

    spec = PhasePersistenceSpec(
        phase_name="phase5_pharmacophore",
        tier=1,
        produces_residue_level_data=True,
    )

    async def persist(
        self,
        phase_result: PhaseResult,
        provenance: ProvenanceContext,
        normalizer: Any,
    ) -> NormalizerResult:
        """Persist Phase 5 pharmacophore results through the Normalizer.

        Args:
            phase_result: Completed Phase 5 pharmacophore output.
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
                "Phase 5 pharmacophore results must be linked to the GNN inference run."
            )

        outputs = phase_result.outputs
        raw_pharmacophores = outputs.get("pharmacophores", [])

        pharmacophores: list[PharmacophoreRecord] = []
        for p in raw_pharmacophores:
            center_xyz = p.get("center_xyz", [0.0, 0.0, 0.0])
            pharmacophores.append(
                PharmacophoreRecord(
                    pocket_index=p["pocket_index"],
                    center_x=center_xyz[0],
                    center_y=center_xyz[1],
                    center_z=center_xyz[2],
                    druggability_score=p["druggability_score"],
                    residue_count=p["residue_count"],
                    residue_indices=p["residue_indices"],
                    allosteric_coupling=p.get("allosteric_coupling", 0.0),
                    volume_estimate=p.get("volume_estimate_A3", 0.0),
                )
            )

        payload = PharmacophorePayload(
            provenance=provenance,
            pharmacophores=pharmacophores,
            druggability_threshold=outputs.get("druggability_threshold", 0.0),
        )

        return await normalizer.normalize_pharmacophores(payload)
