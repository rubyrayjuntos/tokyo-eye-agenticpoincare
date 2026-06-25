"""Phase 3 (witness persistence) adapter conforming to PhasePersistenceAdapter.

Wraps the Phase 3 persistence logic to conform to the unified adapter
protocol, enabling registry-based orchestrator invocation.

Requirements: 1.1, 1.2, 4.2
"""

from __future__ import annotations

from typing import Any

from science.dtie.common.interfaces import PhaseResult
from science.dtie.common.normalizer_payloads import (
    NormalizerResult,
    PersistenceBarcode,
    Phase3PersistencePayload,
    Phase3ResidueContribution,
    ProvenanceContext,
    RunType,
    SourceType,
)
from science.dtie.common.phase_persistence import PhasePersistenceSpec


class Phase3PersistenceAdapter:
    """Adapts Phase 3 witness persistence PhaseResult into Normalizer payloads.

    Conforms to the PhasePersistenceAdapter protocol for registry-based
    invocation by the orchestrator.
    """

    spec = PhasePersistenceSpec(
        phase_name="phase3_persistence",
        tier=1,
        produces_residue_level_data=True,
    )

    async def persist(
        self,
        phase_result: PhaseResult,
        provenance: ProvenanceContext,
        normalizer: Any,
    ) -> NormalizerResult:
        """Persist Phase 3 witness persistence results through the Normalizer.

        Args:
            phase_result: Completed Phase 3 phase output.
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
                "Phase 3 results must be linked to the GNN inference run."
            )

        outputs = phase_result.outputs

        # Extract barcodes
        barcodes = [
            PersistenceBarcode(
                birth=b["birth"],
                death=b["death"],
                dimension=b.get("dimension", 0),
                generator_residues=b.get("generator_residues"),
            )
            for b in outputs.get("barcodes", [])
        ]

        # Extract per-residue contributions
        residue_contributions = None
        if phase_result.residue_contributions:
            residue_contributions = [
                Phase3ResidueContribution(
                    residue_id=rid,
                    persistence_score=score,
                )
                for rid, score in phase_result.residue_contributions.items()
            ]

        payload = Phase3PersistencePayload(
            provenance=provenance,
            barcodes=barcodes,
            max_alpha=outputs.get("max_alpha", 0.0),
            n_witnesses=outputs.get("n_witnesses", 0),
            n_landmarks=outputs.get("n_landmarks", 0),
            hyperbolic_distances_used=outputs.get("hyperbolic_distances_used", False),
            curvature_c=outputs.get("curvature_c"),
            residue_contributions=residue_contributions,
            landmark_to_residue=outputs.get("landmark_to_residue"),
        )

        return await normalizer.normalize_phase3_output(payload)
