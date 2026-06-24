"""Phase 3.5 topological lift persistence adapter.

Converts Phase 3.5 topological lift PhaseResult outputs into a
TopologicalLiftPayload and writes them through the Normalizer's
governed path.

Requirements: 2.1, 2.2
"""

from __future__ import annotations

from typing import Any

from science.dtie.common.interfaces import PhaseResult
from science.dtie.common.normalizer_payloads import (
    LiftedSite,
    NormalizerResult,
    ProvenanceContext,
    TopologicalLiftPayload,
)
from science.dtie.common.phase_persistence import PhasePersistenceSpec


class Phase35LiftAdapter:
    """Adapts Phase 3.5 topological lift PhaseResult into Normalizer payloads.

    Extracts lifted allosteric site coordinates from the PhaseResult outputs
    and constructs a TopologicalLiftPayload for governed persistence.

    This is a Tier 2 adapter — persistence failures won't fail the pipeline.
    """

    spec = PhasePersistenceSpec(
        phase_name="phase35_topological_lift",
        tier=2,
        produces_residue_level_data=False,
    )

    async def persist(
        self,
        phase_result: PhaseResult,
        provenance: ProvenanceContext,
        normalizer: Any,
    ) -> NormalizerResult:
        """Persist Phase 3.5 topological lift results through the Normalizer.

        Args:
            phase_result: Completed Phase 3.5 topological lift output.
            provenance: Provenance context linking to the producing run.
            normalizer: Normalizer instance for governed writes.

        Returns:
            NormalizerResult indicating success and created assets.
        """
        outputs = phase_result.outputs
        raw_sites = outputs.get("lifted_sites", [])

        lifted_sites: list[LiftedSite] = []
        for s in raw_sites:
            barycenter = s.get("barycenter_xyz", [0.0, 0.0, 0.0])
            lifted_sites.append(
                LiftedSite(
                    site_index=s["site_index"],
                    lifted_x=barycenter[0],
                    lifted_y=barycenter[1],
                    lifted_z=barycenter[2],
                    vertex_count=s["vertex_count"],
                    source_method=s.get("source", "unknown"),
                )
            )

        payload = TopologicalLiftPayload(
            provenance=provenance,
            lifted_sites=lifted_sites,
            method=outputs.get("method", "v5_native_ca_lift"),
        )

        return await normalizer.normalize_topological_lift(payload)
