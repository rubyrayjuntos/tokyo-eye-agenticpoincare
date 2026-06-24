"""Allosteric site identification persistence adapter.

Converts allosteric site PhaseResult outputs into an AllostericSitePayload
and writes them through the Normalizer's governed path.

Requirements: 3.1, 3.3, 8.1
"""

from __future__ import annotations

from typing import Any

from science.dtie.common.interfaces import PhaseResult
from science.dtie.common.normalizer_payloads import (
    AllostericSitePayload,
    AllostericSiteRecord,
    NormalizerResult,
    ProvenanceContext,
)
from science.dtie.common.phase_persistence import PhasePersistenceSpec


class AllostericSiteAdapter:
    """Adapts allosteric site identification PhaseResult into Normalizer payloads.

    Extracts predicted allosteric pockets from the PhaseResult outputs
    and constructs an AllostericSitePayload for governed persistence.
    """

    spec = PhasePersistenceSpec(
        phase_name="allosteric_sites",
        tier=1,
        produces_residue_level_data=True,
    )

    async def persist(
        self,
        phase_result: PhaseResult,
        provenance: ProvenanceContext,
        normalizer: Any,
    ) -> NormalizerResult:
        """Persist allosteric site predictions through the Normalizer.

        Args:
            phase_result: Completed allosteric site identification phase output.
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
                "Allosteric site results must be linked to the GNN inference run."
            )

        outputs = phase_result.outputs
        raw_sites = outputs.get("sites", [])

        sites: list[AllostericSiteRecord] = []
        for site in raw_sites:
            centroid = site.get("centroid", [0.0, 0.0, 0.0])
            residue_ids = site.get("residue_ids", [])
            sites.append(
                AllostericSiteRecord(
                    site_id=site["site_id"],
                    residue_ids=residue_ids,
                    centroid_x=centroid[0],
                    centroid_y=centroid[1],
                    centroid_z=centroid[2],
                    confidence_score=site.get("confidence", 0.0),
                    cluster_method=site.get("method", "dbscan"),
                    n_residues=len(residue_ids),
                )
            )

        payload = AllostericSitePayload(
            provenance=provenance,
            sites=sites,
            total_sites=len(sites),
        )

        return await normalizer.normalize_allosteric_sites(payload)
