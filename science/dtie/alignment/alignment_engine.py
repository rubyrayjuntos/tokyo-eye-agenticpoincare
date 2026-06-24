"""Alignment engine — orchestrates SIFTS fetch + persistence + Kabsch computation.

Coordinates the full alignment workflow for a structure:
1. Fetch SIFTS residue mapping for each chain with a UniProt accession
2. Persist residue alignment records via Normalizer
3. Find other structures sharing the same UniProt accession
4. Compute Kabsch superposition against reference structure
5. Persist structural alignment records via Normalizer

Requirements: 7.4, 7.5
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from science.dtie.alignment.kabsch_aligner import (
    ComparableCore,
    ResidueCoordinate,
    SuperpositionResult,
    build_comparable_core,
    compute_kabsch_superposition,
)
from science.dtie.alignment.sifts_mapper import fetch_sifts_mapping
from science.dtie.common.ingest_payloads import (
    AlignmentPayload,
    ResidueAlignmentRecord,
    StructuralAlignmentRecord,
)
from science.dtie.common.normalizer_payloads import (
    ProvenanceContext,
    RunType,
    SourceType,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols for dependencies
# ---------------------------------------------------------------------------


class NormalizerInterface(Protocol):
    """Protocol for the Normalizer write path."""

    async def normalize_alignment(self, payload: AlignmentPayload) -> Any: ...


class StructureLookup(Protocol):
    """Protocol for querying existing structures by UniProt accession."""

    async def get_structures_by_uniprot(
        self, uniprot_accession: str
    ) -> list[str]:
        """Return structure_ids that share this UniProt accession."""
        ...

    async def get_ca_coordinates(
        self, structure_id: str, chain_label: str, residue_ids: list[str]
    ) -> dict[str, np.ndarray]:
        """Return Cα coordinates for given residues. {residue_id: [x,y,z]}"""
        ...

    async def get_chain_label_for_accession(
        self, structure_id: str, uniprot_accession: str
    ) -> str | None:
        """Resolve the chain_label for a given structure + UniProt accession."""
        ...

    async def get_residue_ids_by_uniprot_positions(
        self, structure_id: str, uniprot_accession: str, positions: list[int]
    ) -> dict[int, str]:
        """Return {uniprot_position: residue_id} for the given positions."""
        ...


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AlignmentConfig:
    """Configuration for the alignment engine."""

    reference_structure_id: str | None = None
    reference_version: str = "1.0"
    min_occupancy: float = 0.5
    protocol_version: str = "sifts_kabsch_v1"
    pipeline_name: str = "alignment_sidecar"
    model_version: str = "kabsch_svd_v1"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class AlignmentResult:
    """Result of running the alignment engine for a structure."""

    structure_id: str
    chains_aligned: int
    residue_records_created: int
    structural_alignments_created: int
    warnings: list[str] = field(default_factory=list)
    status: str = "completed"  # completed, partial, skipped


# ---------------------------------------------------------------------------
# Alignment Engine
# ---------------------------------------------------------------------------


class AlignmentEngine:
    """Orchestrates SIFTS fetch + residue alignment persistence + Kabsch.

    Usage:
        engine = AlignmentEngine(
            normalizer=normalizer,
            structure_lookup=db_lookup,
            config=AlignmentConfig(reference_structure_id="4obe"),
        )
        result = await engine.align_structure(
            structure_id="4obe",
            pdb_id="4OBE",
            chains=chain_info,
        )
    """

    def __init__(
        self,
        normalizer: NormalizerInterface,
        structure_lookup: StructureLookup | None = None,
        config: AlignmentConfig | None = None,
    ):
        self._normalizer = normalizer
        self._lookup = structure_lookup
        self._config = config or AlignmentConfig()

    async def align_structure(
        self,
        structure_id: str,
        pdb_id: str,
        chains: list[ChainAlignmentInfo],
    ) -> AlignmentResult:
        """Run full alignment for a newly ingested structure.

        Steps:
        1. For each chain with UniProt accession → fetch SIFTS mapping
        2. Persist residue alignment records via Normalizer
        3. If reference structure configured → compute Kabsch superposition
        4. Persist structural alignment records

        Args:
            structure_id: Canonical structure_id of the ingested structure.
            pdb_id: 4-character PDB ID.
            chains: Chain metadata with UniProt info.

        Returns:
            AlignmentResult summarizing what was done.
        """
        run_id = f"align_{uuid.uuid4().hex[:12]}"
        warnings: list[str] = []
        total_residue_records = 0
        total_structural_alignments = 0
        chains_aligned = 0

        for chain in chains:
            if not chain.uniprot_accession:
                continue

            # Step 1: Fetch SIFTS mapping
            records = await fetch_sifts_mapping(
                pdb_id=pdb_id,
                chain_label=chain.auth_asym_id,
                uniprot_accession=chain.uniprot_accession,
                structure_id=structure_id,
                chain_residue_ids=chain.residue_ids,
            )

            if not records:
                warnings.append(
                    f"No SIFTS mapping for chain {chain.auth_asym_id} "
                    f"↔ {chain.uniprot_accession}"
                )
                continue

            # Step 2: Persist residue alignment via Normalizer
            provenance = ProvenanceContext(
                run_id=run_id,
                structure_id=structure_id,
                model_version=self._config.model_version,
                pipeline_name=self._config.pipeline_name,
                run_type=RunType.ANALYSIS,
                source_type=SourceType.DETERMINISTIC,
                parameters={
                    "pdb_id": pdb_id,
                    "chain_label": chain.auth_asym_id,
                    "uniprot_accession": chain.uniprot_accession,
                    "protocol_version": self._config.protocol_version,
                },
            )

            payload = AlignmentPayload(
                provenance=provenance,
                structure_id=structure_id,
                residue_alignments=records,
            )

            try:
                await self._normalizer.normalize_alignment(payload)
                total_residue_records += len(records)
                chains_aligned += 1
            except Exception as exc:
                warnings.append(
                    f"Failed to persist alignment for chain {chain.auth_asym_id}: {exc}"
                )
                logger.warning(
                    "Alignment persistence failed for %s chain %s: %s",
                    structure_id,
                    chain.auth_asym_id,
                    exc,
                )
                continue

            # Step 3: Structural superposition against reference (if configured)
            if self._config.reference_structure_id and self._lookup:
                if structure_id != self._config.reference_structure_id:
                    sa_result = await self._compute_structural_alignment(
                        query_structure_id=structure_id,
                        query_chain=chain.auth_asym_id,
                        uniprot_accession=chain.uniprot_accession,
                        residue_records=records,
                        run_id=run_id,
                    )
                    if sa_result:
                        total_structural_alignments += 1

        status = "completed" if chains_aligned > 0 else "skipped"
        if warnings and chains_aligned > 0:
            status = "partial"

        return AlignmentResult(
            structure_id=structure_id,
            chains_aligned=chains_aligned,
            residue_records_created=total_residue_records,
            structural_alignments_created=total_structural_alignments,
            warnings=warnings,
            status=status,
        )

    async def _compute_structural_alignment(
        self,
        query_structure_id: str,
        query_chain: str,
        uniprot_accession: str,
        residue_records: list[ResidueAlignmentRecord],
        run_id: str,
    ) -> StructuralAlignmentRecord | None:
        """Compute Kabsch superposition of query against reference.

        Args:
            query_structure_id: The query structure being aligned.
            query_chain: Chain in the query structure.
            uniprot_accession: Shared UniProt accession.
            residue_records: SIFTS mapping for the query chain.
            run_id: Current alignment run ID.

        Returns:
            StructuralAlignmentRecord if superposition succeeded, None otherwise.
        """
        ref_structure_id = self._config.reference_structure_id
        if not ref_structure_id or not self._lookup:
            return None

        # Get mapped positions from query
        mapped_query = [
            r for r in residue_records
            if r.uniprot_position is not None
        ]

        if len(mapped_query) < 3:
            logger.warning(
                "Too few mapped residues for superposition: %s → %s (%d)",
                query_structure_id,
                ref_structure_id,
                len(mapped_query),
            )
            return None

        # Get Cα coordinates for query residues
        query_residue_ids = [r.residue_id for r in mapped_query]
        try:
            query_coords_map = await self._lookup.get_ca_coordinates(
                query_structure_id, query_chain, query_residue_ids
            )
        except Exception as exc:
            logger.warning("Failed to get query Cα coords: %s", exc)
            return None

        # Get Cα coordinates for reference structure
        # We need the reference's residue_ids at the same UniProt positions
        try:
            ref_structures = await self._lookup.get_structures_by_uniprot(
                uniprot_accession
            )
        except Exception as exc:
            logger.warning("Failed to look up reference structures: %s", exc)
            return None

        if ref_structure_id not in ref_structures:
            logger.info(
                "Reference %s not found for %s. Skipping superposition.",
                ref_structure_id,
                uniprot_accession,
            )
            return None

        # Build coordinate arrays for the comparable core
        query_residue_coords: list[ResidueCoordinate] = []
        for record in mapped_query:
            coord = query_coords_map.get(record.residue_id)
            if coord is not None:
                query_residue_coords.append(ResidueCoordinate(
                    residue_id=record.residue_id,
                    uniprot_position=record.uniprot_position,
                    ca_coord=np.array(coord, dtype=np.float64),
                ))

        # For the reference, we need coords at the same UniProt positions
        # This requires a lookup of the reference structure's residue mapping
        # Implementation depends on what's available in structure_lookup
        ref_residue_coords = await self._get_reference_coords(
            ref_structure_id, uniprot_accession, mapped_query
        )

        if not ref_residue_coords:
            logger.warning(
                "No reference coordinates available for %s ↔ %s",
                ref_structure_id,
                uniprot_accession,
            )
            return None

        # Build comparable core and compute superposition
        core_result = build_comparable_core(
            query_residues=query_residue_coords,
            ref_residues=ref_residue_coords,
            min_occupancy=self._config.min_occupancy,
        )

        if core_result is None:
            return None

        query_coords, ref_coords, comparable_core = core_result

        # Compute Kabsch superposition
        sup_result = compute_kabsch_superposition(
            query_coords=query_coords,
            ref_coords=ref_coords,
            comparable_core=comparable_core,
        )

        if sup_result is None:
            return None

        # Build and persist structural alignment record
        sa_record = StructuralAlignmentRecord(
            query_structure_id=query_structure_id,
            reference_structure_id=ref_structure_id,
            uniprot_accession=uniprot_accession,
            rotation_matrix=sup_result.rotation_matrix.tolist(),
            translation=sup_result.translation.tolist(),
            rmsd=sup_result.rmsd,
            aligned_residue_count=sup_result.aligned_residue_count,
            comparable_core=comparable_core.to_dict(),
            protocol_version=self._config.protocol_version,
        )

        # Persist via Normalizer
        provenance = ProvenanceContext(
            run_id=run_id,
            structure_id=query_structure_id,
            model_version=self._config.model_version,
            pipeline_name=self._config.pipeline_name,
            run_type=RunType.ANALYSIS,
            source_type=SourceType.DETERMINISTIC,
            parameters={
                "reference_structure_id": ref_structure_id,
                "uniprot_accession": uniprot_accession,
                "protocol_version": self._config.protocol_version,
            },
        )

        payload = AlignmentPayload(
            provenance=provenance,
            structure_id=query_structure_id,
            residue_alignments=[],  # Already persisted above
            structural_alignments=[sa_record],
        )

        try:
            await self._normalizer.normalize_alignment(payload)
        except Exception as exc:
            logger.warning(
                "Failed to persist structural alignment %s → %s: %s",
                query_structure_id,
                ref_structure_id,
                exc,
            )
            return None

        return sa_record

    async def _get_reference_coords(
        self,
        ref_structure_id: str,
        uniprot_accession: str,
        query_records: list[ResidueAlignmentRecord],
    ) -> list[ResidueCoordinate]:
        """Get Cα coordinates from the reference structure for alignment.

        Looks up the reference's residue alignment (fact_residue_alignment) to find
        residue_ids at matching UniProt positions, then fetches their Cα coordinates
        from dim_atom.

        Args:
            ref_structure_id: The reference structure to get coordinates from.
            uniprot_accession: Shared UniProt accession linking query and reference.
            query_records: SIFTS mapping records from the query structure, used to
                determine which UniProt positions are needed.

        Returns:
            List of ResidueCoordinate objects for the reference structure, one per
            UniProt position that has both an alignment record and a Cα atom.
        """
        if not self._lookup:
            return []

        # Build the set of UniProt positions we need from the query
        needed_positions = sorted(
            {r.uniprot_position for r in query_records if r.uniprot_position is not None}
        )

        if not needed_positions:
            return []

        try:
            # Resolve which chain in the reference maps to this UniProt accession
            ref_chain = await self._lookup.get_chain_label_for_accession(
                ref_structure_id, uniprot_accession
            )
            if not ref_chain:
                logger.debug(
                    "No chain found in %s for accession %s",
                    ref_structure_id,
                    uniprot_accession,
                )
                return []

            # Query fact_residue_alignment for the reference structure to find
            # residue_ids at the needed UniProt positions
            position_to_residue = await self._lookup.get_residue_ids_by_uniprot_positions(
                ref_structure_id, uniprot_accession, needed_positions
            )

            if not position_to_residue:
                logger.debug(
                    "No alignment records found for %s at %d positions",
                    ref_structure_id,
                    len(needed_positions),
                )
                return []

            # Fetch Cα coordinates for those residue_ids
            residue_ids = list(position_to_residue.values())
            coords_map = await self._lookup.get_ca_coordinates(
                ref_structure_id, ref_chain, residue_ids
            )

            # Build ResidueCoordinate objects
            ref_coords: list[ResidueCoordinate] = []
            for position, residue_id in position_to_residue.items():
                coord = coords_map.get(residue_id)
                if coord is not None:
                    ref_coords.append(ResidueCoordinate(
                        residue_id=residue_id,
                        uniprot_position=position,
                        ca_coord=np.array(coord, dtype=np.float64),
                    ))

            logger.debug(
                "Retrieved %d reference Cα coordinates for %s chain %s (%s)",
                len(ref_coords),
                ref_structure_id,
                ref_chain,
                uniprot_accession,
            )
            return ref_coords

        except Exception as exc:
            logger.warning(
                "Failed to get reference coordinates for %s (%s): %s",
                ref_structure_id,
                uniprot_accession,
                exc,
            )
            return []


# ---------------------------------------------------------------------------
# Supporting data model
# ---------------------------------------------------------------------------


@dataclass
class ChainAlignmentInfo:
    """Chain information needed for alignment."""

    auth_asym_id: str
    label_asym_id: str
    entity_id: str
    uniprot_accession: str | None = None
    residue_ids: list[tuple[int, str | None]] | None = None
