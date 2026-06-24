"""SIFTS PDB↔UniProt residue mapping fetcher.

Fetches authoritative SIFTS residue-level mapping from RCSB Data API,
populates ResidueAlignmentRecord instances with reason codes for unmapped
residues (tags, engineered mutations, expression artifacts).

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from science.dtie.common.ingest_payloads import ResidueAlignmentRecord
from science.dtie.common.keys import make_residue_id

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class SIFTSResidue:
    """A single residue mapping from SIFTS."""

    pdb_seq_id: int
    pdb_ins_code: str | None
    pdb_comp_id: str
    uniprot_position: int | None
    uniprot_accession: str | None
    isoform_id: str | None = None
    mapping_confidence: float = 1.0
    reason_code: str | None = None  # None = mapped; "tag", "engineered", "unmapped"


@dataclass
class SIFTSChainMapping:
    """Complete SIFTS mapping for one chain."""

    pdb_id: str
    chain_label: str
    uniprot_accession: str
    residues: list[SIFTSResidue] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Reason code classification
# ---------------------------------------------------------------------------

# Common tag/expression artifact sequences at N/C termini
_TAG_COMP_IDS = frozenset({"ACE", "NH2", "FOR"})

# Residues that are typically engineered for crystallization
_ENGINEERED_INDICATORS = frozenset({
    "expression_tag",
    "linker",
    "cloning_artifact",
})


def _classify_unmapped_reason(
    pdb_comp_id: str,
    pdb_seq_id: int,
    chain_length: int,
    observed_in_expression_system: bool = False,
) -> str:
    """Classify why a residue has no UniProt mapping.

    Returns a reason_code: 'tag', 'engineered', or 'unmapped'.
    """
    # Terminal residues that are capping groups
    if pdb_comp_id in _TAG_COMP_IDS:
        return "tag"

    # Heuristic: residues at extreme N-terminus (negative or very low numbers)
    # that don't map are likely expression tags
    if pdb_seq_id <= 0:
        return "tag"

    # Residues from expression system
    if observed_in_expression_system:
        return "engineered"

    return "unmapped"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def fetch_sifts_mapping(
    pdb_id: str,
    chain_label: str,
    uniprot_accession: str,
    structure_id: str,
    chain_residue_ids: list[tuple[int, str | None]] | None = None,
) -> list[ResidueAlignmentRecord]:
    """Fetch SIFTS PDB↔UniProt residue mapping from RCSB.

    Uses RCSB Data API alignment endpoint to get per-residue mapping.
    Returns empty list with warning if SIFTS data unavailable (Requirement 6.5).

    Args:
        pdb_id: 4-character PDB ID.
        chain_label: Author chain ID (auth_asym_id).
        uniprot_accession: UniProt accession to map against.
        structure_id: Canonical structure_id for building residue keys.
        chain_residue_ids: Optional list of (auth_seq_id, insertion_code) tuples
            for all residues in the chain — used to generate records for
            residues that SIFTS doesn't cover (tags, engineered).

    Returns:
        List of ResidueAlignmentRecord, one per residue. Empty list if
        SIFTS unavailable.
    """
    pdb_id_upper = pdb_id.strip().upper()

    try:
        raw_mapping = await _fetch_sifts_from_rcsb(pdb_id_upper, chain_label, uniprot_accession)
    except Exception as exc:
        logger.warning(
            "SIFTS mapping unavailable for %s chain %s ↔ %s: %s. Skipping alignment.",
            pdb_id_upper,
            chain_label,
            uniprot_accession,
            exc,
        )
        return []

    if not raw_mapping:
        logger.warning(
            "SIFTS returned no mapping for %s chain %s ↔ %s. Skipping alignment.",
            pdb_id_upper,
            chain_label,
            uniprot_accession,
        )
        return []

    # Build the alignment records from SIFTS data
    records = _build_alignment_records(
        raw_mapping=raw_mapping,
        structure_id=structure_id,
        chain_label=chain_label,
        uniprot_accession=uniprot_accession,
        chain_residue_ids=chain_residue_ids,
    )

    logger.info(
        "SIFTS mapping for %s chain %s: %d records (%d mapped, %d unmapped)",
        pdb_id_upper,
        chain_label,
        len(records),
        sum(1 for r in records if r.uniprot_position is not None),
        sum(1 for r in records if r.uniprot_position is None),
    )

    return records


# ---------------------------------------------------------------------------
# Internal: RCSB SIFTS fetch
# ---------------------------------------------------------------------------


async def _fetch_sifts_from_rcsb(
    pdb_id: str,
    chain_label: str,
    uniprot_accession: str,
) -> list[dict] | None:
    """Fetch SIFTS alignment data from RCSB Data API.

    Uses rcsbapi's DataQuery for the polymer entity instance alignment.
    Returns raw list of residue-level mapping dicts, or None on failure.
    """
    from rcsbapi.data import DataQuery

    # Query the alignment annotations for the specific chain instance
    instance_id = f"{pdb_id}.{chain_label}"

    fields = [
        "polymer_entity_instances.rcsb_polymer_entity_instance_container_identifiers.auth_asym_id",
        "polymer_entity_instances.rcsb_polymer_entity_instance_container_identifiers.asym_id",
    ]

    # Use the RCSB alignment API endpoint for SIFTS data
    # The alignment endpoint returns residue-level PDB↔UniProt mappings
    try:
        query = DataQuery(
            input_type="polymer_entity_instances",
            input_ids=[instance_id],
            return_data_list=[
                "polymer_entity_instances.rcsb_polymer_entity_align.reference_database_accession",
                "polymer_entity_instances.rcsb_polymer_entity_align.reference_database_name",
                "polymer_entity_instances.rcsb_polymer_entity_align.aligned_regions.ref_beg_seq_id",
                "polymer_entity_instances.rcsb_polymer_entity_align.aligned_regions.entity_beg_seq_id",
                "polymer_entity_instances.rcsb_polymer_entity_align.aligned_regions.length",
            ],
        )

        response = await query._async_exec()
        query._response = response

        if not response:
            return None

        data = response.get("data", response) if isinstance(response, dict) else response
        if isinstance(data, dict):
            instances = data.get("polymer_entity_instances", [])
            if instances and len(instances) > 0:
                return _extract_alignment_from_response(
                    instances[0], uniprot_accession
                )

    except Exception as exc:
        logger.debug("RCSB SIFTS query failed for %s: %s", instance_id, exc)
        raise

    return None


def _extract_alignment_from_response(
    instance_data: dict,
    target_accession: str,
) -> list[dict] | None:
    """Extract residue-level mappings from RCSB alignment response.

    Converts aligned_regions (contiguous blocks) into per-residue mapping dicts.
    """
    alignments = instance_data.get("rcsb_polymer_entity_align") or []

    for alignment in alignments:
        if not alignment:
            continue

        db_accession = alignment.get("reference_database_accession", "")
        db_name = alignment.get("reference_database_name", "")

        if db_name != "UniProt" or db_accession != target_accession:
            continue

        # Found matching UniProt alignment — expand aligned_regions
        aligned_regions = alignment.get("aligned_regions") or []
        residue_mappings: list[dict] = []

        for region in aligned_regions:
            if not region:
                continue
            ref_beg = region.get("ref_beg_seq_id")
            entity_beg = region.get("entity_beg_seq_id")
            length = region.get("length")

            if ref_beg is None or entity_beg is None or length is None:
                continue

            for offset in range(length):
                residue_mappings.append({
                    "pdb_seq_id": entity_beg + offset,
                    "uniprot_position": ref_beg + offset,
                    "mapping_confidence": 1.0,
                })

        return residue_mappings if residue_mappings else None

    return None


# ---------------------------------------------------------------------------
# Internal: Build alignment records
# ---------------------------------------------------------------------------


def _build_alignment_records(
    raw_mapping: list[dict],
    structure_id: str,
    chain_label: str,
    uniprot_accession: str,
    chain_residue_ids: list[tuple[int, str | None]] | None = None,
) -> list[ResidueAlignmentRecord]:
    """Convert raw SIFTS mapping + chain residue list into alignment records.

    For each residue in the chain:
    - If SIFTS provides a mapping → record with uniprot_position set
    - If no mapping → record with reason_code explaining why

    Args:
        raw_mapping: List of dicts with pdb_seq_id → uniprot_position.
        structure_id: Canonical structure_id.
        chain_label: Author chain label.
        uniprot_accession: Target UniProt accession.
        chain_residue_ids: All residues in the chain as (auth_seq_id, insertion_code).

    Returns:
        List of ResidueAlignmentRecord covering all chain residues.
    """
    # Build lookup: pdb_seq_id → uniprot_position
    sifts_lookup: dict[int, dict] = {}
    for entry in raw_mapping:
        seq_id = entry.get("pdb_seq_id")
        if seq_id is not None:
            sifts_lookup[seq_id] = entry

    records: list[ResidueAlignmentRecord] = []

    if chain_residue_ids:
        # Generate records for ALL residues in the chain
        chain_length = len(chain_residue_ids)
        for auth_seq_id, ins_code in chain_residue_ids:
            residue_id = make_residue_id(
                structure_id, chain_label, auth_seq_id, ins_code
            )

            sifts_entry = sifts_lookup.get(auth_seq_id)
            if sifts_entry and sifts_entry.get("uniprot_position") is not None:
                # Mapped residue
                records.append(ResidueAlignmentRecord(
                    residue_id=residue_id,
                    uniprot_accession=uniprot_accession,
                    uniprot_position=sifts_entry["uniprot_position"],
                    isoform_id=sifts_entry.get("isoform_id"),
                    mapping_source="sifts",
                    mapping_confidence=sifts_entry.get("mapping_confidence", 1.0),
                    reason_code=None,
                ))
            else:
                # Unmapped — classify reason
                reason = _classify_unmapped_reason(
                    pdb_comp_id="UNK",  # comp_id not available here
                    pdb_seq_id=auth_seq_id,
                    chain_length=chain_length,
                )
                records.append(ResidueAlignmentRecord(
                    residue_id=residue_id,
                    uniprot_accession=uniprot_accession,
                    uniprot_position=None,
                    isoform_id=None,
                    mapping_source="sifts",
                    mapping_confidence=0.0,
                    reason_code=reason,
                ))
    else:
        # Only generate records for residues that SIFTS reports
        for entry in raw_mapping:
            seq_id = entry.get("pdb_seq_id")
            if seq_id is None:
                continue
            residue_id = make_residue_id(structure_id, chain_label, seq_id, None)
            records.append(ResidueAlignmentRecord(
                residue_id=residue_id,
                uniprot_accession=uniprot_accession,
                uniprot_position=entry.get("uniprot_position"),
                isoform_id=entry.get("isoform_id"),
                mapping_source="sifts",
                mapping_confidence=entry.get("mapping_confidence", 1.0),
                reason_code=None,
            ))

    return records
