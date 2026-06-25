"""Metadata enricher — query RCSB Data API for entry + entity metadata.

Enriches parsed structures with organism, UniProt accessions, entity hierarchy,
and detects duplicate entity instances (multiple chains same entity_id).
Graceful degradation if RCSB Data API is unreachable.

Requirements: 2.1, 2.2, 2.4, 2.5, 2.6
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class EntityMetadata:
    """Metadata for a single polymer entity."""

    entity_id: str
    organism: str | None = None
    uniprot_accessions: list[str] = field(default_factory=list)
    entity_type: str = "polymer"
    ref_seq_ids: list[str] = field(default_factory=list)


@dataclass
class ChainEntityInfo:
    """Entity-level info associated with a specific chain."""

    auth_asym_id: str
    label_asym_id: str
    entity_id: str
    is_entity_duplicate: bool = False


@dataclass
class StructureMetadata:
    """Aggregated metadata for a structure from RCSB Data API."""

    title: str | None = None
    resolution: float | None = None
    organism: str | None = None
    release_date: str | None = None
    entities: list[EntityMetadata] = field(default_factory=list)
    auth_to_label_mapping: dict[str, str] = field(default_factory=dict)
    chain_entity_info: list[ChainEntityInfo] = field(default_factory=list)
    duplicate_entity_ids: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# RCSB Data API field lists
# ---------------------------------------------------------------------------

# Fields requested for entry-level metadata
_ENTRY_FIELDS = [
    "struct.title",
    "rcsb_entry_info.resolution_combined",
    "rcsb_accession_info.initial_release_date",
]

# Fields requested for polymer entity metadata
_ENTITY_FIELDS = [
    "polymer_entities.rcsb_id",
    "polymer_entities.entity_poly.type",
    "polymer_entities.rcsb_entity_source_organism.ncbi_scientific_name",
    "polymer_entities.rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
    "polymer_entities.rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_name",
]

# Fields for polymer entity instances (chains)
_INSTANCE_FIELDS = [
    "polymer_entity_instances.rcsb_id",
    "polymer_entity_instances.rcsb_polymer_entity_instance_container_identifiers.asym_id",
    "polymer_entity_instances.rcsb_polymer_entity_instance_container_identifiers.auth_asym_id",
    "polymer_entity_instances.rcsb_polymer_entity_instance_container_identifiers.entity_id",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def enrich_metadata(pdb_id: str) -> StructureMetadata | None:
    """Query RCSB Data API via rcsbapi for entry + entity metadata.

    Uses rcsbapi.data.DataQuery with built-in batching and rate limiting.
    Returns None if API unreachable (graceful degradation per Requirement 2.5).

    Args:
        pdb_id: 4-character PDB ID (case-insensitive, will be uppercased).

    Returns:
        StructureMetadata with entities and chain mappings, or None on failure.
    """
    pdb_id_upper = pdb_id.strip().upper()

    try:
        entry_data = await _fetch_entry_metadata(pdb_id_upper)
        if entry_data is None:
            logger.warning(
                "RCSB Data API returned no data for %s, proceeding without metadata",
                pdb_id_upper,
            )
            return None
    except Exception as exc:
        logger.warning(
            "RCSB Data API unreachable for %s: %s. Proceeding without metadata.",
            pdb_id_upper,
            exc,
        )
        return None

    # Parse response into StructureMetadata
    return _parse_entry_response(entry_data, pdb_id_upper)


# ---------------------------------------------------------------------------
# Internal: RCSB API queries
# ---------------------------------------------------------------------------


async def _fetch_entry_metadata(pdb_id: str) -> dict | None:
    """Fetch combined entry + entity data from RCSB Data API.

    Uses rcsbapi's DataQuery with _async_exec for non-blocking execution.
    """
    from rcsbapi.data import DataQuery

    all_fields = _ENTRY_FIELDS + _ENTITY_FIELDS + _INSTANCE_FIELDS

    query = DataQuery(
        input_type="entries",
        input_ids=[pdb_id],
        return_data_list=all_fields,
    )

    response = await query._async_exec()
    query._response = response

    # Extract the entry data from GraphQL response
    if not response:
        return None

    data = response.get("data", response) if isinstance(response, dict) else response
    if isinstance(data, dict):
        entries = data.get("entries", [])
        if entries and len(entries) > 0:
            return entries[0]
    return None


# ---------------------------------------------------------------------------
# Internal: Response parsing
# ---------------------------------------------------------------------------


def _parse_entry_response(entry: dict, pdb_id: str) -> StructureMetadata:
    """Parse RCSB DataQuery response into StructureMetadata."""
    # Entry-level fields
    title = _safe_nested(entry, "struct", "title")
    resolution = _extract_resolution(entry)
    release_date = _safe_nested(entry, "rcsb_accession_info", "initial_release_date")

    # Parse polymer entities
    entities, organism = _parse_entities(entry)

    # Parse polymer entity instances (chain ↔ entity mapping)
    chain_entity_info, auth_to_label = _parse_instances(entry)

    # Detect duplicate entity instances (multiple chains same entity_id)
    duplicate_entity_ids = _detect_duplicate_entities(chain_entity_info)

    # Mark duplicates on chain info
    for chain_info in chain_entity_info:
        chain_info.is_entity_duplicate = chain_info.entity_id in duplicate_entity_ids

    return StructureMetadata(
        title=title,
        resolution=resolution,
        organism=organism,
        release_date=release_date,
        entities=entities,
        auth_to_label_mapping=auth_to_label,
        chain_entity_info=chain_entity_info,
        duplicate_entity_ids=duplicate_entity_ids,
    )


def _parse_entities(entry: dict) -> tuple[list[EntityMetadata], str | None]:
    """Parse polymer_entities from the response.

    Returns (list of EntityMetadata, primary organism).
    """
    entities: list[EntityMetadata] = []
    organism: str | None = None

    polymer_entities = entry.get("polymer_entities") or []
    for pe in polymer_entities:
        if not pe:
            continue

        # Entity ID from rcsb_id (format: "XXXX_N" where N is entity number)
        rcsb_id = pe.get("rcsb_id", "")
        entity_id = rcsb_id.split("_")[-1] if "_" in rcsb_id else rcsb_id

        # Entity type from entity_poly.type
        entity_poly = pe.get("entity_poly") or {}
        entity_type = entity_poly.get("type", "polymer")

        # Organism from rcsb_entity_source_organism
        source_organisms = pe.get("rcsb_entity_source_organism") or []
        entity_organism = None
        if source_organisms:
            first_source = source_organisms[0] if source_organisms[0] else {}
            entity_organism = first_source.get("ncbi_scientific_name")
            # Use first organism found as primary
            if organism is None and entity_organism:
                organism = entity_organism

        # UniProt accessions + reference sequence IDs
        container_ids = pe.get("rcsb_polymer_entity_container_identifiers") or {}
        ref_seq_identifiers = container_ids.get("reference_sequence_identifiers") or []

        uniprot_accessions: list[str] = []
        ref_seq_ids: list[str] = []

        for ref in ref_seq_identifiers:
            if not ref:
                continue
            db_name = ref.get("database_name", "")
            accession = ref.get("database_accession", "")
            if accession:
                ref_seq_ids.append(accession)
                if db_name == "UniProt":
                    uniprot_accessions.append(accession)

        entities.append(EntityMetadata(
            entity_id=entity_id,
            organism=entity_organism,
            uniprot_accessions=uniprot_accessions,
            entity_type=entity_type,
            ref_seq_ids=ref_seq_ids,
        ))

    return entities, organism


def _parse_instances(entry: dict) -> tuple[list[ChainEntityInfo], dict[str, str]]:
    """Parse polymer_entity_instances for chain ↔ entity mapping.

    Returns (list of ChainEntityInfo, auth_asym_id → label_asym_id mapping).
    """
    chain_info_list: list[ChainEntityInfo] = []
    auth_to_label: dict[str, str] = {}

    instances = entry.get("polymer_entity_instances") or []
    for inst in instances:
        if not inst:
            continue

        container = inst.get("rcsb_polymer_entity_instance_container_identifiers") or {}
        asym_id = container.get("asym_id", "")
        auth_asym_id = container.get("auth_asym_id", asym_id)
        entity_id = container.get("entity_id", "")

        if auth_asym_id:
            auth_to_label[auth_asym_id] = asym_id

        chain_info_list.append(ChainEntityInfo(
            auth_asym_id=auth_asym_id,
            label_asym_id=asym_id,
            entity_id=entity_id,
        ))

    return chain_info_list, auth_to_label


def _detect_duplicate_entities(chain_info: list[ChainEntityInfo]) -> set[str]:
    """Detect entity_ids that appear in multiple chains (Requirement 2.4).

    Returns set of entity_ids that have duplicate instances.
    """
    entity_counts: Counter[str] = Counter()
    for ci in chain_info:
        if ci.entity_id:
            entity_counts[ci.entity_id] += 1

    return {eid for eid, count in entity_counts.items() if count > 1}


# ---------------------------------------------------------------------------
# Internal: Utility helpers
# ---------------------------------------------------------------------------


def _safe_nested(data: dict, *keys: str):
    """Safely traverse nested dict keys, returning None if any key is missing."""
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


def _extract_resolution(entry: dict) -> float | None:
    """Extract resolution from rcsb_entry_info.resolution_combined."""
    res_combined = _safe_nested(entry, "rcsb_entry_info", "resolution_combined")
    if isinstance(res_combined, list) and res_combined:
        try:
            return float(res_combined[0])
        except (TypeError, ValueError):
            return None
    elif isinstance(res_combined, (int, float)):
        return float(res_combined)
    return None
