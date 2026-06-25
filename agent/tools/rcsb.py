"""RCSB PDB tools — structure ingestion, search, and metadata fetch.

Provides:
- ingest_structure: Download + parse + populate dimensional model
- search_rcsb: Text/keyword search of RCSB PDB
- fetch_structure_info: Metadata for a single entry
- _fetch_metadata_batch: Batch metadata helper
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from shared.logging import get_logger

logger = get_logger(__name__)

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry"
RCSB_POLYMER_URL = "https://data.rcsb.org/rest/v1/core/polymer_entity"


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


async def ingest_structure(*, pdb_id: str, db: Any) -> dict[str, Any]:
    """Fetch a structure from RCSB and populate the dimensional model.

    This is a lightweight ingestion path for the agent container that does
    NOT require biotite. It uses the RCSB REST API for metadata and the
    ModelServer API for residue-level data.

    Args:
        pdb_id: 4-character PDB identifier (e.g. "6GQ0").
        db: DBAdapter instance (implements IngestionDB protocol).

    Returns:
        Dict with structure_id, run_id, and counts — or error dict.
    """
    from datetime import datetime, timezone

    from science.dtie.common.keys import make_chain_id, make_residue_id, make_structure_id

    pdb_id = pdb_id.strip().upper()
    if len(pdb_id) != 4:
        return {"error": f"Invalid PDB ID: '{pdb_id}' (must be 4 characters)"}

    structure_id = make_structure_id(pdb_id=pdb_id, source="rcsb")

    # Check if already ingested
    existing = await db.fetch_one(
        "SELECT structure_id FROM dim_structure WHERE structure_id = :sid",
        {"sid": structure_id},
    )
    if existing:
        return {
            "structure_id": structure_id,
            "run_id": "",
            "chains_created": 0,
            "residues_created": 0,
            "atoms_created": 0,
            "warnings": ["Already ingested"],
        }

    # Fetch metadata from RCSB
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            # Entry metadata
            entry_resp = await client.get(f"{RCSB_ENTRY_URL}/{pdb_id}")
            if entry_resp.status_code == 404:
                return {"error": f"PDB entry '{pdb_id}' not found on RCSB"}
            if entry_resp.status_code != 200:
                return {"error": f"RCSB unavailable (HTTP {entry_resp.status_code})"}
            entry_data = entry_resp.json()

            # Polymer entities for chain/residue data
            polymer_resp = await client.get(
                f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/1"
            )
            polymer_data = polymer_resp.json() if polymer_resp.status_code == 200 else {}

            # Get all polymer entities to know chain count
            entities_resp = await client.get(
                f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
            )
            entities_data = entities_resp.json() if entities_resp.status_code == 200 else entry_data

    except httpx.TimeoutException:
        return {"error": "RCSB request timed out"}
    except Exception as e:
        return {"error": f"Failed to fetch from RCSB: {e}"}

    # Extract metadata
    struct_info = entry_data.get("struct", {})
    exptl = entry_data.get("exptl", [{}])
    refine = entry_data.get("refine", [{}])

    resolution = refine[0].get("ls_d_res_high") if refine else None
    method = exptl[0].get("method") if exptl else None

    # Get chain/entity mapping from polymer_entities
    polymer_entities = entities_data.get("rcsb_entry_container_identifiers", {})
    auth_chain_ids = polymer_entities.get("polymer_entity_ids", [])

    # Use the entity_poly info to get sequences and chain assignments
    run_id = f"ingest_{structure_id}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

    try:
        await db.begin()

        # 1. Create dim_structure
        await db.execute(
            """
            INSERT INTO dim_structure (structure_id, pdb_id, source, resolution, method)
            VALUES (:structure_id, :pdb_id, :source, :resolution, :method)
            ON CONFLICT (structure_id) DO NOTHING
            """,
            {
                "structure_id": structure_id,
                "pdb_id": pdb_id,
                "source": "rcsb",
                "resolution": resolution,
                "method": method,
            },
        )

        # 2. Create provenance run
        await db.execute(
            """
            INSERT INTO provenance_run (
                run_id, structure_id, model_version, pipeline_name,
                run_type, source_type, started_at
            ) VALUES (
                :run_id, :structure_id, :model_version, :pipeline_name,
                :run_type, :source_type, :started_at
            )
            """,
            {
                "run_id": run_id,
                "structure_id": structure_id,
                "model_version": "ingestion-v1-lightweight",
                "pipeline_name": "structure_ingestion",
                "run_type": "analysis",
                "source_type": "external",
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
        )

        # 3. Fetch per-entity polymer data for chains and residues
        chains_created = 0
        residues_created = 0
        residue_records: list[dict[str, Any]] = []

        async with httpx.AsyncClient(timeout=15.0) as client:
            # Try each entity (typically 1-4 for most structures)
            for entity_num in range(1, 20):  # max 20 entities
                ent_resp = await client.get(
                    f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_num}"
                )
                if ent_resp.status_code != 200:
                    break

                ent_data = ent_resp.json()
                entity_poly = ent_data.get("entity_poly", {})
                sequence = entity_poly.get("pdbx_seq_one_letter_code_can", "")

                # Get auth chain IDs for this entity
                container_ids = ent_data.get(
                    "rcsb_polymer_entity_container_identifiers", {}
                )
                chain_labels = container_ids.get("auth_asym_ids", [])

                if not chain_labels:
                    chain_labels = [chr(64 + entity_num)]  # Fallback: A, B, C...

                for chain_label in chain_labels:
                    chain_id = make_chain_id(structure_id, chain_label)

                    await db.execute(
                        """
                        INSERT INTO dim_chain (chain_id, structure_id, chain_label, entity_type)
                        VALUES (:chain_id, :structure_id, :chain_label, :entity_type)
                        ON CONFLICT (chain_id) DO NOTHING
                        """,
                        {
                            "chain_id": chain_id,
                            "structure_id": structure_id,
                            "chain_label": chain_label,
                            "entity_type": "protein",
                        },
                    )
                    chains_created += 1

                    # Create residue records from sequence
                    for idx, aa in enumerate(sequence, start=1):
                        if aa == "?" or aa == "X":
                            continue
                        residue_id = make_residue_id(structure_id, chain_label, idx)
                        residue_records.append({
                            "residue_id": residue_id,
                            "chain_id": chain_id,
                            "residue_index": idx,
                            "residue_name": aa,
                            "residue_name_3": _ONE_TO_THREE.get(aa, "UNK"),
                        })

        # 4. Batch insert residues
        if residue_records:
            await db.execute_many(
                """
                INSERT INTO dim_residue (
                    residue_id, chain_id, residue_index, residue_name, residue_name_3
                ) VALUES (
                    :residue_id, :chain_id, :residue_index, :residue_name, :residue_name_3
                )
                ON CONFLICT (residue_id) DO NOTHING
                """,
                residue_records,
            )
            residues_created = len(residue_records)

        await db.commit()

    except Exception as e:
        await db.rollback()
        logger.exception("Ingestion failed for %s", pdb_id)
        return {"error": f"Ingestion failed: {e}"}

    logger.info(
        "Ingested structure %s: %d chains, %d residues",
        structure_id,
        chains_created,
        residues_created,
    )

    return {
        "structure_id": structure_id,
        "run_id": run_id,
        "chains_created": chains_created,
        "residues_created": residues_created,
        "atoms_created": 0,
        "warnings": [],
    }


# 1-letter to 3-letter amino acid map (reverse of science/dtie/common/ingestion.py)
_ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


async def search_rcsb(
    *,
    query: str,
    search_type: str = "text",
    organism: str | None = None,
    min_resolution: float | None = None,
    max_resolution: float | None = None,
    max_results: int = 10,
) -> dict[str, Any]:
    """Search RCSB PDB using their Search API v2.

    Args:
        query: Search text/keyword.
        search_type: "text" (only mode currently supported).
        organism: Filter by organism scientific name.
        min_resolution: Minimum resolution filter (Å).
        max_resolution: Maximum resolution filter (Å).
        max_results: Max entries to return.

    Returns:
        Dict with "count" and "results" list, or "error" key on failure.
    """
    # Build RCSB Search API query
    query_nodes: list[dict[str, Any]] = [
        {
            "type": "terminal",
            "service": "full_text",
            "parameters": {"value": query},
        }
    ]

    # Optional organism filter
    if organism:
        query_nodes.append({
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entity_source_organism.taxonomy_lineage.name",
                "operator": "exact_match",
                "value": organism,
            },
        })

    # Optional resolution filter
    if max_resolution is not None:
        query_nodes.append({
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.resolution_combined",
                "operator": "less_or_equal",
                "value": max_resolution,
            },
        })

    if min_resolution is not None:
        query_nodes.append({
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.resolution_combined",
                "operator": "greater_or_equal",
                "value": min_resolution,
            },
        })

    # Combine into group if multiple filters
    if len(query_nodes) == 1:
        search_query = query_nodes[0]
    else:
        search_query = {
            "type": "group",
            "logical_operator": "and",
            "nodes": query_nodes,
        }

    payload = {
        "query": search_query,
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": max_results},
            "sort": [{"sort_by": "score", "direction": "desc"}],
        },
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(RCSB_SEARCH_URL, json=payload)

        if resp.status_code == 204:
            return {"count": 0, "results": []}

        if resp.status_code != 200:
            return {"error": f"RCSB returned {resp.status_code}", "count": 0, "results": []}

        data = resp.json()
        result_ids = [hit["identifier"] for hit in data.get("result_set", [])]
        total = data.get("total_count", len(result_ids))

        # Fetch metadata for results
        metadata = await _fetch_metadata_batch(result_ids)

        return {"count": total, "results": metadata}

    except httpx.TimeoutException:
        return {"error": "RCSB search timed out", "count": 0, "results": []}
    except Exception as e:
        logger.exception("RCSB search failed")
        return {"error": f"RCSB unavailable: {e}", "count": 0, "results": []}


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


async def _fetch_metadata_batch(pdb_ids: list[str]) -> list[dict[str, Any]]:
    """Fetch metadata for a batch of PDB IDs from RCSB REST API.

    Returns a list of dicts with pdb_id, title, resolution, method, organism.
    """
    if not pdb_ids:
        return []

    results: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=10.0) as client:
        tasks = [_fetch_single_metadata(client, pid) for pid in pdb_ids]
        settled = await asyncio.gather(*tasks, return_exceptions=True)

    for item in settled:
        if isinstance(item, dict):
            results.append(item)

    return results


async def _fetch_single_metadata(
    client: httpx.AsyncClient, pdb_id: str
) -> dict[str, Any]:
    """Fetch metadata for a single PDB entry."""
    url = f"{RCSB_ENTRY_URL}/{pdb_id.upper()}"
    resp = await client.get(url)

    if resp.status_code != 200:
        return {"pdb_id": pdb_id.upper(), "title": None, "resolution": None}

    data = resp.json()
    struct = data.get("struct", {})
    cell = data.get("cell", {})
    exptl = data.get("exptl", [{}])
    refine = data.get("refine", [{}])
    source = data.get("rcsb_entity_source_organism", [{}])

    return {
        "pdb_id": pdb_id.upper(),
        "title": struct.get("title"),
        "resolution": refine[0].get("ls_d_res_high") if refine else None,
        "method": exptl[0].get("method") if exptl else None,
        "organism": source[0].get("scientific_name") if source else None,
    }


async def fetch_structure_info(*, pdb_id: str) -> dict[str, Any]:
    """Fetch detailed metadata for a single PDB entry.

    Args:
        pdb_id: 4-character PDB identifier.

    Returns:
        Dict with structure metadata, or error dict.
    """
    pdb_id = pdb_id.strip().upper()

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{RCSB_ENTRY_URL}/{pdb_id}")

        if resp.status_code == 404:
            return {"error": f"PDB entry '{pdb_id}' not found on RCSB"}

        if resp.status_code != 200:
            return {"error": f"RCSB unavailable (HTTP {resp.status_code})"}

        data = resp.json()
        struct = data.get("struct", {})
        exptl = data.get("exptl", [{}])
        refine = data.get("refine", [{}])
        keywords = data.get("struct_keywords", {})
        source = data.get("rcsb_entity_source_organism", [{}])
        entry_info = data.get("rcsb_entry_info", {})

        return {
            "pdb_id": pdb_id,
            "title": struct.get("title"),
            "resolution": refine[0].get("ls_d_res_high") if refine else None,
            "method": exptl[0].get("method") if exptl else None,
            "organism": source[0].get("scientific_name") if source else None,
            "keywords": keywords.get("pdbx_keywords"),
            "deposition_date": entry_info.get("deposit_date"),
            "polymer_entity_count": entry_info.get("polymer_entity_count"),
            "molecular_weight": entry_info.get("molecular_weight"),
        }

    except httpx.TimeoutException:
        return {"error": "RCSB request timed out"}
    except Exception as e:
        logger.exception("Failed to fetch info for %s", pdb_id)
        return {"error": f"RCSB unavailable: {e}"}
