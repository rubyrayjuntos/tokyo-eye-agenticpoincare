"""RCSB PDB tools — search and metadata fetch.

Governed structure onboarding uses POST /api/ingest → science ingest-full only.
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

INGEST_ONLY_MESSAGE = (
    "Lightweight RCSB ingest is disabled. "
    "Use POST /api/ingest (governed ingest-full + onboard pathway scheduler)."
)


# ---------------------------------------------------------------------------
# Ingestion (disabled — canonical path is /api/ingest → ingest-full)
# ---------------------------------------------------------------------------


async def ingest_structure(*, pdb_id: str, db: Any) -> dict[str, Any]:
    """Disabled. Governed ingest must go through POST /api/ingest."""
    _ = (pdb_id, db)
    return {"error": INGEST_ONLY_MESSAGE}


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
