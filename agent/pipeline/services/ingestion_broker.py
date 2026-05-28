"""Ingestion broker for multi-source structure loading."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional, Tuple
from uuid import uuid4

import requests

from gosp.models.data_models import StructureData
from gosp.services.ingestion_cache import IngestionCache
from gosp.services.structure_ingestion import (
    fetch_structure_from_rcsb,
    parse_structure_content,
    build_structure_data,
    StructureIngestionError,
)
from gosp.data.repository import IngestionRepository

SourceId = Literal["rcsb", "pdbe", "alphafold"]
InputType = Literal["pdb", "uniprot"]


@dataclass
class BrokerResult:
    request_id: str
    structure: StructureData
    metadata: Dict[str, object]
    provenance: Dict[str, object]
    warnings: List[str]
    saved_structure_id: Optional[str]


class SourceAdapter:
    source_id: SourceId

    def fetch_structure(
        self,
        *,
        input_value: str,
        fmt: Literal["cif", "bcif", "pdb"],
        chain: Optional[str]
    ) -> StructureData:
        raise NotImplementedError

    def fetch_metadata(self, *, input_value: str) -> Dict[str, object]:
        raise NotImplementedError


class RcsbAdapter(SourceAdapter):
    source_id: SourceId = "rcsb"

    def fetch_structure(
        self,
        *,
        input_value: str,
        fmt: Literal["cif", "bcif", "pdb"],
        chain: Optional[str]
    ) -> StructureData:
        return fetch_structure_from_rcsb(input_value, fmt, chain_id=chain)

    def fetch_metadata(self, *, input_value: str) -> Dict[str, object]:
        return _fetch_rcsb_metadata(input_value)


class PdbeAdapter(SourceAdapter):
    source_id: SourceId = "pdbe"

    def fetch_structure(
        self,
        *,
        input_value: str,
        fmt: Literal["cif", "bcif", "pdb"],
        chain: Optional[str]
    ) -> StructureData:
        url = f"https://www.ebi.ac.uk/pdbe/entry-files/download/{input_value.lower()}.bcif"
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            raise StructureIngestionError(
                f"PDBe download failed ({response.status_code}) for {input_value}"
            )
        structure = parse_structure_content(response.content, fmt)
        return build_structure_data(input_value, structure)

    def fetch_metadata(self, *, input_value: str) -> Dict[str, object]:
        return _fetch_pdbe_metadata(input_value)


class AlphaFoldAdapter(SourceAdapter):
    source_id: SourceId = "alphafold"

    def fetch_structure(
        self,
        *,
        input_value: str,
        fmt: Literal["cif", "bcif", "pdb"],
        chain: Optional[str]
    ) -> StructureData:
        url = f"https://alphafold.ebi.ac.uk/files/AF-{input_value.upper()}-F1-model_v4.bcif"
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            raise StructureIngestionError(
                f"AlphaFold download failed ({response.status_code}) for {input_value}"
            )
        structure = parse_structure_content(response.content, fmt)
        return build_structure_data(f"AF_{input_value.upper()}", structure)

    def fetch_metadata(self, *, input_value: str) -> Dict[str, object]:
        return _fetch_alphafold_metadata(input_value)


_ADAPTERS: Dict[SourceId, SourceAdapter] = {
    "rcsb": RcsbAdapter(),
    "pdbe": PdbeAdapter(),
    "alphafold": AlphaFoldAdapter(),
}


def _merge_metadata(sources: List[Dict[str, object]]) -> Dict[str, object]:
    merged: Dict[str, object] = {}
    for source in sources:
        for key, value in source.items():
            if key not in merged and value not in (None, ""):
                merged[key] = value
    return merged


def _build_metadata(
    structure: StructureData,
    source_id: SourceId,
    fmt: str,
    upstream_metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    base = {
        "pdb_id": structure.pdb_id,
        "uniprot_id": None,
        "title": None,
        "method": None,
        "resolution": structure.resolution,
        "source": source_id,
        "organism": None,
        "total_residues": structure.total_residues,
        "total_atoms": structure.total_atoms,
        "file_type": fmt,
    }
    if upstream_metadata:
        return _merge_metadata([upstream_metadata, base])
    return base


def _fetch_rcsb_metadata(pdb_id: str) -> Dict[str, object]:
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.lower()}"
    response = requests.get(url, timeout=20)
    if response.status_code != 200:
        raise StructureIngestionError(
            f"RCSB metadata failed ({response.status_code}) for {pdb_id}"
        )
    data = response.json()
    entry_info = data.get("rcsb_entry_info", {})
    resolution = None
    if isinstance(entry_info.get("resolution_combined"), list) and entry_info.get("resolution_combined"):
        resolution = entry_info.get("resolution_combined")[0]
    return {
        "pdb_id": pdb_id.upper(),
        "title": data.get("struct", {}).get("title"),
        "method": data.get("exptl", [{}])[0].get("method") if data.get("exptl") else None,
        "resolution": resolution,
        "organism": None,
    }


def _fetch_pdbe_metadata(pdb_id: str) -> Dict[str, object]:
    url = f"https://www.ebi.ac.uk/pdbe/api/pdb/entry/summary/{pdb_id.lower()}"
    response = requests.get(url, timeout=20)
    if response.status_code != 200:
        raise StructureIngestionError(
            f"PDBe metadata failed ({response.status_code}) for {pdb_id}"
        )
    data = response.json()
    entry = (data.get(pdb_id.lower()) or [{}])[0]
    return {
        "pdb_id": pdb_id.upper(),
        "title": entry.get("title"),
        "method": entry.get("experimental_method", [None])[0]
        if isinstance(entry.get("experimental_method"), list)
        else entry.get("experimental_method"),
        "resolution": entry.get("resolution"),
        "organism": entry.get("organism_scientific_name"),
    }


def _fetch_alphafold_metadata(uniprot_id: str) -> Dict[str, object]:
    url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id.upper()}"
    response = requests.get(url, timeout=20)
    if response.status_code != 200:
        raise StructureIngestionError(
            f"AlphaFold metadata failed ({response.status_code}) for {uniprot_id}"
        )
    data = response.json()
    entry = data[0] if isinstance(data, list) and data else {}
    return {
        "pdb_id": f"AF_{uniprot_id.upper()}",
        "uniprot_id": entry.get("uniprotId") or uniprot_id.upper(),
        "title": entry.get("uniprotDescription"),
        "method": "AlphaFold",
        "resolution": None,
        "organism": entry.get("organismScientificName"),
    }


def ingest_structure_broker(
    *,
    input_type: InputType,
    input_value: str,
    preferred_sources: Optional[List[SourceId]] = None,
    combine_sources: bool = False,
    fmt: Literal["cif", "bcif", "pdb"] = "bcif",
    chain: Optional[str] = None,
    save_requested: bool = False,
) -> BrokerResult:
    warnings: List[str] = []
    request_id = f"req_{uuid4().hex}"
    fallback_chain: List[str] = []

    if preferred_sources:
        sources = preferred_sources
    elif input_type == "uniprot":
        sources = ["alphafold", "rcsb", "pdbe"]
    else:
        sources = ["rcsb", "pdbe", "alphafold"]

    structure: Optional[StructureData] = None
    source_used: Optional[SourceId] = None
    metadata_sources: List[Dict[str, object]] = []

    for source_id in sources:
        adapter = _ADAPTERS[source_id]
        if structure is None:
            try:
                fallback_chain.append(source_id)
                structure = adapter.fetch_structure(
                    input_value=input_value,
                    fmt=fmt,
                    chain=chain,
                )
                source_used = source_id
            except NotImplementedError as exc:
                warnings.append(str(exc))
                continue
            except StructureIngestionError as exc:
                warnings.append(f"{source_id} ingestion failed: {exc}")
                continue

        if combine_sources or source_id == source_used:
            try:
                metadata_sources.append(adapter.fetch_metadata(input_value=input_value))
            except NotImplementedError as exc:
                warnings.append(str(exc))
            except StructureIngestionError as exc:
                warnings.append(f"{source_id} metadata failed: {exc}")

    if structure is None or source_used is None:
        raise StructureIngestionError("All sources failed to ingest the structure")

    upstream_metadata = _merge_metadata(metadata_sources) if metadata_sources else None
    metadata = _build_metadata(structure, source_used, fmt, upstream_metadata=upstream_metadata)
    provenance = {
        "request_id": request_id,
        "input_type": input_type,
        "input_value": input_value,
        "source_used": source_used,
        "fallback_chain": fallback_chain,
        "timestamps": {
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        },
        "retries": None,
        "file_hash": None,
    }

    cache = IngestionCache()
    cache_payload = {
        "request_id": request_id,
        "structure": structure.model_dump(),
        "metadata": metadata,
        "provenance": provenance,
        "source_used": source_used,
    }
    cache_entry = cache.write_payload(request_id, cache_payload)
    cache_bytes = cache_entry.path.read_bytes()

    saved_structure_id = None
    if save_requested:
        repo = IngestionRepository()
        result = repo.save_payload(
            request_id=request_id,
            input_type=input_type,
            input_value=input_value,
            source_used=source_used,
            structure=structure.model_dump(),
            metadata=metadata,
            cache_path=str(cache_entry.path),
            cache_bytes=cache_bytes,
            provenance=provenance,
        )
        saved_structure_id = result.structure_id

    return BrokerResult(
        request_id=request_id,
        structure=structure,
        metadata=metadata,
        provenance=provenance,
        warnings=warnings,
        saved_structure_id=saved_structure_id,
    )


def save_cached_ingestion(request_id: str) -> str:
    cache = IngestionCache()
    payload = cache.read_payload(request_id)

    repo = IngestionRepository()
    result = repo.save_payload(
        request_id=request_id,
        input_type=payload.get("provenance", {}).get("input_type", "pdb"),
        input_value=payload.get("provenance", {}).get("input_value", ""),
        source_used=payload.get("source_used", "rcsb"),
        structure=payload.get("structure", {}),
        metadata=payload.get("metadata", {}),
        cache_path=str(cache.root / f"broker_{request_id}.json"),
        cache_bytes=(cache.root / f"broker_{request_id}.json").read_bytes(),
        provenance=payload.get("provenance", {}),
    )
    return result.structure_id
