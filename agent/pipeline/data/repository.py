"""Repository for persisting ingestion data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import Any, Dict, Optional
from uuid import uuid4

from gosp.data.database import SessionLocal
from gosp.data.models import IngestionRequest, StructureMetadata, StructureFile, IngestionProvenance


@dataclass
class PersistResult:
    structure_id: str
    file_id: str
    provenance_id: str


def _hash_file(content: bytes) -> str:
    return sha256(content).hexdigest()


class IngestionRepository:
    """Persistence adapter for ingestion data."""

    def save_payload(
        self,
        *,
        request_id: str,
        input_type: str,
        input_value: str,
        source_used: str,
        structure: Dict[str, Any],
        metadata: Dict[str, Any],
        cache_path: str,
        cache_bytes: bytes,
        provenance: Dict[str, Any],
    ) -> PersistResult:
        session = SessionLocal()
        try:
            structure_id = str(uuid4())
            file_id = str(uuid4())
            provenance_id = str(uuid4())

            session.add(
                IngestionRequest(
                    request_id=request_id,
                    input_type=input_type,
                    input_value=input_value,
                    source_used=source_used,
                    status="saved",
                    error_code=None,
                    created_at=datetime.utcnow(),
                )
            )

            session.add(
                StructureMetadata(
                    structure_id=structure_id,
                    pdb_id=metadata.get("pdb_id"),
                    uniprot_id=metadata.get("uniprot_id"),
                    title=metadata.get("title"),
                    method=metadata.get("method"),
                    resolution=metadata.get("resolution"),
                    source=source_used,
                    organism=metadata.get("organism"),
                    total_residues=metadata.get("total_residues"),
                    total_atoms=metadata.get("total_atoms"),
                    created_at=datetime.utcnow(),
                )
            )

            session.add(
                StructureFile(
                    file_id=file_id,
                    structure_id=structure_id,
                    file_type=metadata.get("file_type", "structure_json"),
                    uri=cache_path,
                    size_bytes=len(cache_bytes),
                    checksum=_hash_file(cache_bytes),
                    created_at=datetime.utcnow(),
                )
            )

            session.add(
                IngestionProvenance(
                    provenance_id=provenance_id,
                    request_id=request_id,
                    structure_id=structure_id,
                    source_chain=provenance.get("fallback_chain"),
                    retries=provenance.get("retries"),
                    file_hash=provenance.get("file_hash"),
                    created_at=datetime.utcnow(),
                )
            )

            session.commit()
            return PersistResult(
                structure_id=structure_id,
                file_id=file_id,
                provenance_id=provenance_id,
            )
        finally:
            session.close()
