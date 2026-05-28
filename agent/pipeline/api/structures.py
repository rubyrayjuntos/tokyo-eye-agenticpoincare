"""
Structure Ingestion API Endpoints

Provides REST API for fetching and parsing protein structures.

Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 15.4
"""

from typing import Literal, Optional, List
from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel, Field

from gosp.models.data_models import StructureData
from gosp.services.structure_ingestion import (
    compress_structure_payload,
    InvalidPDBIDError,
    UnsupportedFormatError,
    StructureIngestionError
)
from gosp.services.error_handling import (
    StructuralError,
    create_error_response
)
from gosp.services.ingestion_broker import ingest_structure_broker, save_cached_ingestion
from gosp.services.ingestion_cache import IngestionCache


SourceId = Literal["rcsb", "pdbe", "alphafold"]


router = APIRouter(prefix="/api/structures", tags=["structures"])


class StructureIngestionResponse(BaseModel):
    """Response model for structure ingestion."""
    structure: StructureData
    compressed_size_bytes: Optional[int] = Field(
        None,
        description="Size of gzip-compressed payload in bytes"
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal warnings about the structure"
    )


class ErrorResponse(BaseModel):
    """Error response model."""
    error: bool = True
    error_type: str
    message: str
    details: dict = Field(default_factory=dict)
    suggested_action: Optional[str] = None
    timestamp: str


class BrokerIngestRequest(BaseModel):
    input_type: Literal["pdb", "uniprot"] = Field(
        "pdb",
        description="Input type: pdb (4-char PDB ID) or uniprot"
    )
    input_value: str = Field(..., description="PDB or UniProt identifier")
    preferred_sources: Optional[List[SourceId]] = Field(
        None,
        description="Preferred ingestion sources in order"
    )
    combine_sources: bool = Field(
        False,
        description="Merge metadata from multiple sources"
    )
    format: Literal["cif", "bcif", "pdb"] = Field(
        "cif",
        description="File format: cif (mmCIF), bcif (BinaryCIF), or pdb"
    )
    chain: Optional[str] = Field(
        None,
        description="Filter to a single chain ID (e.g., 'A')"
    )
    save_requested: bool = Field(
        False,
        description="Persist data only when explicitly requested"
    )


class BrokerIngestResponse(BaseModel):
    request_id: str
    structure: StructureData
    metadata: dict
    provenance: dict
    warnings: list[str] = Field(default_factory=list)
    saved_structure_id: Optional[str] = None


class SaveRequest(BaseModel):
    request_id: str


class SaveResponse(BaseModel):
    structure_id: str


class CacheSummary(BaseModel):
    request_id: str
    created_at: str
    pdb_id: str
    source: str


class CacheListResponse(BaseModel):
    entries: list[CacheSummary]


class CachePayloadResponse(BaseModel):
    request_id: str
    structure: StructureData
    metadata: dict
    provenance: dict


@router.get("/ingest", response_model=StructureIngestionResponse)
async def ingest_structure(
    pdb_id: str = Query(
        ...,
        description="PDB identifier (e.g., '1L2Y')",
        min_length=4,
        max_length=4,
        pattern="^[A-Za-z0-9]{4}$"
    ),
    format: Literal["cif", "bcif", "pdb"] = Query(
        "cif",
        description="File format: cif (mmCIF), bcif (BinaryCIF), or pdb"
    ),
    chain: Optional[str] = Query(
        None,
        description="Filter to a single chain ID (e.g., 'A')"
    ),
    compress: bool = Query(
        False,
        description="Return gzip-compressed payload"
    )
) -> StructureIngestionResponse:
    """
    Fetch and parse a protein structure from RCSB PDB.
    
    This endpoint:
    - Fetches structures from RCSB PDB in mmCIF, BinaryCIF, or PDB formats
    - Parses using Biotite library
    - Extracts coordinates, B-factors, and secondary structure
    - Optionally returns gzip-compressed JSON payload
    - Validates structure size and secondary structure content
    
    Args:
        pdb_id: PDB identifier (4 alphanumeric characters)
        format: File format (cif, bcif, or pdb)
        compress: Whether to return compressed payload
    
    Returns:
        StructureIngestionResponse with parsed structure data and any warnings
    
    Raises:
        HTTPException: If structure cannot be fetched or parsed
    
    Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 15.1, 15.2, 15.4
    """
    warnings = []
    
    try:
        broker_result = ingest_structure_broker(
            input_type="pdb",
            input_value=pdb_id.strip().upper(),
            preferred_sources=["rcsb"],
            combine_sources=False,
            fmt=format,
            chain=chain,
            save_requested=False,
        )
        structure = broker_result.structure

        # Check for secondary structure warning
        secondary_structure_count = sum(
            1
            for chain_id in structure.secondary_structure
            for ss in structure.secondary_structure[chain_id]
            if ss in ['H', 'E']
        )
        ss_fraction = secondary_structure_count / structure.total_residues if structure.total_residues > 0 else 0
        if ss_fraction < 0.20:
            warnings.append(
                f"Insufficient secondary structure - dehydron detection unreliable "
                f"(found {ss_fraction:.1%}, expected ≥20%)"
            )
        
        # Calculate compressed size if requested
        compressed_size = None
        if compress:
            compressed_data = compress_structure_payload(structure)
            compressed_size = len(compressed_data)
        
        return StructureIngestionResponse(
            structure=structure,
            compressed_size_bytes=compressed_size,
            warnings=warnings
        )
    
    except StructuralError as e:
        # Use structured error response
        error_dict = e.to_dict()
        raise HTTPException(
            status_code=400,
            detail=error_dict
        )
    
    except InvalidPDBIDError as e:
        raise HTTPException(
            status_code=400,
            detail=create_error_response(e)
        )
    
    except UnsupportedFormatError as e:
        raise HTTPException(
            status_code=400,
            detail=create_error_response(e)
        )
    
    except StructureIngestionError as e:
        raise HTTPException(
            status_code=500,
            detail=create_error_response(e)
        )


@router.post("/ingest/broker", response_model=BrokerIngestResponse)
async def ingest_structure_broker_endpoint(payload: BrokerIngestRequest) -> BrokerIngestResponse:
    try:
        result = ingest_structure_broker(
            input_type=payload.input_type,
            input_value=payload.input_value.strip().upper(),
            preferred_sources=payload.preferred_sources,
            combine_sources=payload.combine_sources,
            fmt=payload.format,
            chain=payload.chain,
            save_requested=payload.save_requested,
        )
        return BrokerIngestResponse(
            request_id=result.request_id,
            structure=result.structure,
            metadata=result.metadata,
            provenance=result.provenance,
            warnings=result.warnings,
            saved_structure_id=result.saved_structure_id,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=create_error_response(e)
        )


@router.post("/save", response_model=SaveResponse)
async def save_structure(payload: SaveRequest) -> SaveResponse:
    try:
        structure_id = save_cached_ingestion(payload.request_id)
        return SaveResponse(structure_id=structure_id)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=create_error_response(e)
        )


@router.get("/cache", response_model=CacheListResponse)
async def list_cached_structures() -> CacheListResponse:
    cache = IngestionCache()
    entries = cache.list_cached()
    return CacheListResponse(entries=[CacheSummary(**entry) for entry in entries])


@router.get("/cache/{request_id}", response_model=CachePayloadResponse)
async def get_cached_structure(request_id: str) -> CachePayloadResponse:
    try:
        cache = IngestionCache()
        payload = cache.read_payload(request_id)
        return CachePayloadResponse(
            request_id=payload.get("request_id", request_id),
            structure=StructureData(**payload.get("structure", {})),
            metadata=payload.get("metadata", {}),
            provenance=payload.get("provenance", {}),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=create_error_response(e)
        )


@router.get("/ingest/compressed")
async def ingest_structure_compressed(
    pdb_id: str = Query(
        ...,
        description="PDB identifier (e.g., '1L2Y')",
        min_length=4,
        max_length=4,
        pattern="^[A-Za-z0-9]{4}$"
    ),
    format: Literal["cif", "bcif", "pdb"] = Query(
        "cif",
        description="File format: cif (mmCIF), bcif (BinaryCIF), or pdb"
    )
) -> Response:
    """
    Fetch structure and return as gzip-compressed JSON.
    
    This endpoint returns the raw gzip-compressed payload for efficient
    transfer to the frontend.
    
    Args:
        pdb_id: PDB identifier (4 alphanumeric characters)
        format: File format (cif, bcif, or pdb)
    
    Returns:
        Gzip-compressed JSON response
    
    Raises:
        HTTPException: If structure cannot be fetched or parsed
    
    Validates: Requirements 3.4, 15.4
    """
    try:
        broker_result = ingest_structure_broker(
            input_type="pdb",
            input_value=pdb_id.strip().upper(),
            preferred_sources=["rcsb"],
            combine_sources=False,
            fmt=format,
            chain=None,
            save_requested=False,
        )
        structure = broker_result.structure
        
        # Compress payload
        compressed_data = compress_structure_payload(structure)
        
        # Return as binary response with appropriate headers
        return Response(
            content=compressed_data,
            media_type="application/gzip",
            headers={
                "Content-Encoding": "gzip",
                "Content-Disposition": f"attachment; filename={pdb_id}_structure.json.gz"
            }
        )
    
    except (StructuralError, InvalidPDBIDError, UnsupportedFormatError, StructureIngestionError) as e:
        error_dict = create_error_response(e)
        if isinstance(e, StructuralError):
            error_dict = e.to_dict()
        
        raise HTTPException(
            status_code=400 if isinstance(e, (StructuralError, InvalidPDBIDError, UnsupportedFormatError)) else 500,
            detail=error_dict
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=create_error_response(e)
        )
