"""Ingest endpoint for the Science API.

Orchestrates the full BinaryCIF-based structure ingestion pipeline:
idempotency check → download → parse → enrich → canonical keys → score →
quality flags → normalize dimensions → scope upsert → alignment sidecar.

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from data.db import DBAdapter, get_connection
from data.normalizer.core import Normalizer

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    pdb_id: str
    force_reingest: bool = False


class IngestResponse(BaseModel):
    structure_id: str
    chain_count: int
    residue_count: int
    atom_count: int
    computation_scope: dict[str, Any]
    alignment_status: str  # "pending", "skipped", "completed"
    already_existed: bool


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/ingest-full", response_model=IngestResponse)
async def ingest_full(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
) -> IngestResponse:
    """Full structure ingestion pipeline.

    Steps:
    1. Check idempotency (skip if exists and not force)
    2. Download BinaryCIF
    3. Parse with biotite
    4. Enrich metadata from RCSB Data API
    5. Generate canonical keys
    6. Compute chain scores → scope
    7. Derive quality flags (partial_backbone, low_confidence_coords, max_b_factor)
    8. Persist via Normalizer (dimensions + covalent bonds)
    9. Store computation scope (direct upsert)
    10. Fire alignment sidecar (background, non-blocking)
    11. Return summary
    """
    start = time.monotonic()

    from science.dtie.common.keys import (
        make_atom_id,
        make_chain_id,
        make_residue_id,
        make_structure_id,
    )

    pdb_id = request.pdb_id.strip()
    structure_id = make_structure_id(pdb_id=pdb_id, source="rcsb")

    # Step 1: Idempotency check
    async with get_connection() as conn:
        db = DBAdapter(conn)

        if not request.force_reingest:
            existing = await db.fetch_one(
                "SELECT structure_id FROM dim_structure WHERE structure_id = :sid",
                {"sid": structure_id},
            )
            if existing:
                # Return existing metadata
                scope_row = await db.fetch_one(
                    "SELECT * FROM structure_computation_scope WHERE structure_id = :sid",
                    {"sid": structure_id},
                )
                counts = await db.fetch_one(
                    """
                    SELECT
                        (SELECT COUNT(*) FROM dim_chain WHERE structure_id = :sid) as chains,
                        (SELECT COUNT(*) FROM dim_residue r
                         JOIN dim_chain c ON r.chain_id = c.chain_id
                         WHERE c.structure_id = :sid) as residues,
                        (SELECT COUNT(*) FROM dim_atom a
                         JOIN dim_residue r ON a.residue_id = r.residue_id
                         JOIN dim_chain c ON r.chain_id = c.chain_id
                         WHERE c.structure_id = :sid) as atoms
                    """,
                    {"sid": structure_id},
                )
                scope_dict = _scope_row_to_dict(scope_row) if scope_row else {}
                return IngestResponse(
                    structure_id=structure_id,
                    chain_count=counts["chains"] if counts else 0,
                    residue_count=counts["residues"] if counts else 0,
                    atom_count=counts["atoms"] if counts else 0,
                    computation_scope=scope_dict,
                    alignment_status="completed",
                    already_existed=True,
                )

    # Step 2: Download BinaryCIF
    from science.dtie.ingest.downloader import DownloadError, download_bcif

    try:
        download_result = await download_bcif(pdb_id)
    except DownloadError as e:
        raise HTTPException(status_code=502, detail=f"BinaryCIF download failed: {e}")

    # Step 3: Parse with biotite
    from science.dtie.ingest.parser import parse_bcif

    try:
        parsed = parse_bcif(download_result.file_path)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"BinaryCIF parse failed for {pdb_id}: {e}",
        )

    # Step 4: Enrich metadata from RCSB Data API
    from science.dtie.ingest.metadata import enrich_metadata

    metadata = await enrich_metadata(pdb_id)

    # Step 5: Generate canonical keys + build dimension payloads
    from science.dtie.common.ingest_payloads import (
        AtomDimension,
        ChainDimension,
        CovalentBondFact,
        IngestDimensionPayload,
        ResidueDimension,
        StructureDimension,
    )
    from science.dtie.common.normalizer_payloads import (
        ProvenanceContext,
        RunType,
        SourceType,
    )

    # Build chain dimensions
    chain_dims: list[ChainDimension] = []
    residue_dims: list[ResidueDimension] = []
    atom_dims: list[AtomDimension] = []

    # Build entity → uniprot mapping from metadata
    entity_uniprot: dict[str, str | None] = {}
    if metadata:
        for entity in metadata.entities:
            if entity.uniprot_accessions:
                entity_uniprot[entity.entity_id] = entity.uniprot_accessions[0]

    # Detect duplicate entities from metadata
    duplicate_entities: set[str] = set()
    if metadata:
        duplicate_entities = metadata.duplicate_entity_ids

    # Track representatives per entity for is_representative
    entity_representatives: dict[str, str] = {}

    for chain in parsed.chains:
        chain_id = make_chain_id(structure_id, chain.auth_asym_id)

        # Determine label_asym_id from metadata or from parsed
        label_asym_id = chain.label_asym_id
        if metadata and chain.auth_asym_id in metadata.auth_to_label_mapping:
            label_asym_id = metadata.auth_to_label_mapping[chain.auth_asym_id]

        is_duplicate = chain.entity_id in duplicate_entities

        # First chain per entity_id is representative
        if chain.entity_id not in entity_representatives:
            entity_representatives[chain.entity_id] = chain.auth_asym_id
        is_representative = entity_representatives[chain.entity_id] == chain.auth_asym_id

        uniprot_acc = entity_uniprot.get(chain.entity_id)

        chain_dims.append(ChainDimension(
            chain_id=chain_id,
            structure_id=structure_id,
            auth_asym_id=chain.auth_asym_id,
            label_asym_id=label_asym_id,
            entity_id=chain.entity_id,
            entity_type=chain.entity_type,
            sequence_length=len(chain.residues),
            uniprot_accession=uniprot_acc,
            is_entity_duplicate=is_duplicate,
            is_representative=is_representative,
        ))

        # Build residue dimensions for this chain
        for residue in chain.residues:
            residue_id = make_residue_id(
                structure_id,
                chain.auth_asym_id,
                residue.auth_seq_id,
                residue.insertion_code,
            )

            residue_dims.append(ResidueDimension(
                residue_id=residue_id,
                chain_id=chain_id,
                residue_index=residue.auth_seq_id,
                label_seq_id=residue.label_seq_id,
                insertion_code=residue.insertion_code,
                residue_name=residue.residue_name,
                residue_name_3=residue.residue_name_3,
                comp_id=residue.comp_id,
                parent_comp_id=residue.parent_comp_id,
                sse_code=residue.sse_code,
                is_resolved=residue.is_resolved,
                is_modified=residue.is_modified,
                max_b_factor=residue.max_b_factor,
                low_confidence_coords=residue.low_confidence_coords,
                partial_backbone=residue.partial_backbone,
            ))

            # Build atom dimensions for this residue
            for atom in residue.atoms:
                atom_id = make_atom_id(
                    structure_id,
                    chain.auth_asym_id,
                    residue.auth_seq_id,
                    atom.atom_name,
                    residue.insertion_code,
                    atom.altloc,
                )
                atom_dims.append(AtomDimension(
                    atom_id=atom_id,
                    residue_id=residue_id,
                    atom_name=atom.atom_name,
                    element=atom.element,
                    x=atom.x,
                    y=atom.y,
                    z=atom.z,
                    occupancy=atom.occupancy,
                    b_factor=atom.b_factor,
                    altloc=atom.altloc,
                    is_hetero=atom.is_hetero,
                    model_id=atom.model_id,
                ))

    # Build covalent bond facts
    covalent_bonds: list[CovalentBondFact] = []
    for bond in parsed.covalent_bonds:
        try:
            rid1 = make_residue_id(
                structure_id, bond.chain_1, bond.res_seq_1, bond.ins_code_1
            )
            rid2 = make_residue_id(
                structure_id, bond.chain_2, bond.res_seq_2, bond.ins_code_2
            )
            covalent_bonds.append(CovalentBondFact(
                residue_id_1=rid1,
                residue_id_2=rid2,
                atom_name_1=bond.atom_1,
                atom_name_2=bond.atom_2,
                bond_type=bond.bond_type,
            ))
        except ValueError as e:
            logger.warning("Skipping bond with invalid key: %s", e)

    # Get library versions for provenance
    biotite_version = _get_biotite_version()
    rcsbapi_version = _get_rcsbapi_version()

    # Step 6: Compute chain scores → scope
    from science.dtie.ingest.chain_scorer import score_chains

    scope = score_chains(parsed, metadata)

    # Step 7: Quality flags already computed during parsing (partial_backbone,
    # max_b_factor, low_confidence_coords are on ParsedResidue)

    # Step 8: Persist via Normalizer
    run_id = f"ingest_{uuid.uuid4().hex[:12]}"
    provenance = ProvenanceContext(
        run_id=run_id,
        structure_id=structure_id,
        model_version="bcif_ingest_v1",
        pipeline_name="structure_ingestion",
        run_type=RunType.ANALYSIS,
        source_type=SourceType.EMPIRICAL,
        parameters={
            "pdb_id": pdb_id,
            "file_hash": download_result.file_hash,
            "biotite_version": biotite_version,
            "rcsbapi_version": rcsbapi_version,
        },
    )

    structure_dim = StructureDimension(
        structure_id=structure_id,
        pdb_id=parsed.pdb_id,
        source="rcsb",
        method=parsed.method,
        resolution=parsed.resolution,
        r_factor=parsed.r_factor,
        r_free=parsed.r_free,
        title=parsed.title,
        organism=metadata.organism if metadata else parsed.organism,
        release_date=metadata.release_date if metadata else parsed.release_date,
        polymer_composition=parsed.polymer_composition,
        model_count=parsed.model_count,
        assembly_id=parsed.assembly_id,
    )

    payload = IngestDimensionPayload(
        provenance=provenance,
        structure=structure_dim,
        chains=chain_dims,
        residues=residue_dims,
        atoms=atom_dims,
        covalent_bonds=covalent_bonds,
        file_hash=download_result.file_hash,
        biotite_version=biotite_version,
        rcsbapi_version=rcsbapi_version,
    )

    async with get_connection() as conn:
        db = DBAdapter(conn)
        normalizer = Normalizer(db=db, caller_identity="science_api_ingest")

        try:
            await normalizer.normalize_ingest_dimensions(payload)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Normalizer write failed: {e}",
            )

        # Step 9: Store computation scope (direct upsert, not via Normalizer)
        await _upsert_computation_scope(db, structure_id, scope)

    # Step 10: Fire alignment sidecar (background, non-blocking)
    alignment_status = "pending"
    chains_with_uniprot = [
        c for c in chain_dims if c.uniprot_accession
    ]
    if chains_with_uniprot:
        background_tasks.add_task(
            _run_alignment_sidecar,
            structure_id=structure_id,
            pdb_id=pdb_id,
            chain_dims=chains_with_uniprot,
        )
    else:
        alignment_status = "skipped"

    # Step 11: Return summary
    duration_ms = (time.monotonic() - start) * 1000
    logger.info(
        "Ingested %s: chains=%d, residues=%d, atoms=%d, bonds=%d (%.0fms)",
        structure_id,
        len(chain_dims),
        len(residue_dims),
        len(atom_dims),
        len(covalent_bonds),
        duration_ms,
    )

    scope_dict = {
        "primary_chain_ids": scope.primary_chain_ids,
        "reference_chain": scope.reference_chain,
        "exclude_chain_ids": scope.exclude_chain_ids,
        "scope_source": scope.scope_source,
        "selection_reason": scope.selection_reason,
        "normalization_protocol": scope.normalization_protocol,
    }

    return IngestResponse(
        structure_id=structure_id,
        chain_count=len(chain_dims),
        residue_count=len(residue_dims),
        atom_count=len(atom_dims),
        computation_scope=scope_dict,
        alignment_status=alignment_status,
        already_existed=False,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scope_row_to_dict(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a scope DB row to a response-friendly dict."""
    return {
        "primary_chain_ids": row.get("primary_chain_ids", []),
        "reference_chain": row.get("reference_chain", ""),
        "exclude_chain_ids": row.get("exclude_chain_ids", []),
        "scope_source": row.get("scope_source", "auto"),
        "selection_reason": row.get("selection_reason", ""),
        "normalization_protocol": row.get("normalization_protocol", "graph_default"),
    }


async def _upsert_computation_scope(
    db: Any, structure_id: str, scope: Any
) -> None:
    """Upsert computation scope (configuration, direct write not via Normalizer)."""
    import json as _json

    quality_filters = None
    if scope.quality_filters:
        quality_filters = _json.dumps({
            "max_b_factor_threshold": scope.quality_filters.max_b_factor_threshold,
            "min_resolution": scope.quality_filters.min_resolution,
        })

    await db.execute(
        """
        INSERT INTO structure_computation_scope (
            structure_id, primary_chain_ids, reference_chain,
            exclude_chain_ids, scope_source, selection_reason,
            normalization_protocol, quality_filters
        ) VALUES (
            :structure_id, :primary_chain_ids, :reference_chain,
            :exclude_chain_ids, :scope_source, :selection_reason,
            :normalization_protocol, :quality_filters
        )
        ON CONFLICT (structure_id) DO UPDATE SET
            primary_chain_ids = EXCLUDED.primary_chain_ids,
            reference_chain = EXCLUDED.reference_chain,
            exclude_chain_ids = EXCLUDED.exclude_chain_ids,
            scope_source = EXCLUDED.scope_source,
            selection_reason = EXCLUDED.selection_reason,
            normalization_protocol = EXCLUDED.normalization_protocol,
            quality_filters = EXCLUDED.quality_filters,
            updated_at = NOW()
        """,
        {
            "structure_id": structure_id,
            "primary_chain_ids": scope.primary_chain_ids,
            "reference_chain": scope.reference_chain,
            "exclude_chain_ids": scope.exclude_chain_ids,
            "scope_source": scope.scope_source,
            "selection_reason": scope.selection_reason,
            "normalization_protocol": scope.normalization_protocol,
            "quality_filters": quality_filters,
        },
    )


async def _run_alignment_sidecar(
    structure_id: str,
    pdb_id: str,
    chain_dims: list[Any],
) -> None:
    """Run alignment sidecar as a background task (non-blocking).

    Fetches SIFTS mapping and persists residue alignment for each chain
    that has a UniProt accession. Failures are logged but don't propagate.
    """
    from science.dtie.alignment.alignment_engine import (
        AlignmentConfig,
        AlignmentEngine,
        ChainAlignmentInfo,
    )

    try:
        async with get_connection() as conn:
            db = DBAdapter(conn)
            normalizer = Normalizer(db=db, caller_identity="alignment_sidecar")

            from science.dtie.alignment.structure_lookup import StructureLookupDB

            structure_lookup = StructureLookupDB(db=db)

            chain_infos = [
                ChainAlignmentInfo(
                    auth_asym_id=c.auth_asym_id,
                    label_asym_id=c.label_asym_id,
                    entity_id=c.entity_id,
                    uniprot_accession=c.uniprot_accession,
                )
                for c in chain_dims
            ]

            engine = AlignmentEngine(
                normalizer=normalizer,
                structure_lookup=structure_lookup,
                config=AlignmentConfig(pipeline_name="ingest_alignment_sidecar"),
            )

            result = await engine.align_structure(
                structure_id=structure_id,
                pdb_id=pdb_id,
                chains=chain_infos,
            )

            logger.info(
                "Alignment sidecar completed for %s: chains_aligned=%d, "
                "residue_records=%d, status=%s",
                structure_id,
                result.chains_aligned,
                result.residue_records_created,
                result.status,
            )

    except Exception as e:
        logger.error(
            "Alignment sidecar failed for %s: %s",
            structure_id,
            e,
            exc_info=True,
        )


def _get_biotite_version() -> str:
    """Get the installed biotite version string."""
    try:
        import biotite
        return getattr(biotite, "__version__", "unknown")
    except ImportError:
        return "not_installed"


def _get_rcsbapi_version() -> str:
    """Get the installed rcsbapi version string."""
    try:
        import rcsbapi
        return getattr(rcsbapi, "__version__", "unknown")
    except ImportError:
        return "not_installed"
