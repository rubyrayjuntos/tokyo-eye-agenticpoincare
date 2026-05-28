# Migrated from: new (Phase 1/3 implementation) on 2026-05-27
"""Structure ingestion — populates the dimensional model from PDB/mmCIF files.

This module is the entry point for getting structure data into the governed
layer. It:
1. Parses PDB/mmCIF files (using biotite)
2. Generates canonical keys (structure_id, chain_id, residue_id)
3. Populates dim_structure, dim_chain, dim_residue, dim_atom
4. Creates a provenance_run record for the ingestion

This is the foundation that all downstream science code depends on —
GNN inference, phase runners, and dehydron detection all need the
dimensional model populated first.

See: data/DATA_POPULATION_AND_BACKFILL_STRATEGY.md
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from science.dtie.common.keys import (
    make_atom_id,
    make_chain_id,
    make_residue_id,
    make_structure_id,
)

logger = logging.getLogger(__name__)


class IngestionDB(Protocol):
    """Database protocol for ingestion writes."""

    async def execute(self, query: str, params: dict[str, Any]) -> None: ...
    async def execute_many(self, query: str, params_list: list[dict[str, Any]]) -> None: ...
    async def fetch_one(self, query: str, params: dict[str, Any]) -> dict[str, Any] | None: ...
    async def begin(self) -> Any: ...
    async def commit(self) -> None: ...
    async def rollback(self) -> None: ...


@dataclass
class IngestionResult:
    """Result of a structure ingestion."""

    structure_id: str
    run_id: str
    chains_created: int
    residues_created: int
    atoms_created: int
    warnings: list[str] = field(default_factory=list)


@dataclass
class ResidueRecord:
    """Intermediate representation of a residue during ingestion."""

    residue_id: str
    chain_id: str
    residue_index: int
    residue_name: str  # 1-letter
    residue_name_3: str  # 3-letter
    sse_code: str | None = None
    sasa: float | None = None


@dataclass
class AtomRecord:
    """Intermediate representation of an atom during ingestion."""

    atom_id: str
    residue_id: str
    atom_name: str
    element: str
    x: float
    y: float
    z: float
    occupancy: float = 1.0
    b_factor: float = 0.0


class StructureIngestor:
    """Ingests PDB/mmCIF structures into the governed dimensional model.

    Usage:
        ingestor = StructureIngestor(db=connection)
        result = await ingestor.ingest_pdb("4OBE")
        result = await ingestor.ingest_file(Path("structure.pdb"))
    """

    def __init__(self, db: IngestionDB):
        self._db = db

    async def ingest_pdb(
        self,
        pdb_id: str,
        include_atoms: bool = False,
        code_version: str | None = None,
    ) -> IngestionResult:
        """Ingest a structure from RCSB PDB by ID.

        Downloads the structure, parses it, and populates the dimensional model.

        Args:
            pdb_id: 4-character PDB identifier.
            include_atoms: Whether to populate dim_atom (large, often not needed).
            code_version: Git commit for provenance.

        Returns:
            IngestionResult with counts of created records.
        """
        import biotite.database.rcsb as rcsb
        import biotite.structure.io.pdbx as pdbx

        structure_id = make_structure_id(pdb_id=pdb_id, source="rcsb")

        # Check if already ingested
        existing = await self._db.fetch_one(
            "SELECT structure_id FROM dim_structure WHERE structure_id = :sid",
            {"sid": structure_id},
        )
        if existing:
            logger.info("Structure %s already ingested, skipping", structure_id)
            return IngestionResult(
                structure_id=structure_id,
                run_id="",
                chains_created=0,
                residues_created=0,
                atoms_created=0,
                warnings=["Already ingested"],
            )

        # Download and parse
        file_path = rcsb.fetch(pdb_id, "cif", target_path="/tmp")
        cif_file = pdbx.CIFFile.read(file_path)
        structure = pdbx.get_structure(cif_file, model=1)

        return await self._ingest_structure(
            structure=structure,
            structure_id=structure_id,
            pdb_id=pdb_id,
            source="rcsb",
            include_atoms=include_atoms,
            code_version=code_version,
        )

    async def ingest_file(
        self,
        file_path: Path,
        source: str = "user",
        name: str | None = None,
        pdb_id: str | None = None,
        include_atoms: bool = False,
        code_version: str | None = None,
    ) -> IngestionResult:
        """Ingest a structure from a local file.

        Args:
            file_path: Path to PDB or mmCIF file.
            source: Source type ('user', 'rcsb', 'alphafold').
            name: Human-readable name (for user uploads).
            pdb_id: PDB ID if known.
            include_atoms: Whether to populate dim_atom.
            code_version: Git commit for provenance.

        Returns:
            IngestionResult with counts.
        """
        import biotite.structure.io.pdb as pdb_io
        import biotite.structure.io.pdbx as pdbx

        # Generate structure_id
        if source == "rcsb" and pdb_id:
            structure_id = make_structure_id(pdb_id=pdb_id, source="rcsb")
        elif source == "user":
            content_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
            file_name = name or file_path.stem
            structure_id = make_structure_id(
                source="user", name=file_name, content_hash=content_hash
            )
        else:
            content_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
            structure_id = make_structure_id(
                source="user", name=file_path.stem, content_hash=content_hash
            )

        # Parse based on file extension
        suffix = file_path.suffix.lower()
        if suffix in (".cif", ".mmcif"):
            cif_file = pdbx.CIFFile.read(str(file_path))
            structure = pdbx.get_structure(cif_file, model=1)
        elif suffix == ".pdb":
            pdb_file = pdb_io.PDBFile.read(str(file_path))
            structure = pdb_io.get_structure(pdb_file, model=1)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        return await self._ingest_structure(
            structure=structure,
            structure_id=structure_id,
            pdb_id=pdb_id,
            source=source,
            include_atoms=include_atoms,
            code_version=code_version,
        )

    async def _ingest_structure(
        self,
        structure: Any,  # biotite AtomArray
        structure_id: str,
        pdb_id: str | None,
        source: str,
        include_atoms: bool,
        code_version: str | None,
    ) -> IngestionResult:
        """Core ingestion logic — populates dimensional tables."""
        import biotite.structure as struc

        run_id = f"ingest_{structure_id}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        warnings: list[str] = []

        try:
            await self._db.begin()

            # 1. Create dim_structure FIRST (provenance_run has FK to it)
            await self._db.execute(
                """
                INSERT INTO dim_structure (structure_id, pdb_id, source)
                VALUES (:structure_id, :pdb_id, :source)
                ON CONFLICT (structure_id) DO NOTHING
                """,
                {"structure_id": structure_id, "pdb_id": pdb_id, "source": source},
            )

            # 2. Create provenance run (now dim_structure exists)
            await self._db.execute(
                """
                INSERT INTO provenance_run (
                    run_id, structure_id, model_version, pipeline_name,
                    run_type, source_type, code_version, started_at
                ) VALUES (
                    :run_id, :structure_id, :model_version, :pipeline_name,
                    :run_type, :source_type, :code_version, :started_at
                )
                """,
                {
                    "run_id": run_id,
                    "structure_id": structure_id,
                    "model_version": "ingestion-v1",
                    "pipeline_name": "structure_ingestion",
                    "run_type": "analysis",
                    "source_type": "external",
                    "code_version": code_version,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                },
            )

            # 3. Extract chains and residues
            chains_created = 0
            residues_created = 0
            atoms_created = 0

            chain_ids_seen: set[str] = set()
            residue_records: list[dict[str, Any]] = []
            atom_records: list[dict[str, Any]] = []

            # Get per-residue data
            if hasattr(structure, 'chain_id'):
                unique_chains = set(structure.chain_id)
            else:
                unique_chains = {"A"}

            for chain_label in sorted(unique_chains):
                chain_id = make_chain_id(structure_id, chain_label)

                if chain_id not in chain_ids_seen:
                    await self._db.execute(
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
                    chain_ids_seen.add(chain_id)
                    chains_created += 1

                # Get residues for this chain
                chain_mask = structure.chain_id == chain_label
                chain_atoms = structure[chain_mask]

                # Get unique residues
                if hasattr(chain_atoms, 'res_id'):
                    residue_starts = struc.get_residue_starts(chain_atoms)

                    for start_idx in residue_starts:
                        res_idx = int(chain_atoms.res_id[start_idx])
                        res_name = chain_atoms.res_name[start_idx] if hasattr(chain_atoms, 'res_name') else "UNK"

                        residue_id = make_residue_id(structure_id, chain_label, res_idx)

                        # Map 3-letter to 1-letter
                        one_letter = _three_to_one(res_name)

                        residue_records.append({
                            "residue_id": residue_id,
                            "chain_id": chain_id,
                            "residue_index": res_idx,
                            "residue_name": one_letter,
                            "residue_name_3": res_name,
                        })

            # 4. Batch insert residues
            if residue_records:
                await self._db.execute_many(
                    """
                    INSERT INTO dim_residue (residue_id, chain_id, residue_index, residue_name, residue_name_3)
                    VALUES (:residue_id, :chain_id, :residue_index, :residue_name, :residue_name_3)
                    ON CONFLICT (residue_id) DO NOTHING
                    """,
                    residue_records,
                )
                residues_created = len(residue_records)

            await self._db.commit()

        except Exception as e:
            await self._db.rollback()
            logger.error("Ingestion failed for %s: %s", structure_id, e)
            raise

        logger.info(
            "Ingested structure %s: %d chains, %d residues",
            structure_id,
            chains_created,
            residues_created,
        )

        return IngestionResult(
            structure_id=structure_id,
            run_id=run_id,
            chains_created=chains_created,
            residues_created=residues_created,
            atoms_created=atoms_created,
            warnings=warnings,
        )


# Standard amino acid 3-letter to 1-letter mapping
_AA_MAP = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def _three_to_one(three_letter: str) -> str:
    """Convert 3-letter amino acid code to 1-letter."""
    return _AA_MAP.get(three_letter.upper(), "X")
