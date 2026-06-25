"""Database-backed structure lookup for structural alignment.

Implements the StructureLookup protocol defined in alignment_engine.py,
providing queries against dim_chain and dim_atom to resolve structures
by UniProt accession and retrieve Cα coordinates for superposition.

Uses the project's DBAdapter (data.db.DBAdapter) which supports named
parameter syntax (:param) and returns dict rows.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger(__name__)


class DatabaseConnection(Protocol):
    """Database connection protocol matching data.db.DBAdapter."""

    async def fetch_all(self, query: str, params: dict[str, Any]) -> list[dict[str, Any]]: ...
    async def fetch_one(self, query: str, params: dict[str, Any]) -> dict[str, Any] | None: ...


class StructureLookupDB:
    """Concrete implementation of StructureLookup using database queries.

    Queries dim_chain and dim_atom tables to resolve structures sharing
    a UniProt accession and retrieve Cα coordinates for Kabsch alignment.

    Args:
        db: A database adapter implementing the DatabaseConnection protocol
            (typically data.db.DBAdapter).
    """

    def __init__(self, db: DatabaseConnection) -> None:
        self._db = db

    async def get_structures_by_uniprot(self, uniprot_accession: str) -> list[str]:
        """Query dim_chain for distinct structure_ids sharing this UniProt accession.

        Args:
            uniprot_accession: UniProt accession to search for (e.g. "P01116").

        Returns:
            List of distinct structure_ids that have a chain mapped to this accession.
        """
        rows = await self._db.fetch_all(
            """
            SELECT DISTINCT structure_id
            FROM dim_chain
            WHERE uniprot_accession = :uniprot_accession
            """,
            {"uniprot_accession": uniprot_accession},
        )
        structure_ids = [row["structure_id"] for row in rows]
        logger.debug(
            "Found %d structures for UniProt %s",
            len(structure_ids),
            uniprot_accession,
        )
        return structure_ids

    async def get_ca_coordinates(
        self, structure_id: str, chain_label: str, residue_ids: list[str]
    ) -> dict[str, np.ndarray]:
        """Query dim_atom for Cα coordinates of specified residues.

        Args:
            structure_id: Structure identifier (used for logging context).
            chain_label: Chain label within the structure (used for logging context).
            residue_ids: List of residue_ids to fetch coordinates for.

        Returns:
            Dict mapping residue_id → numpy array [x, y, z] for each residue
            that has a CA atom record.
        """
        if not residue_ids:
            return {}

        rows = await self._db.fetch_all(
            """
            SELECT residue_id, x, y, z
            FROM dim_atom
            WHERE residue_id = ANY(:residue_ids)
              AND atom_name = 'CA'
            """,
            {"residue_ids": residue_ids},
        )

        coords: dict[str, np.ndarray] = {}
        for row in rows:
            coords[row["residue_id"]] = np.array(
                [row["x"], row["y"], row["z"]], dtype=np.float64
            )

        logger.debug(
            "Retrieved %d/%d Cα coordinates for %s chain %s",
            len(coords),
            len(residue_ids),
            structure_id,
            chain_label,
        )
        return coords

    async def get_chain_label_for_accession(
        self, structure_id: str, uniprot_accession: str
    ) -> str | None:
        """Resolve the chain_label for a given structure + UniProt accession.

        Queries dim_chain to find which chain in the structure maps to the
        specified UniProt accession.

        Args:
            structure_id: The structure to look in.
            uniprot_accession: The UniProt accession to find.

        Returns:
            The chain_label (auth_asym_id) or None if not found.
        """
        row = await self._db.fetch_one(
            """
            SELECT chain_label
            FROM dim_chain
            WHERE structure_id = :structure_id
              AND uniprot_accession = :uniprot_accession
            LIMIT 1
            """,
            {
                "structure_id": structure_id,
                "uniprot_accession": uniprot_accession,
            },
        )
        if row:
            return row["chain_label"]
        return None

    async def get_residue_ids_by_uniprot_positions(
        self, structure_id: str, uniprot_accession: str, positions: list[int]
    ) -> dict[int, str]:
        """Query fact_residue_alignment for residue_ids at specific UniProt positions.

        Args:
            structure_id: The structure to query alignment records for.
            uniprot_accession: UniProt accession filter.
            positions: List of UniProt positions to resolve.

        Returns:
            Dict mapping uniprot_position → residue_id for positions that have
            alignment records in this structure.
        """
        if not positions:
            return {}

        rows = await self._db.fetch_all(
            """
            SELECT fra.uniprot_position, fra.residue_id
            FROM fact_residue_alignment fra
            JOIN dim_residue dr ON fra.residue_id = dr.residue_id
            JOIN dim_chain dc ON dr.chain_id = dc.chain_id
            WHERE dc.structure_id = :structure_id
              AND fra.uniprot_accession = :uniprot_accession
              AND fra.uniprot_position = ANY(:positions)
            """,
            {
                "structure_id": structure_id,
                "uniprot_accession": uniprot_accession,
                "positions": positions,
            },
        )

        position_map: dict[int, str] = {}
        for row in rows:
            position_map[row["uniprot_position"]] = row["residue_id"]

        logger.debug(
            "Resolved %d/%d UniProt positions to residue_ids for %s (%s)",
            len(position_map),
            len(positions),
            structure_id,
            uniprot_accession,
        )
        return position_map
