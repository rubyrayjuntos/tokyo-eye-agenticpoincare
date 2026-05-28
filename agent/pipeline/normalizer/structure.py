"""Write StructureData to the warehouse dimension tables.

Write path:
    StructureData  →  dim_structure
    Chain          →  dim_chain
    Residue        →  dim_residue
    Atom           →  dim_atom

All primary keys for dim_chain / dim_residue / dim_atom are composite UUIDs
built from the structure_id plus the natural identifier so they are stable and
idempotent across re-ingestions.
"""
import uuid
from typing import Optional

import sqlalchemy as sa

from gosp.models.data_models import StructureData


# ---------------------------------------------------------------------------
# SQL statements
# ---------------------------------------------------------------------------

_INSERT_STRUCTURE = sa.text("""
    INSERT INTO dim_structure
        (structure_id, pdb_id, resolution, num_models, total_residues, total_atoms)
    VALUES
        (:structure_id, :pdb_id, :resolution, :num_models, :total_residues, :total_atoms)
    ON CONFLICT (pdb_id) DO NOTHING
    RETURNING structure_id
""")

_SELECT_STRUCTURE_ID = sa.text("""
    SELECT structure_id FROM dim_structure WHERE pdb_id = :pdb_id
""")

_INSERT_CHAIN = sa.text("""
    INSERT INTO dim_chain
        (chain_id, structure_id, chain_label, sequence)
    VALUES
        (:chain_id, :structure_id, :chain_label, :sequence)
    ON CONFLICT (chain_id) DO NOTHING
""")

_INSERT_RESIDUE = sa.text("""
    INSERT INTO dim_residue
        (residue_id, chain_id, residue_index, residue_name, sse_code)
    VALUES
        (:residue_id, :chain_id, :residue_index, :residue_name, :sse_code)
    ON CONFLICT (residue_id) DO NOTHING
""")

_INSERT_ATOM = sa.text("""
    INSERT INTO dim_atom
        (atom_id, residue_id, atom_name, element, occupancy, b_factor, x, y, z)
    VALUES
        (:atom_id, :residue_id, :atom_name, :element, :occupancy, :b_factor, :x, :y, :z)
    ON CONFLICT (atom_id) DO NOTHING
""")


# ---------------------------------------------------------------------------
# Stable composite UUID helpers
# ---------------------------------------------------------------------------

def _dim_uuid(namespace: uuid.UUID, *parts: str) -> str:
    """Return a deterministic UUID v5 from the given namespace and name parts."""
    name = "|".join(parts)
    return str(uuid.uuid5(namespace, name))


# Use a fixed namespace so IDs are reproducible for the same logical entity.
_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # DNS namespace


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def normalize_structure(
    conn,
    structure_id: str,
    structure: StructureData,
) -> str:
    """Insert StructureData into the four dimension tables.

    Parameters
    ----------
    conn:
        Async SQLAlchemy connection (or any object with an ``execute`` coroutine).
    structure_id:
        UUID v4 generated at ingest time — the root key for all related rows.
    structure:
        Parsed and validated StructureData object.

    Returns
    -------
    str
        The unchanged ``structure_id``.
    """
    # ------------------------------------------------------------------
    # dim_structure (idempotent: ON CONFLICT (pdb_id) DO NOTHING)
    # ------------------------------------------------------------------
    result = await conn.execute(_INSERT_STRUCTURE, {
        "structure_id":  structure_id,
        "pdb_id":        structure.pdb_id,
        "resolution":    structure.resolution,
        "num_models":    structure.num_models,
        "total_residues": structure.total_residues,
        "total_atoms":   structure.total_atoms,
    })
    row = result.fetchone()
    if row is None:
        # Row already existed — fetch the canonical structure_id
        existing = await conn.execute(_SELECT_STRUCTURE_ID, {"pdb_id": structure.pdb_id})
        structure_id = str(existing.scalar_one())

    # Materialise the secondary-structure map for residue-level SSE lookup.
    sse_map: dict[str, list[str]] = structure.secondary_structure or {}

    # ------------------------------------------------------------------
    # dim_chain / dim_residue / dim_atom
    # ------------------------------------------------------------------
    for chain in structure.chains:
        chain_uuid = _dim_uuid(_NS, structure_id, chain.chain_id)

        await conn.execute(_INSERT_CHAIN, {
            "chain_id":    chain_uuid,
            "structure_id": structure_id,
            "chain_label": chain.chain_id,
            "sequence":    chain.sequence,
        })

        chain_sse: list[str] = sse_map.get(chain.chain_id, [])

        for residue in chain.residues:
            residue_uuid = _dim_uuid(_NS, structure_id, chain.chain_id, str(residue.residue_id))

            # residue_id is 1-based sequence number; map to 0-based index for SSE list
            sse_index = residue.residue_id - 1
            sse_code: Optional[str] = (
                chain_sse[sse_index] if 0 <= sse_index < len(chain_sse) else None
            )

            await conn.execute(_INSERT_RESIDUE, {
                "residue_id":    residue_uuid,
                "chain_id":      chain_uuid,
                "residue_index": residue.residue_id,
                "residue_name":  residue.residue_name,
                "sse_code":      sse_code,
            })

            for atom in residue.atoms:
                atom_uuid = _dim_uuid(
                    _NS, structure_id, chain.chain_id,
                    str(residue.residue_id), str(atom.atom_id),
                )

                await conn.execute(_INSERT_ATOM, {
                    "atom_id":   atom_uuid,
                    "residue_id": residue_uuid,
                    "atom_name": atom.atom_name,
                    "element":   atom.element,
                    "occupancy": atom.occupancy,
                    "b_factor":  atom.b_factor,
                    "x":         atom.x,
                    "y":         atom.y,
                    "z":         atom.z,
                })

    return structure_id
