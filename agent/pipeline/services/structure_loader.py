"""Helpers for reconstructing StructureData from warehouse tables."""

from __future__ import annotations

from collections import OrderedDict

import sqlalchemy as sa

from gosp.models.data_models import Atom, Chain, Residue, StructureData


_STRUCTURE_SQL = sa.text(
    """
    SELECT pdb_id, resolution, num_models, total_residues, total_atoms
    FROM dim_structure
    WHERE structure_id = :structure_id
    """
)


_ATOM_ROWS_SQL = sa.text(
    """
    SELECT c.chain_label,
           c.sequence,
           r.residue_index,
           r.residue_name,
           r.sse_code,
           a.atom_name,
           a.element,
           a.occupancy,
           a.b_factor,
           a.x,
           a.y,
           a.z
    FROM dim_chain c
    JOIN dim_residue r ON r.chain_id = c.chain_id
    JOIN dim_atom a ON a.residue_id = r.residue_id
    WHERE c.structure_id = :structure_id
    ORDER BY c.chain_label, r.residue_index, a.atom_name
    """
)


async def load_structure_from_db(conn, structure_id: str) -> StructureData:
    """Rebuild a StructureData payload from dim_* tables."""
    structure_row = (
        await conn.execute(_STRUCTURE_SQL, {"structure_id": structure_id})
    ).mappings().first()
    if structure_row is None:
        raise ValueError(f"Structure not found: {structure_id}")

    atom_rows = (
        await conn.execute(_ATOM_ROWS_SQL, {"structure_id": structure_id})
    ).mappings().all()
    if not atom_rows:
        raise ValueError(f"No atom rows found for structure: {structure_id}")

    chains_map: OrderedDict[str, dict] = OrderedDict()
    atom_serial = 1

    for row in atom_rows:
        chain_id = row["chain_label"]
        residue_index = int(row["residue_index"])

        chain_state = chains_map.setdefault(
            chain_id,
            {
                "sequence": row["sequence"] or "",
                "residues": OrderedDict(),
                "sse": OrderedDict(),
            },
        )

        residue_state = chain_state["residues"].setdefault(
            residue_index,
            {
                "residue_name": row["residue_name"],
                "atoms": [],
            },
        )
        chain_state["sse"][residue_index] = row["sse_code"] or "C"

        residue_state["atoms"].append(
            Atom(
                atom_id=atom_serial,
                atom_name=row["atom_name"],
                residue_name=row["residue_name"],
                chain_id=chain_id,
                residue_id=residue_index,
                x=float(row["x"]),
                y=float(row["y"]),
                z=float(row["z"]),
                occupancy=float(row["occupancy"] or 0.0),
                b_factor=float(row["b_factor"] or 0.0),
                element=(row["element"] or "").strip() or row["atom_name"][0],
            )
        )
        atom_serial += 1

    chains = []
    secondary_structure = {}
    for chain_id, chain_state in chains_map.items():
        residues = []
        ordered_sse = []
        for residue_index, residue_state in chain_state["residues"].items():
            residues.append(
                Residue(
                    residue_id=residue_index,
                    residue_name=residue_state["residue_name"],
                    atoms=residue_state["atoms"],
                )
            )
            ordered_sse.append(chain_state["sse"].get(residue_index, "C"))

        chains.append(
            Chain(
                chain_id=chain_id,
                residues=residues,
                sequence=chain_state["sequence"],
            )
        )
        secondary_structure[chain_id] = ordered_sse

    return StructureData(
        pdb_id=structure_row["pdb_id"],
        num_models=int(structure_row["num_models"] or 1),
        resolution=(
            float(structure_row["resolution"])
            if structure_row["resolution"] is not None
            else None
        ),
        chains=chains,
        total_residues=int(structure_row["total_residues"]),
        total_atoms=int(structure_row["total_atoms"]),
        secondary_structure=secondary_structure,
        missing_residues=None,
    )