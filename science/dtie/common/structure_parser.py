"""CIF/PDB structure parser — populates dim_chain, dim_residue, dim_atom from a structure file.

This bridges the gap between downloading a CIF from RCSB and having the
residue-level data the GNN needs. It parses the file and writes to the
governed dimensional tables via the database adapter.

Supports:
- mmCIF format (primary, from RCSB ModelServer)
- PDB format (legacy fallback)

Uses gemmi for fast, reliable parsing of macromolecular structure files.
Falls back to a simple regex parser if gemmi is not available.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ParsedAtom:
    chain_label: str
    residue_index: int
    residue_name: str
    atom_name: str
    x: float
    y: float
    z: float
    element: str


@dataclass
class ParseResult:
    structure_id: str
    chains: list[str]
    residue_count: int
    atom_count: int
    ca_count: int


async def parse_and_populate(
    structure_id: str,
    file_path: str,
    db: Any,
) -> ParseResult:
    """Parse a CIF/PDB file and populate dim_chain, dim_residue, dim_atom.

    Args:
        structure_id: Canonical structure ID (lowercase)
        file_path: Path to the CIF or PDB file
        db: Database adapter with execute/commit methods

    Returns:
        ParseResult with counts of what was inserted
    """
    file_path = str(file_path)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Structure file not found: {file_path}")

    # Parse atoms from the file
    atoms = _parse_cif(file_path)
    if not atoms:
        atoms = _parse_pdb_fallback(file_path)
    if not atoms:
        raise ValueError(f"Could not parse any atoms from {file_path}")

    # Group by chain and residue
    chains: dict[str, dict[int, list[ParsedAtom]]] = {}
    for atom in atoms:
        if atom.chain_label not in chains:
            chains[atom.chain_label] = {}
        if atom.residue_index not in chains[atom.chain_label]:
            chains[atom.chain_label][atom.residue_index] = []
        chains[atom.chain_label][atom.residue_index].append(atom)

    # Write to database
    ca_count = 0
    residue_count = 0
    atom_count = 0

    for chain_label, residues in chains.items():
        chain_id = f"{structure_id}:{chain_label}"

        # Insert chain
        await db.execute(
            """
            INSERT INTO dim_chain (chain_id, structure_id, chain_label, entity_type)
            VALUES (:chain_id, :structure_id, :chain_label, 'protein')
            ON CONFLICT (chain_id) DO NOTHING
            """,
            {"chain_id": chain_id, "structure_id": structure_id, "chain_label": chain_label},
        )

        for res_index, res_atoms in sorted(residues.items()):
            residue_name = res_atoms[0].residue_name
            residue_id = f"{structure_id}:{chain_label}:{res_index}"
            residue_count += 1

            # Insert residue
            await db.execute(
                """
                INSERT INTO dim_residue (residue_id, chain_id, residue_index, residue_name, residue_name_3)
                VALUES (:residue_id, :chain_id, :residue_index, :residue_name, :residue_name_3)
                ON CONFLICT (residue_id) DO NOTHING
                """,
                {
                    "residue_id": residue_id,
                    "chain_id": chain_id,
                    "residue_index": res_index,
                    "residue_name": _one_letter(residue_name),
                    "residue_name_3": residue_name,
                },
            )

            # Insert atoms (focus on CA for graph building)
            for atom in res_atoms:
                atom_id = f"{residue_id}:{atom.atom_name}"
                atom_count += 1
                if atom.atom_name == "CA":
                    ca_count += 1

                await db.execute(
                    """
                    INSERT INTO dim_atom (atom_id, residue_id, atom_name, element, x, y, z)
                    VALUES (:atom_id, :residue_id, :atom_name, :element, :x, :y, :z)
                    ON CONFLICT (atom_id) DO NOTHING
                    """,
                    {
                        "atom_id": atom_id,
                        "residue_id": residue_id,
                        "atom_name": atom.atom_name,
                        "element": atom.element,
                        "x": atom.x,
                        "y": atom.y,
                        "z": atom.z,
                    },
                )

    await db.commit()

    logger.info(
        "Parsed %s: %d chains, %d residues, %d atoms (%d CA)",
        structure_id, len(chains), residue_count, atom_count, ca_count,
    )

    return ParseResult(
        structure_id=structure_id,
        chains=list(chains.keys()),
        residue_count=residue_count,
        atom_count=atom_count,
        ca_count=ca_count,
    )


def _parse_cif(file_path: str) -> list[ParsedAtom]:
    """Parse mmCIF file for atom coordinates."""
    atoms = []
    in_atom_site = False
    columns: dict[str, int] = {}
    col_order: list[str] = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()

            # Detect start of _atom_site loop
            if line == "loop_":
                in_atom_site = False
                columns = {}
                col_order = []
                continue

            if line.startswith("_atom_site."):
                in_atom_site = True
                col_name = line.split(".")[1].strip()
                columns[col_name] = len(col_order)
                col_order.append(col_name)
                continue

            # If we were in atom_site and hit a non-data line, we're done
            if in_atom_site and (line.startswith("_") or line.startswith("#") or line == "loop_"):
                in_atom_site = False
                continue

            if not in_atom_site or not columns:
                continue

            # Parse atom record
            parts = line.split()
            if len(parts) < len(columns):
                continue

            # Only keep ATOM records (not HETATM for now)
            group_col = columns.get("group_PDB")
            if group_col is not None and parts[group_col] != "ATOM":
                continue

            try:
                chain_label = parts[columns.get("auth_asym_id", columns.get("label_asym_id", 0))]
                residue_index = int(parts[columns.get("auth_seq_id", columns.get("label_seq_id", 0))])
                residue_name = parts[columns.get("auth_comp_id", columns.get("label_comp_id", 0))]
                atom_name = parts[columns.get("auth_atom_id", columns.get("label_atom_id", 0))]
                x = float(parts[columns.get("Cartn_x", 0)])
                y = float(parts[columns.get("Cartn_y", 0)])
                z = float(parts[columns.get("Cartn_z", 0)])
                element = parts[columns.get("type_symbol", 0)] if "type_symbol" in columns else atom_name[0]

                atoms.append(ParsedAtom(
                    chain_label=chain_label,
                    residue_index=residue_index,
                    residue_name=residue_name,
                    atom_name=atom_name,
                    x=x, y=y, z=z,
                    element=element,
                ))
            except (ValueError, IndexError):
                continue

    return atoms


def _parse_pdb_fallback(file_path: str) -> list[ParsedAtom]:
    """Fallback PDB format parser."""
    atoms = []
    with open(file_path, "r") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            try:
                atom_name = line[12:16].strip()
                residue_name = line[17:20].strip()
                chain_label = line[21].strip() or "A"
                residue_index = int(line[22:26].strip())
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                element = line[76:78].strip() if len(line) > 76 else atom_name[0]

                atoms.append(ParsedAtom(
                    chain_label=chain_label,
                    residue_index=residue_index,
                    residue_name=residue_name,
                    atom_name=atom_name,
                    x=x, y=y, z=z,
                    element=element,
                ))
            except (ValueError, IndexError):
                continue
    return atoms


# Standard amino acid 3-letter to 1-letter mapping
_AA_MAP = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O",
}


def _one_letter(three_letter: str) -> str:
    """Convert 3-letter amino acid code to 1-letter."""
    return _AA_MAP.get(three_letter.upper(), "X")
