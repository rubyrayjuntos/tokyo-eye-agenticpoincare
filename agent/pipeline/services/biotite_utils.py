"""Shared Biotite AtomArray construction from StructureData.

Eliminates duplication between dehydron_detection.py and physics_kernel.py.
"""
from __future__ import annotations

import numpy as np
import biotite.structure as struc

from gosp.models.data_models import StructureData


def build_biotite_structure(structure_data: StructureData) -> struc.AtomArray:
    """Construct a Biotite AtomArray from the detailed StructureData model."""
    total_atoms = structure_data.total_atoms
    atom_array = struc.AtomArray(total_atoms)

    atom_array.add_annotation("occupancy", dtype=np.float32)
    atom_array.add_annotation("b_factor", dtype=np.float32)

    atom_index = 0
    for chain in structure_data.chains:
        for residue in chain.residues:
            for atom in residue.atoms:
                atom_array.atom_name[atom_index] = atom.atom_name
                atom_array.res_name[atom_index] = atom.residue_name
                atom_array.chain_id[atom_index] = atom.chain_id
                atom_array.res_id[atom_index] = atom.residue_id
                atom_array.coord[atom_index] = [atom.x, atom.y, atom.z]
                atom_array.occupancy[atom_index] = atom.occupancy
                atom_array.b_factor[atom_index] = atom.b_factor
                atom_array.element[atom_index] = atom.element.upper()
                atom_index += 1

    return atom_array
