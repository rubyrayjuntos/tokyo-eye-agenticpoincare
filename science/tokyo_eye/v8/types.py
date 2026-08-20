"""v8-local atom / residue record types (vendored; not imported from v7/dtie)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AtomRecord:
    atom_name: str
    element: str
    coord: np.ndarray  # shape (3,)
    parent_residue_name: str = ""


@dataclass(frozen=True)
class ResidueRecord:
    chain_label: str
    residue_index: int
    residue_name: str
    atoms: tuple[AtomRecord, ...]
    residue_id: str = ""

    def get_atom(self, name: str) -> AtomRecord | None:
        key = name.strip().upper()
        for atom in self.atoms:
            if atom.atom_name.strip().upper() == key:
                return atom
        return None


__all__ = ["AtomRecord", "ResidueRecord"]
