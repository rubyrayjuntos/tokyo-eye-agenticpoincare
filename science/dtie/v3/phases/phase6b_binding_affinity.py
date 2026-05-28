"""
## CONTRACT

### Reads
- phase6a_result                   <- written by phase6a_virtual_screening.py
  - hits                          : list of hit dicts with smiles, mol, pharmacophore_center
- GDP receptor PDB                 <- user-supplied protein structure
- GTP receptor PDB (optional)      <- user-supplied alternate-state structure

### Writes
- phase6b_result (list of docked hit dicts)
  - delta_G                       : float     GDP-state Vina affinity (kcal/mol)
  - gtp_delta_G                   : float|None GTP-state Vina affinity (kcal/mol)
  - docking_method                : str       "vina"
  - (all Phase 6a fields passed through)

### GNN fields used directly
  - None (operates on Phase 6a screening hits + receptor PDBs)

### What this phase adds
  - AutoDock Vina docking against GDP-state receptor
  - Optional GTP-state docking (for Phase 6d selectivity)
  - Receptor alignment (GTP → GDP frame) when dual-state docking
  - Affinity-ranked output (most negative delta_G first)
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from . import utils

logger_p6b = logging.getLogger("DTIE_Phase6b")


def run_vina_docking(
    mol: Chem.Mol,
    receptor_pdb_path: str,
    center_xyz: List[float],
    box_size: Tuple[float, float, float] = (20.0, 20.0, 20.0),
    exhaustiveness: int = 8,
    num_modes: int = 5,
    reference_pdb: Optional[str] = None,
) -> float:
    """
    Run AutoDock Vina docking for a single ligand against a receptor.

    Returns best binding affinity in kcal/mol (negative = favorable).
    """
    vina_bin = utils._which("vina")
    if vina_bin is None:
        raise RuntimeError("vina binary not found")

    if mol is None:
        raise ValueError("Cannot dock null molecule")

    with tempfile.TemporaryDirectory() as tmpdir:
        lig_pdbqt = os.path.join(tmpdir, "ligand.pdbqt")
        rec_pdbqt = os.path.join(tmpdir, "receptor.pdbqt")
        out_pdbqt = os.path.join(tmpdir, "out.pdbqt")

        # If the receptor is in a different coordinate frame than the reference
        # (GDP structure), align it first so pharmacophore centers map correctly.
        docking_pdb = receptor_pdb_path
        if reference_pdb and reference_pdb != receptor_pdb_path:
            aligned_pdb = os.path.join(tmpdir, "aligned.pdb")
            if utils._align_pdb_to_reference(
                receptor_pdb_path, reference_pdb, aligned_pdb
            ):
                docking_pdb = aligned_pdb
            else:
                raise RuntimeError("Receptor alignment failed")

        utils._write_ligand_pdbqt_meeko(mol, lig_pdbqt)

        # Receptor prep
        utils._prepare_receptor_pdbqt(docking_pdb, rec_pdbqt)

        cx, cy, cz = center_xyz
        sx, sy, sz = box_size

        cmd = [
            vina_bin,
            "--receptor",
            rec_pdbqt,
            "--ligand",
            lig_pdbqt,
            "--out",
            out_pdbqt,
            "--center_x",
            str(cx),
            "--center_y",
            str(cy),
            "--center_z",
            str(cz),
            "--size_x",
            str(sx),
            "--size_y",
            str(sy),
            "--size_z",
            str(sz),
            "--exhaustiveness",
            str(exhaustiveness),
            "--num_modes",
            str(num_modes),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            raise RuntimeError("Vina timed out")
        except Exception as e:
            raise RuntimeError(f"Vina execution error: {e}") from e

        affinity = utils._parse_vina_affinity(result.stdout)
        if affinity is None:
            raise RuntimeError("Vina affinity parse failed")
        return affinity


# ---------------------------------------------------------------------------
# Phase 6b entry point
# ---------------------------------------------------------------------------


def execute_phase_6b_binding_affinity(
    phase_input: "Phase6bInput",
) -> list:
    """
    Phase 6b: Binding Affinity Estimation via AutoDock Vina.

    Docks each hit against the GDP-state receptor (protein_pdb_path).
    If gtp_pdb_path is supplied, also docks against the GTP-state receptor
    in the same loop so Phase 6d only needs to compute ratios (no re-docking).

    Requires valid docking setup for each hit; fails fast on docking errors.
    """
    screening_hits = phase_input.phase6a_result.hits
    protein_pdb_path = phase_input.protein_pdb
    gtp_pdb_path = phase_input.gtp_pdb_path
    method = phase_input.method
    top_n = phase_input.top_n
    hits = screening_hits[:top_n]
    out = []

    for hit in hits:
        mol = hit.get("mol")
        center_xyz = hit.get("pharmacophore_center", [0.0, 0.0, 0.0])

        gdp_dg = run_vina_docking(mol, protein_pdb_path, center_xyz)

        gtp_dg: Optional[float] = None
        if gtp_pdb_path:
            gtp_dg = run_vina_docking(
                mol,
                gtp_pdb_path,
                center_xyz,
                reference_pdb=protein_pdb_path,  # align GTP → GDP frame
            )

        out.append(
            {
                **hit,
                "docking_method": method,
                "delta_G": gdp_dg,
                "gtp_delta_G": gtp_dg,
            }
        )

    # Sort by GDP affinity (best = most negative)
    out.sort(key=lambda x: x["delta_G"])

    logger_p6b.info(
        f"Phase 6b: {len(out)}/{len(out)} hits docked successfully | "
        f"best delta_G = {out[0]['delta_G']:.2f} kcal/mol"
        if out
        else "Phase 6b: no hits to dock"
    )
    return out
