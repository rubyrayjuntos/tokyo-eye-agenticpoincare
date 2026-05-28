"""
## CONTRACT

### Reads
- phase5_output                    <- written by phase5_pharmacophore.py
  - pharmacophores                : list of Pharmacophore dicts with center_xyz,
                                    hbond_donors, hbond_acceptors, atom_type_constraints
- compound library (.sdf)          <- user-supplied screening library on disk

### Writes
- phase6a_result (list of hit dicts)
  - smiles                        : str       canonical SMILES
  - mol                           : RDKit Mol 3D conformer
  - pharm_score                   : float     pharmacophore complementarity [0, 1]
  - mw, logp                     : float     molecular descriptors
  - pharmacophore_idx             : int       index of matched pharmacophore
  - pharmacophore_center          : [3] float center coordinates

### GNN fields used directly
  - None (operates on Phase 5 pharmacophore output)

### What this phase adds
  - 2D pharmacophore feature complementarity scoring (H-bond, hydrophobic, aromatic)
  - Lipinski Rule-of-Five pre-filter
  - Per-compound 3D conformer generation (ETKDGv3 + MMFF)
  - Deduplication by SMILES across pharmacophore centers (best score kept)
"""

from __future__ import annotations

import logging
import numpy as np
from pathlib import Path
from typing import List

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors

logger_p6a = logging.getLogger("DTIE_Phase6a")


def _pharmacophore_match_score(mol: Chem.Mol, pharm: dict) -> float:
    """
    2D pharmacophore feature complementarity score (0.0 – 1.0, higher = better).

    Pocket donors  ↔ ligand acceptors  (complementary H-bond partners)
    Pocket acceptors ↔ ligand donors
    Pocket hydrophobic count → logP proxy
    Pocket aromatic count → n_aromatic_rings
    """
    n_pocket_don = len(pharm.get("hbond_donors", []))
    n_pocket_acc = len(pharm.get("hbond_acceptors", []))
    pocket_hydro = (pharm.get("atom_type_constraints") or {}).get("hydrophobic", 0)

    mol_hbd = rdMolDescriptors.CalcNumHBD(mol)
    mol_hba = rdMolDescriptors.CalcNumHBA(mol)
    logp = Descriptors.MolLogP(mol)
    n_arom = rdMolDescriptors.CalcNumAromaticRings(mol)

    # H-bond complementarity: count satisfied interactions, normalise
    hbond_possible = max(n_pocket_don + n_pocket_acc, 1)
    hbond_score = (
        min(n_pocket_don, mol_hba) + min(n_pocket_acc, mol_hbd)
    ) / hbond_possible

    # Hydrophobic match: pocket_hydro/50 ≈ normalised pocket hydrophobicity
    # logP clamped to [0,5] → ligand hydrophobicity proxy
    pocket_h_norm = min(pocket_hydro / 50.0, 1.0)
    mol_h_norm = min(max(logp, 0.0) / 5.0, 1.0)
    hydro_score = 1.0 - abs(pocket_h_norm - mol_h_norm)

    # Aromatic: 3+ aromatic rings is the upper reference
    arom_score = min(n_arom / 3.0, 1.0)

    return hbond_score * 0.5 + hydro_score * 0.3 + arom_score * 0.2


def execute_phase_6a_virtual_screening(
    phase_input: "Phase6aInput",
) -> list:
    """
    Phase 6a: Virtual Screening.

    2D pharmacophore feature complementarity scoring + Lipinski Ro5.
    Pocket features (H-bond donors/acceptors, hydrophobic count) from Phase 5
    are matched against molecular features to rank compounds for docking.

    Compounds are scored independently against each unique pharmacophore center,
    deduplicated by SMILES (best score kept), then sorted globally by score.
    """
    pharmacophores = phase_input.phase5_result.pharmacophores
    compound_library_path = phase_input.library_path
    top_n = phase_input.top_n
    if not Path(compound_library_path).exists():
        raise FileNotFoundError(f"Compound library not found: {compound_library_path}")

    # Deduplicate pharmacophores by center_xyz to avoid redundant passes
    seen_centers: set = set()
    unique_pharms: list = []
    for p in pharmacophores:
        key = tuple(round(x, 1) for x in p["center_xyz"])
        if key not in seen_centers:
            seen_centers.add(key)
            unique_pharms.append(p)

    logger_p6a.info(
        f"Phase 6a: {len(pharmacophores)} pharmacophores → "
        f"{len(unique_pharms)} unique centers"
    )

    # One pass through the library per unique pharmacophore; deduplicate by SMILES
    best_hits: dict = {}  # smiles → best-scoring hit dict

    for pharm_idx, pharm_spec in enumerate(unique_pharms):
        supplier = Chem.SDMolSupplier(compound_library_path, removeHs=False)
        n_passing = 0

        for mol in supplier:
            if mol is None:
                continue

            # Lipinski Ro5 pre-filter
            mw = rdMolDescriptors.CalcExactMolWt(mol)
            lp = Descriptors.MolLogP(mol)
            hbd = rdMolDescriptors.CalcNumHBD(mol)
            hba = rdMolDescriptors.CalcNumHBA(mol)
            if mw > 500 or lp > 5 or hbd > 5 or hba > 10:
                continue

            smi = Chem.MolToSmiles(mol)
            score = _pharmacophore_match_score(mol, pharm_spec)

            if smi not in best_hits or score > best_hits[smi]["pharm_score"]:
                # Generate a single 3D conformer for docking — once per SMILES
                mol_3d = Chem.AddHs(mol)
                if mol_3d.GetNumConformers() == 0:
                    if AllChem.EmbedMolecule(mol_3d, AllChem.ETKDGv3()) == -1:
                        continue
                    AllChem.MMFFOptimizeMolecule(mol_3d)

                best_hits[smi] = {
                    "smiles": smi,
                    "mol": mol_3d,
                    "pharm_score": score,
                    "mw": mw,
                    "logp": lp,
                    "pharmacophore_idx": pharm_idx,
                    "pharmacophore_center": pharm_spec["center_xyz"],
                }
                n_passing += 1

        logger_p6a.info(
            f"Pharmacophore {pharm_idx} ({pharm_spec['center_xyz']}): "
            f"{n_passing} new Ro5+feature hits"
        )

    # Sort all unique hits by pharmacophore match score, take top_n
    all_hits = sorted(best_hits.values(), key=lambda x: x["pharm_score"], reverse=True)
    top = all_hits[:top_n]

    logger_p6a.info(
        f"Phase 6a: {len(best_hits)} unique Ro5-passing hits | "
        f"top {len(top)} by pharmacophore complementarity carried to Phase 6b"
    )
    return top
