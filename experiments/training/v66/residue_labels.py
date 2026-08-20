"""Residue-level pocket / interface labels for ResidueStage1 supervised heads."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

# RCSB hetero groups treated as non-ligand (solvent, ions, buffers).
_SKIP_HET_CODES = frozenset(
    {
        "HOH",
        "WAT",
        "DOD",
        "NA",
        "CL",
        "K",
        "CA",
        "MG",
        "ZN",
        "MN",
        "FE",
        "CU",
        "CO",
        "NI",
        "CD",
        "HG",
        "SO4",
        "PO4",
        "GOL",
        "EDO",
        "PEG",
        "ACT",
        "BME",
        "DMS",
        "TRS",
        "HEP",
        "MES",
        "FMT",
        "NH4",
    }
)

INTERFACE_CUTOFF_A = 4.5
LIGAND_CUTOFF_A = 5.0
STANDARD_AA = frozenset(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()
)


def _parse_residue_key(chain_id: str, res_id: tuple[Any, ...]) -> str | None:
    if res_id[0] != " ":
        return None
    return f"{chain_id.upper()}:{int(res_id[1])}:"


def _heavy_coords(structure) -> tuple[np.ndarray, list[str]]:
    """All non-H protein + ligand heavy-atom coords with residue keys (or HET:seq)."""
    coords: list[np.ndarray] = []
    keys: list[str] = []
    for model in structure:
        for chain in model:
            for residue in chain:
                resname = residue.get_resname().strip().upper()
                hetflag, seq, _ = residue.get_id()
                if hetflag == " ":
                    if resname not in STANDARD_AA:
                        continue
                    key = _parse_residue_key(chain.id, residue.get_id())
                    if key is None:
                        continue
                else:
                    if resname in _SKIP_HET_CODES:
                        continue
                    key = f"HET:{chain.id}:{seq}:{resname}"
                for atom in residue:
                    if atom.element == "H":
                        continue
                    coords.append(atom.get_coord())
                    keys.append(key)
    if not coords:
        return np.zeros((0, 3), dtype=np.float64), []
    return np.stack(coords, axis=0), keys


def _geometric_pocket_mask(sasa_proxy: np.ndarray, edge_index: torch.Tensor | None) -> np.ndarray:
    """Buried cluster heuristic when no ligand is present."""
    n = len(sasa_proxy)
    if n == 0:
        return np.zeros(0, dtype=bool)
    thresh = float(np.quantile(sasa_proxy, 0.25))
    candidate = sasa_proxy <= thresh
    if edge_index is None or edge_index.numel() == 0:
        return candidate
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    for s, d in zip(src, dst, strict=False):
        if s == d:
            continue
        if candidate[s] and candidate[d]:
            adj[int(s)].add(int(d))
            adj[int(d)].add(int(s))
    visited: set[int] = set()
    best: set[int] = set()
    for start in np.where(candidate)[0]:
        start = int(start)
        if start in visited:
            continue
        stack = [start]
        comp: set[int] = set()
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            comp.add(node)
            stack.extend(adj.get(node, ()))
        if len(comp) > len(best):
            best = comp
    mask = np.zeros(n, dtype=bool)
    if best:
        for idx in best:
            mask[idx] = True
    else:
        mask = candidate
    return mask


def compute_residue_labels(
    pdb_path: Path,
    chain_id: str,
    residue_ids: list[str],
    sasa_proxy: np.ndarray,
    *,
    edge_index: torch.Tensor | None = None,
) -> dict[str, Any]:
    """
    Derive per-residue pocket / interface supervision aligned to ``residue_ids``.

    Pocket priority: ligand proximity → geometric buried cluster.
    Interface: inter-chain Cα/heavy-atom contact on the full asymmetric unit.
    """
    from Bio.PDB import NeighborSearch, PDBParser
    from scipy.spatial.distance import cdist

    n = len(residue_ids)
    pocket = np.zeros(n, dtype=np.float32)
    interface = np.zeros(n, dtype=np.float32)
    pocket_mask = np.ones(n, dtype=bool)
    interface_mask = np.zeros(n, dtype=bool)
    pocket_source = "none"
    interface_source = "none"

    id_to_idx = {rid.upper(): i for i, rid in enumerate(residue_ids)}
    target_chain = chain_id.upper()

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))

    # Target-chain CA coords in graph order
    ca_coords = np.full((n, 3), np.nan, dtype=np.float64)
    for model in structure:
        for chain in model:
            if chain.id.upper() != target_chain:
                continue
            for residue in chain:
                key = _parse_residue_key(chain.id, residue.get_id())
                if key is None or key not in id_to_idx:
                    continue
                if "CA" not in residue:
                    continue
                ca_coords[id_to_idx[key]] = residue["CA"].get_coord()

    valid_ca = np.isfinite(ca_coords).all(axis=1)
    if not valid_ca.any():
        logger.warning("No CA coords matched for label generation on %s chain %s", pdb_path.stem, chain_id)
        return _pack(pocket, interface, pocket_mask, interface_mask, pocket_source, interface_source)

    # ── Interface: contact with another protein chain ─────────────────────
    other_protein_atoms = []
    for model in structure:
        for chain in model:
            if chain.id.upper() == target_chain:
                continue
            for residue in chain:
                if residue.get_id()[0] != " ":
                    continue
                if residue.get_resname().strip().upper() not in STANDARD_AA:
                    continue
                for atom in residue:
                    if atom.element == "H":
                        continue
                    other_protein_atoms.append(atom)

    if other_protein_atoms:
        interface_source = "inter_chain_contact"
        interface_mask[:] = True
        ns = NeighborSearch(other_protein_atoms)
        for i, key in enumerate(residue_ids):
            if not valid_ca[i]:
                continue
            mid = ca_coords[i]
            hits = ns.search(mid, INTERFACE_CUTOFF_A, level="A")
            if hits:
                interface[i] = 1.0

    # ── Pocket: ligand proximity on full structure ────────────────────────
    ligand_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                hetflag, _, _ = residue.get_id()
                if hetflag == " ":
                    continue
                resname = residue.get_resname().strip().upper()
                if resname in _SKIP_HET_CODES:
                    continue
                for atom in residue:
                    if atom.element == "H":
                        continue
                    ligand_atoms.append(atom)

    if ligand_atoms:
        pocket_source = "ligand_proximity"
        lig_coords = np.stack([a.get_coord() for a in ligand_atoms], axis=0)
        dists = cdist(ca_coords[valid_ca], lig_coords).min(axis=1)
        pocket_idx = np.where(valid_ca)[0]
        for j, idx in enumerate(pocket_idx):
            if dists[j] <= LIGAND_CUTOFF_A:
                pocket[idx] = 1.0
        if pocket.sum() == 0:
            nearest = dists.min()
            logger.info(
                "%s chain %s: ligands present but none within %.1f Å (nearest %.1f Å) — geometric pocket",
                pdb_path.stem,
                chain_id,
                LIGAND_CUTOFF_A,
                nearest,
            )
            geo = _geometric_pocket_mask(sasa_proxy, edge_index)
            pocket = geo.astype(np.float32)
            pocket_source = "geometric_buried_cluster"
    else:
        geo = _geometric_pocket_mask(sasa_proxy, edge_index)
        pocket = geo.astype(np.float32)
        pocket_source = "geometric_buried_cluster"

    pos = int(pocket.sum())
    logger.info(
        "%s chain %s labels: pocket=%s (%d/%d pos) interface=%s (%d/%d pos, masked=%d)",
        pdb_path.stem,
        chain_id,
        pocket_source,
        pos,
        n,
        interface_source,
        int(interface.sum()),
        n,
        int(interface_mask.sum()),
    )
    return _pack(pocket, interface, pocket_mask, interface_mask, pocket_source, interface_source)


def _pack(
    pocket: np.ndarray,
    interface: np.ndarray,
    pocket_mask: np.ndarray,
    interface_mask: np.ndarray,
    pocket_source: str,
    interface_source: str,
) -> dict[str, Any]:
    return {
        "target_pocket": torch.tensor(pocket, dtype=torch.float32).unsqueeze(1),
        "target_interface": torch.tensor(interface, dtype=torch.float32).unsqueeze(1),
        "pocket_label_mask": torch.tensor(pocket_mask, dtype=torch.bool),
        "interface_label_mask": torch.tensor(interface_mask, dtype=torch.bool),
        "label_meta": {
            "pocket_source": pocket_source,
            "interface_source": interface_source,
            "pocket_pos_rate": float(pocket.mean()) if len(pocket) else 0.0,
            "interface_pos_rate": float(interface[interface_mask].mean()) if interface_mask.any() else 0.0,
        },
    }


def attach_residue_supervision(prot: dict[str, Any], pdb_dir: Path) -> None:
    """Mutate ``prot`` in place with ResidueStage1 label tensors."""
    from experiments.training.v66._data import _download_pdb

    pdb_id = str(prot["pdb_id"]).upper()
    chain = str(prot.get("chain", "A"))
    pdb_path = _download_pdb(pdb_id, Path(pdb_dir))
    sasa = prot["data"].x[:, 3].detach().cpu().numpy()
    labels = compute_residue_labels(
        pdb_path,
        chain,
        prot["residue_ids"],
        sasa,
        edge_index=prot["data"].edge_index,
    )
    prot.update(labels)
