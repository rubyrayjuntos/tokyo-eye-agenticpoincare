"""
_data.py — Standalone data-loading utilities for v6 retraining.

Copied from experiments/training/v4/train_v4.py (lines 64–301) without the
Gnnv4 import that breaks in the current repo structure.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

TRAINING_TARGETS = {
    # ── KRAS / RAS family ────────────────────────────────────────────────────
    "4OBE": {"gene": "KRAS",  "desc": "WT GDP",           "chain": "A", "stage0": True},
    "4DSO": {"gene": "KRAS",  "desc": "G12D GDP",         "chain": "A", "stage0": False},
    "6OIM": {"gene": "KRAS",  "desc": "G12C GDP",         "chain": "A", "stage0": False},
    "5VQ2": {"gene": "KRAS",  "desc": "G12V GppNHp",      "chain": "A", "stage0": False},
    "3CON": {"gene": "NRAS",  "desc": "Q61R GDP",         "chain": "A", "stage0": False},
    "4G0N": {"gene": "HRAS",  "desc": "WT GppNHp",        "chain": "A", "stage0": False},

    # ── Kinases ──────────────────────────────────────────────────────────────
    "4MNE": {"gene": "BRAF",  "desc": "V600E",            "chain": "A", "stage0": False},
    "1IVO": {"gene": "EGFR",  "desc": "WT kinase",        "chain": "A", "stage0": False},
    "2ITV": {"gene": "EGFR",  "desc": "L858R",            "chain": "A", "stage0": False},
    "4NST": {"gene": "CDK12", "desc": "cyclin binding",   "chain": "A", "stage0": False},
    "3PP0": {"gene": "SRC",   "desc": "active kinase",    "chain": "A", "stage0": False},
    "2OIQ": {"gene": "ABL1",  "desc": "imatinib-bound",   "chain": "A", "stage0": False},

    # ── Phosphatases / adaptors ──────────────────────────────────────────────
    "2SHP": {"gene": "SHP2",  "desc": "WT phosphatase",   "chain": "A", "stage0": False},

    # ── Transcription factors / scaffolds ────────────────────────────────────
    "1BG1": {"gene": "STAT3", "desc": "SH2 domain",       "chain": "A", "stage0": False},
    "2Z6H": {"gene": "CTNNB1","desc": "ARM repeats",      "chain": "A", "stage0": False},

    # ── Methyltransferase ────────────────────────────────────────────────────
    "4GQB": {"gene": "PRMT5", "desc": "methyltransferase","chain": "A", "stage0": False},

    # ── Allosteric / shell-biology exemplars ─────────────────────────────────
    "2HHB": {"gene": "HBB",   "desc": "deoxy haemoglobin","chain": "B", "stage0": False},  # classic allostery
    "1L2Y": {"gene": "TC5b",  "desc": "Trp-cage miniprotein","chain": "A","stage0": False}, # small, all-surface
}

TAU = 13.0
EDGE_CUTOFF = 8.0


def _download_pdb(pdb_id: str, pdb_dir: Path) -> Path:
    import httpx
    local = pdb_dir / f"{pdb_id}.pdb"
    if local.exists():
        return local
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    logger.info("Downloading %s from RCSB...", pdb_id)
    resp = httpx.get(url, follow_redirects=True, timeout=60.0)
    resp.raise_for_status()
    local.write_bytes(resp.content)
    return local


def _extract_chain(pdb_path: Path, chain_id: str) -> Path:
    from Bio.PDB import PDBParser, PDBIO, Select

    class ChainSelect(Select):
        def accept_chain(self, chain):
            return chain.id == chain_id

    out_path = pdb_path.parent / f"{pdb_path.stem}_{chain_id}.pdb"
    if out_path.exists():
        return out_path
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out_path), ChainSelect())
    return out_path


def _compute_rho(residue, all_atoms, wrapping_radius: float = 6.5) -> float:
    from Bio.PDB import NeighborSearch
    try:
        n_atom = residue["N"]
        o_atom = residue["O"]
    except KeyError:
        return -1.0
    mid = (n_atom.get_coord() + o_atom.get_coord()) / 2.0
    ns = NeighborSearch(all_atoms)
    neighbours = ns.search(mid, wrapping_radius, level="A")
    POLAR = {"ARG", "ASN", "ASP", "GLN", "GLU", "HIS", "LYS", "SER", "THR", "TYR", "TRP"}
    count = 0
    for a in neighbours:
        if a.element != "C":
            continue
        if a.get_parent().get_resname().strip() in POLAR:
            continue
        if a.name == "C":
            continue
        count += 1
    return float(count)


def _compute_sasa_proxy(ca_coords: np.ndarray, cutoff: float = 10.0) -> np.ndarray:
    from scipy.spatial.distance import cdist
    dists = cdist(ca_coords, ca_coords)
    neighbor_counts = ((dists < cutoff) & (dists > 0.1)).sum(axis=1).astype(np.float64)
    max_count = neighbor_counts.max()
    if max_count > 0:
        sasa = 1.0 - (neighbor_counts / max_count)
    else:
        sasa = np.full(len(ca_coords), 0.5)
    return sasa


def _compute_ss_geometric(ca_coords: np.ndarray) -> np.ndarray:
    n = len(ca_coords)
    ss = np.ones(n, dtype=np.float64)
    for i in range(2, n - 2):
        v1 = ca_coords[i] - ca_coords[i - 2]
        v2 = ca_coords[i + 2] - ca_coords[i]
        d1 = np.linalg.norm(v1)
        d2 = np.linalg.norm(v2)
        if d1 < 1e-6 or d2 < 1e-6:
            continue
        cos_angle = np.clip(np.dot(v1, v2) / (d1 * d2), -1.0, 1.0)
        if cos_angle < 0.5 and d1 < 7.0:
            ss[i] = 0.0
        elif cos_angle > 0.8:
            ss[i] = 0.5
    return ss


def load_protein_graph(pdb_id: str, chain: str, pdb_dir: Path) -> Optional[Dict]:
    from Bio.PDB import PDBParser
    from science.dtie.v5.gnn.model import precompute_clustering
    from scipy.spatial.distance import cdist

    pdb_path = _download_pdb(pdb_id, pdb_dir)
    chain_path = _extract_chain(pdb_path, chain)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, str(chain_path))

    residues = [r for r in structure.get_residues() if r.get_id()[0] == " "]
    if len(residues) < 10:
        logger.warning("%s chain %s: only %d residues, skipping", pdb_id, chain, len(residues))
        return None

    all_atoms = [a for r in residues for a in r.get_atoms()]

    rho_list, ca_list, res_ids = [], [], []
    for res in residues:
        rho = _compute_rho(res, all_atoms)
        if rho < 0 or "CA" not in res:
            continue
        rho_list.append(rho)
        ca_list.append(res["CA"].get_coord())
        res_ids.append(f"{chain}:{res.get_id()[1]}:")

    if len(rho_list) < 10:
        logger.warning("%s: fewer than 10 valid residues after filtering", pdb_id)
        return None

    rho_arr = np.array(rho_list, dtype=np.float64)
    ca_coords = np.array(ca_list, dtype=np.float64)
    n = len(rho_arr)

    tau_flag = (rho_arr < TAU).astype(np.float64)
    ss_type = _compute_ss_geometric(ca_coords)
    sasa = _compute_sasa_proxy(ca_coords)
    x = np.stack([rho_arr, tau_flag, ss_type, sasa], axis=1).astype(np.float32)

    dists = cdist(ca_coords, ca_coords)
    src, dst = np.where((dists < EDGE_CUTOFF) & (dists > 0.1))
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    rel_pos = ca_coords[dst] - ca_coords[src]
    edge_dist = dists[src, dst]
    edge_attr = np.column_stack([rel_pos, edge_dist]).astype(np.float32)

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
    )
    data = precompute_clustering(data)

    target_rho = torch.tensor(rho_arr, dtype=torch.float32).unsqueeze(1)
    target_dehydron = torch.tensor(tau_flag, dtype=torch.float32).unsqueeze(1)
    ca_tensor = torch.tensor(ca_coords, dtype=torch.float32)

    # SASA as direct training target for cone_depth (surface = disc periphery)
    target_sasa = torch.tensor(sasa, dtype=torch.float32).unsqueeze(1)

    domain_labels = torch.full((n,), -1, dtype=torch.long)
    # RAS-family domain annotations (P-loop, Switch-I/II, α3, α4, C-term)
    if pdb_id in ("4OBE", "4DSO", "6OIM", "5VQ2", "3CON", "4G0N"):
        for i, rid in enumerate(res_ids):
            resnum = int(rid.split(":")[1])
            if 10 <= resnum <= 17:    domain_labels[i] = 0  # P-loop
            elif 25 <= resnum <= 40:  domain_labels[i] = 1  # Switch-I
            elif 57 <= resnum <= 75:  domain_labels[i] = 2  # Switch-II
            elif 87 <= resnum <= 104: domain_labels[i] = 3  # α3
            elif 116 <= resnum <= 126: domain_labels[i] = 4 # α4
            elif 145 <= resnum <= 170: domain_labels[i] = 5 # C-terminal

    return {
        "pdb_id": pdb_id,
        "data": data,
        "target_rho": target_rho,
        "target_dehydron": target_dehydron,
        "target_sasa": target_sasa,
        "ca_coords": ca_tensor,
        "domain_labels": domain_labels if (domain_labels >= 0).any() else None,
        "residue_ids": res_ids,
        "n_residues": n,
    }
