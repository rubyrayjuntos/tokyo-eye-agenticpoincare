"""Run PeSTo i_v4_1 protein-protein interface inference on a single PDB chain."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch as pt

DEFAULT_PESTO_ROOT = Path(os.environ.get("PESTO_ROOT", "/tmp/PeSTo"))
MODEL_SAVE = "model/save/i_v4_1_2021-09-07_11-21"


def _chain_mask(structure: dict, chain: str) -> np.ndarray:
    prefixes = (f"{chain}:", f"{chain.upper()}:", f"{chain.lower()}:")
    names = structure["chain_name"]
    return np.array([str(n).startswith(prefixes) for n in names], dtype=bool)


def _apply_mask(structure: dict, mask: np.ndarray) -> dict:
    return {key: structure[key][mask] for key in structure}


def run_pesto_pp_interface(
    pdb_path: str | Path,
    *,
    chain: str = "A",
    pesto_root: Path = DEFAULT_PESTO_ROOT,
    device: str = "cpu",
) -> tuple[dict[int, float], dict]:
    """Return per-residue PP interface scores keyed by PDB residue number."""
    pesto_root = Path(pesto_root)
    save_path = pesto_root / MODEL_SAVE
    model_filepath = save_path / "model_ckpt.pt"
    if not model_filepath.exists():
        raise FileNotFoundError(
            f"PeSTo model not found at {model_filepath}. "
            f"Clone https://github.com/LBM-EPFL/PeSTo to PESTO_ROOT."
        )

    src_path = str(pesto_root)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    save_str = str(save_path)
    if save_str not in sys.path:
        sys.path.insert(0, save_str)

    from config import config_model  # type: ignore[import-not-found]
    from model import Model  # type: ignore[import-not-found]
    from src.data_encoding import encode_features, encode_structure, extract_topology
    from src.dataset import collate_batch_features
    from src.structure import clean_structure, concatenate_chains
    from src.structure_io import read_pdb

    raw = read_pdb(str(pdb_path))
    mask = _chain_mask(raw, chain)
    if not mask.any():
        raise ValueError(f"chain {chain} not found in {pdb_path}")
    raw = _apply_mask(raw, mask)
    pdb_resid_per_atom = raw["resid"].copy()

    structure = clean_structure(raw)
    subunits = {chain: structure}
    structure = concatenate_chains(subunits)

    dev = pt.device(device)
    model = Model(config_model)
    model.load_state_dict(pt.load(model_filepath, map_location="cpu"))
    model.train(False)
    model.to(dev)

    with pt.no_grad():
        X, M = encode_structure(structure, device=dev)
        q = encode_features(structure, device=dev)[0]
        ids_topk, _, _, _, _ = extract_topology(X, 64)
        X, ids_topk, q, M = collate_batch_features([[X, ids_topk, q, M]])
        z = model(X.to(dev), ids_topk.to(dev), q.to(dev), M.float().to(dev))
        p = pt.sigmoid(z).detach().cpu().numpy().reshape(-1)

    clean_resids = structure["resid"]
    unique_seq = np.unique(clean_resids)
    scores: dict[int, float] = {}
    for j, seq_id in enumerate(unique_seq):
        atom_idx = int(np.where(clean_resids == seq_id)[0][0])
        pdb_resnum = int(pdb_resid_per_atom[atom_idx])
        scores[pdb_resnum] = float(p[min(j, len(p) - 1)])

    meta = {
        "pesto_root": str(pesto_root),
        "model": MODEL_SAVE,
        "device": device,
        "pdb_path": str(pdb_path),
        "chain": chain,
        "n_residues_scored": len(scores),
    }
    return scores, meta
