"""
_data.py — Training data loading via governed DB (single ingest writer path).

Corpus manifest: manifests/v6_corpus_120.json via experiments.training.v6.corpus

Training loads protein graphs from the database only — ingest must complete first
via POST /api/ingest. This module does not compute MASTER features at train time
unless TRAINING_ALLOW_AUTO_INGEST=1 (dev escape hatch only).
"""

from __future__ import annotations

import asyncio
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch_geometric.data import Data

logger = logging.getLogger(__name__)
_MISSING_BARCODE_WARNED: set[str] = set()
_LEGACY_BARCODE_LOAD_WARNED = False

TRAINING_TARGETS = {
    "4OBE": {"gene": "KRAS",  "desc": "WT GDP",           "chain": "A", "stage0": True},
    "4DSO": {"gene": "KRAS",  "desc": "G12D GDP",         "chain": "A", "stage0": False},
    "6OIM": {"gene": "KRAS",  "desc": "G12C GDP",         "chain": "A", "stage0": False},
    "5VQ2": {"gene": "KRAS",  "desc": "G12V GppNHp",      "chain": "A", "stage0": False},
    "3CON": {"gene": "NRAS",  "desc": "Q61R GDP",         "chain": "A", "stage0": False},
    "4G0N": {"gene": "HRAS",  "desc": "WT GppNHp",        "chain": "A", "stage0": False},
    "4MNE": {"gene": "BRAF",  "desc": "V600E",            "chain": "A", "stage0": False},
    "1IVO": {"gene": "EGFR",  "desc": "WT kinase",        "chain": "A", "stage0": False},
    "2ITV": {"gene": "EGFR",  "desc": "L858R",            "chain": "A", "stage0": False},
    "4NST": {"gene": "CDK12", "desc": "cyclin binding",   "chain": "A", "stage0": False},
    "3PP0": {"gene": "SRC",   "desc": "active kinase",    "chain": "A", "stage0": False},
    "2OIQ": {"gene": "ABL1",  "desc": "imatinib-bound",   "chain": "A", "stage0": False},
    "2SHP": {"gene": "SHP2",  "desc": "WT phosphatase",   "chain": "A", "stage0": False},
    "1BG1": {"gene": "STAT3", "desc": "multidomain (coiled-coil+SH2)", "chain": "A", "stage0": False},
    "2Z6H": {"gene": "CTNNB1","desc": "ARM repeats",      "chain": "A", "stage0": False},
    "4GQB": {"gene": "PRMT5", "desc": "methyltransferase","chain": "A", "stage0": False},
    "2HHB": {"gene": "HBB",   "desc": "deoxy haemoglobin","chain": "B", "stage0": False},
}

TAU = 13.0
EDGE_CUTOFF = 8.0

from science.dtie.common import residue_features as rf


def _barcode_sidecar_path(barcode_dir: Path, pdb_id: str, chain: str) -> Path:
    from science.dtie.common.dehydron_barcode_features import (
        barcode_sidecar_filename,
    )

    return Path(barcode_dir) / barcode_sidecar_filename(pdb_id, chain)


def _missing_barcode_payload(n_residues: int, *, use_binned: bool) -> dict[str, np.ndarray | None]:
    from science.dtie.common.dehydron_barcode_features import BINNED_DIM, SCALAR_DIM

    return {
        "scalars": np.zeros((n_residues, SCALAR_DIM), dtype=np.float32),
        "binned": np.zeros((n_residues, BINNED_DIM), dtype=np.float32) if use_binned else None,
        "missing": np.ones((n_residues, 1), dtype=np.float32),
    }


def _barcode_payload_to_numpy(barcode: dict) -> dict[str, np.ndarray | None]:
    def to_array(value, *, dtype) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
        return np.asarray(value, dtype=dtype)

    out: dict[str, np.ndarray | None] = {
        "scalars": to_array(barcode["scalars"], dtype=np.float32),
        "missing": to_array(barcode["missing"], dtype=np.float32),
        "binned": (
            None
            if barcode.get("binned") is None
            else to_array(barcode["binned"], dtype=np.float32)
        ),
    }
    if barcode.get("residue_indices") is not None:
        out["residue_indices"] = to_array(barcode["residue_indices"], dtype=np.int32)
    return out


def _load_barcode_sidecar(sidecar: Path) -> dict:
    """Load a sidecar with the restricted torch loader when available."""
    global _LEGACY_BARCODE_LOAD_WARNED

    try:
        return torch.load(sidecar, map_location="cpu", weights_only=True)
    except (TypeError, pickle.UnpicklingError):
        if not _LEGACY_BARCODE_LOAD_WARNED:
            logger.warning(
                "Loading legacy dehydron barcode sidecar with unsafe torch.load; "
                "re-run precompute-dehydron-barcodes so sidecars are tensor-only."
            )
            _LEGACY_BARCODE_LOAD_WARNED = True
        return torch.load(sidecar, map_location="cpu", weights_only=False)


def _graph_residue_indices(prot: Dict, n_residues: int) -> list[int]:
    """Parse ``prot['residue_ids']`` (``chain:resseq:``) into resseq ints."""
    residue_ids = prot.get("residue_ids") or []
    if len(residue_ids) != n_residues:
        raise ValueError(
            f"{prot.get('pdb_id')}:{prot.get('chain')} residue_ids length "
            f"{len(residue_ids)} != data.x rows {n_residues}"
        )
    out: list[int] = []
    for rid in residue_ids:
        parts = str(rid).split(":")
        if len(parts) < 2:
            raise ValueError(f"unparseable residue_id {rid!r}")
        out.append(int(parts[1]))
    return out


def attach_dehydron_barcode_features(
    prot: Dict,
    *,
    barcode_dir: Path,
    use_binned: bool,
) -> Dict:
    """Attach dehydron barcode sidecar features to ``prot['data'].x``."""
    from science.dtie.common.dehydron_barcode_features import (
        align_barcode_to_graph_residues,
        stack_node_features_with_barcode,
    )

    pdb_id = str(prot.get("pdb_id", "")).upper()
    chain = str(prot.get("chain", "A"))
    data = prot["data"]
    base_x = data.x.detach().cpu().numpy()
    if base_x.ndim == 2 and base_x.shape[1] > 3:
        base_x = base_x[:, :3]
    n_residues = int(base_x.shape[0])
    sidecar = _barcode_sidecar_path(Path(barcode_dir), pdb_id, chain)

    if sidecar.is_file():
        barcode = _load_barcode_sidecar(sidecar)
        barcode = _barcode_payload_to_numpy(barcode)
        graph_resseqs = _graph_residue_indices(prot, n_residues)
        barcode = align_barcode_to_graph_residues(
            barcode,
            graph_resseqs,
            use_binned=use_binned,
        )
    else:
        warn_key = f"{pdb_id}:{chain}"
        if warn_key not in _MISSING_BARCODE_WARNED:
            logger.warning(
                "%s missing dehydron barcode sidecar %s; using zero barcode with missing=1",
                warn_key,
                sidecar,
            )
            _MISSING_BARCODE_WARNED.add(warn_key)
        barcode = _missing_barcode_payload(n_residues, use_binned=use_binned)

    x = stack_node_features_with_barcode(base_x, barcode, use_binned=use_binned)
    data.x = torch.as_tensor(x, dtype=data.x.dtype, device=data.x.device)
    prot["data"] = data
    return prot


def attach_dehydron_edge_barcode_lookup(
    prot: Dict,
    *,
    barcode_dir: Path,
    pdb_dir: Path | None = None,
) -> Dict:
    """Load per-pair local dehydron barcode features for role-edge injection."""
    from science.dtie.common.dehydron_barcode_features import (
        EDGE_BARCODE_DIM,
        align_edge_barcode_to_graph,
        featurize_chain_dehydron_barcode,
    )

    pdb_id = str(prot.get("pdb_id", "")).upper()
    chain = str(prot.get("chain", "A"))
    n_residues = int(prot["data"].x.shape[0])
    sidecar = _barcode_sidecar_path(Path(barcode_dir), pdb_id, chain)
    graph_resseqs = _graph_residue_indices(prot, n_residues)

    edge_pairs: np.ndarray | None = None
    edge_scalars: np.ndarray | None = None
    barcode_residue_indices: np.ndarray | None = None

    if sidecar.is_file():
        barcode = _load_barcode_sidecar(sidecar)
        if barcode.get("edge_pairs") is not None and barcode.get("edge_scalars") is not None:
            edge_pairs = np.asarray(barcode["edge_pairs"], dtype=np.int32)
            edge_scalars = np.asarray(barcode["edge_scalars"], dtype=np.float32)
            if barcode.get("residue_indices") is not None:
                barcode_residue_indices = np.asarray(
                    barcode["residue_indices"], dtype=np.int32
                )

    if edge_pairs is None or edge_scalars is None or barcode_residue_indices is None:
        warn_key = f"{pdb_id}:{chain}:edge"
        if warn_key not in _MISSING_BARCODE_WARNED:
            logger.warning(
                "%s sidecar %s missing edge barcode block; recomputing from PDB",
                warn_key,
                sidecar,
            )
            _MISSING_BARCODE_WARNED.add(warn_key)
        pdb_path = prot.get("pdb_path")
        if pdb_path is None and pdb_dir is not None:
            pdb_path = _download_pdb(pdb_id, Path(pdb_dir))
        if pdb_path is None:
            prot["dehydron_edge_lookup"] = {}
            return prot
        payload = featurize_chain_dehydron_barcode(Path(pdb_path), chain)
        edge_pairs = np.asarray(payload["edge_pairs"], dtype=np.int32)
        edge_scalars = np.asarray(payload["edge_scalars"], dtype=np.float32)
        barcode_residue_indices = np.asarray(payload["residue_indices"], dtype=np.int32)

    if edge_scalars.ndim != 2 or edge_scalars.shape[1] != EDGE_BARCODE_DIM:
        raise ValueError(
            f"{pdb_id}:{chain} edge_scalars must be [M, {EDGE_BARCODE_DIM}], "
            f"got {edge_scalars.shape}"
        )

    lookup = align_edge_barcode_to_graph(
        edge_pairs,
        edge_scalars,
        barcode_residue_indices,
        graph_resseqs,
    )
    prot["dehydron_edge_lookup"] = lookup
    return prot


def _use_db_load() -> bool:
    return os.environ.get("TRAINING_LOAD_FROM_PDB", "").lower() not in ("1", "true", "yes")


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError(
        "load_protein_graph cannot be called from a running event loop; "
        "use load_protein_graph_async instead"
    )


async def load_protein_graph_async(
    pdb_id: str,
    chain: str,
    pdb_dir: Path,
    *,
    force_reingest: bool = False,
) -> Optional[Dict]:
    """Read training graph from governed DB (read-only after ingest).

    Production training must not re-ingest or re-persist MASTER features.
    Run ``make gate-p-feature-01`` after ingest to prove DB round-trip parity.
    """
    from data.db import DBAdapter, get_connection
    from science.dtie.common.keys import make_structure_id
    from science.dtie.common.load_graph_from_db import load_protein_graph_from_db
    from science.dtie.common.structure_readiness import (
        check_master_features_ready,
        ensure_structure_ready,
    )

    pdb_id = pdb_id.upper()
    allow_auto_ingest = os.environ.get("TRAINING_ALLOW_AUTO_INGEST", "").lower() in (
        "1",
        "true",
        "yes",
    )

    async with get_connection() as conn:
        db = DBAdapter(conn)
        if allow_auto_ingest or force_reingest:
            structure_id = await ensure_structure_ready(
                db,
                pdb_id,
                chain,
                pdb_dir,
                force_reingest=force_reingest,
            )
        else:
            structure_id = make_structure_id(pdb_id=pdb_id, source="rcsb")
            readiness = await check_master_features_ready(db, structure_id, chain)
            if not readiness.ready:
                logger.error(
                    "%s:%s not ingestion-complete (%s) — run POST /api/ingest, "
                    "then make gate-p-feature-01 before training",
                    pdb_id,
                    chain,
                    readiness.reason,
                )
                return None
        graph = await load_protein_graph_from_db(db, structure_id, pdb_id, chain)
        if graph is not None:
            from science.dtie.v66.chem_edge_graph import fetch_covalent_bonds

            graph["covalent_bonds"] = await fetch_covalent_bonds(db, structure_id)
            # Deposited PDB path for train-side parsers (barcode / chem / hierarchy).
            for name in (f"{pdb_id.upper()}.pdb", f"{pdb_id.lower()}.pdb"):
                local = Path(pdb_dir) / name
                if local.is_file():
                    graph["pdb_path"] = str(local)
                    break
            graph["pdb_dir"] = str(pdb_dir)
        return graph


def load_protein_graph(pdb_id: str, chain: str, pdb_dir: Path) -> Optional[Dict]:
    """Load protein graph for training — read-only from DB after ingest."""
    if not _use_db_load():
        return load_protein_graph_from_pdb_legacy(pdb_id, chain, pdb_dir)
    return _run_async(load_protein_graph_async(pdb_id, chain, pdb_dir))


def _download_pdb(pdb_id: str, pdb_dir: Path) -> Path:
    local = pdb_dir / f"{pdb_id}.pdb"
    if local.exists():
        return local
    import urllib.request
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    logger.info("Downloading %s from RCSB...", pdb_id)
    with urllib.request.urlopen(url, timeout=60) as resp:
        local.write_bytes(resp.read())
    return local


def _chain_cache_dir(pdb_dir: Path) -> Path:
    candidate = pdb_dir / ".chain_cache"
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate
    except OSError:
        fallback = Path("/tmp/dtie_pdb_cache/chain_cache")
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def _extract_chain(pdb_path: Path, chain_id: str, cache_dir: Path | None = None) -> Path:
    from Bio.PDB import PDBParser, PDBIO, Select

    class ChainSelect(Select):
        def accept_chain(self, chain):
            return chain.id == chain_id

    work_dir = cache_dir or pdb_path.parent
    out_path = work_dir / f"{pdb_path.stem}_{chain_id}.pdb"
    if out_path.exists():
        return out_path
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out_path), ChainSelect())
    return out_path


def load_protein_graph_from_pdb_legacy(pdb_id: str, chain: str, pdb_dir: Path) -> Optional[Dict]:
    """Legacy PDB-local load (TRAINING_LOAD_FROM_PDB=1 only — not for production training)."""
    from Bio.PDB import PDBParser
    from science.dtie.v5.gnn.model import precompute_clustering
    from scipy.spatial.distance import cdist

    pdb_path = _download_pdb(pdb_id, pdb_dir)
    chain_path = _extract_chain(pdb_path, chain, _chain_cache_dir(pdb_dir))

    node_feats = rf.build_from_pdb_chain(chain_path, chain, mode=rf.FeatureMode.MASTER)
    if len(node_feats) < 10:
        logger.warning("%s chain %s: fewer than 10 valid residues after filtering", pdb_id, chain)
        return None

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, str(chain_path))
    res_by_idx = {
        int(r.get_id()[1]): r
        for r in structure.get_residues()
        if r.get_id()[0] == " " and r.parent.id == chain
    }

    ca_list, bf_list, bf_present, res_ids = [], [], [], []
    for feat in node_feats:
        res = res_by_idx.get(feat.residue_index)
        if res is None or "CA" not in res:
            logger.warning("%s: missing CA for residue %s", pdb_id, feat.residue_index)
            return None
        ca = res["CA"]
        bf = float(ca.get_bfactor())
        has_bf = np.isfinite(bf)
        ca_list.append(ca.get_coord())
        bf_list.append(bf if has_bf else float("nan"))
        bf_present.append(has_bf)
        res_ids.append(f"{chain}:{feat.residue_index}:")

    rho_arr = np.array([f.rho for f in node_feats], dtype=np.float64)
    ca_coords = np.array(ca_list, dtype=np.float64)
    n = len(rho_arr)

    tau_flag = np.array([f.tau_flag for f in node_feats], dtype=np.float64)
    ss_type = np.array([f.ss_type for f in node_feats], dtype=np.float64)
    sasa = np.array([f.sasa for f in node_feats], dtype=np.float64)
    x = rf.stack_gnn_node_features(rho_arr, tau_flag, ss_type, sasa)

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
    data.sasa = torch.tensor(sasa, dtype=torch.float32)
    data = precompute_clustering(data)

    target_rho = torch.tensor(rho_arr, dtype=torch.float32).unsqueeze(1)
    target_dehydron = torch.tensor(tau_flag, dtype=torch.float32).unsqueeze(1)
    ca_tensor = torch.tensor(ca_coords, dtype=torch.float32)
    target_sasa = torch.tensor(sasa, dtype=torch.float32).unsqueeze(1)
    b_factor_ca = torch.tensor(bf_list, dtype=torch.float32).unsqueeze(1)
    b_factor_present = torch.tensor(bf_present, dtype=torch.bool)

    domain_labels = torch.full((n,), -1, dtype=torch.long)
    if pdb_id in ("4OBE", "4DSO", "6OIM", "5VQ2", "3CON", "4G0N"):
        for i, rid in enumerate(res_ids):
            resnum = int(rid.split(":")[1])
            if 10 <= resnum <= 17:
                domain_labels[i] = 0
            elif 25 <= resnum <= 40:
                domain_labels[i] = 1
            elif 57 <= resnum <= 75:
                domain_labels[i] = 2
            elif 87 <= resnum <= 104:
                domain_labels[i] = 3
            elif 116 <= resnum <= 126:
                domain_labels[i] = 4
            elif 145 <= resnum <= 170:
                domain_labels[i] = 5

    return {
        "pdb_id": pdb_id,
        "chain": chain,
        "data": data,
        "target_rho": target_rho,
        "target_dehydron": target_dehydron,
        "target_sasa": target_sasa,
        "b_factor_ca": b_factor_ca,
        "b_factor_present": b_factor_present,
        "ca_coords": ca_tensor,
        "domain_labels": domain_labels if (domain_labels >= 0).any() else None,
        "residue_ids": res_ids,
        "n_residues": n,
        "source": "pdb_legacy",
        "covalent_bonds": [],
        # Full deposited PDB (not chain extract) for train-side parsers.
        "pdb_path": str(pdb_path),
        "pdb_dir": str(pdb_dir),
    }
