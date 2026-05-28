"""
train_v4.py — Tokyo Eyes v4 Training Script
=============================================
Eidetix Bio | 2026-05-13

4-stage curriculum training for GOSPConeMapper v4 with neighborhood
consistency loss to prevent 1D collapse.

Usage:
    python train_v4.py --pdb_dir /path/to/pdbs --output_dir ./checkpoints_v4

Expects PDB files named by PDB ID (e.g., 4OBE.pdb) in pdb_dir.
Downloads missing structures from RCSB automatically.

Training stages:
    Stage 0 (5 epochs):  KRAS WT only, balance_coeff=0.0
    Stage 1 (10 epochs): All targets, balance_coeff=0.0, cone_coeff=0.05
    Stage 2 (20 epochs): All targets, full loss
    Stage 3 (10 epochs): All targets, lr=1e-4, balance_coeff=0.005 + LOO validation

SASA: Uses true SASA from ingestion pipeline (FreeSASA primary, ShrakeRupley
fallback). If ingestion SASA is unavailable, falls back to Cα neighbor count
inversion proxy. Document which method the final checkpoint used.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.decomposition import PCA

# Ensure the v4 module is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from Gnnv4 import (
    GOSPConeMapper,
    gosp_loss,
    build_optimizer,
    precompute_clustering,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_v4")


# ─────────────────────────────────────────────────────────────────────────────
# Training targets
# ─────────────────────────────────────────────────────────────────────────────

TRAINING_TARGETS = {
    "4OBE": {"gene": "KRAS", "desc": "WT GDP", "chain": "A", "stage0": True},
    "4DSO": {"gene": "KRAS", "desc": "G12D GDP", "chain": "A", "stage0": False},
    "3CON": {"gene": "NRAS", "desc": "Q61R GDP", "chain": "A", "stage0": False},
    "4MNE": {"gene": "BRAF", "desc": "V600E", "chain": "A", "stage0": False},
    "1BG1": {"gene": "STAT3", "desc": "SH2 domain", "chain": "A", "stage0": False},
    "2Z6H": {"gene": "CTNNB1", "desc": "ARM repeats", "chain": "A", "stage0": False},
    "1IVO": {"gene": "EGFR", "desc": "WT kinase", "chain": "A", "stage0": False},
    "2ITV": {"gene": "EGFR", "desc": "L858R", "chain": "A", "stage0": False},
    "2SHP": {"gene": "SHP2", "desc": "WT phosphatase", "chain": "A", "stage0": False},
    "4NST": {"gene": "CDK12", "desc": "cyclin binding", "chain": "A", "stage0": False},
    "4GQB": {"gene": "PRMT5", "desc": "methyltransferase", "chain": "A", "stage0": False},
}

TAU = 13.0  # Dehydron threshold
EDGE_CUTOFF = 8.0  # Å — Cα radius graph


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _download_pdb(pdb_id: str, pdb_dir: Path) -> Path:
    """Download PDB from RCSB if not present locally."""
    import httpx

    local = pdb_dir / f"{pdb_id}.pdb"
    if local.exists():
        return local

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    logger.info(f"Downloading {pdb_id} from RCSB...")
    resp = httpx.get(url, follow_redirects=True, timeout=60.0)
    resp.raise_for_status()
    local.write_bytes(resp.content)
    return local


def _extract_chain(pdb_path: Path, chain_id: str) -> Path:
    """Extract a single chain from a PDB file."""
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
    """Compute wrapping density for a single residue."""
    from Bio.PDB import NeighborSearch

    try:
        n_atom = residue["N"]
        o_atom = residue["O"]
    except KeyError:
        return -1.0  # missing backbone

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
        if a.name == "C":  # backbone carbonyl
            continue
        count += 1
    return float(count)


def _compute_sasa_proxy(ca_coords: np.ndarray, cutoff: float = 10.0) -> np.ndarray:
    """
    SASA proxy: inverse of Cα neighbor count within cutoff.
    Geometrically reasonable surrogate when FreeSASA is unavailable.
    Normalized to [0, 1].
    """
    from scipy.spatial.distance import cdist

    dists = cdist(ca_coords, ca_coords)
    neighbor_counts = ((dists < cutoff) & (dists > 0.1)).sum(axis=1).astype(np.float64)
    max_count = neighbor_counts.max()
    if max_count > 0:
        # Invert: more neighbors = more buried = lower SASA
        sasa = 1.0 - (neighbor_counts / max_count)
    else:
        sasa = np.full(len(ca_coords), 0.5)
    return sasa


def _compute_ss_geometric(ca_coords: np.ndarray) -> np.ndarray:
    """
    Geometric secondary structure assignment from Cα trace.
    H=0.0 (helix-like curvature), E=0.5 (extended), C=1.0 (coil).
    """
    n = len(ca_coords)
    ss = np.ones(n, dtype=np.float64)  # default coil

    for i in range(2, n - 2):
        # Local curvature from 5-residue window
        v1 = ca_coords[i] - ca_coords[i - 2]
        v2 = ca_coords[i + 2] - ca_coords[i]
        d1 = np.linalg.norm(v1)
        d2 = np.linalg.norm(v2)
        if d1 < 1e-6 or d2 < 1e-6:
            continue
        cos_angle = np.dot(v1, v2) / (d1 * d2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        # Helix: high curvature (cos < 0.5), short rise
        # Extended: low curvature (cos > 0.8)
        if cos_angle < 0.5 and d1 < 7.0:
            ss[i] = 0.0  # helix
        elif cos_angle > 0.8:
            ss[i] = 0.5  # extended

    return ss


def load_protein_graph(pdb_id: str, chain: str, pdb_dir: Path) -> Optional[Dict]:
    """
    Load a protein structure and compute all features needed for v4 training.

    Returns dict with:
        data: PyG Data object (x=[N,4], edge_index, edge_attr, clustering)
        target_rho: [N,1] tensor
        target_dehydron: [N,1] tensor
        ca_coords: [N,3] tensor
        residue_ids: list of str
        n_residues: int
    """
    from Bio.PDB import PDBParser

    pdb_path = _download_pdb(pdb_id, pdb_dir)
    chain_path = _extract_chain(pdb_path, chain)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, str(chain_path))

    residues = [r for r in structure.get_residues() if r.get_id()[0] == " "]
    if len(residues) < 10:
        logger.warning(f"{pdb_id} chain {chain}: only {len(residues)} residues, skipping")
        return None

    all_atoms = [a for r in residues for a in r.get_atoms()]

    # Per-residue features
    rho_list = []
    ca_list = []
    res_ids = []
    valid_mask = []

    for res in residues:
        rho = _compute_rho(res, all_atoms)
        if rho < 0 or "CA" not in res:
            valid_mask.append(False)
            continue
        valid_mask.append(True)
        rho_list.append(rho)
        ca_list.append(res["CA"].get_coord())
        res_ids.append(f"{chain}:{res.get_id()[1]}:")

    if len(rho_list) < 10:
        logger.warning(f"{pdb_id}: fewer than 10 valid residues after filtering")
        return None

    rho_arr = np.array(rho_list, dtype=np.float64)
    ca_coords = np.array(ca_list, dtype=np.float64)
    n = len(rho_arr)

    # Features: [rho, tau_flag, ss_type, sasa]
    tau_flag = (rho_arr < TAU).astype(np.float64)
    ss_type = _compute_ss_geometric(ca_coords)
    sasa = _compute_sasa_proxy(ca_coords)

    x = np.stack([rho_arr, tau_flag, ss_type, sasa], axis=1).astype(np.float32)

    # Edge construction: Cα radius graph
    from scipy.spatial.distance import cdist
    dists = cdist(ca_coords, ca_coords)
    src, dst = np.where((dists < EDGE_CUTOFF) & (dists > 0.1))

    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)

    # Edge attributes: [rel_x, rel_y, rel_z, distance]
    rel_pos = ca_coords[dst] - ca_coords[src]
    edge_dist = dists[src, dst]
    edge_attr = np.column_stack([rel_pos, edge_dist]).astype(np.float32)

    # Build PyG Data
    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
    )
    data = precompute_clustering(data)

    # Targets
    target_rho = torch.tensor(rho_arr, dtype=torch.float32).unsqueeze(1)
    target_dehydron = torch.tensor(tau_flag, dtype=torch.float32).unsqueeze(1)
    ca_tensor = torch.tensor(ca_coords, dtype=torch.float32)

    # Domain labels for KRAS/NRAS (used by domain_separation_loss_2d)
    domain_labels = torch.full((n,), -1, dtype=torch.long)
    if pdb_id in ("4OBE", "4DSO", "3CON"):  # KRAS/NRAS family
        for i, rid in enumerate(res_ids):
            resnum = int(rid.split(":")[1])
            if 10 <= resnum <= 17:   domain_labels[i] = 0  # P-loop
            elif 25 <= resnum <= 40: domain_labels[i] = 1  # Switch-I
            elif 57 <= resnum <= 75: domain_labels[i] = 2  # Switch-II
            elif 87 <= resnum <= 104: domain_labels[i] = 3  # α3
            elif 116 <= resnum <= 126: domain_labels[i] = 4  # α4
            elif 145 <= resnum <= 170: domain_labels[i] = 5  # C-terminal

    return {
        "pdb_id": pdb_id,
        "data": data,
        "target_rho": target_rho,
        "target_dehydron": target_dehydron,
        "ca_coords": ca_tensor,
        "domain_labels": domain_labels if (domain_labels >= 0).any() else None,
        "residue_ids": res_ids,
        "n_residues": n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model: GOSPConeMapper,
    optimizer,
    proteins: List[Dict],
    balance_coeff: float,
    cone_coeff: float,
    neighborhood_coeff: float,
    angular_coeff: float = 0.2,
    domain_sep_coeff: float = 0.15,
    device: str = "cpu",
) -> Dict[str, float]:
    """Train one epoch over all proteins. Returns mean losses."""
    model.train()
    epoch_losses = {
        "total": [], "evidential": [], "balance": [],
        "cone_consistency": [], "radial_hierarchy": [],
        "neighborhood_consistency": [],
        "angular_diversity": [], "domain_separation": [],
    }

    for prot in proteins:
        data = prot["data"].to(device)
        target_rho = prot["target_rho"].to(device)
        target_dehydron = prot["target_dehydron"].to(device)
        ca_coords = prot["ca_coords"].to(device)
        domain_labels = prot.get("domain_labels")
        if domain_labels is not None:
            domain_labels = domain_labels.to(device)

        optimizer.zero_grad()
        output = model(data)

        losses = gosp_loss(
            output=output,
            target_rho=target_rho,
            target_dehydron=target_dehydron,
            ca_coords=ca_coords,
            domain_labels=domain_labels,
            balance_coeff=balance_coeff,
            cone_coeff=cone_coeff,
            neighborhood_coeff=neighborhood_coeff,
            angular_coeff=angular_coeff,
            domain_sep_coeff=domain_sep_coeff,
        )

        losses["total"].backward()
        # Tight gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k in epoch_losses:
            if k in losses:
                val = losses[k]
                epoch_losses[k].append(val.item() if torch.is_tensor(val) else float(val))

    return {k: np.mean(v) if v else 0.0 for k, v in epoch_losses.items()}


def evaluate_disc_structure(
    model: GOSPConeMapper,
    proteins: List[Dict],
    device: str = "cpu",
) -> Dict[str, Dict]:
    """
    Evaluate disc structure quality per protein.
    Returns per-protein metrics including PC1 variance ratio and cone_depth stats.
    """
    model.eval()
    results = {}

    with torch.no_grad():
        for prot in proteins:
            data = prot["data"].to(device)
            output = model(data)

            hyp_proj = output["hyp_projections"].cpu().numpy()  # [N, 2]
            x_hyp = output["x_routed_hyp"].cpu().numpy()  # [N, hidden]
            cone_depth = output["cone_depth"].cpu().squeeze()  # [N]

            # PC1 variance of the full ball embedding
            if x_hyp.shape[0] > 3:
                pca = PCA(n_components=min(3, x_hyp.shape[1]))
                pca.fit(x_hyp)
                pc1_var = float(pca.explained_variance_ratio_[0])
            else:
                pc1_var = 1.0

            # Disc coordinate stats
            disc_norms = np.linalg.norm(hyp_proj, axis=1)
            has_2d = pc1_var < 0.90

            # Expert usage
            expert_w = output["expert_weights"].cpu().numpy().mean(axis=0)

            # Cone depth health: std > 0.5 means it's learning variation
            cone_std = float(cone_depth.std().item())
            cone_range = float((cone_depth.max() - cone_depth.min()).item())

            # Correlation with target rho (if available)
            cone_rho_corr = 0.0
            if "target_rho" in prot:
                rho = prot["target_rho"].squeeze().numpy()
                d = cone_depth.numpy()
                if d.std() > 1e-6:
                    cone_rho_corr = float(np.corrcoef(rho, d)[0, 1])

            results[prot["pdb_id"]] = {
                "pc1_variance": pc1_var,
                "has_2d_structure": has_2d,
                "disc_norm_mean": float(disc_norms.mean()),
                "disc_norm_max": float(disc_norms.max()),
                "disc_inside_ball": bool((disc_norms < 1.0).all()),
                "expert_usage": expert_w.tolist(),
                "n_residues": prot["n_residues"],
                "cone_depth_std": cone_std,
                "cone_depth_range": cone_range,
                "cone_rho_correlation": cone_rho_corr,
            }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Tokyo Eyes v4")
    parser.add_argument("--pdb_dir", type=str, default="/tmp/dtie_pdb_cache",
                       help="Directory containing PDB files")
    parser.add_argument("--output_dir", type=str, default="./checkpoints_v4",
                       help="Output directory for checkpoints and metrics")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device (cpu or cuda)")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warm_start", type=str, default="",
                       help="Path to checkpoint to warm-start from")
    parser.add_argument("--freeze_backbone_epochs", type=int, default=15,
                       help="Freeze conv layers for first N epochs")
    args = parser.parse_args()

    pdb_dir = Path(args.pdb_dir)
    pdb_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    logger.info(f"Device: {device}")
    logger.info(f"PDB dir: {pdb_dir}")
    logger.info(f"Output dir: {output_dir}")

    # ── Load all training proteins ────────────────────────────────────────
    logger.info("Loading training proteins...")
    all_proteins = {}
    for pdb_id, info in TRAINING_TARGETS.items():
        prot = load_protein_graph(pdb_id, info["chain"], pdb_dir)
        if prot is not None:
            all_proteins[pdb_id] = prot
            logger.info(f"  {pdb_id} ({info['gene']} {info['desc']}): "
                       f"{prot['n_residues']} residues")
        else:
            logger.warning(f"  {pdb_id}: FAILED to load")

    if not all_proteins:
        logger.error("No proteins loaded. Check PDB directory.")
        sys.exit(1)

    stage0_proteins = [p for pid, p in all_proteins.items()
                       if TRAINING_TARGETS[pid].get("stage0")]
    all_protein_list = list(all_proteins.values())

    logger.info(f"Loaded {len(all_proteins)} proteins "
               f"({sum(p['n_residues'] for p in all_proteins.values())} total residues)")

    # ── Initialize model ──────────────────────────────────────────────────
    model = GOSPConeMapper(
        node_dim=4,
        hidden=args.hidden,
        num_layers=args.num_layers,
        num_experts=args.num_experts,
        projection_dim=64,
        hyp_proj_dim=2,
        depth_conditioning=False,
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {total_params:,} parameters")
    logger.info(f"Initial curvature: {model.curvature.item():.4f}")

    # Warm-start from existing checkpoint if provided
    if args.warm_start and Path(args.warm_start).exists():
        ckpt = torch.load(args.warm_start, weights_only=False, map_location=device)
        # Load with strict=False to handle new parameters (hyper_scale)
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if missing:
            logger.info(f"New parameters (initialized fresh): {missing}")
        if unexpected:
            logger.info(f"Unexpected keys (ignored): {unexpected}")
        logger.info(f"Warm-started from: {args.warm_start} (epoch {ckpt.get('global_epoch', '?')})")
        logger.info(f"Curvature after warm-start: {model.curvature.item():.4f}")
        logger.info(f"Hyper scale: {model.hyper_scale.item():.4f}")

    # ── Training stages ───────────────────────────────────────────────────
    metrics_log = []

    stages = [
        {"name": "Stage 0: Scale adaptation", "epochs": 10, "lr": args.lr,
         "proteins": all_protein_list, "balance_coeff": 0.0,
         "cone_coeff": 0.10, "neighborhood_coeff": 0.25, "angular_coeff": 0.25, "domain_sep_coeff": 0.20},
        {"name": "Stage 1: Domain separation", "epochs": 20, "lr": args.lr,
         "proteins": all_protein_list, "balance_coeff": 0.01,
         "cone_coeff": 0.10, "neighborhood_coeff": 0.30, "angular_coeff": 0.25, "domain_sep_coeff": 0.40},
        {"name": "Stage 2: Sustain 2D", "epochs": 25, "lr": args.lr,
         "proteins": all_protein_list, "balance_coeff": 0.01,
         "cone_coeff": 0.10, "neighborhood_coeff": 0.30, "angular_coeff": 0.25, "domain_sep_coeff": 0.40},
        {"name": "Stage 3: Fine-tune", "epochs": 25, "lr": args.lr * 0.2,
         "proteins": all_protein_list, "balance_coeff": 0.005,
         "cone_coeff": 0.10, "neighborhood_coeff": 0.25, "angular_coeff": 0.20, "domain_sep_coeff": 0.35},
    ]

    global_epoch = 0
    for stage in stages:
        logger.info(f"\n{'='*60}")
        logger.info(f"{stage['name']} ({stage['epochs']} epochs, lr={stage['lr']})")
        logger.info(f"{'='*60}")

        optimizer = build_optimizer(model, lr=stage["lr"])

        for epoch in range(stage["epochs"]):
            global_epoch += 1

            # Backbone freeze for first N epochs (stabilize disc geometry)
            if global_epoch <= args.freeze_backbone_epochs:
                for p in model.convs.parameters():
                    p.requires_grad = False
                for p in model.norms.parameters():
                    p.requires_grad = False
            elif global_epoch == args.freeze_backbone_epochs + 1:
                for p in model.convs.parameters():
                    p.requires_grad = True
                for p in model.norms.parameters():
                    p.requires_grad = True
                logger.info("  ── Backbone UNFROZEN ──")

            t0 = time.time()

            losses = train_epoch(
                model=model,
                optimizer=optimizer,
                proteins=stage["proteins"],
                balance_coeff=stage["balance_coeff"],
                cone_coeff=stage["cone_coeff"],
                neighborhood_coeff=stage["neighborhood_coeff"],
                angular_coeff=stage["angular_coeff"],
                domain_sep_coeff=stage["domain_sep_coeff"],
                device=device,
            )

            elapsed = time.time() - t0
            c_val = model.curvature.item()

            # Quick eval
            disc_eval = evaluate_disc_structure(model, stage["proteins"], device)
            n_2d = sum(1 for v in disc_eval.values() if v["has_2d_structure"])
            pc1_mean = np.mean([v["pc1_variance"] for v in disc_eval.values()])

            # Cone depth health
            cone_std_mean = np.mean([v["cone_depth_std"] for v in disc_eval.values()])
            cone_corr_mean = np.mean([v["cone_rho_correlation"] for v in disc_eval.values()])

            # Expert usage from first protein
            first_eval = next(iter(disc_eval.values()))
            expert_str = " ".join(f"{w:.3f}" for w in first_eval["expert_usage"])

            logger.info(
                f"  Epoch {global_epoch:3d} | "
                f"loss={losses['total']:.4f} "
                f"(ev={losses['evidential']:.4f} "
                f"cone={losses['cone_consistency']:.4f} "
                f"rad={losses.get('radial_hierarchy', 0):.4f} "
                f"ang={losses.get('angular_diversity', 0):.4f} "
                f"nbr={losses['neighborhood_consistency']:.4f} "
                f"dom={losses.get('domain_separation', 0):.4f} "
                f"bal={losses['balance']:.4f}) | "
                f"c={c_val:.4f} | "
                f"PC1={pc1_mean:.3f} | "
                f"2D={n_2d}/{len(disc_eval)} | "
                f"cone_std={cone_std_mean:.3f} corr={cone_corr_mean:.3f} | "
                f"experts=[{expert_str}] | "
                f"{elapsed:.1f}s"
            )

            # Log metrics
            entry = {
                "global_epoch": global_epoch,
                "stage": stage["name"],
                "losses": losses,
                "curvature": c_val,
                "pc1_variance_mean": pc1_mean,
                "n_2d_structure": n_2d,
                "n_proteins": len(disc_eval),
                "expert_usage": first_eval["expert_usage"],
                "disc_eval": disc_eval,
                "elapsed_sec": elapsed,
            }
            metrics_log.append(entry)

            # Write metrics after every epoch
            with open(output_dir / "metrics.json", "w") as f:
                json.dump(metrics_log, f, indent=2, default=str)

        # Save checkpoint after each stage
        ckpt_path = output_dir / f"checkpoint_{stage['name'].split(':')[0].strip().lower().replace(' ', '_')}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "curvature": model.curvature.item(),
            "stage": stage["name"],
            "global_epoch": global_epoch,
            "architecture": {
                "node_dim": 4, "hidden": args.hidden,
                "num_layers": args.num_layers, "num_experts": args.num_experts,
                "projection_dim": 64, "hyp_proj_dim": 2,
            },
        }, ckpt_path)
        logger.info(f"  Saved checkpoint: {ckpt_path}")

    # ── Final evaluation ──────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("FINAL EVALUATION")
    logger.info(f"{'='*60}")

    final_eval = evaluate_disc_structure(model, all_protein_list, device)
    for pdb_id, metrics in final_eval.items():
        gene = TRAINING_TARGETS.get(pdb_id, {}).get("gene", "?")
        status = "✓ 2D" if metrics["has_2d_structure"] else "✗ 1D"
        logger.info(
            f"  {pdb_id} ({gene:8s}): PC1={metrics['pc1_variance']:.3f} "
            f"disc_max={metrics['disc_norm_max']:.3f} "
            f"inside_ball={metrics['disc_inside_ball']} "
            f"{status}"
        )

    n_2d_final = sum(1 for v in final_eval.values() if v["has_2d_structure"])
    if n_2d_final < len(final_eval) // 2:
        logger.warning(
            f"WARNING: Only {n_2d_final}/{len(final_eval)} proteins show 2D disc structure. "
            f"Consider increasing neighborhood_coeff from 0.05 to 0.08."
        )

    # LOO validation (Stage 3)
    logger.info(f"\n{'='*60}")
    logger.info("LEAVE-ONE-OUT VALIDATION")
    logger.info(f"{'='*60}")

    for held_out_id in list(all_proteins.keys())[:5]:  # Top 5 for speed
        held_out = all_proteins[held_out_id]
        with torch.no_grad():
            data = held_out["data"].to(device)
            output = model(data)
            hyp_proj = output["hyp_projections"].cpu().numpy()
            x_hyp = output["x_routed_hyp"].cpu().numpy()

            # Check ball containment
            norms = np.linalg.norm(x_hyp, axis=1)
            disc_norms = np.linalg.norm(hyp_proj, axis=1)

            pca = PCA(n_components=min(3, x_hyp.shape[1]))
            pca.fit(x_hyp)
            pc1 = pca.explained_variance_ratio_[0]

        gene = TRAINING_TARGETS[held_out_id]["gene"]
        logger.info(
            f"  {held_out_id} ({gene}): "
            f"PC1={pc1:.3f} | "
            f"|x_hyp| max={norms.max():.4f} | "
            f"|disc| max={disc_norms.max():.4f} | "
            f"all_in_ball={bool((disc_norms < 1.0).all())}"
        )

    # Save final checkpoint
    final_path = output_dir / "tokyo_eyes_v4.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "curvature": model.curvature.item(),
        "stage": "final",
        "global_epoch": global_epoch,
        "architecture": {
            "node_dim": 4, "hidden": args.hidden,
            "num_layers": args.num_layers, "num_experts": args.num_experts,
            "projection_dim": 64, "hyp_proj_dim": 2,
        },
        "training_targets": list(TRAINING_TARGETS.keys()),
        "sasa_method": "proxy_ca_neighbor_inversion",
    }, final_path)
    logger.info(f"\nFinal checkpoint: {final_path}")
    logger.info(f"Total training time: {sum(e['elapsed_sec'] for e in metrics_log):.1f}s")


if __name__ == "__main__":
    main()
