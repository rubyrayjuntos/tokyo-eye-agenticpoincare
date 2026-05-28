"""
gnn_runner.py
=============
Eidetix Bio — DTIE Pipeline v3.0

Runs the GOSPConeMapper GNN on ingestion outputs for both GDP and GTP
conditions, writing all predictions to a single gnn_output.npz file.

## CONTRACT

### Reads
- ingestion_gdp.npz            <- written by ingestion.py
  - rho            : [N]       wrapping density
  - tau_flag       : [N]       dehydron flag
  - ss_type        : [N]       secondary structure encoding
  - sasa           : [N]       normalized SASA
  - ca_coords      : [N, 3]    Cα positions (for graph edges)
  - no_midpoints   : [N, 3]    dehydron positions (passthrough)
  - residue_ids    : [N]       residue identifiers (passthrough)
- ingestion_gtp.npz            <- written by ingestion.py
  - (same keys as above)
- robust_experts.pt            <- GNN checkpoint
  - model_state_dict           containing log_c parameter

### Writes
- gnn_output.npz               -> read by Phases 1, 2, 4, 5
  - {gdp,gtp}_residue_ids     : [N]       str
  - {gdp,gtp}_ca_coords       : [N, 3]    float64
  - {gdp,gtp}_no_midpoints    : [N, 3]    float64  (passthrough)
  - {gdp,gtp}_cone_depth      : [N]       float64
  - {gdp,gtp}_cone_width      : [N]       float64
  - {gdp,gtp}_projections     : [N, 64]   float64
  - {gdp,gtp}_epistemic       : [N]       float64
  - {gdp,gtp}_aleatoric       : [N]       float64
  - {gdp,gtp}_total_uncertainty: [N]      float64
  - {gdp,gtp}_expert_weights  : [N, 4]    float64
  - {gdp,gtp}_evidence_mu     : [N]       float64
  - {gdp,gtp}_evidence_nu     : [N]       float64
  - {gdp,gtp}_evidence_alpha  : [N]       float64
  - {gdp,gtp}_evidence_beta   : [N]       float64
  - curvature_c                : scalar    float64
  - audit_trail                : scalar    str (JSON)

### GNN fields used directly
  - All — this module produces them

### What this phase adds
    - [N, 28] node feature construction from ingestion arrays
  - Cα radius graph (8Å cutoff) with edge_attr [rel_x, rel_y, rel_z, dist]
  - GOSPConeMapper forward pass (evidential uncertainty, cone geometry, MoE)
  - NaN/Inf validation via scan_tensors_for_invalid
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

try:
    from .Gnnv3 import GOSPConeMapper, precompute_clustering, scan_tensors_for_invalid
except Exception:
    from Gnnv3 import GOSPConeMapper, precompute_clustering, scan_tensors_for_invalid  # type: ignore

logger = logging.getLogger("DTIE_GNNRunner")

EDGE_RADIUS: float = 8.0  # Angstrom cutoff for Cα graph
NODE_FEATURE_DIM: int = 28

_AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
_AA_INDEX = {aa: i for i, aa in enumerate(_AA_ORDER)}
_RES3_TO_1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def build_node_features(ingestion: dict) -> torch.Tensor:
    """
    Build canonical [N, 28] node features from ingestion output:
      [0]      rho
      [1]      tau_flag
      [2:22]   one-hot amino acid identity (20 dims)
      [22:25]  Cα position, globally mean-centered
      [25:28]  one-hot secondary structure (H/E/C)

    This intentionally avoids any projection-to-4 fallback; the redesign
    requires robust 28-dim inputs for cross-application generalization.
    """
    required = ["rho", "tau_flag", "resnames", "ca_coords", "ss_type"]
    missing = [k for k in required if k not in ingestion]
    if missing:
        raise KeyError(f"ingestion is missing required keys for 28-dim features: {missing}")

    rho = np.asarray(ingestion["rho"], dtype=np.float32)
    tau_flag = np.asarray(ingestion["tau_flag"], dtype=np.float32)
    ca_coords = np.asarray(ingestion["ca_coords"], dtype=np.float32)
    ss_type = np.asarray(ingestion["ss_type"], dtype=np.float32)
    resnames = np.asarray(ingestion["resnames"], dtype=object)

    n = len(rho)
    if len(tau_flag) != n or len(ca_coords) != n or len(ss_type) != n or len(resnames) != n:
        raise ValueError("Ingestion arrays have inconsistent lengths for 28-dim feature assembly")

    features = np.zeros((n, NODE_FEATURE_DIM), dtype=np.float32)
    ca_centered = ca_coords - ca_coords.mean(axis=0, keepdims=True)

    features[:, 0] = rho
    features[:, 1] = tau_flag
    features[:, 22:25] = ca_centered

    for i in range(n):
        res3 = str(resnames[i]).strip().upper()
        aa1 = _RES3_TO_1.get(res3)
        if aa1 in _AA_INDEX:
            features[i, 2 + _AA_INDEX[aa1]] = 1.0

        ss = float(ss_type[i])
        if abs(ss - 0.0) < 1e-6:
            features[i, 25] = 1.0  # H
        elif abs(ss - 0.5) < 1e-6:
            features[i, 26] = 1.0  # E
        else:
            features[i, 27] = 1.0  # C/unknown

    logger.info(f"Built canonical node features: [{features.shape[0]}, {features.shape[1]}]")
    return torch.tensor(features, dtype=torch.float32)


def build_edges(ca_coords: np.ndarray, radius: float = EDGE_RADIUS):
    """
    Build radius graph from Cα coordinates.

    Returns:
        edge_index: [2, E] long tensor
        edge_attr:  [E, 4] float tensor — [rel_x, rel_y, rel_z, distance]
    """
    coords = np.asarray(ca_coords, dtype=np.float64)
    n = len(coords)
    src_list, dst_list = [], []

    for i in range(n):
        diff = coords - coords[i]  # [N, 3]
        dists = np.linalg.norm(diff, axis=1)  # [N]
        mask = (dists > 0) & (dists <= radius)
        neighbours = np.where(mask)[0]
        for j in neighbours:
            src_list.append(i)
            dst_list.append(int(j))

    if not src_list:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 4), dtype=torch.float32)
        return edge_index, edge_attr

    src = np.array(src_list, dtype=np.int64)
    dst = np.array(dst_list, dtype=np.int64)

    rel_pos = coords[dst] - coords[src]  # [E, 3]
    dists_e = np.linalg.norm(rel_pos, axis=1, keepdims=True)  # [E, 1]
    ea = np.concatenate([rel_pos, dists_e], axis=1).astype(np.float32)  # [E, 4]

    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    edge_attr = torch.tensor(ea, dtype=torch.float32)
    return edge_index, edge_attr


def _build_pyg_data(ingestion: dict) -> Data:
    """Build a PyG Data object from ingestion output, with clustering."""
    x = build_node_features(ingestion)
    edge_index, edge_attr = build_edges(ingestion["ca_coords"])

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data = precompute_clustering(data)
    return data


def _save_pyg_graph(data: Data, output_path: Path, label: str) -> None:
    """Save the PyG Data object as a .pt file for downstream graph comparison."""
    graph_path = output_path / f"graph_{label}.pt"
    torch.save(data, str(graph_path))
    logger.info(f"Saved PyG graph: {graph_path}")


def _extract_curvature(checkpoint: dict) -> float:
    """
    Extract curvature c from checkpoint via softplus(log_c) + 1e-4.

    Returns the scalar float value.
    """
    sd = checkpoint.get("model_state_dict", {})
    if "log_c" not in sd:
        raise KeyError("log_c not found in model_state_dict")
    log_c = sd["log_c"]
    c = F.softplus(log_c) + 1e-4
    return c.item()


def _checkpoint_input_dim(checkpoint: dict) -> int:
    """Infer expected input node_dim from checkpoint node embedding weights."""
    sd = checkpoint.get("model_state_dict", {})
    w = sd.get("node_emb.weight")
    if w is None or not hasattr(w, "shape") or len(w.shape) != 2:
        raise KeyError("node_emb.weight not found in checkpoint model_state_dict")
    return int(w.shape[1])


def _run_single_condition(
    model: GOSPConeMapper, ingestion: dict, device: torch.device
) -> dict:
    """
    Run GNN forward pass for one condition (GDP or GTP).

    Returns dict of numpy arrays for all GNN outputs.
    """
    data = _build_pyg_data(ingestion)
    data = data.to(device)

    with torch.no_grad():
        output = model(data)

    # Validate outputs
    findings = scan_tensors_for_invalid(output)
    if findings:
        raise RuntimeError(
            f"GNN forward pass produced invalid values: {findings}"
        )

    def _np(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy().astype(np.float64)

    def _np_squeeze(t: torch.Tensor) -> np.ndarray:
        return t.squeeze(-1).detach().cpu().numpy().astype(np.float64)

    return {
        "cone_depth": _np_squeeze(output["cone_depth"]),
        "cone_width": _np_squeeze(output["cone_width"]),
        "projections": _np(output["projections"]),
        "epistemic": _np_squeeze(output["uncertainty"]["epistemic"]),
        "aleatoric": _np_squeeze(output["uncertainty"]["aleatoric"]),
        "total_uncertainty": _np_squeeze(output["uncertainty"]["total"]),
        "expert_weights": _np(output["expert_weights"]),
        "evidence_mu": _np_squeeze(output["evidence"]["mu"]),
        "evidence_nu": _np_squeeze(output["evidence"]["nu"]),
        "evidence_alpha": _np_squeeze(output["evidence"]["alpha"]),
        "evidence_beta": _np_squeeze(output["evidence"]["beta"]),
        "audit_trail": output.get("audit_trail", {}),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_gnn(
    gdp_ingestion: dict,
    gtp_ingestion: dict,
    checkpoint_path: Path,
    output_path: Path,
) -> dict:
    """
    Run GOSPConeMapper on both GDP and GTP conditions, write gnn_output.npz.

    Parameters
    ----------
    gdp_ingestion : dict
        Output from ingest_pdb() for the GDP condition.
    gtp_ingestion : dict
        Output from ingest_pdb() for the GTP condition.
    checkpoint_path : Path
        Path to robust_experts.pt (or compatible checkpoint).
    output_path : Path
        Directory where gnn_output.npz will be written.

    Returns
    -------
    dict with all output arrays (same keys as gnn_output.npz).
    """
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    device = torch.device("cpu")

    # --- Load checkpoint and extract curvature ---
    if checkpoint_path.exists():
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(
            str(checkpoint_path), map_location=device, weights_only=False
        )
        curvature_c = _extract_curvature(checkpoint)
        logger.info(f"Extracted curvature c = {curvature_c:.6f}")
    else:
        logger.warning(
            f"Checkpoint {checkpoint_path} not found — using fallback c=1.2. "
            "This is not suitable for production runs."
        )
        checkpoint = None
        curvature_c = 1.2

    # --- Instantiate model ---
    # Enforce canonical 28-dim feature contract.
    sample_features = build_node_features(gdp_ingestion)
    actual_node_dim = sample_features.shape[1]
    if actual_node_dim != NODE_FEATURE_DIM:
        raise ValueError(
            f"Canonical GNN feature dimension must be {NODE_FEATURE_DIM}, got {actual_node_dim}"
        )

    if checkpoint is not None:
        ckpt_dim = _checkpoint_input_dim(checkpoint)
        if ckpt_dim != NODE_FEATURE_DIM:
            raise ValueError(
                f"Checkpoint input dim mismatch: expected {NODE_FEATURE_DIM}, checkpoint has {ckpt_dim}. "
                "Use a 28-dim checkpoint or retrain the model with canonical inputs."
            )

    model = GOSPConeMapper(
        node_dim=NODE_FEATURE_DIM,
        hidden=128,
        num_layers=6,
        num_experts=4,
        projection_dim=64,
    ).to(device)

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        logger.info("Loaded GOSPConeMapper state dict")

    model.eval()

    # --- Run inference for both conditions ---
    logger.info(
        f"Running GNN: GDP ({len(gdp_ingestion['rho'])} residues), "
        f"GTP ({len(gtp_ingestion['rho'])} residues)"
    )

    gdp_out = _run_single_condition(model, gdp_ingestion, device)
    gtp_out = _run_single_condition(model, gtp_ingestion, device)

    # Save PyG graphs for downstream graph comparison (edge sym diff, centrality, etc.)
    gdp_data = _build_pyg_data(gdp_ingestion)
    gtp_data = _build_pyg_data(gtp_ingestion)
    _save_pyg_graph(gdp_data, output_path, "gdp")
    _save_pyg_graph(gtp_data, output_path, "gtp")

    # --- Build combined output dict ---
    result = {}

    # Prefixed per-condition arrays
    for prefix, ingestion, gnn_out in [
        ("gdp_", gdp_ingestion, gdp_out),
        ("gtp_", gtp_ingestion, gtp_out),
    ]:
        # Passthrough from ingestion
        result[f"{prefix}residue_ids"] = np.asarray(
            ingestion["residue_ids"], dtype=object
        )
        result[f"{prefix}ca_coords"] = np.asarray(
            ingestion["ca_coords"], dtype=np.float64
        )
        result[f"{prefix}no_midpoints"] = np.asarray(
            ingestion["no_midpoints"], dtype=np.float64
        )

        # GNN outputs
        result[f"{prefix}cone_depth"] = gnn_out["cone_depth"]
        result[f"{prefix}cone_width"] = gnn_out["cone_width"]
        result[f"{prefix}projections"] = gnn_out["projections"]
        result[f"{prefix}epistemic"] = gnn_out["epistemic"]
        result[f"{prefix}aleatoric"] = gnn_out["aleatoric"]
        result[f"{prefix}total_uncertainty"] = gnn_out["total_uncertainty"]
        result[f"{prefix}expert_weights"] = gnn_out["expert_weights"]
        result[f"{prefix}evidence_mu"] = gnn_out["evidence_mu"]
        result[f"{prefix}evidence_nu"] = gnn_out["evidence_nu"]
        result[f"{prefix}evidence_alpha"] = gnn_out["evidence_alpha"]
        result[f"{prefix}evidence_beta"] = gnn_out["evidence_beta"]

    # Scalar curvature
    result["curvature_c"] = np.float64(curvature_c)

    # Audit trail as JSON string
    audit = {
        "gdp_audit": _serialize_audit(gdp_out.get("audit_trail", {})),
        "gtp_audit": _serialize_audit(gtp_out.get("audit_trail", {})),
        "curvature_c": curvature_c,
        "checkpoint": str(checkpoint_path),
        "gdp_n_residues": len(gdp_ingestion["rho"]),
        "gtp_n_residues": len(gtp_ingestion["rho"]),
    }
    result["audit_trail"] = np.array(json.dumps(audit), dtype=object)

    # --- Write gnn_output.npz ---
    output_path.mkdir(parents=True, exist_ok=True)
    npz_path = output_path / "gnn_output.npz"
    np.savez(str(npz_path), **result)
    logger.info(f"Wrote {npz_path}")

    return result


def _serialize_audit(audit: dict) -> dict:
    """Convert torch tensors in audit trail to Python scalars for JSON."""
    out = {}
    for k, v in audit.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.item() if v.numel() == 1 else v.tolist()
        elif isinstance(v, dict):
            out[k] = _serialize_audit(v)
        else:
            out[k] = v
    return out
