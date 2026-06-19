"""
shell_signal_diagnostics.py — Outer Shell Signal Diagnostics for GNNv5 / DTIE v6
==================================================================================
Eidetix Bio | 2026-06-19

Tests whether GNNv5 (the "v6" manifold) is still inferring on the outer shell
— the high-SASA, high-uncertainty, solvent-exposed communication layer that was
explicitly rendered in GNNv2 but has since become implicit.

Four diagnostic probes, each targeting a different failure mode:

  PROBE 1 — Uncertainty vs. Solvent Accessibility Correlation
    If the shell signal is alive, high epistemic uncertainty + high cone_depth
    should correlate positively with high SASA (node feature index 3).
    Weak or inverted correlation = shell signal diluted.

  PROBE 2 — Core-Only Ablation
    Strip out all residues with SASA > 0.25 or cone_depth above the 75th
    percentile, then compare inference quality metrics.  A measurable drop in
    uncertainty calibration (epistemic spread, depth variance) when the shell
    is removed confirms the model was relying on it.

  PROBE 3 — Surface Hotspot Alignment
    Take the top-N residues by composite peripheral score (high epistemic *
    high cone_depth, i.e. the "persistent-leak" proxy) and check what fraction
    are actually surface-exposed (SASA > 0.20) and what fraction fall in known
    functional surface regions (Switch-I/II, P-loop for KRAS).

  PROBE 4 — Radial Gradient Strength
    Color residues by cone_depth on the 2D disc.  The shell-alive signal
    is a clear radial gradient: low cone_depth at the disc centre (core) and
    high cone_depth at the periphery (shell).  Quantify this as Pearson r
    between |hyp_proj_2d| (disc radius) and cone_depth.  r > 0.6 = shell
    visible; r < 0.3 = gradient collapsed.

Usage:
    # Inside the science container:
    docker compose run science python -m experiments.diagnostics.shell_signal_diagnostics \\
        --checkpoint /app/checkpoints/v5/tokyo_eyes_v5.pt \\
        --pdb_dir /tmp/dtie_pdb_cache \\
        --structure_id 9O0R --chain A

    # Or from repo root directly:
    python -m experiments.diagnostics.shell_signal_diagnostics \\
        --checkpoint checkpoints/v5/tokyo_eyes_v5.pt \\
        --pdb_dir /tmp/dtie_pdb_cache

    # Or with a pre-computed JSON export (skips inference):
    python -m experiments.diagnostics.shell_signal_diagnostics \\
        --precomputed_json /path/to/disc_data.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared constants — feature engineering (must match training)
# ---------------------------------------------------------------------------

TAU          = 13.0   # dehydron wrapping threshold
EDGE_CUTOFF  = 8.0    # Å — Cα radius graph cutoff
SASA_CUTOFF  = 10.0   # Å — neighbour shell for SASA proxy

SASA_SURFACE_THRESHOLD = 0.20   # residues with SASA ≥ this are "surface-exposed"
SASA_OUTER_THRESHOLD   = 0.25   # ablation: treat SASA ≥ this as "outer shell"
DEPTH_OUTER_PERCENTILE = 75     # ablation: also treat cone_depth ≥ p75 as "outer shell"

# KRAS domain annotations used for Probe 3 functional surface check.
KRAS_DOMAINS = {
    "P-loop":    list(range(10, 18)),
    "Switch-I":  list(range(25, 41)),
    "Switch-II": list(range(57, 76)),
}
KRAS_FUNCTIONAL_SURFACE = {r for rng in KRAS_DOMAINS.values() for r in rng}

SEP = "=" * 72


# ---------------------------------------------------------------------------
# Protein graph builder — v5-compatible, no v4 model imports
# ---------------------------------------------------------------------------

def _download_pdb(pdb_id: str, pdb_dir: Path) -> Path:
    import urllib.request
    pdb_dir.mkdir(parents=True, exist_ok=True)
    local = pdb_dir / f"{pdb_id}.pdb"
    if local.exists():
        return local
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    logger.info("Downloading %s from RCSB...", pdb_id)
    urllib.request.urlretrieve(url, local)
    return local


def _extract_chain(pdb_path: Path, chain_id: str) -> Path:
    from Bio.PDB import PDBParser, PDBIO, Select

    class _ChainSelect(Select):
        def accept_chain(self, chain):
            return chain.id == chain_id

    out = pdb_path.parent / f"{pdb_path.stem}_{chain_id}.pdb"
    if out.exists():
        return out
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out), _ChainSelect())
    return out


def _compute_rho(residue, all_atoms: list, radius: float = 6.5) -> float:
    from Bio.PDB import NeighborSearch
    try:
        mid = (residue["N"].get_coord() + residue["O"].get_coord()) / 2.0
    except KeyError:
        return -1.0
    POLAR = {"ARG","ASN","ASP","GLN","GLU","HIS","LYS","SER","THR","TYR","TRP"}
    ns = NeighborSearch(all_atoms)
    count = sum(
        1 for a in ns.search(mid, radius, level="A")
        if a.element == "C"
        and a.get_parent().get_resname().strip() not in POLAR
        and a.name != "C"
    )
    return float(count)


def _compute_sasa_proxy(ca_coords: np.ndarray) -> np.ndarray:
    from scipy.spatial.distance import cdist
    dists = cdist(ca_coords, ca_coords)
    nc = ((dists < SASA_CUTOFF) & (dists > 0.1)).sum(axis=1).astype(np.float64)
    mx = nc.max()
    return 1.0 - (nc / mx) if mx > 0 else np.full(len(ca_coords), 0.5)


def _compute_ss_geometric(ca_coords: np.ndarray) -> np.ndarray:
    n = len(ca_coords)
    ss = np.ones(n, dtype=np.float64)
    for i in range(2, n - 2):
        v1 = ca_coords[i] - ca_coords[i - 2]
        v2 = ca_coords[i + 2] - ca_coords[i]
        d1, d2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if d1 < 1e-6 or d2 < 1e-6:
            continue
        cos_a = np.clip(np.dot(v1, v2) / (d1 * d2), -1.0, 1.0)
        if cos_a < 0.5 and d1 < 7.0:
            ss[i] = 0.0   # helix
        elif cos_a > 0.8:
            ss[i] = 0.5   # extended
    return ss


def build_protein_graph(pdb_id: str, chain: str, pdb_dir: Path) -> Optional[Dict]:
    """
    Build a PyG Data object with node features [rho, tau_flag, ss_type, sasa]
    using the same feature engineering as v5 training — but importing only
    from the v5 model package (GOSPConeMapper, precompute_clustering).
    """
    import torch
    from torch_geometric.data import Data
    from scipy.spatial.distance import cdist
    from Bio.PDB import PDBParser

    # Import ONLY from the v5 model package.
    from science.dtie.v5.gnn.model import precompute_clustering

    pdb_path = _download_pdb(pdb_id, pdb_dir)
    chain_path = _extract_chain(pdb_path, chain)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, str(chain_path))
    residues = [r for r in structure.get_residues() if r.get_id()[0] == " "]

    if len(residues) < 10:
        logger.warning("%s chain %s: only %d residues", pdb_id, chain, len(residues))
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
        logger.warning("%s: fewer than 10 valid residues", pdb_id)
        return None

    rho_arr  = np.array(rho_list, dtype=np.float64)
    ca_coords = np.array(ca_list,  dtype=np.float64)

    tau_flag = (rho_arr < TAU).astype(np.float64)
    ss_type  = _compute_ss_geometric(ca_coords)
    sasa     = _compute_sasa_proxy(ca_coords)

    x = np.stack([rho_arr, tau_flag, ss_type, sasa], axis=1).astype(np.float32)

    dists = cdist(ca_coords, ca_coords)
    src, dst = np.where((dists < EDGE_CUTOFF) & (dists > 0.1))
    rel_pos  = ca_coords[dst] - ca_coords[src]
    edge_attr = np.column_stack([rel_pos, dists[src, dst]]).astype(np.float32)

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor(np.stack([src, dst]), dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
    )
    data = precompute_clustering(data)

    return {
        "pdb_id":      pdb_id,
        "data":        data,
        "ca_coords":   torch.tensor(ca_coords, dtype=torch.float32),
        "residue_ids": res_ids,
        "n_residues":  len(rho_arr),
    }


# ---------------------------------------------------------------------------
# Inference runner — v5 only
# ---------------------------------------------------------------------------

def _patch_radial_head_if_needed(state: dict, hidden: int = 128) -> None:
    """
    Detect the RadialHead architecture from the checkpoint state dict and
    monkey-patch science.dtie.v5.gnn.model.RadialHead to match if it differs
    from the current code.

    The checkpoint was trained with a 2-layer RadialHead (hidden → hidden//2 → 1).
    The current model.py defines a 3-layer RadialHead (hidden → hidden//2 → hidden//4 → 1).
    If the checkpoint is missing 'radial_head.net.4.weight', apply the 2-layer patch.
    """
    import torch.nn as nn
    import torch.nn.functional as F
    import science.dtie.v5.gnn.model as _model_mod

    has_3layer = any("radial_head.net.4" in k for k in state)

    if not has_3layer:
        logger.info(
            "Checkpoint has 2-layer RadialHead (hidden→hidden//2→1). "
            "Patching model module to match checkpoint architecture."
        )

        class _RadialHead2Layer(nn.Module):
            def __init__(self, hidden_dim: int):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.SiLU(),
                    nn.Linear(hidden_dim // 2, 1),
                )
                self.radial_scale = nn.Parameter(
                    __import__("torch").tensor(0.5)
                )

            def forward(self, x):
                raw = self.net(x)
                return F.softplus(raw) * F.softplus(self.radial_scale)

        _model_mod.RadialHead = _RadialHead2Layer
        logger.info("RadialHead patched to 2-layer variant.")


def run_v5_inference(
    structure_id: str,
    chain: str,
    checkpoint_path: str,
    pdb_dir: Path,
    device: str,
) -> Tuple[dict, dict]:
    """Load the v5 checkpoint, run inference, return (probe_dict, prot_dict)."""
    import torch
    from science.dtie.v5.gnn.model import GOSPConeMapper

    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(ckpt, map_location=device, weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]

    # Patch RadialHead in the model module BEFORE instantiating GOSPConeMapper.
    # GOSPConeMapper.__init__ calls RadialHead(hidden) via its module namespace,
    # so replacing the name there is sufficient — no reload needed.
    _patch_radial_head_if_needed(state)

    model = GOSPConeMapper(node_dim=4, hidden=128, num_experts=4)
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    prot = build_protein_graph(structure_id, chain, pdb_dir)
    if prot is None:
        raise RuntimeError(f"Could not build graph for {structure_id}:{chain}")

    data = prot["data"].to(device)
    with torch.no_grad():
        raw = model(data)

    out = {
        "cone_depth":   raw["cone_depth"].squeeze().cpu().numpy(),
        "cone_width":   raw["cone_width"].squeeze().cpu().numpy(),
        "epistemic":    raw["uncertainty"]["epistemic"].cpu().numpy(),
        "aleatoric":    raw["uncertainty"]["aleatoric"].cpu().numpy(),
        "hyp_proj_2d":  raw["hyp_projections_2d"].cpu().numpy(),
        "x_routed_hyp": raw["x_routed_hyp"].cpu().numpy(),
        "sasa":         data.x[:, 3].cpu().numpy(),
        "rho":          data.x[:, 0].cpu().numpy(),
        "residue_ids":  prot["residue_ids"],
    }
    if "expert_weights" in raw:
        out["expert_weights"] = raw["expert_weights"].cpu().numpy()

    return out, prot


def load_from_json(json_path: str) -> dict:
    """Load pre-computed outputs from a JSON export (export_for_viewer format)."""
    with open(json_path) as f:
        data = json.load(f)
    nodes = data.get("nodes", data)
    return {
        "cone_depth":  np.array([nd.get("cone_depth", 0.0)           for nd in nodes]),
        "cone_width":  np.array([nd.get("cone_width", 0.0)            for nd in nodes]),
        "epistemic":   np.array([nd.get("epistemic_uncertainty", 0.0) for nd in nodes]),
        "aleatoric":   np.array([nd.get("aleatoric_uncertainty", 0.0) for nd in nodes]),
        "hyp_proj_2d": np.array([nd.get("disc_pos", [0.0, 0.0])       for nd in nodes]),
        "sasa":        np.array([nd.get("sasa", 0.0)                  for nd in nodes]),
        "rho":         np.array([nd.get("rho", 0.0)                   for nd in nodes]),
        "residue_ids": [nd.get("residue_id", f"?:{i}:") for i, nd in enumerate(nodes)],
    }


# ---------------------------------------------------------------------------
# PROBE 1 — Uncertainty vs. Solvent Accessibility Correlation
# ---------------------------------------------------------------------------

def probe1_uncertainty_sasa_correlation(out: dict) -> bool:
    print(f"\n{SEP}")
    print("PROBE 1: Epistemic Uncertainty & cone_depth vs. SASA")
    print(SEP)
    print(
        "Hypothesis: if the outer shell is alive, high epistemic uncertainty\n"
        "and high cone_depth should co-occur with high SASA (node feature 3).\n"
    )

    sasa      = out["sasa"]
    depth     = out["cone_depth"]
    epistemic = out["epistemic"]
    aleatoric = out["aleatoric"]

    r_epi_sasa   = _pearson(epistemic, sasa)
    r_depth_sasa = _pearson(depth, sasa)
    r_ale_sasa   = _pearson(aleatoric, sasa)
    r_epi_depth  = _pearson(epistemic, depth)

    print(f"  {'Pair':<36} {'Pearson r':>10}  Interpretation")
    print(f"  {'-'*72}")
    _prow("epistemic_uncertainty × SASA",  r_epi_sasa,
          "> 0.40 = shell alive;  < 0.20 = signal diluted")
    _prow("cone_depth × SASA",             r_depth_sasa,
          "> 0.40 = radial gradient intact; < 0.20 = collapsed")
    _prow("aleatoric_uncertainty × SASA",  r_ale_sasa,
          "> 0.30 = model sees surface noise")
    _prow("epistemic × cone_depth",        r_epi_depth,
          "> 0.50 = uncertainty tracks depth (well calibrated)")

    # Stratum stats: core / intermediate / shell
    labels = _shell_labels(sasa, depth)
    _print_stratum_stats(out, labels, ["epistemic", "aleatoric", "cone_depth"])

    verdict = (r_epi_sasa > 0.40) and (r_depth_sasa > 0.40)
    print(f"\n  VERDICT: outer shell signal is "
          f"{'ALIVE ✓' if verdict else 'WEAK or ABSENT ✗'} "
          f"(r_epi_sasa={r_epi_sasa:.3f}, r_depth_sasa={r_depth_sasa:.3f})")
    return verdict


# ---------------------------------------------------------------------------
# PROBE 2 — Core-Only Ablation
# ---------------------------------------------------------------------------

def probe2_core_ablation(
    out: dict,
    checkpoint_path: Optional[str],
    prot: Optional[dict],
    device: str,
) -> bool:
    print(f"\n{SEP}")
    print("PROBE 2: Core-Only Ablation (mask outer shell residues)")
    print(SEP)
    print(
        f"Outer shell = SASA ≥ {SASA_OUTER_THRESHOLD} OR "
        f"cone_depth ≥ p{DEPTH_OUTER_PERCENTILE}.\n"
        "A ≥10% drop in epistemic spread when the shell is excluded\n"
        "confirms the model was relying on shell information.\n"
    )

    sasa  = out["sasa"]
    depth = out["cone_depth"]
    n     = len(sasa)

    depth_cutoff = float(np.percentile(depth, DEPTH_OUTER_PERCENTILE))
    shell_mask   = (sasa >= SASA_OUTER_THRESHOLD) | (depth >= depth_cutoff)
    core_mask    = ~shell_mask

    print(f"  Total residues : {n}")
    print(f"  Outer shell    : {shell_mask.sum()} ({100*shell_mask.mean():.1f}%)  "
          f"[depth cutoff = {depth_cutoff:.3f}]")
    print(f"  Core           : {core_mask.sum()} ({100*core_mask.mean():.1f}%)")

    full_epi_spread  = float(out["epistemic"].std())
    full_depth_std   = float(depth.std())
    core_epi_spread  = float(out["epistemic"][core_mask].std())
    core_depth_std   = float(depth[core_mask].std())

    print(f"\n  {'Metric':<28} {'Full':>8}  {'Core-only':>10}  {'Δ':>8}")
    print(f"  {'-'*60}")
    epi_drop   = (full_epi_spread - core_epi_spread)  / (full_epi_spread  + 1e-9)
    depth_drop = (full_depth_std  - core_depth_std)   / (full_depth_std   + 1e-9)
    print(f"  {'epistemic spread (σ)':<28} {full_epi_spread:>8.4f}  "
          f"{core_epi_spread:>10.4f}  {epi_drop:>+7.1%}")
    print(f"  {'cone_depth spread (σ)':<28} {full_depth_std:>8.4f}  "
          f"{core_depth_std:>10.4f}  {depth_drop:>+7.1%}")
    print(f"  {'epistemic mean':<28} {out['epistemic'].mean():>8.4f}  "
          f"{out['epistemic'][core_mask].mean():>10.4f}")

    # Live re-inference: zero shell nodes and rerun the full forward pass.
    if checkpoint_path and prot is not None:
        try:
            _probe2_live_ablation(out, prot, checkpoint_path, core_mask, device)
        except Exception as exc:
            print(f"\n  [live re-inference skipped: {exc}]")
    else:
        print(
            "\n  NOTE: pass --checkpoint + a live prot to enable causal re-inference."
        )

    verdict = abs(epi_drop) > 0.10
    print(f"\n  VERDICT: shell contribution is "
          f"{'SIGNIFICANT ✓' if verdict else 'MARGINAL ✗'} "
          f"(epistemic spread Δ = {epi_drop:+.1%})")
    return verdict


def _probe2_live_ablation(
    out: dict, prot: dict, checkpoint_path: str, core_mask: np.ndarray, device: str
) -> None:
    import torch
    from science.dtie.v5.gnn.model import GOSPConeMapper, precompute_clustering

    model = GOSPConeMapper(node_dim=4, hidden=128, num_experts=4)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    data_core = precompute_clustering(prot["data"].clone()).to(device)
    shell_t = torch.tensor(~core_mask, dtype=torch.bool, device=device)
    data_core.x[shell_t] = 0.0   # zero all shell node features

    with torch.no_grad():
        raw_core = model(data_core)

    abl_epi   = raw_core["uncertainty"]["epistemic"].cpu().numpy()
    abl_depth = raw_core["cone_depth"].squeeze().cpu().numpy()

    full_epi_spread = float(out["epistemic"].std())
    full_depth_std  = float(out["cone_depth"].std())
    live_epi_drop   = (full_epi_spread - abl_epi.std())   / (full_epi_spread + 1e-9)
    live_depth_drop = (full_depth_std  - abl_depth.std()) / (full_depth_std  + 1e-9)

    print(f"\n  Live-ablated (shell nodes zeroed — causal effect):")
    print(f"  {'epistemic spread (σ)':<28} {full_epi_spread:>8.4f}  "
          f"{abl_epi.std():>10.4f}  {live_epi_drop:>+7.1%}")
    print(f"  {'cone_depth spread (σ)':<28} {full_depth_std:>8.4f}  "
          f"{abl_depth.std():>10.4f}  {live_depth_drop:>+7.1%}")


# ---------------------------------------------------------------------------
# PROBE 3 — Surface Hotspot Alignment
# ---------------------------------------------------------------------------

def probe3_surface_hotspot_alignment(out: dict, top_n: int = 30) -> bool:
    print(f"\n{SEP}")
    print(f"PROBE 3: Surface Hotspot Alignment (top-{top_n} peripheral residues)")
    print(SEP)
    print(
        "Composite score = epistemic_uncertainty × cone_depth.\n"
        "Checks whether the top residues are surface-exposed and overlap\n"
        "known KRAS functional surface regions (Switch-I/II, P-loop).\n"
    )

    sasa      = out["sasa"]
    depth     = out["cone_depth"]
    epistemic = out["epistemic"]
    res_ids   = out.get("residue_ids", [])

    score   = epistemic * depth
    ranked  = np.argsort(score)[::-1]
    top_idx = ranked[:top_n]

    n_surface      = int((sasa[top_idx] >= SASA_SURFACE_THRESHOLD).sum())
    bg_surface_rate = float((sasa >= SASA_SURFACE_THRESHOLD).mean())
    top_surface_rate = n_surface / len(top_idx)
    enrichment = top_surface_rate / (bg_surface_rate + 1e-9)

    print(f"  Top-{top_n} surface-exposed (SASA≥{SASA_SURFACE_THRESHOLD}): "
          f"{n_surface}/{top_n} = {top_surface_rate:.1%}  "
          f"(background: {bg_surface_rate:.1%})")
    print(f"  Surface enrichment: {enrichment:.2f}×  "
          f"({'ENRICHED ✓' if enrichment > 1.5 else 'not enriched ✗'})")

    functional_hits = 0
    if res_ids:
        print(f"\n  {'Rank':<5} {'ResID':<14} {'Score':>8} {'SASA':>6} "
              f"{'Depth':>7} {'Epi':>7} {'KRAS func?':>11}")
        print(f"  {'-'*66}")
        for rank, i in enumerate(top_idx, 1):
            rid    = res_ids[i] if i < len(res_ids) else f"?:{i}:"
            resnum = _parse_resnum(rid)
            is_func = resnum in KRAS_FUNCTIONAL_SURFACE if resnum else False
            if is_func:
                functional_hits += 1
            marker = "★" if is_func else " "
            print(f"  {rank:<5} {rid:<14} {score[i]:>8.4f} {sasa[i]:>6.3f} "
                  f"{depth[i]:>7.3f} {epistemic[i]:>7.4f} {marker:>11}")

        func_rate = functional_hits / top_n
        print(f"\n  Overlap with KRAS functional surface "
              f"(Switch-I/II, P-loop): {functional_hits}/{top_n} = {func_rate:.1%}")

    verdict = enrichment > 1.4
    print(f"\n  VERDICT: top peripheral cluster "
          f"{'SURFACE-ENRICHED ✓' if verdict else 'NOT surface-enriched ✗'} "
          f"(enrichment = {enrichment:.2f}×)")
    return verdict


# ---------------------------------------------------------------------------
# PROBE 4 — Radial Gradient Strength
# ---------------------------------------------------------------------------

def probe4_radial_gradient(out: dict) -> bool:
    print(f"\n{SEP}")
    print("PROBE 4: Radial Gradient in the 2D Poincaré Disc")
    print(SEP)
    print(
        "Shell-alive: low cone_depth → disc centre (core),\n"
        "             high cone_depth → disc periphery (outer shell).\n"
        "Pearson r between |hyp_proj_2d| and cone_depth:\n"
        "  r > 0.60 = gradient intact\n"
        "  r < 0.30 = gradient collapsed\n"
    )

    hyp   = out["hyp_proj_2d"]
    depth = out["cone_depth"]
    sasa  = out["sasa"]

    disc_r = np.linalg.norm(hyp, axis=1) if hyp.ndim == 2 else np.abs(hyp)

    r_r_depth = _pearson(disc_r, depth)
    r_r_epi   = _pearson(disc_r, out["epistemic"])
    r_r_sasa  = _pearson(disc_r, sasa)

    print(f"  {'Pair':<36} {'Pearson r':>10}  Strength")
    print(f"  {'-'*68}")
    _prow("|proj| × cone_depth",   r_r_depth,
          "> 0.60 strong; 0.30–0.60 moderate; < 0.30 collapsed")
    _prow("|proj| × epistemic",    r_r_epi,
          "high = uncertainty lives at disc periphery")
    _prow("|proj| × SASA",         r_r_sasa,
          "high = Poincaré periphery maps to molecular surface")

    # Bin by radius percentile and report mean depth, SASA, epistemic per bin.
    pctiles    = [0, 20, 40, 60, 80, 100]
    thresholds = np.percentile(disc_r, pctiles)
    print(f"\n  Radial bins (disc radius percentile → mean values per bin):")
    print(f"  {'Bin':<20} {'N':>5} {'mean depth':>11} "
          f"{'mean SASA':>10} {'mean epi':>9}")
    print(f"  {'-'*60}")
    for plo, phi, lo, hi in zip(
        pctiles[:-1], pctiles[1:], thresholds[:-1], thresholds[1:]
    ):
        mask = (disc_r >= lo) & (disc_r < hi if phi < 100 else disc_r <= hi)
        cnt  = int(mask.sum())
        if cnt == 0:
            continue
        print(f"  p{plo:>2}–p{phi:<2} (r={lo:.3f}–{hi:.3f})"
              f"  {cnt:>5}  {depth[mask].mean():>11.4f}"
              f"  {sasa[mask].mean():>10.4f}  {out['epistemic'][mask].mean():>9.4f}")

    verdict = r_r_depth > 0.50
    print(f"\n  VERDICT: radial gradient is "
          f"{'INTACT ✓' if verdict else 'WEAK or ABSENT ✗'} "
          f"(r = {r_r_depth:.3f})")
    return verdict


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(verdicts: List[Tuple[str, bool]]) -> None:
    print(f"\n{SEP}")
    print("SUMMARY — Outer Shell Signal Health in GNNv5 / DTIE v6")
    print(SEP)
    passed = sum(v for _, v in verdicts)
    for label, ok in verdicts:
        print(f"  {'✓' if ok else '✗'}  {label}")
    print(f"\n  Score: {passed}/{len(verdicts)} probes passed")
    if passed == len(verdicts):
        print("\n  The outer shell is a first-class citizen in the current model. ✓")
    elif passed >= 2:
        print(
            "\n  Partial shell signal — the model is using the shell implicitly\n"
            "  but not as an explicit communicative layer. Consider adding:\n"
            "    • Explicit RSA / dehydron wrapper node features\n"
            "    • Auxiliary shell-membership loss\n"
            "    • Dual-manifold message passing (separate shell attention head)"
        )
    else:
        print(
            "\n  Shell signal is significantly degraded. GNNv5 is primarily\n"
            "  inferring on the inner core. The outer shell needs to be restored\n"
            "  as a first-class structural compartment to recover the v2 insight."
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 3:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _prow(label: str, r: float, note: str) -> None:
    r_str = f"{r:>10.3f}" if np.isfinite(r) else f"{'NaN':>10}"
    print(f"  {label:<36} {r_str}  {note}")


def _shell_labels(sasa: np.ndarray, depth: np.ndarray) -> np.ndarray:
    d75, d25 = np.percentile(depth, 75), np.percentile(depth, 25)
    labels = np.full(len(sasa), "intermediate", dtype=object)
    labels[(sasa < SASA_SURFACE_THRESHOLD) & (depth < d25)] = "core"
    labels[(sasa >= SASA_SURFACE_THRESHOLD) & (depth >= d75)] = "shell"
    return labels


def _print_stratum_stats(
    out: dict, labels: np.ndarray, fields: List[str]
) -> None:
    header = f"\n  {'Stratum':<14} {'N':>5}"
    for f in fields:
        header += f"  {f[:12]:>12}"
    print(header)
    print(f"  {'-' * (22 + 14 * len(fields))}")
    for s in ["core", "intermediate", "shell"]:
        mask = labels == s
        if not mask.any():
            continue
        row = f"  {s:<14} {int(mask.sum()):>5}"
        for f in fields:
            arr = out.get(f)
            if arr is not None:
                row += f"  {float(arr[mask].mean()):>12.4f}"
        print(row)


def _parse_resnum(res_id: str) -> Optional[int]:
    try:
        parts = res_id.split(":")
        return int(parts[1]) if len(parts) >= 2 else int(parts[0])
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="GNNv5 Outer Shell Signal Diagnostics"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="checkpoints/v5/tokyo_eyes_v5.pt",
        help="Path to v5 checkpoint (.pt file)",
    )
    parser.add_argument(
        "--pdb_dir", type=str, default="/tmp/dtie_pdb_cache",
        help="PDB cache directory (structures downloaded here if missing)",
    )
    parser.add_argument(
        "--structure_id", type=str, default="4OBE",
        help="PDB structure ID to analyse",
    )
    parser.add_argument(
        "--chain", type=str, default="A",
        help="Chain to analyse",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Torch device: cpu or cuda:0",
    )
    parser.add_argument(
        "--precomputed_json", type=str, default=None,
        help="Skip inference: load pre-computed GNN outputs from a JSON export",
    )
    parser.add_argument(
        "--top_n", type=int, default=30,
        help="Number of top peripheral residues to inspect in Probe 3",
    )
    args = parser.parse_args()

    print(f"\n{'#' * 72}")
    print("  GNNv5 / DTIE v6 — Outer Shell Signal Diagnostics")
    print(f"  Structure : {args.structure_id}:{args.chain}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"{'#' * 72}")

    prot: Optional[dict] = None
    ckpt: Optional[str]  = None

    if args.precomputed_json:
        print(f"\nLoading pre-computed outputs from {args.precomputed_json}")
        out = load_from_json(args.precomputed_json)
    else:
        print(f"\nRunning v5 inference on {args.structure_id}:{args.chain} ...")
        out, prot = run_v5_inference(
            args.structure_id, args.chain,
            args.checkpoint, Path(args.pdb_dir), args.device,
        )
        ckpt = args.checkpoint
        print(f"  {len(out['sasa'])} residues loaded and embedded.")

    # Run all four probes and collect pass/fail.
    v1 = probe1_uncertainty_sasa_correlation(out)
    v2 = probe2_core_ablation(out, ckpt, prot, args.device)
    v3 = probe3_surface_hotspot_alignment(out, top_n=args.top_n)
    v4 = probe4_radial_gradient(out)

    print_summary([
        ("Probe 1 — epistemic+depth correlate with SASA",      v1),
        ("Probe 2 — shell contributes to uncertainty spread",   v2),
        ("Probe 3 — top peripheral residues surface-enriched",  v3),
        ("Probe 4 — radial gradient intact in Poincaré disc",   v4),
    ])
    print()


if __name__ == "__main__":
    main()
