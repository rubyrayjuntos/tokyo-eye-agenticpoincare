"""
diagnose_v4.py — Model Interpretability Diagnostics for Tokyo Eyes v4
======================================================================
Eidetix Bio | 2026-05-14

Probes what the GNN has learned by testing its representations against
known protein biology. Runs on KRAS (4OBE WT, 4DSO G12D) as the primary
test case since we have ground-truth domain annotations.

Probes:
  1. Known biology vs learned depth (do catalytic/functional residues cluster?)
  2. Feature ablation sensitivity (which inputs drive the embedding?)
  3. Nearest-neighbor retrieval in hyperbolic space (spatial vs functional?)
  4. Expert routing analysis (do experts specialize by motif?)
  5. WT vs G12D differential (what changes in the embedding?)

Usage:
    uv run python DTIE_GNN_ORCHESTRATION/TokyoEyesv4/diagnose_v4.py \
        --checkpoint ./checkpoints_v4_cone_fix2/tokyo_eyes_v4.pt \
        --pdb_dir /tmp/dtie_pdb_cache
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from Gnnv4 import GOSPConeMapper, precompute_clustering
from train_v4 import load_protein_graph


# ─────────────────────────────────────────────────────────────────────────────
# KRAS domain annotations (ground truth)
# ─────────────────────────────────────────────────────────────────────────────

KRAS_DOMAINS = {
    "P-loop": range(10, 18),       # Phosphate-binding loop (GxxxxGKS/T)
    "Switch-I": range(25, 41),     # Effector binding, conformational switch
    "Switch-II": range(57, 76),    # GAP interaction, allosteric relay
    "α3-helix": range(87, 105),    # Core structural helix
    "α4-helix": range(116, 127),   # Membrane-proximal
    "C-terminal": range(145, 170), # HVR region
}

# Functionally critical residues in KRAS
KRAS_FUNCTIONAL = {
    "catalytic": [12, 13, 16, 17, 60, 61],  # GTPase active site
    "effector_contact": [32, 33, 34, 36, 37, 38, 40],  # RAF/PI3K binding
    "allosteric_hub": [12, 59, 60, 61, 63, 64],  # Switch-II relay
    "membrane_anchor": [165, 166, 167, 168, 169],  # CAAX-proximal
}


def get_residue_number(res_id: str) -> int:
    """Extract residue number from 'A:123:' format."""
    parts = res_id.split(":")
    return int(parts[1])


def assign_domain(resnum: int) -> str:
    """Assign a KRAS residue to its functional domain."""
    for name, rng in KRAS_DOMAINS.items():
        if resnum in rng:
            return name
    return "other"


# ─────────────────────────────────────────────────────────────────────────────
# Probe 1: Known biology vs learned representations
# ─────────────────────────────────────────────────────────────────────────────

def probe_biology_vs_depth(out: dict, prot: dict):
    """Check if functionally important residues have distinct depth/uncertainty."""
    print("\n" + "=" * 70)
    print("PROBE 1: Known Biology vs Learned Representations")
    print("=" * 70)

    depth = out["cone_depth"].squeeze().numpy()
    epistemic = out["uncertainty"]["epistemic"].squeeze().numpy()
    aleatoric = out["uncertainty"]["aleatoric"].squeeze().numpy()
    rho = prot["target_rho"].squeeze().numpy()
    res_ids = prot["residue_ids"]

    # Per-domain statistics
    print(f"\n{'Domain':<14} {'N':>3} {'depth_mean':>10} {'depth_std':>9} "
          f"{'epi_mean':>9} {'ale_mean':>9} {'rho_mean':>9}")
    print("-" * 70)

    domain_depths = {}
    for i, rid in enumerate(res_ids):
        resnum = get_residue_number(rid)
        domain = assign_domain(resnum)
        domain_depths.setdefault(domain, []).append(i)

    for domain in list(KRAS_DOMAINS.keys()) + ["other"]:
        indices = domain_depths.get(domain, [])
        if not indices:
            continue
        idx = np.array(indices)
        print(f"{domain:<14} {len(idx):>3} "
              f"{depth[idx].mean():>10.4f} {depth[idx].std():>9.4f} "
              f"{epistemic[idx].mean():>9.4f} {aleatoric[idx].mean():>9.4f} "
              f"{rho[idx].mean():>9.1f}")

    # Functional residue groups
    print(f"\n{'Functional Group':<18} {'N':>3} {'depth_mean':>10} "
          f"{'epi_mean':>9} {'ale_mean':>9}")
    print("-" * 55)

    for group_name, resnums in KRAS_FUNCTIONAL.items():
        indices = [i for i, rid in enumerate(res_ids)
                   if get_residue_number(rid) in resnums]
        if not indices:
            continue
        idx = np.array(indices)
        print(f"{group_name:<18} {len(idx):>3} "
              f"{depth[idx].mean():>10.4f} "
              f"{epistemic[idx].mean():>9.4f} {aleatoric[idx].mean():>9.4f}")

    # Key question: are catalytic residues deeper than surface residues?
    catalytic_idx = [i for i, rid in enumerate(res_ids)
                     if get_residue_number(rid) in KRAS_FUNCTIONAL["catalytic"]]
    surface_idx = [i for i, rid in enumerate(res_ids) if rho[i] > 20]

    if catalytic_idx and surface_idx:
        cat_depth = depth[np.array(catalytic_idx)].mean()
        surf_depth = depth[np.array(surface_idx)].mean()
        print(f"\n  Catalytic residues mean depth: {cat_depth:.4f}")
        print(f"  Surface residues (ρ>20) mean depth: {surf_depth:.4f}")
        print(f"  Δ = {cat_depth - surf_depth:.4f} "
              f"({'catalytic DEEPER ✓' if cat_depth > surf_depth else 'UNEXPECTED: surface deeper'})")


# ─────────────────────────────────────────────────────────────────────────────
# Probe 2: Feature ablation sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def probe_feature_ablation(model: GOSPConeMapper, prot: dict, device: str = "cpu"):
    """Zero out each input feature and measure embedding displacement."""
    print("\n" + "=" * 70)
    print("PROBE 2: Feature Ablation Sensitivity")
    print("=" * 70)
    print("(Which input features drive the learned representation?)\n")

    data = prot["data"].clone()
    model.eval()

    # Baseline
    with torch.no_grad():
        baseline = model(data.to(device))
    base_hyp = baseline["x_routed_hyp"].cpu().numpy()
    base_depth = baseline["cone_depth"].squeeze().cpu().numpy()

    feature_names = ["rho", "tau_flag", "ss_type", "sasa"]

    print(f"{'Feature zeroed':<14} {'Δ embedding (L2)':>16} {'Δ depth (MAE)':>14} "
          f"{'Δ expert routing':>17}")
    print("-" * 65)

    for feat_idx, feat_name in enumerate(feature_names):
        # Clone and zero one feature
        data_ablated = data.clone()
        data_ablated.x = data_ablated.x.clone()
        data_ablated.x[:, feat_idx] = 0.0

        with torch.no_grad():
            ablated = model(data_ablated.to(device))

        abl_hyp = ablated["x_routed_hyp"].cpu().numpy()
        abl_depth = ablated["cone_depth"].squeeze().cpu().numpy()
        abl_experts = ablated["expert_weights"].cpu().numpy()
        base_experts = baseline["expert_weights"].cpu().numpy()

        # Embedding displacement
        displacement = np.linalg.norm(abl_hyp - base_hyp, axis=1).mean()
        # Depth change
        depth_mae = np.abs(abl_depth - base_depth).mean()
        # Expert routing change (L1 distance in probability space)
        expert_shift = np.abs(abl_experts - base_experts).sum(axis=1).mean()

        print(f"{feat_name:<14} {displacement:>16.4f} {depth_mae:>14.4f} {expert_shift:>17.4f}")

    # Also test: randomize all features (should maximally disrupt)
    data_random = data.clone()
    data_random.x = torch.randn_like(data_random.x)
    with torch.no_grad():
        random_out = model(data_random.to(device))
    rand_hyp = random_out["x_routed_hyp"].cpu().numpy()
    rand_disp = np.linalg.norm(rand_hyp - base_hyp, axis=1).mean()
    print(f"{'ALL RANDOM':<14} {rand_disp:>16.4f} {'—':>14} {'—':>17}")


# ─────────────────────────────────────────────────────────────────────────────
# Probe 3: Nearest-neighbor retrieval in hyperbolic space
# ─────────────────────────────────────────────────────────────────────────────

def probe_nearest_neighbors(out: dict, prot: dict):
    """For key residues, find nearest neighbors in ball and check if they're
    spatially adjacent or functionally related."""
    print("\n" + "=" * 70)
    print("PROBE 3: Nearest-Neighbor Retrieval in Hyperbolic Space")
    print("=" * 70)
    print("(Are neighbors in the ball physically adjacent or functionally related?)\n")

    from geoopt.manifolds.stereographic import math as pmath

    x_hyp = out["x_routed_hyp"].cpu()
    c = torch.tensor(out["audit_trail"]["curvature_value"])
    k = -c
    res_ids = prot["residue_ids"]
    ca_coords = prot["ca_coords"].numpy()
    rho = prot["target_rho"].squeeze().numpy()

    # Query residues: G12, K16 (P-loop), T35 (Switch-I), Q61 (Switch-II), D119 (α4)
    query_resnums = [12, 16, 35, 61, 119]
    query_indices = []
    for qr in query_resnums:
        for i, rid in enumerate(res_ids):
            if get_residue_number(rid) == qr:
                query_indices.append(i)
                break

    K = 8  # neighbors to retrieve

    for qi in query_indices:
        resnum = get_residue_number(res_ids[qi])
        domain = assign_domain(resnum)

        # Hyperbolic distances from this residue to all others
        qi_hyp = x_hyp[qi:qi+1].expand(len(x_hyp), -1)
        hyp_dists = pmath.dist(qi_hyp, x_hyp, k=k).numpy()
        hyp_dists[qi] = np.inf  # exclude self

        # Physical distances
        phys_dists = np.linalg.norm(ca_coords - ca_coords[qi], axis=1)
        phys_dists[qi] = np.inf

        # Top-K in hyperbolic space
        top_k_hyp = np.argsort(hyp_dists)[:K]
        # Top-K in physical space
        top_k_phys = np.argsort(phys_dists)[:K]

        # Overlap: how many hyperbolic neighbors are also physical neighbors?
        overlap = len(set(top_k_hyp) & set(top_k_phys))

        print(f"  Residue {resnum} ({domain}, ρ={rho[qi]:.0f}):")
        print(f"    Hyperbolic {K}-NN: ", end="")
        for ni in top_k_hyp:
            nn_resnum = get_residue_number(res_ids[ni])
            nn_domain = assign_domain(nn_resnum)
            nn_phys_dist = phys_dists[ni]
            marker = "★" if nn_domain == domain else " "
            print(f"{nn_resnum}({nn_domain[:3]},{nn_phys_dist:.0f}Å){marker}", end=" ")
        print()
        print(f"    Overlap with physical {K}-NN: {overlap}/{K} "
              f"({'spatial' if overlap >= K//2 else 'FUNCTIONAL — non-local neighbors'})")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Probe 4: Expert routing analysis
# ─────────────────────────────────────────────────────────────────────────────

def probe_expert_routing(out: dict, prot: dict):
    """Analyze which expert handles which domain/functional class."""
    print("\n" + "=" * 70)
    print("PROBE 4: Expert Routing Specialization")
    print("=" * 70)
    print("(Do experts specialize by structural motif or functional role?)\n")

    expert_weights = out["expert_weights"].cpu().numpy()  # [N, num_experts]
    res_ids = prot["residue_ids"]
    rho = prot["target_rho"].squeeze().numpy()
    num_experts = expert_weights.shape[1]

    # Dominant expert per residue
    dominant = expert_weights.argmax(axis=1)

    # Per-domain expert distribution
    print(f"{'Domain':<14}", end="")
    for e in range(num_experts):
        print(f" {'Exp'+str(e):>6}", end="")
    print(f" {'Dominant':>9}")
    print("-" * (14 + 7 * num_experts + 10))

    domain_expert_counts = {}
    for i, rid in enumerate(res_ids):
        resnum = get_residue_number(rid)
        domain = assign_domain(resnum)
        domain_expert_counts.setdefault(domain, np.zeros(num_experts))
        domain_expert_counts[domain][dominant[i]] += 1

    for domain in list(KRAS_DOMAINS.keys()) + ["other"]:
        counts = domain_expert_counts.get(domain)
        if counts is None:
            continue
        total = counts.sum()
        print(f"{domain:<14}", end="")
        for e in range(num_experts):
            print(f" {counts[e]/total:>6.2f}", end="")
        print(f" {'Exp'+str(int(counts.argmax())):>9}")

    # Expert specialization score: entropy of domain distribution per expert
    print(f"\n  Expert specialization (lower entropy = more specialized):")
    for e in range(num_experts):
        # Which domains does this expert serve?
        expert_mask = dominant == e
        if not expert_mask.any():
            print(f"    Expert {e}: UNUSED")
            continue
        domain_dist = defaultdict(int)
        for i in np.where(expert_mask)[0]:
            resnum = get_residue_number(res_ids[i])
            domain_dist[assign_domain(resnum)] += 1
        total = sum(domain_dist.values())
        probs = np.array([v/total for v in domain_dist.values()])
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        top_domain = max(domain_dist, key=domain_dist.get)
        print(f"    Expert {e}: H={entropy:.2f} | "
              f"serves {total} residues | "
              f"top domain: {top_domain} ({domain_dist[top_domain]/total:.0%})")

    # Burial correlation: does any expert preferentially handle buried residues?
    print(f"\n  Expert vs burial (mean ρ per expert):")
    for e in range(num_experts):
        mask = dominant == e
        if mask.any():
            mean_rho = rho[mask].mean()
            print(f"    Expert {e}: mean ρ = {mean_rho:.1f} "
                  f"({'buried' if mean_rho < 13 else 'exposed'})")


# ─────────────────────────────────────────────────────────────────────────────
# Probe 5: WT vs G12D differential
# ─────────────────────────────────────────────────────────────────────────────

def probe_differential(model: GOSPConeMapper, pdb_dir: Path, device: str = "cpu"):
    """Compare WT and G12D embeddings — what moves?"""
    print("\n" + "=" * 70)
    print("PROBE 5: WT (4OBE) vs G12D (4DSO) Differential")
    print("=" * 70)
    print("(Which residues move most in the embedding between states?)\n")

    from geoopt.manifolds.stereographic import math as pmath

    prot_wt = load_protein_graph("4OBE", "A", pdb_dir)
    prot_g12d = load_protein_graph("4DSO", "A", pdb_dir)

    if prot_wt is None or prot_g12d is None:
        print("  ERROR: Could not load both structures")
        return

    model.eval()
    with torch.no_grad():
        out_wt = model(prot_wt["data"].to(device))
        out_g12d = model(prot_g12d["data"].to(device))

    # Align by residue number (structures may have different lengths)
    wt_resnums = {get_residue_number(r): i for i, r in enumerate(prot_wt["residue_ids"])}
    g12d_resnums = {get_residue_number(r): i for i, r in enumerate(prot_g12d["residue_ids"])}
    common = sorted(set(wt_resnums.keys()) & set(g12d_resnums.keys()))

    if len(common) < 50:
        print(f"  WARNING: Only {len(common)} common residues")
        return

    wt_hyp = out_wt["x_routed_hyp"].cpu()
    g12d_hyp = out_g12d["x_routed_hyp"].cpu()
    c = torch.tensor(model.curvature.item())
    k = -c

    # Compute per-residue hyperbolic displacement
    displacements = []
    depth_changes = []
    wt_depths = out_wt["cone_depth"].squeeze().cpu().numpy()
    g12d_depths = out_g12d["cone_depth"].squeeze().cpu().numpy()

    for resnum in common:
        wi = wt_resnums[resnum]
        gi = g12d_resnums[resnum]
        d = pmath.dist(wt_hyp[wi:wi+1], g12d_hyp[gi:gi+1], k=k).item()
        displacements.append((resnum, d))
        depth_changes.append((resnum, g12d_depths[gi] - wt_depths[wi]))

    displacements.sort(key=lambda x: -x[1])
    depth_changes.sort(key=lambda x: -abs(x[1]))

    # Null distribution: what's the typical displacement?
    all_d = np.array([d for _, d in displacements])
    p95 = np.percentile(all_d, 95)
    p50 = np.percentile(all_d, 50)

    print(f"  Displacement statistics:")
    print(f"    median = {p50:.4f}")
    print(f"    95th percentile = {p95:.4f}")
    print(f"    max = {all_d.max():.4f}")
    print()

    # Top 15 most displaced residues
    print(f"  {'Rank':<5} {'Res#':>5} {'Domain':<12} {'Displacement':>13} {'> 95th?':>8}")
    print("  " + "-" * 48)
    for rank, (resnum, d) in enumerate(displacements[:15], 1):
        domain = assign_domain(resnum)
        sig = "★" if d > p95 else ""
        print(f"  {rank:<5} {resnum:>5} {domain:<12} {d:>13.4f} {sig:>8}")

    # Domain-level summary
    print(f"\n  Per-domain mean displacement:")
    domain_disps = defaultdict(list)
    for resnum, d in displacements:
        domain_disps[assign_domain(resnum)].append(d)

    for domain in list(KRAS_DOMAINS.keys()) + ["other"]:
        vals = domain_disps.get(domain, [])
        if vals:
            print(f"    {domain:<14}: {np.mean(vals):.4f} ± {np.std(vals):.4f} "
                  f"(n={len(vals)})")

    # Key question: does Switch-I show significantly more displacement?
    switch_i_d = domain_disps.get("Switch-I", [])
    other_d = [d for dom, vals in domain_disps.items()
               if dom != "Switch-I" for d in vals]
    if switch_i_d and other_d:
        sw_mean = np.mean(switch_i_d)
        oth_mean = np.mean(other_d)
        print(f"\n  Switch-I mean displacement: {sw_mean:.4f}")
        print(f"  All other residues mean:    {oth_mean:.4f}")
        print(f"  Ratio: {sw_mean/oth_mean:.2f}x "
              f"({'Switch-I DIFFERENTIALLY DISPLACED ✓' if sw_mean > oth_mean * 1.2 else 'no significant difference'})")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GNN v4 Interpretability Diagnostics")
    parser.add_argument("--checkpoint", type=str,
                       default="./checkpoints_v4_cone_fix2/tokyo_eyes_v4.pt")
    parser.add_argument("--pdb_dir", type=str, default="/tmp/dtie_pdb_cache")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    pdb_dir = Path(args.pdb_dir)
    device = args.device

    # Load model
    print("Loading checkpoint:", args.checkpoint)
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
    model = GOSPConeMapper(
        node_dim=4, hidden=128, num_layers=6,
        num_experts=4, projection_dim=64, hyp_proj_dim=2,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    print(f"Curvature: {model.curvature.item():.4f}")
    print(f"Epoch: {ckpt.get('global_epoch', '?')}")

    # Load KRAS WT
    print("\nLoading KRAS WT (4OBE)...")
    prot = load_protein_graph("4OBE", "A", pdb_dir)
    if prot is None:
        print("ERROR: Could not load 4OBE")
        sys.exit(1)

    # Run inference
    with torch.no_grad():
        out = model(prot["data"].to(device))

    # Run all probes
    probe_biology_vs_depth(out, prot)
    probe_feature_ablation(model, prot, device)
    probe_nearest_neighbors(out, prot)
    probe_expert_routing(out, prot)
    probe_differential(model, pdb_dir, device)

    print("\n" + "=" * 70)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
