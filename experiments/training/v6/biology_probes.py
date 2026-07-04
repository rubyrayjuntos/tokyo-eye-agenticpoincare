"""
biology_probes.py — KRAS-grounded interpretability probes for v6 checkpoints.

Ports v4 diagnose_v4 Probes 1, 4, 5 onto GOSPConeMapperV6 + training graph loader.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.training.v6._data import load_protein_graph
from experiments.training.v6.assess_checkpoint import collect_protein_bundle, load_v6_model
from experiments.training.v6.corpus import load_training_proteins
from experiments.training.v6.train_loop import attach_v6_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("biology_probes")

KRAS_DOMAINS = {
    "P-loop": range(10, 18),
    "Switch-I": range(25, 41),
    "Switch-II": range(57, 76),
    "α3-helix": range(87, 105),
    "α4-helix": range(116, 127),
    "C-terminal": range(145, 170),
}

SWITCH_I_MOBILE = list(range(29, 41))
CORE_STABLE = list(range(87, 105))


def _resnum(res_id: str) -> int:
    return int(str(res_id).split(":")[1])


def _assign_domain(resnum: int) -> str:
    for name, rng in KRAS_DOMAINS.items():
        if resnum in rng:
            return name
    return "other"


def _domain_indices(residue_ids: list[str]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = defaultdict(list)
    for i, rid in enumerate(residue_ids):
        out[_assign_domain(_resnum(rid))].append(i)
    return out


def probe_biology_vs_depth(bundle: dict[str, Any], prot: dict[str, Any]) -> dict[str, Any]:
    depth = np.asarray(bundle["cone_depth"], dtype=float)
    epi = np.asarray(bundle["epistemic"], dtype=float)
    ale = np.asarray(bundle["aleatoric"], dtype=float)
    sasa = np.asarray(bundle["sasa"], dtype=float)
    domains = _domain_indices(prot["residue_ids"])

    domain_stats = {}
    for domain in list(KRAS_DOMAINS.keys()) + ["other"]:
        idx = domains.get(domain, [])
        if not idx:
            continue
        ix = np.asarray(idx)
        domain_stats[domain] = {
            "n": int(len(ix)),
            "depth_mean": float(depth[ix].mean()),
            "depth_std": float(depth[ix].std()),
            "epi_mean": float(epi[ix].mean()),
            "ale_mean": float(ale[ix].mean()),
            "sasa_mean": float(sasa[ix].mean()),
        }

    switch_i = domains.get("Switch-I", [])
    alpha3 = domains.get("α3-helix", [])
    switch_epi = float(epi[np.asarray(switch_i)].mean()) if switch_i else None
    alpha3_epi = float(epi[np.asarray(alpha3)].mean()) if alpha3 else None

    return {
        "pdb_id": prot.get("pdb_id"),
        "domain_stats": domain_stats,
        "switch1_epi_mean": switch_epi,
        "alpha3_epi_mean": alpha3_epi,
        "switch1_gt_alpha3_epi": (
            switch_epi > alpha3_epi if switch_epi is not None and alpha3_epi is not None else None
        ),
    }


def probe_expert_routing_by_domain(
    model: torch.nn.Module, prot: dict[str, Any], device: str
) -> dict[str, Any]:
    with torch.no_grad():
        out = model(attach_v6_features(prot["data"].to(device)))
    weights = out["expert_weights"].cpu().numpy()
    if weights.ndim == 1:
        weights = weights.reshape(-1, 1)
    n_experts = weights.shape[1]
    domains = _domain_indices(prot["residue_ids"])

    per_domain: dict[str, Any] = {}
    dominant_counts: dict[int, int] = defaultdict(int)
    for domain, idx in domains.items():
        if not idx or domain == "other":
            continue
        mean_w = weights[np.asarray(idx)].mean(axis=0)
        dom = int(np.argmax(mean_w))
        dominant_counts[dom] += 1
        per_domain[domain] = {
            "n": len(idx),
            "expert_load_mean": [float(x) for x in mean_w.tolist()],
            "dominant_expert": dom,
        }

    return {
        "pdb_id": prot.get("pdb_id"),
        "num_experts": n_experts,
        "per_domain": per_domain,
        "dominant_expert_by_domain": {str(k): v for k, v in dominant_counts.items()},
    }


def probe_wt_mutant_differential(
    model: torch.nn.Module,
    pdb_dir: Path,
    device: str,
    *,
    wt_pdb: str = "4OBE",
    mut_pdb: str = "4DSO",
    chain: str = "A",
) -> dict[str, Any]:
    from geoopt.manifolds.stereographic import math as pmath

    prot_wt = load_protein_graph(wt_pdb, chain, pdb_dir)
    prot_mut = load_protein_graph(mut_pdb, chain, pdb_dir)
    if prot_wt is None or prot_mut is None:
        return {"error": f"Could not load {wt_pdb} and/or {mut_pdb}"}

    model.eval()
    with torch.no_grad():
        out_wt = model(attach_v6_features(prot_wt["data"].to(device)))
        out_mut = model(attach_v6_features(prot_mut["data"].to(device)))

    wt_map = {_resnum(r): i for i, r in enumerate(prot_wt["residue_ids"])}
    mut_map = {_resnum(r): i for i, r in enumerate(prot_mut["residue_ids"])}
    common = sorted(set(wt_map) & set(mut_map))
    if len(common) < 50:
        return {"error": f"Only {len(common)} common residues"}

    wt_hyp = out_wt["x_routed_hyp"].cpu()
    mut_hyp = out_mut["x_routed_hyp"].cpu()
    k = -model.curvature.detach().cpu()

    displacements: list[tuple[int, float]] = []
    for resnum in common:
        wi, gi = wt_map[resnum], mut_map[resnum]
        d = float(pmath.dist(wt_hyp[wi : wi + 1], mut_hyp[gi : gi + 1], k=k).item())
        displacements.append((resnum, d))

    domain_disps: dict[str, list[float]] = defaultdict(list)
    for resnum, d in displacements:
        domain_disps[_assign_domain(resnum)].append(d)

    domain_summary = {
        dom: {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
        for dom, vals in domain_disps.items()
        if vals
    }

    switch_i_d = domain_disps.get("Switch-I", [])
    other_vals = [d for dom, vals in domain_disps.items() if dom != "Switch-I" for d in vals]
    switch_mean = float(np.mean(switch_i_d)) if switch_i_d else None
    other_mean = float(np.mean(other_vals)) if other_vals else None
    ratio = (switch_mean / other_mean) if switch_mean and other_mean and other_mean > 1e-9 else None

    mobile_wt = [wt_map[r] for r in SWITCH_I_MOBILE if r in wt_map and r in mut_map]
    mobile_mut = [mut_map[r] for r in SWITCH_I_MOBILE if r in wt_map and r in mut_map]
    stable_wt = [wt_map[r] for r in CORE_STABLE if r in wt_map and r in mut_map]
    stable_mut = [mut_map[r] for r in CORE_STABLE if r in wt_map and r in mut_map]
    mobile_disp = stable_disp = None
    if mobile_wt and mobile_mut:
        mobile_disp = float(pmath.dist(wt_hyp[mobile_wt], mut_hyp[mobile_mut], k=k).mean().item())
    if stable_wt and stable_mut:
        stable_disp = float(pmath.dist(wt_hyp[stable_wt], mut_hyp[stable_mut], k=k).mean().item())
    ms_ratio = (
        mobile_disp / stable_disp
        if mobile_disp is not None and stable_disp is not None and stable_disp > 1e-9
        else None
    )

    return {
        "wt_pdb": wt_pdb,
        "mut_pdb": mut_pdb,
        "n_common": len(common),
        "domain_displacement": domain_summary,
        "switch1_mean_displacement": switch_mean,
        "non_switch1_mean_displacement": other_mean,
        "switch1_displacement_ratio": ratio,
        "switch1_differential_ok": ratio is not None and ratio >= 1.2,
        "mobile_vs_stable_hyperbolic": {
            "mobile_mean": mobile_disp,
            "stable_mean": stable_disp,
            "ratio": ms_ratio,
        },
        "top_displaced": [
            {"resnum": r, "domain": _assign_domain(r), "displacement": d}
            for r, d in sorted(displacements, key=lambda x: -x[1])[:10]
        ],
    }


def run_biology_probes(
    checkpoint: Path,
    pdb_dir: Path,
    device: str = "cpu",
    *,
    kras_pdb: str = "4OBE",
    kras_chain: str = "A",
) -> dict[str, Any]:
    model = load_v6_model(checkpoint, device)
    proteins, _ = load_training_proteins(
        pdb_dir,
        Path("manifests/v6_corpus_stage_a_small_v1.json"),
        max_proteins=12,
        use_cache=True,
    )
    kras_prot = next((p for p in proteins if str(p["pdb_id"]).upper() == kras_pdb.upper()), None)
    if kras_prot is None:
        kras_prot = load_protein_graph(kras_pdb, kras_chain, pdb_dir)
        if kras_prot is None:
            raise RuntimeError(f"Could not load KRAS structure {kras_pdb}")

    bundle = collect_protein_bundle(model, kras_prot, device)
    return {
        "checkpoint": str(checkpoint),
        "probe1_biology_vs_depth": probe_biology_vs_depth(bundle, kras_prot),
        "probe4_expert_routing": probe_expert_routing_by_domain(model, kras_prot, device),
        "probe5_wt_mutant": probe_wt_mutant_differential(
            model, pdb_dir, device, wt_pdb="4OBE", mut_pdb="4DSO", chain=kras_chain
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="V6 KRAS biology probes")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pdb-dir", type=Path, default=Path("/tmp/dtie_pdb_cache"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--compare", type=Path, default=None)
    args = parser.parse_args()

    report = run_biology_probes(args.checkpoint, args.pdb_dir, args.device)
    if args.compare and args.compare.is_file():
        report["compare"] = run_biology_probes(args.compare, args.pdb_dir, args.device)

    out = args.output or Path("checkpoints/v6/diagnostics/biology_probes.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
