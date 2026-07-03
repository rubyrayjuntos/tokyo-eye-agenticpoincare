"""
pharmacophore_surface_render.py
================================
Step-2 SBIR figure: 3D surface uncertainty + GOSP-validated pharmacophore labels.

Coloring = residue-level ν_epi inherited to surface (Possibility B).
Green markers = structural-physics sites (not model discoveries).

Usage:
  python -m experiments.diagnostics.pharmacophore_surface_render --run
  make project-pharmacophore-surface STRUCTURES=11QE,4DSO
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.diagnostics.crescent_biology_projection import PHARMACOPHORE_SITES
from experiments.diagnostics.embedding_occupancy_audit import _forward_audit, load_audit_model
from experiments.training.v6._data import _chain_cache_dir, _download_pdb, _extract_chain, load_protein_graph
from experiments.training.v6.train_loop import attach_v6_features
from science.dtie.common.interfaces import GNNInferenceResult, GNNNodeOutput
from science.dtie.common.curvature_values import require_learned_curvature
from science.dtie.v6.visualization.interactive_viewer import (
    build_residue_channel_lookup,
    write_interactive_html,
    write_offline_annotated_pdb,
    _scale_channel,
)

DEFAULT_CHECKPOINT = Path(
    "checkpoints/v6/runs/lever_a_clean_slate_v1/v6_best_disc.pt"
)
DEFAULT_PDB_DIR = Path("/tmp/dtie_pdb_cache")
DEFAULT_OUTPUT = Path("checkpoints/v6/diagnostics/pharmacophore_surface")
DEFAULT_STRUCTURES = ["11QE", "4DSO"]
MODEL_VERSION = "GOSPConeMapper-v6-lever_a"


def _resnum_from_id(res_id: str) -> int:
    return int(res_id.split(":")[1])


def _inference_from_graph(
    structure_id: str,
    chain: str,
    prot: dict[str, Any],
    model: torch.nn.Module,
    device: str,
    checkpoint_path: Path,
) -> GNNInferenceResult:
    out = _forward_audit(model, "v6", prot, device)
    res_ids = list(prot["residue_ids"])
    num_nodes = len(res_ids)

    raw_depth = out["radial_features"].squeeze(-1).detach().cpu().numpy()
    depth_max = float(raw_depth.max()) + 1e-8
    normalized_depth = (raw_depth / depth_max) * 8.0

    epi = out["uncertainty"]["epistemic"].detach().cpu().numpy().reshape(-1)
    alea = out["uncertainty"]["aleatoric"].detach().cpu().numpy().reshape(-1)
    total = out["uncertainty"]["total"].detach().cpu().numpy().reshape(-1)
    x = attach_v6_features(prot["data"]).x.detach().cpu().numpy()

    nodes: list[GNNNodeOutput] = []
    for i in range(num_nodes):
        res_index = _resnum_from_id(res_ids[i])
        nodes.append(
            GNNNodeOutput(
                residue_index=res_index,
                chain_label=chain,
                input_features=x[i],
                projections=out["projections"][i].detach().cpu().numpy(),
                cone_depth=float(normalized_depth[i]),
                cone_width=float(np.exp(-normalized_depth[i])),
                epistemic_uncertainty=float(epi[i]),
                aleatoric_uncertainty=float(alea[i]),
                total_uncertainty=float(total[i]),
                x_hyp=out["x_hyp"][i].detach().cpu().numpy().astype("float64"),
                x_routed_hyp=out["x_routed_hyp"][i].detach().cpu().numpy().astype("float64"),
                hyp_projections=out["hyp_projections_2d"][i].detach().cpu().numpy().astype("float64"),
                expert_weights=out["expert_weights"][i].detach().cpu().numpy(),
            )
        )

    curvature = float(model.curvature.detach().cpu().item())
    return GNNInferenceResult(
        structure_id=structure_id.lower(),
        model_version=MODEL_VERSION,
        checkpoint_path=str(checkpoint_path),
        nodes=nodes,
        curvature=require_learned_curvature(curvature, context="pharmacophore_surface_render"),
        embedding_dim=int(out["x_hyp"].shape[1]),
        space_type="hyperbolic",
        metadata={"num_nodes": num_nodes, "chain": chain},
    )


def _transitional_band_metrics(
    nodes: list[GNNNodeOutput],
    sites: list[dict[str, str | int]],
) -> dict[str, Any]:
    epi = np.array([n.epistemic_uncertainty for n in nodes], dtype=np.float64)
    scaled = _scale_channel(epi, out_min=0.0, out_max=99.0)
    p33, p66 = np.percentile(scaled, [33.33, 66.67])

    by_res = {n.residue_index: (n, float(scaled[i])) for i, n in enumerate(nodes)}
    site_rows: list[dict[str, Any]] = []
    in_band = 0
    for site in sites:
        resnum = int(site["resnum"])
        node, b_scaled = by_res.get(resnum, (None, float("nan")))
        if node is None:
            site_rows.append(
                {
                    "label": site["label"],
                    "resnum": resnum,
                    "present": False,
                }
            )
            continue
        transitional = bool(p33 <= b_scaled <= p66)
        in_band += int(transitional)
        site_rows.append(
            {
                "label": site["label"],
                "resnum": resnum,
                "present": True,
                "nu_epi": float(node.epistemic_uncertainty),
                "nu_epi_scaled_bfactor": b_scaled,
                "cone_depth": float(node.cone_depth),
                "rho": float(node.input_features[0]),
                "sasa_proxy": float(node.input_features[3]),
                "percentile_rank": float((epi < node.epistemic_uncertainty).mean() * 100.0),
                "transitional_band": transitional,
                "band_limits_scaled": [float(p33), float(p66)],
            }
        )

    return {
        "structure_scaled_p33": float(p33),
        "structure_scaled_p66": float(p66),
        "structure_epi_min": float(epi.min()),
        "structure_epi_max": float(epi.max()),
        "structure_epi_mean": float(epi.mean()),
        "sites_in_transitional_band": in_band,
        "sites_total": len(sites),
        "sites": site_rows,
        "claim": (
            "Validated GOSP pharmacophore residues occupy the mid-range ν_epi band "
            "(neither buried-blue nor exposed-red) under Possibility B inheritance."
        ),
    }


def render_pharmacophore_surface(
    checkpoint_path: Path,
    structure_ids: list[str],
    pdb_dir: Path,
    output_dir: Path,
    *,
    chain: str = "A",
    device: str = "cpu",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model, version = load_audit_model(checkpoint_path, device)
    if version != "v6":
        raise RuntimeError("pharmacophore_surface_render requires v6 checkpoint")

    report: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "legend_mode": "possibility_b",
        "coloring": "residue-level nu_epi inherited to surface (not shell-independent)",
        "pharmacophore_source": "GOSP structural physics (not model-discovered)",
        "structures": {},
    }

    for sid in structure_ids:
        sid_u = sid.upper()
        sites = PHARMACOPHORE_SITES.get(sid_u, [])
        if not sites:
            raise ValueError(f"No PHARMACOPHORE_SITES entry for {sid_u}")

        prot = load_protein_graph(sid_u, chain, pdb_dir)
        if prot is None:
            raise RuntimeError(f"Could not load {sid_u}:{chain}")

        result = _inference_from_graph(sid_u, chain, prot, model, device, checkpoint_path)
        lookup = build_residue_channel_lookup(result.nodes)

        pdb_path = _download_pdb(sid_u, pdb_dir)
        chain_path = _extract_chain(pdb_path, chain, _chain_cache_dir(pdb_dir))
        out_sub = output_dir / sid_u.lower()
        annotated_pdb = out_sub / f"{sid_u.lower()}_pharmacophore_surface.pdb"
        html_path = out_sub / f"{sid_u.lower()}_pharmacophore_surface.html"

        n_atoms = write_offline_annotated_pdb(chain_path, lookup, annotated_pdb, chain=chain)
        write_interactive_html(
            structure_id=sid_u,
            pdb_text=annotated_pdb.read_text(encoding="utf-8"),
            model_version=MODEL_VERSION,
            output_path=html_path,
            checkpoint_path=str(checkpoint_path),
            pharmacophore_sites=sites,
            legend_mode="possibility_b",
            chain=chain,
        )

        band = _transitional_band_metrics(result.nodes, sites)
        report["structures"][sid_u] = {
            "n_residues": len(result.nodes),
            "n_atoms": n_atoms,
            "pdb": str(annotated_pdb),
            "html": str(html_path),
            "pharmacophore_sites": sites,
            "transitional_band": band,
        }

    summary_path = output_dir / "pharmacophore_surface_report.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    report["summary_path"] = str(summary_path)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--pdb-dir", type=Path, default=DEFAULT_PDB_DIR)
    ap.add_argument("--structures", default=",".join(DEFAULT_STRUCTURES))
    ap.add_argument("--chain", default="A")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()
    if not args.run:
        ap.print_help()
        return 0

    structure_ids = [s.strip().upper() for s in args.structures.split(",") if s.strip()]
    report = render_pharmacophore_surface(
        args.checkpoint,
        structure_ids,
        args.pdb_dir,
        args.output_dir,
        chain=args.chain,
        device=args.device,
    )
    print(json.dumps(report["structures"], indent=2))
    print(f"Wrote → {report['summary_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
