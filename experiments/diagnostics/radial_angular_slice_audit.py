"""
Extended radial×angular / disc projection slice audit.

Reports lift stages, pre- vs post-routing disc, legacy hard projection,
and recombination counterfactuals on a loaded checkpoint.

Usage:
  make diagnose-radial-angular-slice \\
      CHECKPOINT=checkpoints/v6/runs/shell_p2_disc_occupancy/phase_2_disc_occupancy_recovery.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from geoopt.manifolds.stereographic import math as pmath

from experiments.training.v6._data import load_protein_graph
from experiments.training.v6.assess_checkpoint import load_v6_model
from experiments.training.v6.train_loop import attach_v6_features
from science.dtie.common.poincare_conventions import rescale_tangent_before_expmap
from science.dtie.v6.gnn.hyperbolic_moe import project_disc_2d, project_disc_2d_legacy
from science.dtie.v6.gnn.model import GOSPConeMapperV6
from science.dtie.v6.gnn.slice_diagnostics import (
    centered_svd_stats,
    cone_slice_stats,
    disc_2d_stats,
    recompute_tangent,
)


def _parse_structures(raw: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for item in raw.split(","):
        item = item.strip()
        if ":" in item:
            pdb, chain = item.split(":", 1)
        else:
            pdb, chain = item, "A"
        out.append((pdb.upper(), chain))
    return out


def _tensor_stats(name: str, t: torch.Tensor | None) -> dict[str, Any] | None:
    if t is None:
        return None
    arr = t.detach().cpu().numpy()
    if arr.ndim == 2 and arr.shape[1] == 2:
        return {"layer": name, **disc_2d_stats(arr)}
    return {"layer": name, **centered_svd_stats(arr)}


def _project_disc_variants(
    model: GOSPConeMapperV6,
    x_hyp: torch.Tensor,
    x_routed_hyp: torch.Tensor,
    *,
    k: torch.Tensor,
    c: torch.Tensor,
    softness: float,
) -> dict[str, np.ndarray]:
    pre_raw = model.hyp_proj_head_2d(x_hyp, c=c)
    post_raw = model.hyp_proj_head_2d(x_routed_hyp, c=c)
    pre_soft, _ = project_disc_2d(pre_raw, k=k, softness=softness)
    post_soft, _ = project_disc_2d(post_raw, k=k, softness=softness)
    post_legacy, _ = project_disc_2d_legacy(post_raw, k=k)
    return {
        "disc_pre_routing_soft": pre_soft.detach().cpu().numpy(),
        "disc_post_routing_soft": post_soft.detach().cpu().numpy(),
        "disc_post_routing_legacy": post_legacy.detach().cpu().numpy(),
    }


def _fmt_disc_row(label: str, stats: dict[str, Any]) -> str:
    sr = stats.get("sigma_ratio") or []
    s2s1 = sr[1] if len(sr) > 1 else 0.0
    thick = stats.get("line_thickness_rms") or 0.0
    span = stats.get("origin_angular_span_p5_p95_deg")
    span_s = f"{span:.1f}°" if span is not None else "n/a"
    return (
        f"    {label:<28} thick={thick:.4f}  origin_span={span_s}  "
        f"σ₂/σ₁={s2s1:.3f}  eff_rank={stats.get('eff_rank', 0):.3f}"
    )


@torch.inference_mode()
def audit_structure(
    model: GOSPConeMapperV6,
    prot: dict[str, Any],
    device: str,
    *,
    counterfactuals: list[str],
) -> dict[str, Any]:
    data = attach_v6_features(prot["data"].to(device))
    out_fwd = model(data)

    radial = out_fwd["radial_features"]
    angular = out_fwd["angular_features"]
    c = model.curvature
    k = -c
    softness = float(getattr(model, "disc_proj_softness", 0.95))

    x_hyp = out_fwd["x_hyp"]
    x_routed_hyp = out_fwd.get("x_routed_hyp")
    cone = cone_slice_stats(radial, angular, hidden=model.hidden)

    lift_stages: dict[str, Any] = {
        "angular_direction": centered_svd_stats(angular.cpu().numpy()),
        "tangent_multiply": centered_svd_stats((radial * angular).cpu().numpy()),
        "x_hyp": centered_svd_stats(x_hyp.cpu().numpy()),
    }
    if x_routed_hyp is not None:
        lift_stages["x_routed_hyp"] = centered_svd_stats(x_routed_hyp.cpu().numpy())

    disc_layers: dict[str, Any] = {}
    pre_fwd = out_fwd.get("hyp_projections_2d_pre")
    post_fwd = out_fwd.get("hyp_projections_2d_post")
    if pre_fwd is not None:
        disc_layers["disc_pre_routing_soft"] = disc_2d_stats(pre_fwd.detach().cpu().numpy())
    if post_fwd is not None:
        disc_layers["disc_post_routing_soft"] = disc_2d_stats(post_fwd.detach().cpu().numpy())
    if x_routed_hyp is not None:
        depth_routed = pmath.dist0(x_routed_hyp, k=k, keepdim=True)
        post_raw_mobius = model.hyp_proj_head_2d(x_routed_hyp, c=c)
        post_mobius_soft, _ = project_disc_2d(post_raw_mobius, k=k, softness=softness)
        disc_layers["disc_post_routing_mobius"] = disc_2d_stats(
            post_mobius_soft.detach().cpu().numpy()
        )
        if str(getattr(model, "disc_radial_source", "mobius")) != "mobius":
            post_raw_override = model._post_routing_disc_raw(
                x_routed_hyp,
                depth_routed=depth_routed,
                c=c,
                k=k,
            )
            post_override_soft, _ = project_disc_2d(post_raw_override, k=k, softness=softness)
            disc_layers["disc_post_routing_override"] = disc_2d_stats(
                post_override_soft.detach().cpu().numpy()
            )
        post_legacy, _ = project_disc_2d_legacy(post_raw_mobius, k=k)
        disc_layers["disc_post_routing_legacy"] = disc_2d_stats(post_legacy.detach().cpu().numpy())

    model_disc = out_fwd.get("hyp_projections_2d")
    if model_disc is not None:
        disc_layers["disc_model_output"] = disc_2d_stats(model_disc.detach().cpu().numpy())

    teacher = out_fwd.get("hyp_projections_2d_legacy_teacher")
    if teacher is not None:
        disc_layers["disc_legacy_teacher"] = disc_2d_stats(teacher.detach().cpu().numpy())

    counter: dict[str, Any] = {}
    fusion = getattr(model, "radial_angular_fusion", None)
    for mode in counterfactuals:
        t = recompute_tangent(radial, angular, mode=mode, fusion=fusion)
        t = rescale_tangent_before_expmap(t, c)
        x_cf = pmath.project(pmath.expmap0(t, k=k), k=k)
        raw = model.hyp_proj_head_2d(x_cf, c=c)
        xy, _ = project_disc_2d(raw, k=k, softness=softness)
        counter[mode] = {
            "tangent": centered_svd_stats(t.cpu().numpy()),
            "x_hyp": centered_svd_stats(x_cf.cpu().numpy()),
            "disc_pre_routing_soft": disc_2d_stats(xy.detach().cpu().numpy()),
        }

    routing_entropy = float(out_fwd.get("routing_entropy", torch.tensor(0.0)).detach().cpu())
    expert_load = out_fwd.get("expert_load")
    load_spread = None
    if expert_load is not None:
        loads = expert_load.detach().cpu().numpy()
        load_spread = float(loads.max() - loads.min())

    return {
        "structure_id": prot.get("pdb_id", "?"),
        "n_residues": int(prot.get("n_residues", data.x.shape[0])),
        "legacy_disc_projection": bool(getattr(model, "legacy_disc_projection", False)),
        "disc_radial_source": str(getattr(model, "disc_radial_source", "mobius")),
        "disc_proj_softness": softness,
        "routing_entropy": routing_entropy,
        "expert_load_spread": load_spread,
        "cone_slice": cone,
        "lift_stages": lift_stages,
        "disc_layers": disc_layers,
        "counterfactuals": counter,
    }


def _print_report(audit: dict[str, Any]) -> None:
    sid = audit["structure_id"]
    print("=" * 88)
    print(
        f"STRUCTURE {sid}  n={audit['n_residues']}  "
        f"legacy={audit['legacy_disc_projection']}  "
        f"disc_radial={audit.get('disc_radial_source', 'mobius')}  "
        f"softness={audit['disc_proj_softness']:.2f}  "
        f"routing_H={audit.get('routing_entropy', 0):.3f}  "
        f"expert_load_spread={audit.get('expert_load_spread')}"
    )
    cone = audit["cone_slice"]
    print(
        f"  CONE: tangent_eff={cone['tangent_eff_rank']:.3f}  "
        f"angular_eff={cone['angular_eff_rank']:.3f}  "
        f"pc1_frac={cone['cone_pc1_frac']:.3f}"
    )
    print("  --- lift stages (high-D) ---")
    for name, stats in audit["lift_stages"].items():
        sr = stats.get("sigma_ratio") or []
        s2s1 = sr[1] if len(sr) > 1 else 0.0
        thick = stats.get("line_thickness_rms")
        thick_s = f"{thick:.4f}" if thick is not None else "n/a"
        print(
            f"    {name:<22} eff_rank={stats.get('eff_rank', 0):.3f}  "
            f"σ₂/σ₁={s2s1:.3f}  thick={thick_s}  pc1={stats.get('pc1_energy_frac', 0):.3f}"
        )
    print("  --- disc layers (pre vs post routing) ---")
    for name, stats in audit.get("disc_layers", {}).items():
        print(_fmt_disc_row(name, stats))
    if audit.get("counterfactuals"):
        print("  --- counterfactual recombination → pre-routing soft disc ---")
        for mode, block in audit["counterfactuals"].items():
            print(_fmt_disc_row(mode, block["disc_pre_routing_soft"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extended disc projection slice audit")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--structures", default="11QE:A,4OBE:A,1IVO:A")
    parser.add_argument("--pdb-dir", default="/tmp/dtie_pdb_cache")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--legacy-disc-projection", action="store_true")
    parser.add_argument("--no-legacy-disc-projection", action="store_true")
    parser.add_argument(
        "--counterfactual",
        default="multiply,mlp_fusion,angular_only,angular_lift",
    )
    parser.add_argument(
        "--disc-radial-source",
        choices=("mobius", "radial_depth", "dist0_x_hyp"),
        default="mobius",
        help="Override pre-routing 2D radius (Lever A counterfactual on loaded checkpoint)",
    )
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    legacy_override: bool | None = None
    if args.legacy_disc_projection:
        legacy_override = True
    if args.no_legacy_disc_projection:
        legacy_override = False

    counterfactuals = [m.strip() for m in args.counterfactual.split(",") if m.strip()]
    pdb_dir = Path(args.pdb_dir)

    model = load_v6_model(args.checkpoint, args.device, legacy_disc_projection=legacy_override)
    from science.dtie.v6.gnn.model import resolve_disc_radial_source

    model.disc_radial_source = resolve_disc_radial_source(args.disc_radial_source)

    if not hasattr(model, "radial_angular_fusion") or model.radial_angular_fusion is None:
        hidden = model.hidden
        fusion = torch.nn.Sequential(
            torch.nn.Linear(hidden + 1, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden),
        )
        torch.nn.init.zeros_(fusion[-1].weight)
        torch.nn.init.zeros_(fusion[-1].bias)
        model.radial_angular_fusion = fusion

    model.train(False)
    reports: list[dict[str, Any]] = []
    for pdb_id, chain in _parse_structures(args.structures):
        prot = load_protein_graph(pdb_id, chain, pdb_dir)
        if prot is None:
            raise SystemExit(f"Failed to load {pdb_id}:{chain}")
        audit = audit_structure(
            model, prot, args.device, counterfactuals=counterfactuals
        )
        reports.append(audit)
        _print_report(audit)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(reports, indent=2))
        print(f"Wrote JSON -> {args.json_out}")


if __name__ == "__main__":
    main()
