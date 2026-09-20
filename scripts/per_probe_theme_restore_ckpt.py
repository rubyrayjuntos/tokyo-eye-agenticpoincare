"""Per-structure sat/spread/AUPRC for a theme_restore ckpt."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from experiments.training.v8.equ_geoopt_restore import assert_geoopt_restore_preflight
from experiments.training.v8.equ_lift_radius import enrich_rim_probe_row, hyp_radius
from experiments.training.v8.equ_theme_restore import (
    DEFAULT_FRONTEND_CKPT,
    DEFAULT_MANIFEST,
    PINNED_FRONTEND_SHA256,
    RV_TAU_CLAMP_MODE,
    RV_TAU_END,
    assert_frontend_bank,
    load_boot_split,
    theme_dehydron_auprc,
)
from experiments.training.v8.run_tokyo_eye_equ_theme_restore import (
    _load_train_batch,
    _resolve_entry,
)
from experiments.training.v8.run_v8_experiment import build_equiformer_pool_system
from science.tokyo_eye.v8.equiformer_frontend import DEFAULT_WEIGHT_MAP, load_weight_map
from science.tokyo_eye.v8.r0_r5_graph import set_dehydron_wrap_max

PDB_DIR = Path("/tmp/dtie_pdb_cache")
GRAPH_CACHE = Path("data/graph_cache")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    args = ap.parse_args()
    set_dehydron_wrap_max(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert_frontend_bank(DEFAULT_FRONTEND_CKPT, expected=PINNED_FRONTEND_SHA256)
    cfg = dict(load_weight_map(DEFAULT_WEIGHT_MAP))
    system, _ = build_equiformer_pool_system(
        cfg, equiformer_ckpt=DEFAULT_FRONTEND_CKPT, device=device, freeze_backbone=True, max_neighbors=50
    )
    assert_geoopt_restore_preflight(system)
    system.spine.tau_clamp_mode = RV_TAU_CLAMP_MODE
    blob = torch.load(args.ckpt, map_location=device, weights_only=False)
    system.load_state_dict(blob["model"], strict=False)
    c = float(system.spine.c.detach().cpu()) if hasattr(system.spine.c, "detach") else float(system.spine.c)
    split = load_boot_split(DEFAULT_MANIFEST)
    rows = []
    for raw in split["probe"]:
        entry = _resolve_entry(raw, PDB_DIR)
        theme = raw.get("theme") or entry["pdb_id"]
        batch = _load_train_batch(entry, device=device, pdb_dir=PDB_DIR, graph_cache_dir=GRAPH_CACHE)
        system.eval()
        with torch.no_grad():
            out = system(batch["x"], batch["edge_index"], batch["edge_type"], tau_ceiling=RV_TAU_END)
        z = out["z_hyp"].detach().float()
        r = torch.linalg.vector_norm(z, dim=-1)
        r_h = hyp_radius(z, c=c)
        bio = theme_dehydron_auprc(system=system, batch=batch, tau_ceiling=RV_TAU_END)
        row = {
            "theme": theme,
            "pdb_id": entry["pdb_id"],
            "chain": entry["chain"],
            "n": int(z.shape[0]),
            "mean_ball_radius": float(r.mean()),
            "std_ball_radius": float(r.std(unbiased=False)),
            "std_hyp_radius": float(r_h.std(unbiased=False)) if r_h.numel() > 1 else 0.0,
            "mean_hyp_radius": float(r_h.mean()),
            "theme_auprc": bio["theme_auprc"],
            "dehydron_frac": bio["dehydron_frac"],
        }
        rows.append(row)
        print(
            f"{row['pdb_id']}:{row['chain']} [{theme}] "
            f"sat={row['mean_ball_radius']:.4f} spread_rH={row['std_hyp_radius']:.4f} "
            f"auprc={row['theme_auprc']:.4f} dh={row['dehydron_frac']:.3f}"
        )
    # panel aggregates matching seal formulas
    mean_sat = sum(r["mean_ball_radius"] for r in rows) / len(rows)
    # sealed spread is mean of per-structure spreads? or pooled? check evaluate_probe_hygiene
    print("PANEL mean_sat", mean_sat)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
