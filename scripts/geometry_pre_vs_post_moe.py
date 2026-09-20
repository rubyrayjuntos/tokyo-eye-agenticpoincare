"""Verify whether sealed sat/spread were pre-MoE or post-MoE coupled."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch

from experiments.training.v8.equ_geoopt_restore import (
    DEFAULT_FRONTEND_CKPT, DEFAULT_MANIFEST, PINNED_FRONTEND_SHA256,
    RV_TAU_CLAMP_MODE, RV_TAU_END, assert_frontend_bank, load_boot_split,
)
from experiments.training.v8.run_tokyo_eye_equ_geoopt_restore import _load_train_batch, _resolve_entry
from experiments.training.v8.run_v8_experiment import build_equiformer_pool_system
from science.tokyo_eye.v8.equiformer_frontend import DEFAULT_WEIGHT_MAP, load_weight_map
from science.tokyo_eye.v8.r0_r5_graph import set_dehydron_wrap_max

BEST = Path("checkpoints/tokyoeye/runs/eqf_equ_geoopt_restore_20260916/tokyoeye_best.pt")
OUT = Path("data/local_objects/frontend_ablation/geometry_pre_vs_post_moe.json")
PDB_DIR = Path("/tmp/dtie_pdb_cache")
GRAPH_CACHE = Path("data/graph_cache")
SAT_CEIL = 0.50

def radius_stats(z: torch.Tensor) -> dict:
    r = torch.linalg.vector_norm(z.detach().float(), dim=-1).cpu().numpy()
    return {
        "mean_r": float(r.mean()),
        "std_r": float(r.std()),
        "min_r": float(r.min()),
        "max_r": float(r.max()),
        "sat_frac": float((r >= SAT_CEIL).mean()),
        "n_unique_r_rounded_3": int(len(np.unique(np.round(r, 3)))),
        "hist20": np.histogram(r, bins=20, range=(0, 0.6))[0].tolist(),
    }

def main():
    set_dehydron_wrap_max(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert_frontend_bank(DEFAULT_FRONTEND_CKPT, expected=PINNED_FRONTEND_SHA256)
    cfg = dict(load_weight_map(DEFAULT_WEIGHT_MAP))
    system, _ = build_equiformer_pool_system(
        cfg, equiformer_ckpt=DEFAULT_FRONTEND_CKPT, device=device,
        freeze_backbone=True, max_neighbors=50,
    )
    system.spine.tau_clamp_mode = RV_TAU_CLAMP_MODE
    blob = torch.load(BEST, map_location=device, weights_only=False)
    system.load_state_dict(blob["model"], strict=False)
    system.eval()

    split = load_boot_split(DEFAULT_MANIFEST)
    entries = [_resolve_entry(e, PDB_DIR) for e in (split["train"] + split["probe"])]
    rows = []
    agg = {k: [] for k in ("lift", "attn", "moe")}
    with torch.no_grad():
        for entry in entries:
            batch = _load_train_batch(entry, device=device, pdb_dir=PDB_DIR, graph_cache_dir=GRAPH_CACHE)
            out = system(batch["x"], batch["edge_index"], batch["edge_type"], tau_ceiling=RV_TAU_END)
            stats = {
                "pdb": f"{entry['pdb_id']}:{entry['chain']}",
                "lift": radius_stats(out["z_lift"]),
                "attn": radius_stats(out["z_attn"]),
                "moe": radius_stats(out["z_hyp"]),  # sealed name = post-MoE
            }
            # per-structure spread = std(r) as used in gates roughly
            for k in ("lift", "attn", "moe"):
                agg[k].append(stats[k])
            print(
                f"{stats['pdb']} lift_sat={stats['lift']['sat_frac']:.3f} std={stats['lift']['std_r']:.3f} | "
                f"attn_sat={stats['attn']['sat_frac']:.3f} std={stats['attn']['std_r']:.3f} | "
                f"moe_sat={stats['moe']['sat_frac']:.3f} std={stats['moe']['std_r']:.3f} "
                f"unique_r≈{stats['moe']['n_unique_r_rounded_3']}"
            )
            rows.append(stats)

    def mean_key(lst, key):
        return float(np.mean([x[key] for x in lst]))

    summary = {
        "ckpt": str(BEST),
        "sealed_z_hyp_is": "post_MoE (model.py returns z_moe as z_hyp)",
        "equiv_compares": "spine(...)[z_hyp] = post_MoE",
        "mean_sat_frac": {k: mean_key(agg[k], "sat_frac") for k in agg},
        "mean_std_r": {k: mean_key(agg[k], "std_r") for k in agg},
        "mean_r": {k: mean_key(agg[k], "mean_r") for k in agg},
        "finding": (
            "If moe std_r / sat differ sharply from lift/attn, sealed geometry HOLD "
            "was coupled to MoE radius parking — QUALIFIED does not stand independent."
        ),
        "per_structure": rows,
    }
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print("SUMMARY sat", summary["mean_sat_frac"])
    print("SUMMARY std_r", summary["mean_std_r"])
    print("WROTE", OUT)

if __name__ == "__main__":
    main()
