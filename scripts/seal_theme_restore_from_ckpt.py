"""Re-seal theme_restore from a chosen ckpt (last vs best) — no retrain."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.training.v8.equ_geoopt_restore import assert_geoopt_restore_preflight
from experiments.training.v8.equ_theme_restore import (
    DEFAULT_FRONTEND_CKPT,
    DEFAULT_MANIFEST,
    PINNED_FRONTEND_SHA256,
    RV_TAU_CLAMP_MODE,
    RV_TAU_END,
    assert_frontend_bank,
    assert_pure_hyp_strict,
    build_theme_restore_stamp,
    evaluate_lift_spine_equivariance,
    evaluate_probe_hygiene,
    evaluate_theme_restore,
    load_boot_split,
    theme_dehydron_auprc,
)
from experiments.training.v8.run_tokyo_eye_equ_theme_restore import (
    _load_train_batch,
    _observe_panel,
    _resolve_entry,
)
from experiments.training.v8.run_v8_experiment import build_equiformer_pool_system
from science.tokyo_eye.v8.engine import PoincareDiagnosticsEngine
from science.tokyo_eye.v8.equiformer_frontend import DEFAULT_WEIGHT_MAP, load_weight_map
from science.tokyo_eye.v8.r0_r5_graph import set_dehydron_wrap_max

OUT_DIR = Path("checkpoints/tokyoeye/runs/eqf_equ_theme_restore_20260916")
MLFLOW_RUN = "0fc2986e0f49476694dfff3a4a955af5"
PDB_DIR = Path("/tmp/dtie_pdb_cache")
GRAPH_CACHE = Path("data/graph_cache")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--label", type=str, default="")
    args = p.parse_args()

    set_dehydron_wrap_max(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    digest = assert_frontend_bank(DEFAULT_FRONTEND_CKPT, expected=PINNED_FRONTEND_SHA256)
    cfg = dict(load_weight_map(DEFAULT_WEIGHT_MAP))
    system, load_info = build_equiformer_pool_system(
        cfg,
        equiformer_ckpt=DEFAULT_FRONTEND_CKPT,
        device=device,
        freeze_backbone=True,
        max_neighbors=50,
    )
    assert_geoopt_restore_preflight(system)
    pure = assert_pure_hyp_strict(system)
    system.spine.tau_clamp_mode = RV_TAU_CLAMP_MODE

    blob = torch.load(args.ckpt, map_location=device, weights_only=False)
    system.load_state_dict(blob["model"], strict=False)
    epoch = blob.get("epoch")
    print("loaded", args.ckpt, "epoch", epoch, "label", args.label or "")

    curvature = float(system.spine.c.detach().cpu()) if hasattr(system.spine.c, "detach") else float(system.spine.c)
    split = load_boot_split(DEFAULT_MANIFEST)
    probe = [_resolve_entry(e, PDB_DIR) for e in split["probe"]]
    # attach theme names from manifest if present
    for e, raw in zip(probe, split["probe"]):
        e["theme"] = raw.get("theme") or raw.get("flags", {}).get("theme") or e.get("pdb_id")

    diagnostics = PoincareDiagnosticsEngine(boundary_radius=0.80)
    final_probe_rows = _observe_panel(
        probe, system=system, curvature=curvature, device=device,
        pdb_dir=PDB_DIR, diagnostics=diagnostics, tau_ceil=RV_TAU_END,
    )
    # enrich with theme AUPRC
    for entry, row in zip(probe, final_probe_rows):
        batch = _load_train_batch(entry, device=device, pdb_dir=PDB_DIR, graph_cache_dir=GRAPH_CACHE)
        bio = theme_dehydron_auprc(system=system, batch=batch, tau_ceiling=RV_TAU_END)
        row.update(bio)
        row["theme"] = entry.get("theme")
        row["pdb_id"] = entry["pdb_id"]
        row["loaded"] = True

    hygiene = evaluate_probe_hygiene(final_probe_rows)
    biology = evaluate_theme_restore(final_probe_rows)
    print("hygiene", {k: hygiene[k] for k in ("mean_sat", "mean_spread", "probe_sat_gate", "radius_spread_gate", "finite_h2_gate")})
    print("biology", {k: biology[k] for k in ("mean_theme_auprc", "n_themes_pass_floor", "min_theme_auprc", "theme_auprc_coverage_gate", "mean_theme_auprc_gate", "theme_auprc_floor_gate")})
    print("per_theme", biology.get("per_theme"))

    equiv_residual = full_system_residual = None
    for entry in probe:
        try:
            batch = _load_train_batch(entry, device=device, pdb_dir=PDB_DIR, graph_cache_dir=GRAPH_CACHE)
            eq = evaluate_lift_spine_equivariance(system, batch, tau_ceiling=RV_TAU_END, seed=0)
            equiv_residual = float(eq["lift_spine_residual"])
            full_system_residual = float(eq["full_system_residual"])
            print(f"equiv {entry['pdb_id']} lift={equiv_residual:.6e} full={full_system_residual:.6e}")
            break
        except Exception as exc:
            print("equiv skip", entry["pdb_id"], exc)

    stamp = build_theme_restore_stamp(
        pure_hyp=pure,
        hygiene=hygiene,
        biology=biology,
        equiv_residual=equiv_residual,
        full_system_residual=full_system_residual,
        frontend_sha256=digest,
        mlflow_run_id=MLFLOW_RUN,
        extra={
            "reseal_from": str(args.ckpt),
            "reseal_epoch": epoch,
            "reseal_label": args.label,
            "note": "diagnostic reseal — does not overwrite official stamp",
        },
    )
    out = OUT_DIR / f"reseal_{args.label or args.ckpt.stem}.json"
    out.write_text(json.dumps(stamp, indent=2) + "\n")
    print("RESEAL", stamp["status"], stamp["execution_state"], "->", out)
    print("gates", {k: v["pass"] for k, v in stamp["gates"].items()})


if __name__ == "__main__":
    main()
