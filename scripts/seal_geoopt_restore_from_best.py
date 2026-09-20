"""Seal tokyo_eye_equ_geoopt_restore from saved best ckpt (no retrain)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from experiments.training.v8.equ_geoopt_restore import (
    DEFAULT_FRONTEND_CKPT,
    DEFAULT_MANIFEST,
    DEFAULT_STAMP,
    PINNED_FRONTEND_SHA256,
    RV_TAU_CLAMP_MODE,
    RV_TAU_END,
    assert_frontend_bank,
    assert_geoopt_restore_preflight,
    assert_pure_hyp_strict,
    build_geoopt_restore_stamp,
    evaluate_lift_spine_equivariance,
    evaluate_probe_hygiene,
    load_boot_split,
)
from experiments.training.v8.run_b0_topology_observation import _device, observe_one
from experiments.training.v8.run_tokyo_eye_equ_geoopt_restore import (
    _load_train_batch,
    _observe_panel,
    _resolve_entry,
)
from experiments.training.v8.run_v8_experiment import build_equiformer_pool_system
from science.tokyo_eye.v8.equiformer_frontend import DEFAULT_WEIGHT_MAP, load_weight_map
from science.tokyo_eye.v8.r0_r5_graph import set_dehydron_wrap_max

OUT_DIR = Path("checkpoints/tokyoeye/runs/eqf_equ_geoopt_restore_20260916")
BEST = OUT_DIR / "tokyoeye_best.pt"
MLFLOW_RUN = "b0480b85e87f492f965f2993a9298dba"


def main() -> None:
    set_dehydron_wrap_max(1)
    device = _device("cuda")
    pdb_dir = Path("/tmp/dtie_pdb_cache")
    graph_cache_dir = Path("data/graph_cache")
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
    pure_report = assert_pure_hyp_strict(system)
    system.spine.tau_clamp_mode = RV_TAU_CLAMP_MODE

    blob = torch.load(BEST, map_location=device, weights_only=False)
    missing, unexpected = system.load_state_dict(blob["model"], strict=False)
    print("loaded best", BEST, "epoch", blob.get("epoch"), "missing", len(missing), "unexpected", len(unexpected))

    curvature = float(system.spine.c.detach().cpu()) if hasattr(system.spine.c, "detach") else float(system.spine.c)
    split = load_boot_split(DEFAULT_MANIFEST)
    probe = [_resolve_entry(e, pdb_dir) for e in split["probe"]]
    home_entry = {"pdb_id": "1HNG", "chain": "A"}

    from science.tokyo_eye.v8.engine import PoincareDiagnosticsEngine
    diagnostics = PoincareDiagnosticsEngine(boundary_radius=0.80)

    final_probe_rows = _observe_panel(
        probe, system=system, curvature=curvature, device=device,
        pdb_dir=pdb_dir, diagnostics=diagnostics, tau_ceil=RV_TAU_END,
    )
    home_rows = _observe_panel(
        [home_entry], system=system, curvature=curvature, device=device,
        pdb_dir=pdb_dir, diagnostics=diagnostics, tau_ceil=RV_TAU_END,
    )
    print("probe_rows", len(final_probe_rows))
    for r in final_probe_rows:
        print(" ", r.get("pdb_id"), "sat", r.get("mean_ball_radius"), "spread", r.get("radius_spread"), "h_norm", r.get("h_norm"))

    equiv_residual = None
    full_system_residual = None
    for entry in probe:
        try:
            batch = _load_train_batch(entry, device=device, pdb_dir=pdb_dir, graph_cache_dir=graph_cache_dir)
            eq = evaluate_lift_spine_equivariance(system, batch, tau_ceiling=RV_TAU_END, seed=0)
            equiv_residual = float(eq["lift_spine_residual"])
            full_system_residual = float(eq["full_system_residual"])
            print(f"equiv on {entry['pdb_id']}: lift={equiv_residual:.6e} full={full_system_residual:.6e}")
            break
        except Exception as exc:
            print("equiv skip", entry["pdb_id"], exc)

    hygiene = evaluate_probe_hygiene(final_probe_rows)
    print("hygiene", hygiene)

    prior = json.loads(DEFAULT_STAMP.read_text()) if DEFAULT_STAMP.is_file() else {}
    stamp = build_geoopt_restore_stamp(
        pure_hyp=pure_report,
        hygiene=hygiene,
        equiv_residual=equiv_residual,
        full_system_residual=full_system_residual,
        frontend_sha256=digest,
        mlflow_run_id=MLFLOW_RUN,
        prior=prior,
        extra={
            "best_checkpoint": str(BEST),
            "best_epoch": blob.get("epoch"),
            "best_mean_loss": blob.get("mean_loss"),
            "out_dir": str(OUT_DIR),
            "load_info": load_info,
            "home_probe": home_rows[0] if home_rows else None,
            "per_probe": final_probe_rows,
            "freeze": "entire_equiformer_pool_geometry_pass",
            "seal_note": "sealed_from_best_after_stamp_kwargs_fix",
        },
    )
    DEFAULT_STAMP.write_text(json.dumps(stamp, indent=2) + "\n")
    (OUT_DIR / "run_summary.json").write_text(json.dumps({
        "best_epoch": blob.get("epoch"),
        "best_mean_loss": blob.get("mean_loss"),
        "status": stamp.get("status"),
        "gates": stamp.get("gates"),
        "mlflow_run_id": MLFLOW_RUN,
        "equiv_residual": equiv_residual,
        "full_system_residual": full_system_residual,
        "hygiene": hygiene,
    }, indent=2) + "\n")
    print("SEAL", stamp["status"], stamp["execution_state"])
    print("gates", {k: v["pass"] for k, v in stamp["gates"].items()})


if __name__ == "__main__":
    main()
