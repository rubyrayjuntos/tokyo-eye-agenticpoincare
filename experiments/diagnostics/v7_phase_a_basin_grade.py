#!/usr/bin/env python3
"""Grade Phase A nucleotide basin continue — cheap hooks + migration spot-check.

Pre-reg: docs/specs/tokyo-eye-v7/phase-a-nucleotide-basin-train-prereg.md
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.diagnostics.fix1_champion_hub_knockout_sweep import _align_prot_features
from experiments.training.v66._data import load_protein_graph_from_pdb_legacy
from experiments.training.v6.train_loop import prepare_training_batch, residue_node_count
from experiments.training.v7.healthy_bprime import HEALTHY_V7_CKPT
from science.dtie.common.kras_g12_graft import neighborhood_n12
from science.dtie.common.kras_topo_matrix import residue_index_map
from science.tokyo_eye.basin_contrastive import (
    ball_distance_embeddings,
    r_star_indices,
    structure_embedding_logmap0,
)
from science.tokyo_eye.hyp_mp_telemetry import attach_hyp_mp_telemetry
from science.training.gnn_lineage import load_model_from_checkpoint

PREREG = Path("data/gates/tokyo_eye_v7_phase_a_basin_train_prereg.json")
DEFAULT_CLOSEOUT = Path("data/gates/tokyo_eye_v7_phase_a_basin_train_closeout.json")
DEFAULT_OUT = Path(
    "checkpoints/v7/diagnostics/phase_a_basin/phase_a_basin_grade.json"
)
ROSTER = {
    "OFF": [("4LPK", "A"), ("5US4", "A")],
    "ON": [("6GOD", "A"), ("6GOF", "A")],
}
SEALED_MIGRATION = Path(
    "checkpoints/v7/diagnostics/hub_migration/kras_hub_migration_4obe_4dso.json"
)


def _curv(out: dict[str, Any], model: torch.nn.Module) -> float:
    c = (out.get("audit_trail") or {}).get("curvature_value")
    if c is None:
        return float(model.curvature.detach().cpu().reshape(-1)[0])
    return float(c.detach().cpu()) if torch.is_tensor(c) else float(c)


def _disc_r_mean(out: dict[str, Any]) -> float:
    disc = out.get("hyp_projections_2d")
    if disc is None:
        disc = out.get("hyp_proj_2d")
    if disc is None or not torch.is_tensor(disc):
        return float("nan")
    return float(torch.linalg.vector_norm(disc.float(), dim=-1).mean().item())


def _embed(
    model: torch.nn.Module,
    pdb_id: str,
    chain: str,
    pdb_dir: Path,
    device: str,
    partner: dict[str, Any] | None,
) -> tuple[torch.Tensor, float, dict[str, Any], dict[str, Any]]:
    prot = load_protein_graph_from_pdb_legacy(pdb_id, chain, pdb_dir)
    if prot is None:
        raise FileNotFoundError(f"{pdb_id}:{chain}")
    prot = _align_prot_features(model, prot)
    data = prepare_training_batch(model, prot, device)
    n = residue_node_count(data, prot)
    with torch.no_grad():
        out = model(data)
    attach_hyp_mp_telemetry(out, data, k_frac=0.10)
    c = _curv(out, model)
    idx_map = residue_index_map(list(prot.get("residue_ids") or [])[:n])
    present = set(idx_map.keys())
    n12: list[int] = []
    try:
        if partner is not None:
            n12 = neighborhood_n12(partner, prot)
        else:
            n12 = neighborhood_n12(prot, prot)
    except Exception:  # noqa: BLE001
        n12 = []
    idxs = [idx_map[r] for r in r_star_indices(present, n12) if r in idx_map]
    z = structure_embedding_logmap0(out["x_hyp"][:n], idxs, curvature=c)
    return z.cpu(), c, out, prot


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "checkpoints/v7/runs/tokyo_eye_v7_phase_a_nucleotide_basin_v1/v7_phase_a_best.pt"
        ),
    )
    p.add_argument("--pdb-dir", type=Path, default=Path("/tmp/dtie_pdb_cache"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--closeout", type=Path, default=DEFAULT_CLOSEOUT)
    p.add_argument("--skip-migration-spotcheck", action="store_true")
    p.add_argument("--basin-sep-gap-min", type=float, default=0.10)
    p.add_argument("--disc-r-mean-min", type=float, default=0.25)
    args = p.parse_args(argv)
    os.environ.setdefault("GNN_INPUT_MODE", "topology_three_vector")
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    if not PREREG.is_file():
        raise SystemExit(f"missing prereg {PREREG}")
    if not args.checkpoint.is_file():
        raise SystemExit(f"missing checkpoint {args.checkpoint}")

    model = load_model_from_checkpoint(args.checkpoint, args.device)
    model.eval()

    embeds: dict[str, torch.Tensor] = {}
    curvatures: dict[str, float] = {}
    disc_rs: list[float] = []
    telemetry: dict[str, Any] = {}
    prots: dict[str, dict[str, Any]] = {}

    # Load all first for N12 partners
    for basin, items in ROSTER.items():
        for pdb_id, chain in items:
            prot = load_protein_graph_from_pdb_legacy(pdb_id, chain, args.pdb_dir)
            if prot is None:
                raise FileNotFoundError(pdb_id)
            prots[pdb_id] = prot

    for basin, items in ROSTER.items():
        for pdb_id, chain in items:
            partner = None
            if basin == "OFF":
                partner = prots.get("5US4") if pdb_id == "4LPK" else prots.get("4LPK")
            else:
                partner = prots.get("6GOF") if pdb_id == "6GOD" else prots.get("6GOD")
            print(f"embed {pdb_id} ...", flush=True)
            z, c, out, _ = _embed(
                model, pdb_id, chain, args.pdb_dir, args.device, partner
            )
            embeds[pdb_id] = z
            curvatures[pdb_id] = c
            disc_rs.append(_disc_r_mean(out))
            tel = out.get("hyp_mp_telemetry") or {}
            telemetry[pdb_id] = {
                "n_hubs": tel.get("n_hubs"),
                "strength_cv": tel.get("strength_cv"),
                "n_edges": tel.get("n_edges"),
                "hyp_mp_primary": bool((out.get("audit_trail") or {}).get("hyp_mp_primary")),
            }

    c_ref = float(np.mean(list(curvatures.values())))
    # Cross OFF-ON distances
    cross = []
    for o, _ in ROSTER["OFF"]:
        for n, _ in ROSTER["ON"]:
            cross.append(
                ball_distance_embeddings(embeds[o], embeds[n], curvature=c_ref)
            )
    same = []
    same.append(ball_distance_embeddings(embeds["4LPK"], embeds["5US4"], curvature=c_ref))
    same.append(ball_distance_embeddings(embeds["6GOD"], embeds["6GOF"], curvature=c_ref))
    mean_cross = float(np.mean(cross))
    mean_same = float(np.mean(same))
    gap = mean_cross - mean_same
    disc_mean = float(np.nanmean(disc_rs))

    basin_pass = bool(gap > float(args.basin_sep_gap_min))
    disc_pass = bool(np.isfinite(disc_mean) and disc_mean >= float(args.disc_r_mean_min))
    hyp_ok = all(bool(t.get("hyp_mp_primary")) for t in telemetry.values())

    migration = None
    if not args.skip_migration_spotcheck:
        from experiments.diagnostics.v7_kras_g12d_hub_migration import main as mig_main

        mig_out = Path(
            "checkpoints/v7/diagnostics/phase_a_basin/kras_hub_migration_spotcheck.json"
        )
        mig_out.parent.mkdir(parents=True, exist_ok=True)
        rc = mig_main(
            [
                "--checkpoint",
                str(args.checkpoint),
                "--pdb-dir",
                str(args.pdb_dir),
                "--device",
                args.device,
                "--output",
                str(mig_out),
            ]
        )
        mig = json.loads(mig_out.read_text())
        sealed = None
        if SEALED_MIGRATION.is_file():
            sealed = json.loads(SEALED_MIGRATION.read_text()).get("verdict")
        migration = {
            "returncode": rc,
            "verdict": mig.get("verdict"),
            "sealed_baseline_verdict": sealed,
            "artifact": str(mig_out),
        }

    hard_pass = bool(basin_pass and disc_pass and hyp_ok)
    verdict = {
        "pass": hard_pass,
        "outcome": "Pass" if hard_pass else "Fail",
        "basin_sep_gap": gap,
        "basin_sep_gap_min": float(args.basin_sep_gap_min),
        "mean_cross_off_on": mean_cross,
        "mean_same_basin": mean_same,
        "disc_r_mean": disc_mean,
        "disc_r_mean_min": float(args.disc_r_mean_min),
        "hyp_mp_primary_all": hyp_ok,
        "migration_spotcheck": "report_only",
    }

    out = {
        "schema_version": 1,
        "probe": "tokyo_eye_v7_phase_a_basin_grade",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "prereg": str(PREREG),
        "telemetry": telemetry,
        "basin_distances": {
            "cross_off_on": cross,
            "same_basin": same,
            "mean_cross": mean_cross,
            "mean_same": mean_same,
            "gap": gap,
        },
        "migration_spotcheck": migration,
        "verdict": verdict,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")

    closeout = {
        "schema_version": 1,
        "gate": "tokyo_eye_v7_phase_a_basin_train_closeout",
        "recorded_at": out["recorded_at"],
        "prereg": str(PREREG),
        "checkpoint": str(args.checkpoint),
        "grade_artifact": str(args.output),
        "verdict": verdict,
        "spec": "docs/specs/tokyo-eye-v7/phase-a-nucleotide-basin-train-prereg.md",
    }
    args.closeout.write_text(json.dumps(closeout, indent=2) + "\n")
    print(json.dumps(verdict, indent=2))
    print(f"wrote {args.output}")
    print(f"wrote {args.closeout}")
    return 0 if hard_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
