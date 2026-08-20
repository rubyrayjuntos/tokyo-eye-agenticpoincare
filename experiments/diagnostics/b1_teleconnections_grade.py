#!/usr/bin/env python3
"""B1 teleconnections grade — AlleleSens on conduit vs scramble (sealed v7 Θ).

Pre-reg: ``docs/specs/tokyo-eye-v7/b1-teleconnections-prereg.md``
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.diagnostics.fix1_champion_hub_knockout_sweep import _align_prot_features
from experiments.training.v66._data import load_protein_graph_from_pdb_legacy
from experiments.training.v6.train_loop import prepare_training_batch, residue_node_count
from experiments.training.v7.healthy_bprime import HEALTHY_V7_CKPT
from science.dtie.common.kras_g12_graft import (
    GRAFT_RESSEQ,
    distal_scramble_donors,
    neighborhood_n12,
    residue_index_map,
)
from science.tokyo_eye.investigation_metrics import allele_sens, ball_distance_pair
from science.training.gnn_lineage import load_model_from_checkpoint

SPEC = "docs/specs/tokyo-eye-v7/b1-teleconnections-prereg.md"
PREREG = "data/gates/tokyo_eye_v7_b1_teleconnections_prereg.json"
PROBE = "tokyo_eye_v7_b1_teleconnections"
DEFAULT_OUTPUT = "checkpoints/v7/diagnostics/b1_teleconnections/b1_teleconnections.json"
DEFAULT_CLOSEOUT = "data/gates/tokyo_eye_v7_b1_teleconnections_closeout.json"
OFF_NOISE_FLOOR = 1e-4

# Literature / prior-hub locks for investigation-only destination reporting.
HUB_RESSEQS = (151, 163)
POCKET_RESSEQS = (14, 15, 16, 17, 30, 32, 34, 35, 61, 116, 117, 119, 146)

ARMS = {
    "off": {"wt": "4LPK", "mut": "5US4", "chain": "A"},
    "on": {"wt": "6GOD", "mut": "6GOF", "chain": "A"},
}

FIX1_HUB_MONITOR = (
    "checkpoints/v66/diagnostics/routing_sparsity/kras_hub_migration_4obe_4dso.json"
)


def _load(pdb_id: str, chain: str, pdb_dir: Path) -> dict[str, Any]:
    prot = load_protein_graph_from_pdb_legacy(pdb_id, chain, pdb_dir)
    if prot is None:
        raise FileNotFoundError(f"could not load {pdb_id}:{chain}")
    return prot


def _forward_x_hyp(
    model: torch.nn.Module,
    prot: dict[str, Any],
    device: str,
) -> tuple[torch.Tensor, dict[int, int], float, int, np.ndarray]:
    prot = _align_prot_features(model, prot)
    data = prepare_training_batch(model, prot, device)
    n = residue_node_count(data, prot)
    ids = list(prot.get("residue_ids") or [])[:n]
    idx_map = residue_index_map(ids)
    with torch.no_grad():
        out = model(data)
    x_hyp = out["x_hyp"].detach().cpu()[:n]
    c_t = (out.get("audit_trail") or {}).get("curvature_value")
    if c_t is None:
        c = float(model.curvature.detach().cpu().reshape(-1)[0])
    else:
        c = float(c_t.detach().cpu()) if torch.is_tensor(c_t) else float(c_t)
    ca = prot.get("ca_coords")
    ca_np = ca.detach().cpu().numpy() if torch.is_tensor(ca) else np.asarray(ca)
    ca_np = np.asarray(ca_np[:n], dtype=np.float64)
    return x_hyp, idx_map, c, n, ca_np


def _graph_idx(idx_map: dict[int, int], resseqs: list[int]) -> torch.Tensor:
    missing = [r for r in resseqs if r not in idx_map]
    if missing:
        raise KeyError(f"resseqs missing from graph map: {missing}")
    return torch.tensor([idx_map[r] for r in resseqs], dtype=torch.long)


def _align_pair_maps(
    map_wt: dict[int, int],
    map_mut: dict[int, int],
    resseqs: list[int],
) -> list[int]:
    return [r for r in resseqs if r in map_wt and r in map_mut]


def _mean_ca_dist(
    ca_wt: np.ndarray,
    ca_mut: np.ndarray,
    map_wt: dict[int, int],
    map_mut: dict[int, int],
    resseqs: list[int],
) -> float:
    ds: list[float] = []
    for r in resseqs:
        i, j = map_wt[r], map_mut[r]
        ds.append(float(np.linalg.norm(ca_wt[i] - ca_mut[j])))
    return float(np.mean(ds)) if ds else float("nan")


def _destination_block(
    *,
    x_wt: torch.Tensor,
    x_mut: torch.Tensor,
    map_wt: dict[int, int],
    map_mut: dict[int, int],
    curvature: float,
    candidates: tuple[int, ...],
    label: str,
) -> dict[str, Any]:
    present = _align_pair_maps(map_wt, map_mut, list(candidates))
    per: dict[str, float] = {}
    for r in present:
        i_wt, i_mut = map_wt[r], map_mut[r]
        d = ball_distance_pair(
            x_wt[i_wt : i_wt + 1],
            x_mut[i_mut : i_mut + 1],
            curvature=curvature,
        ).reshape(())
        per[str(r)] = float(d)
    mean_d = float(np.mean(list(per.values()))) if per else float("nan")
    return {
        "label": label,
        "candidates": list(candidates),
        "present_resseqs": present,
        "per_resseq_d_ball": per,
        "mean_d_ball": mean_d,
    }


def grade_arm(
    *,
    model: torch.nn.Module,
    device: str,
    pdb_dir: Path,
    arm: str,
) -> dict[str, Any]:
    cfg = ARMS[arm]
    wt_id, mut_id, chain = cfg["wt"], cfg["mut"], cfg["chain"]
    print(f"=== B1 {arm.upper()}: {wt_id} vs {mut_id} ===", flush=True)

    wt = _align_prot_features(model, _load(wt_id, chain, pdb_dir))
    mut = _align_prot_features(model, _load(mut_id, chain, pdb_dir))
    n12 = neighborhood_n12(mut, wt, seed_resseq=GRAFT_RESSEQ)
    scramble_raw = distal_scramble_donors(mut, n_needed=len(n12))

    print(f"  forward WT {wt_id} ...", flush=True)
    x_wt, map_wt, c_wt, n_wt, ca_wt = _forward_x_hyp(model, wt, device)
    print(f"  forward mut {mut_id} ...", flush=True)
    x_mut, map_mut, c_mut, n_mut, ca_mut = _forward_x_hyp(model, mut, device)
    curvature = float(c_wt)

    n12 = _align_pair_maps(map_wt, map_mut, n12)
    scramble = _align_pair_maps(map_wt, map_mut, scramble_raw)
    if len(scramble) < len(n12):
        # Pad from next distal candidates if some missing on WT.
        extra = distal_scramble_donors(mut, n_needed=len(n12) + 32)
        scramble = _align_pair_maps(map_wt, map_mut, extra)[: len(n12)]
    if len(n12) == 0 or len(scramble) != len(n12):
        raise RuntimeError(
            f"{arm}: bad neighborhoods |N12|={len(n12)} |scramble|={len(scramble)}"
        )

    idx_n12 = _graph_idx(map_wt, n12)
    # AlleleSens indexes into each tensor separately — use WT indices for wt rows
    # and mut indices for mut rows via aligned resseq lists.
    idx_n12_mut = _graph_idx(map_mut, n12)
    idx_scr_wt = _graph_idx(map_wt, scramble)
    idx_scr_mut = _graph_idx(map_mut, scramble)

    # Gather aligned embeddings for AlleleSens (same order).
    x_wt_n12 = x_wt.index_select(0, idx_n12)
    x_mut_n12 = x_mut.index_select(0, idx_n12_mut)
    x_wt_scr = x_wt.index_select(0, idx_scr_wt)
    x_mut_scr = x_mut.index_select(0, idx_scr_mut)

    as_n12 = float(
        allele_sens(
            x_wt_n12,
            x_mut_n12,
            torch.arange(x_wt_n12.shape[0]),
            curvature=curvature,
        )
    )
    as_scr = float(
        allele_sens(
            x_wt_scr,
            x_mut_scr,
            torch.arange(x_wt_scr.shape[0]),
            curvature=curvature,
        )
    )

    per_res = {}
    d_rows = ball_distance_pair(x_wt_n12, x_mut_n12, curvature=curvature)
    for r, d in zip(n12, d_rows.tolist()):
        per_res[str(r)] = float(d)

    hub = _destination_block(
        x_wt=x_wt,
        x_mut=x_mut,
        map_wt=map_wt,
        map_mut=map_mut,
        curvature=curvature,
        candidates=HUB_RESSEQS,
        label="hub",
    )
    pocket = _destination_block(
        x_wt=x_wt,
        x_mut=x_mut,
        map_wt=map_wt,
        map_mut=map_mut,
        curvature=curvature,
        candidates=POCKET_RESSEQS,
        label="pocket",
    )
    hub_m = hub["mean_d_ball"]
    poc_m = pocket["mean_d_ball"]
    if math.isfinite(hub_m) and math.isfinite(poc_m):
        dest_winner = "hub" if hub_m >= poc_m else "pocket"
    elif math.isfinite(hub_m):
        dest_winner = "hub"
    elif math.isfinite(poc_m):
        dest_winner = "pocket"
    else:
        dest_winner = "none"

    euc_n12 = _mean_ca_dist(ca_wt, ca_mut, map_wt, map_mut, n12)
    finite = bool(
        torch.isfinite(x_wt).all()
        and torch.isfinite(x_mut).all()
        and math.isfinite(as_n12)
        and math.isfinite(as_scr)
        and math.isfinite(curvature)
    )

    return {
        "arm": arm,
        "wt": wt_id,
        "mut": mut_id,
        "chain": chain,
        "n12": n12,
        "scramble": scramble,
        "curvature": curvature,
        "n_residues_wt": n_wt,
        "n_residues_mut": n_mut,
        "allele_sens_n12": as_n12,
        "allele_sens_scramble": as_scr,
        "conduit_beats_scramble": bool(as_n12 > as_scr),
        "finite": finite,
        "investigation": {
            "absolute_allele_sens": {
                "n12": as_n12,
                "scramble": as_scr,
            },
            "conduit_per_resseq_d_ball": per_res,
            "hub": hub,
            "pocket": pocket,
            "destination_winner_investigation": dest_winner,
            "prior_belief": "hub_hub",
            "euc_mean_ca_dist_n12": euc_n12,
        },
    }


def _verdict(off: dict[str, Any], on: dict[str, Any]) -> dict[str, Any]:
    as_on = float(on["allele_sens_n12"])
    as_off = float(off["allele_sens_n12"])
    as_on_scr = float(on["allele_sens_scramble"])
    as_off_scr = float(off["allele_sens_scramble"])

    bar_on_gt_off = bool(as_on > as_off)
    bar_on_scr = bool(as_on > as_on_scr)
    off_near_zero = bool(as_off < OFF_NOISE_FLOOR)
    bar_off_scr = bool(as_off > as_off_scr) or off_near_zero
    hygiene = bool(off["finite"] and on["finite"])

    passed = bar_on_gt_off and bar_on_scr and bar_off_scr and hygiene
    return {
        "pass": passed,
        "bars": {
            "on_conduit_gt_off_conduit": {
                "pass": bar_on_gt_off,
                "allele_sens_on": as_on,
                "allele_sens_off": as_off,
            },
            "on_conduit_gt_on_scramble": {
                "pass": bar_on_scr,
                "allele_sens_n12": as_on,
                "allele_sens_scramble": as_on_scr,
            },
            "off_conduit_gt_off_scramble_or_near_zero_waiver": {
                "pass": bar_off_scr,
                "allele_sens_n12": as_off,
                "allele_sens_scramble": as_off_scr,
                "near_zero_waiver": off_near_zero,
                "noise_floor": OFF_NOISE_FLOOR,
            },
            "hygiene_finite": {"pass": hygiene},
        },
    }


def _fix1_hub_soft_monitor() -> dict[str, Any]:
    path = Path(FIX1_HUB_MONITOR)
    if not path.is_file():
        return {"available": False, "path": str(path), "note": "missing artifact"}
    try:
        blob = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"available": False, "path": str(path), "error": str(exc)}
    return {
        "available": True,
        "path": str(path),
        "soft_monitor_only": True,
        "summary": {
            k: blob.get(k)
            for k in ("probe", "pass", "status", "R_4DSO", "R_4OBE", "structures")
            if k in blob
        },
    }


def grade(*, checkpoint: Path, pdb_dir: Path, device: str) -> dict[str, Any]:
    model = load_model_from_checkpoint(str(checkpoint), device)
    model.eval()

    off = grade_arm(model=model, device=device, pdb_dir=pdb_dir, arm="off")
    on = grade_arm(model=model, device=device, pdb_dir=pdb_dir, arm="on")
    verdict = _verdict(off, on)

    dest_note = (
        f"investigation destination winner: ON={on['investigation']['destination_winner_investigation']}, "
        f"OFF={off['investigation']['destination_winner_investigation']} "
        f"(prior hub_hub; not Pass/Fail)"
    )
    report = {
        "schema_version": 1,
        "probe": PROBE,
        "spec": SPEC,
        "prereg": PREREG,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "status": "PASS" if verdict["pass"] else "FAIL",
        "verdict": verdict,
        "arms": {"off": off, "on": on},
        "soft_monitors": {
            "absolute_allele_sens": {
                "off_n12": off["allele_sens_n12"],
                "on_n12": on["allele_sens_n12"],
                "off_scramble": off["allele_sens_scramble"],
                "on_scramble": on["allele_sens_scramble"],
            },
            "fix1_hub_migration": _fix1_hub_soft_monitor(),
        },
        "investigation_note": dest_note,
        "forbidden_claims": [
            "destination_type_pass",
            "fix1_hub_migration_pass",
            "absolute_allele_sens_floor_pass",
            "biology_champion_vs_fix1",
        ],
    }
    return report


def write_closeout(report: dict[str, Any], path: Path) -> None:
    stamp = {
        "schema_version": 1,
        "gate": "tokyo_eye_v7_b1_teleconnections_closeout",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "status": report["status"],
        "prereg": PREREG,
        "spec": SPEC,
        "checkpoint": report["checkpoint"],
        "artifact": None,
        "verdict": report["verdict"],
        "allele_sens": report["soft_monitors"]["absolute_allele_sens"],
        "investigation_note": report["investigation_note"],
        "note": (
            "B1 teleconnections closeout. Destination hub/pocket is investigation-only."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stamp, indent=2) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=HEALTHY_V7_CKPT)
    p.add_argument("--pdb-dir", type=Path, default=Path("/tmp/dtie_pdb_cache"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    p.add_argument("--closeout", type=Path, default=Path(DEFAULT_CLOSEOUT))
    p.add_argument("--no-closeout", action="store_true")
    args = p.parse_args()

    prereg_ok = Path(PREREG).is_file() or Path("/app").joinpath(PREREG).is_file()
    if not prereg_ok:
        raise SystemExit(f"missing prereg stamp {PREREG}")

    report = grade(
        checkpoint=args.checkpoint,
        pdb_dir=args.pdb_dir,
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output)}, indent=2))
    print("verdict bars:", json.dumps(report["verdict"]["bars"], indent=2))
    if not args.no_closeout:
        write_closeout(report, args.closeout)
        # Patch artifact path into closeout
        close = json.loads(args.closeout.read_text())
        close["artifact"] = str(args.output)
        args.closeout.write_text(json.dumps(close, indent=2) + "\n")
        print(f"closeout → {args.closeout}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
