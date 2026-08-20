#!/usr/bin/env python3
"""Forward-only first-ckpt grade for chem_mvp_euc_reach_v1 vs chem-MVP baseline.

Metrics (ablation §5 / chem-mvp-reengage):
  - cone_depth P@K on frozen S          (tag: hyperbolic_inference)
  - ‖encoder_h‖₂ P@K on frozen S         (tag: euclidean_construction)

K = max(10, ceil(0.15 * N)). Suite: 1BE9 / 1GPW(A+B) / 1F88.

Does not interrupt training. Does not overwrite the baseline checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

from experiments.diagnostics.jacobian_flow_influence import _capture_layers
from experiments.training.v66.manifold_ssot import (
    EUCLIDEAN_REACH_SUITE as SUITE,
    PDB_DIRS,
    find_pdb as _find_pdb,
    load_suite_prot,
)
from experiments.training.v66.train_loop import prepare_training_batch, residue_node_count
from science.training.gnn_lineage import load_model_from_checkpoint

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "checkpoints/v66/diagnostics/euclidean_reach"
SITE_DIR = DEFAULT_OUT / "site_lists"
BASELINE_CKPT = REPO / "checkpoints/v66/runs/chem_mvp_stage_a12_cold_v1/v66_best.pt"
TREATMENT_DIR = REPO / "checkpoints/v66/runs/chem_mvp_euc_reach_v1"


def k_rule(n: int) -> int:
    return max(10, int(math.ceil(0.15 * n)))


def _parse_chain_auth(rid: Any) -> tuple[str, int]:
    parts = str(rid).split(":")
    return parts[0], int(parts[1])


def load_frozen_S(path: Path) -> list[tuple[str, int]]:
    blob = json.loads(path.read_text())
    if not blob.get("frozen"):
        raise RuntimeError(f"site list not frozen: {path}")
    hubs = blob.get("hubs") or []
    keys = [(str(h["chain_id"]), int(h["auth_seq"])) for h in hubs]
    if not keys and blob.get("unique_keys"):
        for k in blob["unique_keys"]:
            c, a = str(k).split(":")
            keys.append((c, int(a)))
    return keys


def resolve_treatment_ckpt(run_dir: Path) -> Path:
    ep0 = run_dir / "epochs" / "epoch_000.pt"
    if ep0.is_file():
        return ep0
    best = run_dir / "v66_best.pt"
    if best.is_file():
        return best
    raise FileNotFoundError(f"No epoch_000.pt or v66_best.pt under {run_dir}")


def enrichment(
    salience: np.ndarray,
    S_indices: list[int],
    *,
    residue_ids: list[Any],
    metric: str,
    tag: str,
) -> dict[str, Any]:
    n = int(salience.shape[0])
    k = k_rule(n)
    order = np.argsort(-salience)
    topk = order[:k]
    S_set = set(S_indices)
    hits = [
        f"{_parse_chain_auth(residue_ids[i])[0]}:{_parse_chain_auth(residue_ids[i])[1]}"
        for i in topk
        if int(i) in S_set
    ]
    n_hits = len(hits)
    mask_s = np.zeros(n, dtype=bool)
    mask_s[S_indices] = True
    mean_s = float(np.mean(salience[mask_s])) if S_indices else float("nan")
    mean_c = float(np.mean(salience[~mask_s])) if (~mask_s).any() else float("nan")
    return {
        "tag": tag,
        "salience_metric": metric,
        "N": n,
        "K": k,
        "K_rule": "max(10, ceil(0.15*N))",
        "|S|": len(S_indices),
        "hits_in_topk": hits,
        "n_hits": n_hits,
        "precision_at_K": n_hits / k,
        "recall_at_K": n_hits / max(len(S_indices), 1),
        "mean_salience_on_S": mean_s,
        "mean_salience_on_complement": mean_c,
        "salience_gap_S_minus_complement": mean_s - mean_c,
    }


def forward_metrics(
    model: torch.nn.Module, data: Any
) -> tuple[np.ndarray, np.ndarray, float | None]:
    handles, captured = _capture_layers(model)
    try:
        with torch.no_grad():
            out = model(data)
    finally:
        for h in handles:
            h.remove()
    if "encoder_h" not in captured:
        raise RuntimeError("failed to capture encoder_h")
    h = captured["encoder_h"].detach().float().cpu().numpy()
    depth = out["cone_depth"].detach().float().cpu().numpy().reshape(-1)
    route_h: float | None = None
    if "routing_entropy" in out:
        rh = out["routing_entropy"]
        route_h = float(rh.detach().cpu()) if torch.is_tensor(rh) else float(rh)
    return depth, h, route_h


def grade_arm(
    *,
    name: str,
    checkpoint: Path,
    prot: dict[str, Any],
    S_keys: list[tuple[str, int]],
    device: str,
) -> dict[str, Any]:
    model = load_model_from_checkpoint(checkpoint, device)
    model.eval()
    data = prepare_training_batch(model, prot, device)
    n = residue_node_count(data, prot)
    residue_ids = list(prot["residue_ids"])[:n]
    idx_map = {_parse_chain_auth(r): i for i, r in enumerate(residue_ids)}
    missing = [f"{c}:{a}" for c, a in S_keys if (c, a) not in idx_map]
    if missing:
        raise RuntimeError(f"{name}/{prot.get('pdb_id')}: S missing from graph: {missing}")
    S_indices = [idx_map[k] for k in S_keys]

    depth, h, route_h = forward_metrics(model, data)
    depth = depth[:n]
    h = h[:n]
    trunk = np.linalg.norm(h, axis=1)

    cone = enrichment(
        depth,
        S_indices,
        residue_ids=residue_ids,
        metric="cone_depth",
        tag="hyperbolic_inference",
    )
    trunk_e = enrichment(
        trunk,
        S_indices,
        residue_ids=residue_ids,
        metric="encoder_h_l2_norm",
        tag="euclidean_construction",
    )
    return {
        "arm": name,
        "checkpoint": str(checkpoint),
        "euclidean_shortcut_mp": bool(getattr(model, "euclidean_shortcut_mp", False)),
        "chem_edge_mp": bool(getattr(model, "chem_edge_mp", False)),
        "N": n,
        "route_H": route_h,
        "S_keys": [f"{c}:{a}" for c, a in S_keys],
        "cone_depth": cone,
        "trunk_encoder_h_l2": trunk_e,
    }


def _delta(a: dict[str, Any], b: dict[str, Any], key: str) -> float:
    return float(b[key] - a[key])


def grade_structure(
    *,
    pdb_id: str,
    tier: str,
    chains: tuple[str, ...],
    baseline_ckpt: Path,
    treatment_ckpt: Path,
    device: str,
    out_dir: Path,
    per_structure_name: str | None = None,
) -> dict[str, Any]:
    site_path = SITE_DIR / f"{pdb_id.lower()}_pathway_residues.json"
    S_keys = load_frozen_S(site_path)
    prot = load_suite_prot(pdb_id, chains)
    print(f"=== {pdb_id} N={prot['n_residues']} chains={chains} |S|={len(S_keys)} ===", flush=True)

    base = grade_arm(
        name="chem_mvp_baseline",
        checkpoint=baseline_ckpt,
        prot=prot,
        S_keys=S_keys,
        device=device,
    )
    treat = grade_arm(
        name="chem_mvp_euc_reach_v1",
        checkpoint=treatment_ckpt,
        prot=prot,
        S_keys=S_keys,
        device=device,
    )

    row = {
        "schema_version": 1,
        "ssot": "docs/specs/v66_chem_MVP/ablation_euclidean_reach.md",
        "structure": pdb_id,
        "tier": tier,
        "chains": list(chains),
        "site_list": str(site_path.relative_to(REPO)),
        "site_list_frozen": True,
        "|S|": len(S_keys),
        "baseline_checkpoint": str(baseline_ckpt),
        "treatment_checkpoint": str(treatment_ckpt),
        "baseline": base,
        "treatment": treat,
        "delta_cone_depth_precision_at_K": _delta(
            base["cone_depth"], treat["cone_depth"], "precision_at_K"
        ),
        "delta_cone_depth_recall_at_K": _delta(
            base["cone_depth"], treat["cone_depth"], "recall_at_K"
        ),
        "delta_trunk_precision_at_K": _delta(
            base["trunk_encoder_h_l2"], treat["trunk_encoder_h_l2"], "precision_at_K"
        ),
        "delta_trunk_recall_at_K": _delta(
            base["trunk_encoder_h_l2"], treat["trunk_encoder_h_l2"], "recall_at_K"
        ),
        "knockout": {
            "status": "not_run",
            "note": "First-ckpt grade is enrichment-only; knockout left for full suite.",
        },
        "honesty": {
            "partial_is_not_pass": True,
            "instruments_complete_for_enrichment": True,
            "instruments_complete_for_suite_pass": False,
            "missing_for_suite_pass": ["knockout_out_effect_on_S", "rim_non_regression"],
        },
    }
    fname = per_structure_name or f"grade_{pdb_id.lower()}.json"
    out_path = out_dir / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(row, indent=2) + "\n")
    row["_artifact_rel"] = str(out_path.relative_to(REPO)) if out_path.is_relative_to(REPO) else str(out_path)
    print(
        f"  cone P@K base={base['cone_depth']['precision_at_K']:.4f} "
        f"treat={treat['cone_depth']['precision_at_K']:.4f} "
        f"Δ={row['delta_cone_depth_precision_at_K']:+.4f}",
        flush=True,
    )
    print(
        f"  trunk P@K base={base['trunk_encoder_h_l2']['precision_at_K']:.4f} "
        f"treat={treat['trunk_encoder_h_l2']['precision_at_K']:.4f} "
        f"Δ={row['delta_trunk_precision_at_K']:+.4f}",
        flush=True,
    )
    return row


def suite_verdict(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by = {r["structure"]: r for r in rows}
    gpw = by.get("1GPW")
    f88 = by.get("1F88")
    be9 = by.get("1BE9")

    def _mean_delta(metric: str) -> float | None:
        vals = []
        for r in (gpw, f88):
            if r is None:
                return None
            vals.append(float(r[metric]))
        return float(np.mean(vals))

    mean_cone = _mean_delta("delta_cone_depth_precision_at_K")
    mean_trunk = _mean_delta("delta_trunk_precision_at_K")
    be9_cone = float(be9["delta_cone_depth_precision_at_K"]) if be9 else None
    be9_trunk = float(be9["delta_trunk_precision_at_K"]) if be9 else None

    cone_ok = (
        mean_cone is not None
        and be9_cone is not None
        and mean_cone >= 0.05
        and be9_cone >= 0.0
    )
    trunk_ok = (
        mean_trunk is not None
        and be9_trunk is not None
        and mean_trunk >= 0.0
        and be9_trunk >= 0.0
    )
    # Suite Pass requires knockout too — enrichment-only → Partial at best.
    if cone_ok and trunk_ok:
        verdict = "Partial"
        reason = (
            "Enrichment bars hold on cone+trunk, but knockout + rim floors "
            "not graded → Partial≠Pass."
        )
    else:
        verdict = "Fail"
        reason = "Enrichment Δ bars not met (or incomplete suite)."

    return {
        "mean_delta_cone_P@K_1gpw_1f88": mean_cone,
        "mean_delta_trunk_P@K_1gpw_1f88": mean_trunk,
        "delta_cone_P@K_1be9": be9_cone,
        "delta_trunk_P@K_1be9": be9_trunk,
        "cone_enrichment_bar": cone_ok,
        "trunk_enrichment_bar": trunk_ok,
        "suite_verdict": verdict,
        "reason": reason,
        "pass_bars_ssot": {
            "cone": "mean Δ{1GPW,1F88} ≥ +0.05; 1BE9 Δ ≥ 0",
            "trunk": "mean Δ{1GPW,1F88} ≥ 0; 1BE9 Δ ≥ 0",
            "knockout": "not graded here",
        },
    }


def _treatment_epoch_meta(checkpoint: Path) -> dict[str, Any]:
    """Best-effort epoch / score from checkpoint payload (CPU, no model load)."""
    meta: dict[str, Any] = {"path": str(checkpoint)}
    try:
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ck = torch.load(checkpoint, map_location="cpu")
    if not isinstance(ck, dict):
        return meta
    for k in ("global_epoch", "epoch", "phase", "phase_name", "score"):
        if k in ck:
            meta[k] = ck[k]
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, default=BASELINE_CKPT)
    ap.add_argument("--treatment-dir", type=Path, default=TREATMENT_DIR)
    ap.add_argument("--treatment-ckpt", type=Path, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--suite-out",
        type=str,
        default="grade_first_ckpt_chem_mvp_euc_reach_v1.json",
        help="Aggregate JSON filename under --out-dir",
    )
    ap.add_argument(
        "--grade-kind",
        type=str,
        default="first_checkpoint_forward_only",
    )
    ap.add_argument(
        "--per-structure-subdir",
        type=str,
        default="",
        help="If set, write per-PDB grades under out-dir/<subdir>/ (preserves root first-ckpt artifacts).",
    )
    ap.add_argument(
        "--write-grade-suite-alias",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also write grade_suite.json (disable for best-ckpt to avoid clobbering first-ckpt alias).",
    )
    args = ap.parse_args()

    treatment = args.treatment_ckpt or resolve_treatment_ckpt(args.treatment_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_dir = args.out_dir
    if args.per_structure_subdir:
        per_dir = args.out_dir / args.per_structure_subdir
        per_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for spec in SUITE:
        rows.append(
            grade_structure(
                pdb_id=spec["pdb_id"],
                tier=spec["tier"],
                chains=spec["chains"],
                baseline_ckpt=args.baseline,
                treatment_ckpt=treatment,
                device=args.device,
                out_dir=per_dir,
            )
        )

    treatment_meta = _treatment_epoch_meta(treatment)
    aggregate = {
        "schema_version": 1,
        "ssot": "docs/specs/v66_chem_MVP/ablation_euclidean_reach.md",
        "run_id": "chem_mvp_euc_reach_v1",
        "grade_kind": args.grade_kind,
        "baseline_checkpoint": str(args.baseline),
        "treatment_checkpoint": str(treatment),
        "treatment_checkpoint_meta": treatment_meta,
        "device": args.device,
        "structures": {
            r["structure"]: {
                "tier": r["tier"],
                "N": r["baseline"]["N"],
                "K": r["baseline"]["cone_depth"]["K"],
                "|S|": r["|S|"],
                "cone_P@K_baseline": r["baseline"]["cone_depth"]["precision_at_K"],
                "cone_P@K_treatment": r["treatment"]["cone_depth"]["precision_at_K"],
                "delta_cone_P@K": r["delta_cone_depth_precision_at_K"],
                "trunk_P@K_baseline": r["baseline"]["trunk_encoder_h_l2"]["precision_at_K"],
                "trunk_P@K_treatment": r["treatment"]["trunk_encoder_h_l2"]["precision_at_K"],
                "delta_trunk_P@K": r["delta_trunk_precision_at_K"],
                "artifact": r.get(
                    "_artifact_rel",
                    f"grade_{r['structure'].lower()}.json",
                ),
            }
            for r in rows
        },
        "aggregate": suite_verdict(rows),
        "honesty": {
            "partial_is_not_pass": True,
            "knockout_not_run": True,
            "rim_non_regression_not_run": True,
        },
    }
    suite_path = args.out_dir / args.suite_out
    suite_path.write_text(json.dumps(aggregate, indent=2) + "\n")
    if args.write_grade_suite_alias:
        # Also write grade_suite.json alias expected by ablation §8
        (args.out_dir / "grade_suite.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    print(f"wrote {suite_path}", flush=True)
    print(json.dumps(aggregate["aggregate"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
