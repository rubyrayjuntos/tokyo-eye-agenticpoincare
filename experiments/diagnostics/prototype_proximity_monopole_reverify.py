"""Re-run prototype-proximity vs monopole correlation on 3-D and legacy 4-D ge30.

Replicates ``monopole_and_seed_dilution_audit.audit_monopole_proto`` and the
``WEAK_OR_NO_PROTOTYPE_PROXIMITY_LINK`` verdict logic from the 4-D foundation
audit, so the causal finding can be re-scored on the correct three-vector input.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.diagnostics.prototype_repulsion_epoch import committed_majority_partition
from experiments.training.v6.corpus import load_training_proteins
from experiments.training.v6.train_loop import prepare_training_batch
from science.training.gnn_lineage import load_model_from_checkpoint

DEVICE = "cuda"

RUNS: list[tuple[str, Path, str]] = [
    (
        "three_vector_seed1",
        Path(
            "/app/checkpoints/v66/runs/fix1_s4_stack_three_vector_cold_seed1_v1/"
            "epochs/epoch_030.pt"
        ),
        "topology_three_vector",
    ),
    (
        "three_vector_seed2",
        Path(
            "/app/checkpoints/v66/runs/fix1_s4_stack_three_vector_cold_seed2_v1/"
            "epochs/epoch_030.pt"
        ),
        "topology_three_vector",
    ),
    (
        "legacy4d_seed1",
        Path(
            "/app/checkpoints/v66/runs/fix1_s4_proto_repulsion_scale_l2_stage_a12_cold_v1/"
            "epochs/epoch_030.pt"
        ),
        "legacy_four_vector",
    ),
    (
        "legacy4d_seed2",
        Path(
            "/app/checkpoints/v66/runs/fix1_s4_proto_repulsion_scale_l2_stage_a12_cold_seed2_v1/"
            "epochs/epoch_030.pt"
        ),
        "legacy_four_vector",
    ),
]

LEGACY_ARTIFACT = Path(
    "/app/checkpoints/v66/diagnostics/proto_repulsion_scale_l2_stack/"
    "monopole_and_seed_dilution_audit.json"
)


def audit_monopole_proto_for_mode(ckpt: Path, name: str, input_mode: str) -> dict:
    """Like monopole_and_seed_dilution_audit.audit_monopole_proto but reloads corpus per mode."""
    os.environ["GNN_INPUT_MODE"] = input_mode
    proteins, _ = load_training_proteins(
        Path("/tmp/dtie_pdb_cache"),
        Path("/app/manifests/v6_corpus_stage_a_small_v1.json"),
        max_proteins=12,
    )
    model = load_model_from_checkpoint(ckpt, DEVICE)
    model.to(DEVICE)
    model.core_capacity_quota_tau = 0.0
    model.eval()
    rows = []
    with torch.no_grad():
        for prot in proteins:
            data = prepare_training_batch(model, prot, DEVICE)
            in_f = int(getattr(model.node_emb, "in_features", data.x.size(-1)))
            if data.x.size(-1) > in_f:
                data.x = data.x[:, :in_f].contiguous()
            out = model(data)
            w = out["expert_weights"].float()
            pre = getattr(model.gate, "_last_pre_softmax", {}) or {}
            dists = pre.get("dists")
            if dists is None:
                continue
            dists = dists.float()
            top2 = torch.topk(dists, k=2, largest=False, dim=-1).values
            margin = top2[:, 1] - top2[:, 0]
            nearest = dists.argmin(dim=-1)
            max_p = w.max(dim=-1).values
            committed = max_p >= 0.60
            dh = prot.get("target_dehydron")
            if dh is None and data.x.size(1) > 1:
                dh = data.x[:, 1]
            if dh is None:
                dh = torch.zeros(w.shape[0], device=w.device)
            dh = dh.float().reshape(-1)[: w.shape[0]]
            part = committed_majority_partition(w.cpu().numpy(), dh.cpu().numpy())
            maj_e = part["majority_expert"]
            if maj_e is None or part["n_committed"] < 20:
                continue
            maj_mask = committed & (w.argmax(dim=-1) == int(maj_e))
            oth_mask = committed & (w.argmax(dim=-1) != int(maj_e))
            d_to_maj = dists[:, int(maj_e)]
            pdb = str(prot.get("pdb_id") or "?").upper()
            mean_d_all = dists.mean(dim=0).cpu().numpy()
            frac_nearest_maj = (
                float((nearest[maj_mask] == int(maj_e)).float().mean())
                if maj_mask.any()
                else float("nan")
            )
            rows.append(
                {
                    "pdb_id": pdb,
                    "majority_expert": int(maj_e),
                    "committed_hard_share_max": float(part["majority_share"]),
                    "n_committed": int(part["n_committed"]),
                    "mean_margin_maj": float(margin[maj_mask].mean())
                    if maj_mask.any()
                    else float("nan"),
                    "mean_margin_oth_committed": float(margin[oth_mask].mean())
                    if oth_mask.any()
                    else float("nan"),
                    "mean_d_to_maj_proto_on_maj": float(d_to_maj[maj_mask].mean())
                    if maj_mask.any()
                    else float("nan"),
                    "mean_d_to_maj_proto_on_all": float(d_to_maj.mean()),
                    "frac_maj_nearest_is_maj_proto": frac_nearest_maj,
                    "corpus_mean_d_to_each_proto": [float(x) for x in mean_d_all],
                    "monopole": bool(part["majority_share"] >= 0.56),
                }
            )

    shares = np.array([r["committed_hard_share_max"] for r in rows])
    margins = np.array([r["mean_margin_maj"] for r in rows])
    d_maj = np.array([r["mean_d_to_maj_proto_on_maj"] for r in rows])
    frac_nn = np.array([r["frac_maj_nearest_is_maj_proto"] for r in rows])

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 3 or a[m].std() == 0 or b[m].std() == 0:
            return float("nan")
        return float(np.corrcoef(a[m], b[m])[0, 1])

    mono = [r for r in rows if r["monopole"]]
    non = [r for r in rows if not r["monopole"]]
    return {
        "run": name,
        "n_structures": len(rows),
        "n_monopole": len(mono),
        "corr_share_vs_margin_maj": corr(shares, margins),
        "corr_share_vs_d_to_maj_proto": corr(shares, d_maj),
        "corr_share_vs_frac_nearest_maj": corr(shares, frac_nn),
        "mean_margin_maj_monopole": float(np.nanmean([r["mean_margin_maj"] for r in mono]))
        if mono
        else None,
        "mean_margin_maj_non_monopole": float(
            np.nanmean([r["mean_margin_maj"] for r in non])
        )
        if non
        else None,
        "mean_d_maj_monopole": float(
            np.nanmean([r["mean_d_to_maj_proto_on_maj"] for r in mono])
        )
        if mono
        else None,
        "mean_d_maj_non_monopole": float(
            np.nanmean([r["mean_d_to_maj_proto_on_maj"] for r in non])
        )
        if non
        else None,
        "per_structure": rows,
    }


def _verdict_from_corr(c: float) -> str:
    if c == c and c > 0.3:
        return "STRUCTURE_SPECIFIC_PROTOTYPE_CAPTURE"
    if c != c or abs(c) < 0.2:
        return "WEAK_OR_NO_PROTOTYPE_PROXIMITY_LINK"
    return "PARTIAL_PROTOTYPE_PROXIMITY_LINK"


def _frac_nearest_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fracs = np.array([r["frac_maj_nearest_is_maj_proto"] for r in rows], dtype=float)
    at_one = float(np.mean(np.isclose(fracs, 1.0, rtol=0, atol=1e-6)))
    return {
        "mean_frac_maj_nearest_is_maj_proto": float(np.nanmean(fracs)),
        "min_frac_maj_nearest_is_maj_proto": float(np.nanmin(fracs)),
        "frac_structures_all_maj_nearest_eq_1": at_one,
        "n_structures": len(rows),
    }


def _distance_contrast(rows: list[dict[str, Any]]) -> dict[str, Any]:
    mono = [r for r in rows if r["monopole"]]
    non = [r for r in rows if not r["monopole"]]
    d_mono = float(np.nanmean([r["mean_d_to_maj_proto_on_maj"] for r in mono])) if mono else None
    d_non = float(np.nanmean([r["mean_d_to_maj_proto_on_maj"] for r in non])) if non else None
    delta = (d_mono - d_non) if d_mono is not None and d_non is not None else None
    return {
        "mean_d_maj_monopole": d_mono,
        "mean_d_maj_non_monopole": d_non,
        "delta_monopole_minus_non_monopole": delta,
        "monopole_closer_than_non": bool(delta is not None and delta < 0),
        "monopole_farther_than_non": bool(delta is not None and delta > 0),
    }


def _headline_verdict(report: dict[str, Any]) -> dict[str, Any]:
    """Synthesize across corr metrics + distance contrast + frac saturation."""
    c_frac = report.get("corr_share_vs_frac_nearest_maj")
    c_dist = report.get("corr_share_vs_d_to_maj_proto")
    c_margin = report.get("corr_share_vs_margin_maj")
    frac_sum = report.get("frac_nearest_summary") or {}
    dist = report.get("distance_contrast") or {}

    notes: list[str] = []
    if frac_sum.get("frac_structures_all_maj_nearest_eq_1", 0) >= 0.99:
        notes.append(
            "frac_maj_nearest_is_maj_proto saturated at 1.0 — cross-structure corr on "
            "that axis is uninformative (near-zero variance)"
        )

    if c_dist == c_dist and c_dist > 0.3:
        mechanism = "HIGHER_SHARE_FARTHER_FROM_MAJ_PROTO"
        notes.append(
            "positive corr(share, d_to_maj_proto): higher share → farther from maj proto on maj residues"
        )
    elif c_dist == c_dist and c_dist < -0.3:
        mechanism = "PROTOTYPE_CAPTURE_DISTANCE_LIKELY"
        notes.append(
            "negative corr(share, d_to_maj_proto): higher share → closer to maj proto on maj residues"
        )
    elif dist.get("monopole_farther_than_non"):
        mechanism = "MONOPOLE_FARTHER_FROM_MAJ_PROTO"
        notes.append("mean distance contrast: monopole structures farther from maj proto on maj residues")
    elif dist.get("monopole_closer_than_non"):
        mechanism = "MONOPOLE_CLOSER_TO_MAJ_PROTO"
        notes.append("mean distance contrast: monopole structures closer to maj proto on maj residues")
    else:
        mechanism = "NO_CLEAR_DISTANCE_CONTRAST"

    legacy_verdict = _verdict_from_corr(float(c_frac) if c_frac == c_frac else float("nan"))

    return {
        "legacy_axis_verdict_frac_nearest_corr": legacy_verdict,
        "mechanism_read_distance_axis": mechanism,
        "corr_share_vs_frac_nearest_maj": c_frac,
        "corr_share_vs_d_to_maj_proto": c_dist,
        "corr_share_vs_margin_maj": c_margin,
        "notes": notes,
    }


def main() -> None:
    out_path = Path(
        "/app/checkpoints/v66/diagnostics/three_vector_stack_battery_reverify/"
        "prototype_proximity_monopole_reverify.json"
    )
    results: dict[str, Any] = {}
    for label, ckpt, input_mode in RUNS:
        if not ckpt.exists():
            results[label] = {"error": f"missing checkpoint {ckpt}"}
            continue
        audit = audit_monopole_proto_for_mode(ckpt, label, input_mode)
        rows = audit["per_structure"]
        summary = {k: v for k, v in audit.items() if k != "per_structure"}
        summary["gnn_input_mode"] = input_mode
        summary["frac_nearest_summary"] = _frac_nearest_summary(rows)
        summary["distance_contrast"] = _distance_contrast(rows)
        summary["headline"] = _headline_verdict({**summary, "per_structure": rows})
        summary["per_structure"] = rows
        results[label] = summary

    legacy_ref: dict[str, Any] | None = None
    if LEGACY_ARTIFACT.exists():
        legacy_ref = json.loads(LEGACY_ARTIFACT.read_text())

    three_d = {
        k: results[k]["headline"]
        for k in ("three_vector_seed1", "three_vector_seed2")
        if k in results and "error" not in results[k]
    }
    legacy_rerun = {
        k: results[k]["headline"]
        for k in ("legacy4d_seed1", "legacy4d_seed2")
        if k in results and "error" not in results[k]
    }

    report = {
        "tag": "PROTOTYPE_PROXIMITY_MONOPOLE_REVERIFY",
        "purpose": (
            "Re-score monopole_and_seed_dilution_audit prototype-proximity correlations "
            "on three-vector ge30 before choosing next lever"
        ),
        "verdict_logic": {
            "corr_share_vs_frac_nearest_maj_gt_0_3": "STRUCTURE_SPECIFIC_PROTOTYPE_CAPTURE",
            "abs_corr_lt_0_2_or_nan": "WEAK_OR_NO_PROTOTYPE_PROXIMITY_LINK",
            "else": "PARTIAL_PROTOTYPE_PROXIMITY_LINK",
        },
        "runs": results,
        "legacy_artifact_reference": (
            {
                "seed1": legacy_ref["monopole_prototype_proximity"]["seed1"],
                "seed2": legacy_ref["monopole_prototype_proximity"]["seed2"],
                "verdict_monopole_geometry": legacy_ref.get("verdicts", {}).get(
                    "monopole_geometry"
                ),
            }
            if legacy_ref
            else None
        ),
        "synthesis": {
            "three_vector_headlines": three_d,
            "legacy_rerun_headlines": legacy_rerun,
            "transfer_check": {
                "seed2_legacy_frac_corr_verdict": (
                    legacy_ref.get("verdicts", {}).get("monopole_geometry")
                    if legacy_ref
                    else None
                ),
                "seed2_three_vector_frac_corr_verdict": results.get(
                    "three_vector_seed2", {}
                )
                .get("headline", {})
                .get("legacy_axis_verdict_frac_nearest_corr"),
                "seed2_three_vector_mechanism_read": results.get(
                    "three_vector_seed2", {}
                )
                .get("headline", {})
                .get("mechanism_read_distance_axis"),
            },
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print("WROTE", out_path)
    print(json.dumps(report["synthesis"], indent=2))


if __name__ == "__main__":
    main()
