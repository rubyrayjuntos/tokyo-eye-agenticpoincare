"""Board-feature distinctiveness of monopolized/fallback maj residues vs rest.

Separate from Gram-conditioning pre-reg — checks whether the monopole
population is the least distinctive on the gate board within each structure.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiments.diagnostics.prototype_repulsion_epoch import (
    committed_majority_partition,
)
from experiments.training.v6.corpus import load_training_proteins
from experiments.training.v6.train_loop import prepare_training_batch
from science.training.gnn_lineage import load_model_from_checkpoint

DEVICE = "cuda"
RUNS = {
    "seed1": Path(
        "/app/checkpoints/v66/runs/fix1_s4_proto_repulsion_scale_l2_stage_a12_cold_v1"
    ),
    "seed2": Path(
        "/app/checkpoints/v66/runs/fix1_s4_proto_repulsion_scale_l2_stage_a12_cold_seed2_v1"
    ),
}

# Hand-feature names for topo board (topology_only + SASA; no disc in last_topo
# when disc is separate — _last_topo_features is pre-MLP concatenated board).
# Base 8 + optional SASA; disc_xy/r appended when use_disc_position.
BOARD_BASE = [
    "clustering",
    "cone_depth",
    "norm_degree",
    "norm_rho",
    "tau_flag",
    "ss_0",
    "ss_1",
    "ss_2",
]


proteins, _ = load_training_proteins(
    Path("/tmp/dtie_pdb_cache"),
    Path("/app/manifests/v6_corpus_stage_a_small_v1.json"),
    max_proteins=12,
)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 3 or float(a[m].std()) == 0.0 or float(b[m].std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 5 or b.size < 5:
        return float("nan")
    pooled = np.sqrt(
        ((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1))
        / max(a.size + b.size - 2, 1)
    )
    if pooled < 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def audit_board(ckpt: Path, name: str) -> dict:
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
            board = getattr(model.gate, "_last_topo_features", None)
            if board is None:
                continue
            board = board.float().cpu().numpy()
            # Drop disc columns if present: use first topo_dim continuous-ish cols
            topo_dim = int(getattr(model.gate, "topo_dim", board.shape[1]))
            feat = board[:, :topo_dim]
            # Within-structure z-score
            mu = feat.mean(axis=0, keepdims=True)
            sd = feat.std(axis=0, keepdims=True) + 1e-8
            z = (feat - mu) / sd
            dist_cent = np.linalg.norm(z, axis=1)
            # Local uniqueness: mean L2 to 5 nearest board neighbors in-structure
            # (excluding self)
            n = z.shape[0]
            if n > 6:
                # pairwise on z (n small, ≤~500)
                dmat = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=-1)
                np.fill_diagonal(dmat, np.inf)
                knn5 = np.sort(dmat, axis=1)[:, :5].mean(axis=1)
            else:
                knn5 = np.full(n, np.nan)

            max_p = w.max(dim=-1).values.cpu().numpy()
            hard = w.argmax(dim=-1).cpu().numpy()
            committed = max_p >= 0.60
            dh = prot.get("target_dehydron")
            if dh is None and data.x.size(1) > 1:
                dh = data.x[:, 1]
            if dh is None:
                dh = torch.zeros(w.shape[0])
            dh = dh.float().reshape(-1)[: w.shape[0]].cpu().numpy()
            part = committed_majority_partition(w.cpu().numpy(), dh)
            maj_e = part["majority_expert"]
            if maj_e is None or part["n_committed"] < 20:
                continue
            maj = committed & (hard == int(maj_e))
            oth_c = committed & (hard != int(maj_e))
            rest = ~maj

            def mean_on(mask, arr):
                return float(arr[mask].mean()) if mask.any() else float("nan")

            # Per continuous feature |cohen d| maj vs rest (skip one-hots ss/tau roughly)
            # Features 0..3 + sasa if present are continuous; tau=4; ss=5,6,7; sasa=8
            cont_idx = [0, 1, 2, 3]
            if topo_dim >= 9:
                cont_idx.append(8)  # sasa
            feat_ds = []
            for i in cont_idx:
                feat_ds.append(abs(_cohen_d(feat[maj, i], feat[rest, i])))
            mean_abs_d = float(np.nanmean(feat_ds)) if feat_ds else float("nan")

            rows.append(
                {
                    "pdb_id": str(prot.get("pdb_id") or "?").upper(),
                    "majority_expert": int(maj_e),
                    "committed_hard_share_max": float(part["majority_share"]),
                    "monopole": bool(part["majority_share"] >= 0.56),
                    "n_maj": int(maj.sum()),
                    "n_oth_committed": int(oth_c.sum()),
                    "mean_dist_cent_maj": mean_on(maj, dist_cent),
                    "mean_dist_cent_oth_committed": mean_on(oth_c, dist_cent),
                    "mean_dist_cent_rest": mean_on(rest, dist_cent),
                    "mean_knn5_maj": mean_on(maj, knn5),
                    "mean_knn5_oth_committed": mean_on(oth_c, knn5),
                    "mean_knn5_rest": mean_on(rest, knn5),
                    "mean_abs_cohen_d_maj_vs_rest_cont": mean_abs_d,
                    "ratio_dist_cent_maj_over_oth": (
                        mean_on(maj, dist_cent) / (mean_on(oth_c, dist_cent) + 1e-12)
                        if oth_c.any()
                        else float("nan")
                    ),
                    "ratio_knn5_maj_over_oth": (
                        mean_on(maj, knn5) / (mean_on(oth_c, knn5) + 1e-12)
                        if oth_c.any()
                        else float("nan")
                    ),
                }
            )

    shares = np.array([r["committed_hard_share_max"] for r in rows])
    d_maj = np.array([r["mean_dist_cent_maj"] for r in rows])
    d_oth = np.array([r["mean_dist_cent_oth_committed"] for r in rows])
    knn_maj = np.array([r["mean_knn5_maj"] for r in rows])
    knn_oth = np.array([r["mean_knn5_oth_committed"] for r in rows])
    ratios = np.array([r["ratio_dist_cent_maj_over_oth"] for r in rows])
    mono = [r for r in rows if r["monopole"]]
    non = [r for r in rows if not r["monopole"]]

    def mean_key(key, subset):
        vals = [r[key] for r in subset if np.isfinite(r.get(key, float("nan")))]
        return float(np.mean(vals)) if vals else None

    # Distinctiveness deficit: maj closer to centroid / less unique than oth
    mean_ratio = float(np.nanmean(ratios))
    mono_ratio = mean_key("ratio_dist_cent_maj_over_oth", mono)
    non_ratio = mean_key("ratio_dist_cent_maj_over_oth", non)

    if mean_ratio == mean_ratio and mean_ratio < 0.85:
        verdict = "FALLBACK_BOARD_LESS_DISTINCTIVE"
    elif (
        mono_ratio is not None
        and non_ratio is not None
        and mono_ratio + 0.05 < non_ratio
        and mono_ratio < 0.95
    ):
        verdict = "MONOPOLE_STRUCTURES_MAJ_LESS_DISTINCTIVE"
    elif mean_ratio == mean_ratio and mean_ratio > 1.15:
        verdict = "FALLBACK_BOARD_MORE_DISTINCTIVE_UNEXPECTED"
    else:
        verdict = "NO_CLEAR_BOARD_DISTINCTIVENESS_DEFICIT"

    return {
        "run": name,
        "n_structures": len(rows),
        "n_monopole": len(mono),
        "verdict": verdict,
        "mean_ratio_dist_cent_maj_over_oth": mean_ratio,
        "mean_ratio_knn5_maj_over_oth": float(
            np.nanmean([r["ratio_knn5_maj_over_oth"] for r in rows])
        ),
        "corr_share_vs_dist_cent_maj": _corr(shares, d_maj),
        "corr_share_vs_knn5_maj": _corr(shares, knn_maj),
        "corr_share_vs_ratio_dist": _corr(shares, ratios),
        "mean_dist_cent_maj_monopole": mean_key("mean_dist_cent_maj", mono),
        "mean_dist_cent_maj_non_monopole": mean_key("mean_dist_cent_maj", non),
        "mean_dist_cent_oth_monopole": mean_key("mean_dist_cent_oth_committed", mono),
        "mean_ratio_monopole": mono_ratio,
        "mean_ratio_non_monopole": non_ratio,
        "mean_abs_cohen_d_maj_vs_rest": mean_key(
            "mean_abs_cohen_d_maj_vs_rest_cont", rows
        ),
        "per_structure": rows,
    }


def main() -> None:
    out_audits = {}
    for name, run in RUNS.items():
        out_audits[name] = audit_board(run / "epochs" / "epoch_030.pt", name)
    report = {
        "seed1": {k: v for k, v in out_audits["seed1"].items() if k != "per_structure"},
        "seed2": {k: v for k, v in out_audits["seed2"].items() if k != "per_structure"},
        "seed1_per_structure": out_audits["seed1"]["per_structure"],
        "seed2_per_structure": out_audits["seed2"]["per_structure"],
        "verdicts": {
            "seed1": out_audits["seed1"]["verdict"],
            "seed2": out_audits["seed2"]["verdict"],
            "note": (
                "ratio_dist_cent_maj_over_oth < 1 ⇒ maj residues closer to "
                "structure board centroid than other committed (less distinctive)."
            ),
        },
    }
    path = Path("/tmp/monopole_board_distinctiveness_audit.json")
    path.write_text(json.dumps(report, indent=2) + "\n")
    print("REPORT_JSON_BEGIN")
    print(json.dumps(report))
    print("REPORT_JSON_END")
    print(
        json.dumps(
            {
                "verdicts": report["verdicts"],
                "seed1": report["seed1"],
                "seed2": report["seed2"],
            },
            indent=2,
        )
    )
    print("WROTE", path)


if __name__ == "__main__":
    main()
