"""(1) Monopole vs prototype proximity  (2) seed1 dilution vs repulsion trajectory."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiments.diagnostics.prototype_repulsion_epoch import (
    committed_majority_partition,
    measure_prototype_repulsion_epoch,
)
from experiments.training.v6.corpus import load_training_proteins
from experiments.training.v6.train_loop import prepare_training_batch
from science.training.gnn_lineage import load_model_from_checkpoint

DEVICE = "cuda"
proteins, _ = load_training_proteins(
    Path("/tmp/dtie_pdb_cache"),
    Path("/app/manifests/v6_corpus_stage_a_small_v1.json"),
    max_proteins=12,
)


def audit_monopole_proto(ckpt: Path, name: str) -> dict:
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

    def corr(a, b):
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


def seed_compare() -> dict:
    out = {}
    for name, run in [
        (
            "seed1",
            Path(
                "/app/checkpoints/v66/runs/fix1_s4_proto_repulsion_scale_l2_stage_a12_cold_v1"
            ),
        ),
        (
            "seed2",
            Path(
                "/app/checkpoints/v66/runs/fix1_s4_proto_repulsion_scale_l2_stage_a12_cold_seed2_v1"
            ),
        ),
    ]:
        rows = [
            json.loads(l)
            for l in (run / "prototype_repulsion_per_epoch.jsonl")
            .read_text()
            .splitlines()
            if l.strip()
        ]
        by = {int(r["global_epoch"]): r for r in rows}
        traj = [
            {
                "ge": ge,
                "nearest_d": by[ge].get("nearest_pair_hyp_dist"),
                "twin_d": by[ge].get("historical_twin_hyp_dist"),
                "frac": by[ge].get("frac_max_p_ge_0_60"),
            }
            for ge in sorted(by)
            if ge in (0, 5, 10, 15, 20, 25, 30)
        ]
        ckpt = run / "epochs" / "epoch_030.pt"
        model = load_model_from_checkpoint(ckpt, DEVICE)
        model.to(DEVICE)
        model.core_capacity_quota_tau = 0.0
        model.eval()
        report = measure_prototype_repulsion_epoch(model, proteins, DEVICE)
        purity = report.get("dehydron_partition_purity") or {}
        out[name] = {
            "nearest_d_ge0": by[0].get("nearest_pair_hyp_dist"),
            "nearest_d_ge30": by[30].get("nearest_pair_hyp_dist"),
            "twin_d_ge30": by[30].get("historical_twin_hyp_dist"),
            "delta_nearest_0_to_30": float(by[30].get("nearest_pair_hyp_dist") or 0)
            - float(by[0].get("nearest_pair_hyp_dist") or 0),
            "frac_ge30": report.get("frac_max_p_ge_0_60"),
            "struct_ch_ge30": report.get("per_structure_committed_hard_max"),
            "relative_purity": {
                "verdict": purity.get("verdict"),
                "passes": purity.get("passes"),
                "n_eligible": purity.get("n_eligible_structures"),
                "n_blurred": len(purity.get("blurred_pdb_ids") or []),
                "floor_mode": purity.get("floor_mode"),
                "blurred": purity.get("blurred_pdb_ids"),
            },
            "trajectory": traj,
        }
    return out


def main() -> None:
    mono1 = audit_monopole_proto(
        Path(
            "/app/checkpoints/v66/runs/fix1_s4_proto_repulsion_scale_l2_stage_a12_cold_v1/epochs/epoch_030.pt"
        ),
        "seed1",
    )
    mono2 = audit_monopole_proto(
        Path(
            "/app/checkpoints/v66/runs/fix1_s4_proto_repulsion_scale_l2_stage_a12_cold_seed2_v1/epochs/epoch_030.pt"
        ),
        "seed2",
    )
    seeds = seed_compare()
    s1_weak = seeds["seed1"]["nearest_d_ge30"] < seeds["seed2"]["nearest_d_ge30"]
    c = mono2["corr_share_vs_frac_nearest_maj"]
    if c == c and c > 0.3:
        mono_v = "STRUCTURE_SPECIFIC_PROTOTYPE_CAPTURE"
    elif c != c or abs(c) < 0.2:
        mono_v = "WEAK_OR_NO_PROTOTYPE_PROXIMITY_LINK"
    else:
        mono_v = "PARTIAL_PROTOTYPE_PROXIMITY_LINK"

    report = {
        "monopole_prototype_proximity": {
            "seed1": {k: v for k, v in mono1.items() if k != "per_structure"},
            "seed2": {k: v for k, v in mono2.items() if k != "per_structure"},
            "seed1_per_structure": mono1["per_structure"],
            "seed2_per_structure": mono2["per_structure"],
        },
        "seed1_vs_seed2_repulsion_and_purity": seeds,
        "verdicts": {
            "monopole_geometry": mono_v,
            "seed1_dilution_from_weak_repulsion": (
                "YES_WEAKER_REPULSION"
                if s1_weak
                else "NO_SEED1_HAS_STRONGER_NEAREST_PAIR_SEP"
            ),
            "note_monopole": (
                "corr(committed_share, frac maj residues nearest to maj proto); "
                "also margin / distance contrasts mono vs non-mono structures"
            ),
            "note_seed1": (
                "If seed1 nearest_d@ge30 > seed2, dilution is not from weaker "
                "prototype repulsion trajectory"
            ),
        },
    }
    # Container user often cannot write under /app/checkpoints; dump to /tmp for host copy.
    out = Path("/tmp/monopole_and_seed_dilution_audit.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("REPORT_JSON_BEGIN")
    print(json.dumps(report))
    print("REPORT_JSON_END")
    print(
        json.dumps(
            {
                "verdicts": report["verdicts"],
                "seed1_mono": report["monopole_prototype_proximity"]["seed1"],
                "seed2_mono": report["monopole_prototype_proximity"]["seed2"],
                "seeds": {
                    k: {kk: seeds[k][kk] for kk in seeds[k] if kk != "trajectory"}
                    for k in seeds
                },
            },
            indent=2,
        )
    )
    print("WROTE", out)


if __name__ == "__main__":
    main()
