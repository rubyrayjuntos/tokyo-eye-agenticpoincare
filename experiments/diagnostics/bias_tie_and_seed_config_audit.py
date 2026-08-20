"""Cheap pre-draft checks: (1) monopole equidistance + expert_bias ties
(2) seed1 dilution vs ge0/ge30 prototype configuration beyond nearest-pair.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from geoopt.manifolds.stereographic import math as pmath

from experiments.diagnostics.prototype_repulsion_epoch import (
    committed_majority_partition,
    measure_prototype_pairwise,
    measure_prototype_repulsion_epoch,
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


def _proto_config(model: torch.nn.Module) -> dict:
    """Beyond nearest-pair: full pairwise bank geometry in hyp + tangent."""
    pw = measure_prototype_pairwise(model)
    dists = np.array([p["hyp_dist"] for p in pw["pairwise"]], dtype=float)
    cos = np.array([p["tangent_cos"] for p in pw["pairwise"]], dtype=float)
    gate = model.gate
    k = -model.curvature
    with torch.no_grad():
        proto_tan = gate.prototype_bank.prototype_tangent.detach().float()
        tn = F.normalize(proto_tan, dim=-1)
        # Gram of unit tangents — eigenvalues of 4x4 cos matrix
        gram = (tn @ tn.T).cpu().numpy()
        evals = np.linalg.eigvalsh(gram)
        bias = gate.expert_bias.detach().float().cpu().numpy()
        scale = float(F.softplus(gate.logit_scale).item())
        floor = getattr(gate, "logit_softplus_floor", None)
        if floor is not None:
            scale = max(scale, float(floor))
        proto_hyp = gate.prototype_bank(k)
        # Centroid of bank in tangent; radius of each proto from centroid
        centroid = proto_tan.mean(dim=0, keepdim=True)
        rad = torch.norm(proto_tan - centroid, dim=-1).cpu().numpy()
        # Pairwise angles via acos(clip cos)
        angles = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return {
        "nearest_pair": pw["nearest_pair"],
        "nearest_pair_hyp_dist": pw["nearest_pair_hyp_dist"],
        "twin_hyp_dist": pw["historical_twin_hyp_dist"],
        "pairwise_hyp_dists": [float(x) for x in dists],
        "pairwise_tangent_cos": [float(x) for x in cos],
        "pairwise_angles_deg": [float(x) for x in angles],
        "hyp_dist_mean": float(dists.mean()),
        "hyp_dist_std": float(dists.std()),
        "hyp_dist_cv": float(dists.std() / (dists.mean() + 1e-12)),
        "hyp_dist_min_over_max": float(dists.min() / (dists.max() + 1e-12)),
        "angle_mean_deg": float(angles.mean()),
        "angle_std_deg": float(angles.std()),
        "angle_min_deg": float(angles.min()),
        "gram_eigenvalues": [float(x) for x in evals],
        "gram_eig_min": float(evals.min()),
        "gram_eig_max": float(evals.max()),
        "gram_condition": float(evals.max() / (abs(evals.min()) + 1e-12)),
        "proto_radius_from_centroid": [float(x) for x in rad],
        "proto_radius_cv": float(rad.std() / (rad.mean() + 1e-12)),
        "expert_bias": [float(x) for x in bias],
        "bias_argmax": int(np.argmax(bias)),
        "bias_argmin": int(np.argmin(bias)),
        "bias_range": float(bias.max() - bias.min()),
        "logit_scale_softplus": scale,
        "pairwise_detail": pw["pairwise"],
    }


def audit_equidistance_and_bias(ckpt: Path, name: str) -> dict:
    model = load_model_from_checkpoint(ckpt, DEVICE)
    model.to(DEVICE)
    model.core_capacity_quota_tau = 0.0
    model.eval()
    cfg = _proto_config(model)
    bias = np.array(cfg["expert_bias"], dtype=float)
    bias_argmax = int(cfg["bias_argmax"])

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
            scale = float(cfg["logit_scale_softplus"])
            logits_nobias = -scale * dists
            logits_bias = logits_nobias + torch.as_tensor(
                bias, device=dists.device, dtype=dists.dtype
            )
            arg_nobias = logits_nobias.argmax(dim=-1)
            arg_bias = logits_bias.argmax(dim=-1)
            top2_d = torch.topk(dists, k=2, largest=False, dim=-1).values
            margin_d = top2_d[:, 1] - top2_d[:, 0]
            # Equidistance: std of distances across experts (low = ambiguous)
            d_std = dists.std(dim=-1)
            d_range = dists.max(dim=-1).values - dists.min(dim=-1).values
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
            # Near-tie among maj: distance margin below thresholds
            near_tie_01 = (margin_d[maj_mask] < 0.10).float().mean() if maj_mask.any() else float("nan")
            near_tie_05 = (margin_d[maj_mask] < 0.05).float().mean() if maj_mask.any() else float("nan")
            # Would bias flip maj residues relative to distance-only argmax?
            flip_on_maj = (
                float((arg_bias[maj_mask] != arg_nobias[maj_mask]).float().mean())
                if maj_mask.any()
                else float("nan")
            )
            # Distance-only majority among committed
            hard_nobias = arg_nobias[committed]
            if hard_nobias.numel() > 0:
                counts = torch.bincount(hard_nobias, minlength=4)
                maj_nobias = int(counts.argmax().item())
            else:
                maj_nobias = None
            pdb = str(prot.get("pdb_id") or "?").upper()
            rows.append(
                {
                    "pdb_id": pdb,
                    "majority_expert": int(maj_e),
                    "majority_expert_nobias": maj_nobias,
                    "majority_matches_bias_argmax": bool(int(maj_e) == bias_argmax),
                    "committed_hard_share_max": float(part["majority_share"]),
                    "monopole": bool(part["majority_share"] >= 0.56),
                    "mean_d_std_maj": float(d_std[maj_mask].mean()) if maj_mask.any() else float("nan"),
                    "mean_d_range_maj": float(d_range[maj_mask].mean())
                    if maj_mask.any()
                    else float("nan"),
                    "mean_margin_maj": float(margin_d[maj_mask].mean())
                    if maj_mask.any()
                    else float("nan"),
                    "frac_maj_near_tie_margin_lt_0_10": float(near_tie_01)
                    if not isinstance(near_tie_01, float)
                    else near_tie_01,
                    "frac_maj_near_tie_margin_lt_0_05": float(near_tie_05)
                    if not isinstance(near_tie_05, float)
                    else near_tie_05,
                    "frac_maj_bias_flips_argmax": flip_on_maj,
                    "mean_max_p_maj": float(max_p[maj_mask].mean()) if maj_mask.any() else float("nan"),
                }
            )

    shares = np.array([r["committed_hard_share_max"] for r in rows])
    d_stds = np.array([r["mean_d_std_maj"] for r in rows])
    margins = np.array([r["mean_margin_maj"] for r in rows])
    ties = np.array([r["frac_maj_near_tie_margin_lt_0_10"] for r in rows])
    mono = [r for r in rows if r["monopole"]]
    non = [r for r in rows if not r["monopole"]]
    maj_experts = [r["majority_expert"] for r in rows]
    match_bias = sum(1 for r in rows if r["majority_matches_bias_argmax"])
    maj_changed_by_bias = sum(
        1
        for r in rows
        if r["majority_expert_nobias"] is not None
        and r["majority_expert"] != r["majority_expert_nobias"]
    )

    # Equidistance contrast
    def _mean(key, subset):
        vals = [r[key] for r in subset if np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else None

    return {
        "run": name,
        "expert_bias": cfg["expert_bias"],
        "bias_argmax": bias_argmax,
        "bias_range": cfg["bias_range"],
        "logit_scale_softplus": cfg["logit_scale_softplus"],
        "n_structures": len(rows),
        "n_monopole": len(mono),
        "frac_structures_maj_equals_bias_argmax": match_bias / max(len(rows), 1),
        "n_structures_maj_changed_by_bias": maj_changed_by_bias,
        "majority_expert_counts": {
            str(i): int(sum(1 for m in maj_experts if m == i)) for i in range(4)
        },
        "corr_share_vs_d_std_maj": _corr(shares, d_stds),
        "corr_share_vs_margin_maj": _corr(shares, margins),
        "corr_share_vs_frac_near_tie": _corr(shares, ties),
        "mean_d_std_maj_monopole": _mean("mean_d_std_maj", mono),
        "mean_d_std_maj_non_monopole": _mean("mean_d_std_maj", non),
        "mean_margin_maj_monopole": _mean("mean_margin_maj", mono),
        "mean_margin_maj_non_monopole": _mean("mean_margin_maj", non),
        "mean_frac_near_tie_0_10_monopole": _mean("frac_maj_near_tie_margin_lt_0_10", mono),
        "mean_frac_near_tie_0_10_non_monopole": _mean(
            "frac_maj_near_tie_margin_lt_0_10", non
        ),
        "mean_frac_bias_flips_on_maj_monopole": _mean("frac_maj_bias_flips_argmax", mono),
        "mean_frac_bias_flips_on_maj_non_monopole": _mean(
            "frac_maj_bias_flips_argmax", non
        ),
        "per_structure": rows,
        "proto_config_ge30": {
            k: cfg[k]
            for k in cfg
            if k
            not in (
                "pairwise_detail",
                "pairwise_hyp_dists",
                "pairwise_tangent_cos",
                "pairwise_angles_deg",
            )
        },
    }


def audit_seed_proto_configs() -> dict:
    """Compare seed1 vs seed2 at early (epoch_001≈ge1) and ge30."""
    out: dict = {}
    for name, run in RUNS.items():
        early_ckpt = run / "epochs" / "epoch_001.pt"
        late_ckpt = run / "epochs" / "epoch_030.pt"
        early = load_model_from_checkpoint(early_ckpt, DEVICE)
        early.to(DEVICE)
        early.core_capacity_quota_tau = 0.0
        early.eval()
        late = load_model_from_checkpoint(late_ckpt, DEVICE)
        late.to(DEVICE)
        late.core_capacity_quota_tau = 0.0
        late.eval()
        cfg_early = _proto_config(early)
        cfg_late = _proto_config(late)
        late_report = measure_prototype_repulsion_epoch(late, proteins, DEVICE)
        purity = late_report.get("dehydron_partition_purity") or {}
        out[name] = {
            "early_epoch": 1,
            "early": {
                k: cfg_early[k]
                for k in (
                    "nearest_pair",
                    "nearest_pair_hyp_dist",
                    "twin_hyp_dist",
                    "hyp_dist_mean",
                    "hyp_dist_std",
                    "hyp_dist_cv",
                    "hyp_dist_min_over_max",
                    "angle_mean_deg",
                    "angle_std_deg",
                    "angle_min_deg",
                    "gram_eigenvalues",
                    "gram_eig_min",
                    "gram_condition",
                    "proto_radius_cv",
                    "expert_bias",
                    "bias_argmax",
                    "bias_range",
                    "pairwise_hyp_dists",
                    "pairwise_angles_deg",
                )
            },
            "ge30": {
                k: cfg_late[k]
                for k in (
                    "nearest_pair",
                    "nearest_pair_hyp_dist",
                    "twin_hyp_dist",
                    "hyp_dist_mean",
                    "hyp_dist_std",
                    "hyp_dist_cv",
                    "hyp_dist_min_over_max",
                    "angle_mean_deg",
                    "angle_std_deg",
                    "angle_min_deg",
                    "gram_eigenvalues",
                    "gram_eig_min",
                    "gram_condition",
                    "proto_radius_cv",
                    "expert_bias",
                    "bias_argmax",
                    "bias_range",
                    "pairwise_hyp_dists",
                    "pairwise_angles_deg",
                )
            },
            "relative_purity_ge30": {
                "verdict": purity.get("verdict"),
                "passes": purity.get("passes"),
                "n_blurred": len(purity.get("blurred_pdb_ids") or []),
                "floor_mode": purity.get("floor_mode"),
            },
            "frac_ge30": late_report.get("frac_max_p_ge_0_60"),
            "struct_ch_ge30": late_report.get("per_structure_committed_hard_max"),
        }
        del early, late
        torch.cuda.empty_cache()
    return out


def _verdicts(bias_audits: dict, configs: dict) -> dict:
    # Bias tipping: if maj often == bias_argmax AND bias flips many residues OR changes maj
    s2 = bias_audits["seed2"]
    s1 = bias_audits["seed1"]
    bias_tipping = False
    for a in (s1, s2):
        if (
            a["frac_structures_maj_equals_bias_argmax"] >= 0.75
            and (
                a["n_structures_maj_changed_by_bias"] >= 3
                or (a["mean_frac_bias_flips_on_maj_monopole"] or 0) > 0.05
            )
        ):
            bias_tipping = True
    # Soft: majority collapses onto bias_argmax even if flips rare (bias only tips ties)
    soft_bias = any(
        a["frac_structures_maj_equals_bias_argmax"] >= 0.75
        and a["bias_range"] > 1e-4
        for a in (s1, s2)
    )
    # Equidistance: monopoles have higher d_std or more near-ties?
    # Weak-signal fallback: MORE equidistant (higher near-tie frac, LOWER margin) on mono
    eq_signals = []
    for a in (s1, s2):
        m_tie = a["mean_frac_near_tie_0_10_monopole"]
        n_tie = a["mean_frac_near_tie_0_10_non_monopole"]
        m_mar = a["mean_margin_maj_monopole"]
        n_mar = a["mean_margin_maj_non_monopole"]
        if m_tie is not None and n_tie is not None and m_tie > n_tie + 0.05:
            eq_signals.append("more_near_ties_on_monopole")
        if m_mar is not None and n_mar is not None and m_mar + 0.02 < n_mar:
            eq_signals.append("smaller_margin_on_monopole")
        # Actually prior finding was farther from maj — check if MORE ambiguous (higher d_std)
        if (
            a["mean_d_std_maj_monopole"] is not None
            and a["mean_d_std_maj_non_monopole"] is not None
            and a["mean_d_std_maj_monopole"] < a["mean_d_std_maj_non_monopole"] - 0.02
        ):
            # lower std = more equidistant
            eq_signals.append("lower_d_std_on_monopole_equidistant")

    if bias_tipping:
        bias_v = "BIAS_TIPS_NEAR_TIES"
    elif soft_bias and not bias_tipping:
        # maj lands on bias_argmax but removing bias wouldn't change — coincidence or soft?
        # Check if maj_changed is 0
        if all(a["n_structures_maj_changed_by_bias"] == 0 for a in (s1, s2)):
            bias_v = "BIAS_ALIGN_BUT_NOT_CAUSAL"
        else:
            bias_v = "PARTIAL_BIAS_ALIGNMENT"
    else:
        bias_v = "BIAS_NOT_DRIVING_MAJORITY"

    if "lower_d_std_on_monopole_equidistant" in eq_signals or (
        "more_near_ties_on_monopole" in eq_signals
    ):
        eq_v = "MONOPOLE_MORE_EQUIDISTANT_WEAK_SIGNAL"
    elif not eq_signals:
        eq_v = "NO_CLEAR_EQUIDISTANCE_CONTRAST"
    else:
        eq_v = "MIXED_EQUIDISTANCE_SIGNALS"

    # Seed1 vs seed2 early config divergence
    e1, e2 = configs["seed1"]["early"], configs["seed2"]["early"]
    l1, l2 = configs["seed1"]["ge30"], configs["seed2"]["ge30"]
    early_diffs = {
        "nearest_d": abs(e1["nearest_pair_hyp_dist"] - e2["nearest_pair_hyp_dist"]),
        "hyp_cv": abs(e1["hyp_dist_cv"] - e2["hyp_dist_cv"]),
        "angle_std": abs(e1["angle_std_deg"] - e2["angle_std_deg"]),
        "gram_condition": abs(e1["gram_condition"] - e2["gram_condition"]),
        "angle_min": abs(e1["angle_min_deg"] - e2["angle_min_deg"]),
        "radius_cv": abs(e1["proto_radius_cv"] - e2["proto_radius_cv"]),
    }
    late_diffs = {
        "nearest_d": abs(l1["nearest_pair_hyp_dist"] - l2["nearest_pair_hyp_dist"]),
        "hyp_cv": abs(l1["hyp_dist_cv"] - l2["hyp_dist_cv"]),
        "angle_std": abs(l1["angle_std_deg"] - l2["angle_std_deg"]),
        "gram_condition": abs(l1["gram_condition"] - l2["gram_condition"]),
        "angle_min": abs(l1["angle_min_deg"] - l2["angle_min_deg"]),
        "radius_cv": abs(l1["proto_radius_cv"] - l2["proto_radius_cv"]),
    }
    # If early already differs a lot on angular/gram while nearest similar → init config
    early_nearest_close = early_diffs["nearest_d"] < 0.02
    early_config_apart = (
        early_diffs["gram_condition"] > 0.5
        or early_diffs["angle_std"] > 5.0
        or early_diffs["hyp_cv"] > 0.1
    )
    late_config_apart = (
        late_diffs["gram_condition"] > 1.0
        or late_diffs["angle_std"] > 10.0
        or late_diffs["nearest_d"] > 0.1
    )
    if early_nearest_close and early_config_apart:
        seed_v = "SEED1_DILUTION_TRACKS_EARLY_BANK_CONFIG"
    elif (not early_config_apart) and late_config_apart:
        seed_v = "SEED1_DIVERGES_DURING_TRAINING_NOT_INIT"
    elif early_nearest_close and (not early_config_apart) and (not late_config_apart):
        seed_v = "CONFIG_SIMILAR_DILUTION_ELSEWHERE"
    else:
        seed_v = "MIXED_INIT_AND_TRAJECTORY"

    return {
        "equidistance": eq_v,
        "equidistance_signals": eq_signals,
        "bias": bias_v,
        "seed1_dilution_config": seed_v,
        "early_diffs_seed1_vs_seed2": early_diffs,
        "late_diffs_seed1_vs_seed2": late_diffs,
        "note": (
            "Only two seeds — 'correlates' means comparative divergence on "
            "angular/gram metrics beyond nearest-pair, not a multi-seed regression."
        ),
    }


def main() -> None:
    bias_audits = {}
    for name, run in RUNS.items():
        bias_audits[name] = audit_equidistance_and_bias(
            run / "epochs" / "epoch_030.pt", name
        )
    configs = audit_seed_proto_configs()
    verdicts = _verdicts(bias_audits, configs)
    report = {
        "equidistance_and_bias": {
            "seed1": {k: v for k, v in bias_audits["seed1"].items() if k != "per_structure"},
            "seed2": {k: v for k, v in bias_audits["seed2"].items() if k != "per_structure"},
            "seed1_per_structure": bias_audits["seed1"]["per_structure"],
            "seed2_per_structure": bias_audits["seed2"]["per_structure"],
        },
        "seed_proto_configs": configs,
        "verdicts": verdicts,
    }
    out = Path("/tmp/bias_tie_and_seed_config_audit.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("REPORT_JSON_BEGIN")
    print(json.dumps(report))
    print("REPORT_JSON_END")
    summary = {
        "verdicts": verdicts,
        "seed1_bias": {
            k: bias_audits["seed1"][k]
            for k in (
                "expert_bias",
                "bias_argmax",
                "bias_range",
                "frac_structures_maj_equals_bias_argmax",
                "n_structures_maj_changed_by_bias",
                "majority_expert_counts",
                "corr_share_vs_d_std_maj",
                "corr_share_vs_frac_near_tie",
                "mean_d_std_maj_monopole",
                "mean_d_std_maj_non_monopole",
                "mean_margin_maj_monopole",
                "mean_margin_maj_non_monopole",
                "mean_frac_near_tie_0_10_monopole",
                "mean_frac_near_tie_0_10_non_monopole",
                "mean_frac_bias_flips_on_maj_monopole",
            )
        },
        "seed2_bias": {
            k: bias_audits["seed2"][k]
            for k in (
                "expert_bias",
                "bias_argmax",
                "bias_range",
                "frac_structures_maj_equals_bias_argmax",
                "n_structures_maj_changed_by_bias",
                "majority_expert_counts",
                "corr_share_vs_d_std_maj",
                "corr_share_vs_frac_near_tie",
                "mean_d_std_maj_monopole",
                "mean_d_std_maj_non_monopole",
                "mean_margin_maj_monopole",
                "mean_margin_maj_non_monopole",
                "mean_frac_near_tie_0_10_monopole",
                "mean_frac_near_tie_0_10_non_monopole",
                "mean_frac_bias_flips_on_maj_monopole",
            )
        },
        "early_seed1": configs["seed1"]["early"],
        "early_seed2": configs["seed2"]["early"],
        "ge30_seed1": configs["seed1"]["ge30"],
        "ge30_seed2": configs["seed2"]["ge30"],
    }
    print(json.dumps(summary, indent=2))
    print("WROTE", out)


if __name__ == "__main__":
    main()
