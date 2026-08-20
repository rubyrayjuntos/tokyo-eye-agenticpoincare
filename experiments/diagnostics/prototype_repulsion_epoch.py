"""Per-epoch prototype nearest-pair repulsion standing metrics (PROTO_SEP pre-reg)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from geoopt.manifolds.stereographic import math as pmath

from experiments.training.v6.train_loop import prepare_training_batch
from science.dtie.common.residue_features import residue_sasa_from_data

# SASA baseline underwrap twin (historical continuity).
HISTORICAL_TWIN = (1, 3)


def measure_prototype_pairwise(
    model: torch.nn.Module,
    *,
    twin: tuple[int, int] = HISTORICAL_TWIN,
) -> dict[str, Any]:
    """All six pairwise hyp dists + tangent cos; identify nearest pair."""
    gate = model.gate
    k = -model.curvature
    e = int(gate.num_experts)
    with torch.no_grad():
        proto_hyp = gate.prototype_bank(k)
        proto_tan = gate.prototype_bank.prototype_tangent.detach()
        tn = F.normalize(proto_tan.float(), dim=-1)
        cos = (tn @ tn.T).cpu()
        pairs: list[dict[str, Any]] = []
        for i in range(e):
            for j in range(i + 1, e):
                d = float(pmath.dist(proto_hyp[i : i + 1], proto_hyp[j : j + 1], k=k).squeeze())
                pairs.append(
                    {
                        "pair": [i, j],
                        "hyp_dist": d,
                        "tangent_cos": float(cos[i, j]),
                        "is_historical_twin": sorted([i, j]) == sorted(list(twin)),
                    }
                )
    pairs_sorted = sorted(pairs, key=lambda p: p["hyp_dist"])
    twin_row = next(p for p in pairs if p["is_historical_twin"])
    other = [p for p in pairs if not p["is_historical_twin"]]
    nearest = pairs_sorted[0]
    e0e2 = next((p for p in pairs if p["pair"] == [0, 2]), None)
    # Unit-tangent Gram conditioning (full-bank; not nearest-pair alone).
    gram = cos.numpy() if hasattr(cos, "numpy") else np.asarray(cos)
    eigs = np.linalg.eigvalsh(gram)
    # Numerical floor for logdet monitor (matches training ε).
    eps = 1e-4
    logdet = float(np.linalg.slogdet(gram + eps * np.eye(e))[1])
    tau_logdet = -1.15
    hinge_raw = float(max(0.0, tau_logdet - logdet))
    return {
        "twin_pair": list(twin),
        "pairwise": pairs_sorted,
        "historical_twin_hyp_dist": float(twin_row["hyp_dist"]),
        "historical_twin_tangent_cos": float(twin_row["tangent_cos"]),
        "nearest_pair": nearest["pair"],
        "nearest_pair_hyp_dist": float(nearest["hyp_dist"]),
        "e0_e2_hyp_dist": None if e0e2 is None else float(e0e2["hyp_dist"]),
        "other_hyp_dist_mean": float(np.mean([p["hyp_dist"] for p in other])),
        "other_hyp_dist_min": float(np.min([p["hyp_dist"] for p in other])),
        "twin_over_other_mean": float(
            twin_row["hyp_dist"] / (float(np.mean([p["hyp_dist"] for p in other])) + 1e-12)
        ),
        "collateral_other_below_0_05": bool(
            any(p["hyp_dist"] < 0.05 for p in other)
        ),
        "gram_eig_min": float(eigs.min()),
        "gram_eig_max": float(eigs.max()),
        "gram_condition": float(eigs.max() / (abs(eigs.min()) + 1e-12)),
        "gram_logdet": logdet,
        "gram_logdet_hinge": hinge_raw**2,
        "gram_logdet_tau": tau_logdet,
    }


def measure_degree_rival_sensitivity(
    model: torch.nn.Module,
    proteins: list[dict[str, Any]],
    device: str,
    *,
    twin: tuple[int, int] = HISTORICAL_TWIN,
    max_twin_residues: int = 400,
    seed: int = 0,
) -> dict[str, Any]:
    """Re-measure degree→rival-median swap (same protocol as attractor diagnostic)."""
    gate = model.gate
    k = -model.curvature
    a, b = twin
    records: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for prot in proteins:
            data = prepare_training_batch(model, prot, device)
            out = model(data)
            w = out["expert_weights"].float()
            hard = w.argmax(-1)
            pre = gate._last_pre_softmax
            dists = pre["dists"].float()
            raw = pre["raw_logits"].float()
            sasa = residue_sasa_from_data(data).float().view(-1)
            clustering = data.clustering.float().view(-1)
            degree = data.degree.float().view(-1)
            rho = data.rho.float().view(-1)
            depth = out.get("cone_depth")
            if depth is None:
                depth = pmath.dist0(out["x_hyp"], k=k, keepdim=True)
            depth = depth.float().view(-1)
            tau = (
                data.x[:, 1].float()
                if data.x.size(1) > 1
                else torch.zeros(w.size(0), device=device)
            )
            ss = data.ss_onehot.float()
            x_hyp = out["x_hyp"].float()
            disc = out.get("disc_coords")
            if disc is not None:
                disc = disc.float()
                disc_xy = disc[:, :2]
                disc_r = disc.norm(dim=-1, keepdim=True)
            else:
                disc_xy = disc_r = None
            for i in range(w.size(0)):
                h = int(hard[i])
                if h not in twin:
                    continue
                records.append(
                    {
                        "hard": h,
                        "w": w[i].cpu().numpy(),
                        "dists": dists[i].cpu().numpy(),
                        "raw": raw[i].cpu().numpy(),
                        "clustering": float(clustering[i]),
                        "degree": float(degree[i]),
                        "sasa": float(sasa[i]),
                        "rho": float(rho[i]),
                        "depth": float(depth[i]),
                        "tau": float(tau[i]),
                        "ss": ss[i].cpu().float(),
                        "x_hyp": x_hyp[i].cpu().float(),
                        "disc_xy": None if disc_xy is None else disc_xy[i].cpu().float(),
                        "disc_r": None if disc_r is None else disc_r[i].cpu().float(),
                    }
                )

    if not records:
        return {
            "n_twin_residues": 0,
            "mean_soft_on_rival_twin": float("nan"),
            "degree_swap_mean_abs_delta_logit_gap": float("nan"),
            "degree_swap_mean_abs_delta_soft_self": float("nan"),
        }

    med = {
        e: {
            "degree": float(np.median([r["degree"] for r in records if r["hard"] == e])),
        }
        for e in twin
    }
    rival_soft = []
    for r in records:
        rival = b if r["hard"] == a else a
        rival_soft.append(float(r["w"][rival]))

    rng = np.random.default_rng(seed)
    idx = np.arange(len(records))
    if len(idx) > max_twin_residues:
        idx = rng.choice(idx, size=max_twin_residues, replace=False)
    sample = [records[i] for i in idx]

    def run_gate_batch(recs: list[dict[str, Any]], deg: np.ndarray | None = None):
        x_hyp = torch.stack([r["x_hyp"] for r in recs]).to(device=device, dtype=torch.float32)
        clustering = torch.tensor(
            [r["clustering"] for r in recs], device=device, dtype=torch.float32
        )
        degree = torch.tensor(
            [r["degree"] if deg is None else float(deg[i]) for i, r in enumerate(recs)],
            device=device,
            dtype=torch.float32,
        )
        sasa = torch.tensor([r["sasa"] for r in recs], device=device, dtype=torch.float32)
        rho = torch.tensor([r["rho"] for r in recs], device=device, dtype=torch.float32)
        depth = torch.tensor(
            [r["depth"] for r in recs], device=device, dtype=torch.float32
        ).unsqueeze(-1)
        tau = torch.tensor([r["tau"] for r in recs], device=device, dtype=torch.float32)
        ss = torch.stack([r["ss"] for r in recs]).to(device=device, dtype=torch.float32)
        kw: dict[str, Any] = {}
        if recs[0]["disc_xy"] is not None:
            kw["disc_xy"] = torch.stack([r["disc_xy"] for r in recs]).to(
                device=device, dtype=torch.float32
            )
            kw["disc_r"] = (
                torch.stack([r["disc_r"] for r in recs])
                .to(device=device, dtype=torch.float32)
                .view(-1, 1)
            )
        scores, _, _ = gate(
            x_hyp=x_hyp,
            k=k,
            clustering=clustering,
            cone_depth=depth,
            degree=degree,
            rho=rho,
            ss_onehot=ss,
            tau_flag=tau,
            sasa=sasa,
            **kw,
        )
        pre = gate._last_pre_softmax
        return {
            "scores": scores.detach().float().cpu().numpy(),
            "raw": pre["raw_logits"].detach().float().cpu().numpy(),
        }

    with torch.no_grad():
        base = run_gate_batch(sample)
        deg_swap = np.array(
            [med[b if r["hard"] == a else a]["degree"] for r in sample], dtype=float
        )
        pert = run_gate_batch(sample, deg=deg_swap)

    hard = np.array([r["hard"] for r in sample])
    logit_b, logit_p, soft_b, soft_p = [], [], [], []
    for i, h in enumerate(hard):
        o = b if h == a else a
        logit_b.append(base["raw"][i, h] - base["raw"][i, o])
        logit_p.append(pert["raw"][i, h] - pert["raw"][i, o])
        soft_b.append(base["scores"][i, h])
        soft_p.append(pert["scores"][i, h])
    logit_b = np.asarray(logit_b)
    logit_p = np.asarray(logit_p)
    soft_b = np.asarray(soft_b)
    soft_p = np.asarray(soft_p)

    return {
        "n_twin_residues": len(records),
        "n_sensitivity_sample": len(sample),
        "mean_soft_on_rival_twin": float(np.mean(rival_soft)),
        "degree_swap_mean_abs_delta_logit_gap": float(np.mean(np.abs(logit_p - logit_b))),
        "degree_swap_mean_abs_delta_soft_self": float(np.mean(np.abs(soft_p - soft_b))),
        "twin_feature_medians_degree": med,
    }


# SASA / stack protein-spread discipline (frozen for seed-2 committed subset).
COMMITTED_PROTEIN_MIN_COUNT = 8
COMMITTED_PROTEIN_SHARE_MAX = 0.35
# Historical monopole band lower edge — committed hard must stay below this.
MONOPOLE_HARD_SHARE_FLOOR = 0.56

# Majority-conditional load hinge + dehydron-partition purity.
# Loss conditions on committed majority *share*, not on dehydron=0; purity is monitor-only.
MAJORITY_SHARE_HINGE_TAU = MONOPOLE_HARD_SHARE_FLOOR  # 0.56
MAJORITY_DEHYDRON_RATE_MAX = 0.05  # ≤5% of the other class on the majority side
MINORITY_DEHYDRON_RATE_MIN = 0.90  # ≥90% of its class on the minority side
PURITY_MIN_COMMITTED = 20
PURITY_MIN_MINORITY = 5
# Standing purity (2026-07-15): relative to per-structure corpus majority.
PURITY_FLOOR_MODE = "relative"  # "relative" | "fixed_core_majority" (legacy)


def _relative_purity_ok(
    maj_dh: float,
    min_dh: float,
    corpus_dehydron_frac: float,
    *,
    maj_rate_max: float = MAJORITY_DEHYDRON_RATE_MAX,
    min_rate_min: float = MINORITY_DEHYDRON_RATE_MIN,
) -> tuple[bool, str]:
    """Committed majority matches corpus majority direction; minority the other."""
    if corpus_dehydron_frac > 0.5:
        ok = maj_dh >= min_rate_min and min_dh <= maj_rate_max
        return ok, "dehydron"
    if corpus_dehydron_frac < 0.5:
        ok = maj_dh <= maj_rate_max and min_dh >= min_rate_min
        return ok, "core"
    core_ok = maj_dh <= maj_rate_max and min_dh >= min_rate_min
    dh_ok = maj_dh >= min_rate_min and min_dh <= maj_rate_max
    return (core_ok or dh_ok), "tie"


def _effective_experts_from_shares(shares: np.ndarray) -> float:
    """exp(H) effective-expert count from a share vector (sums to ~1)."""
    p = np.asarray(shares, dtype=float)
    if p.size == 0 or not np.isfinite(p).all() or float(p.sum()) <= 0:
        return float("nan")
    p = np.clip(p, 1e-8, None)
    p = p / p.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


def score_committed_distribution(
    report: dict[str, Any],
    *,
    n_proteins_expected: int = 12,
    min_proteins_with_committed: int = COMMITTED_PROTEIN_MIN_COUNT,
    max_protein_share: float = COMMITTED_PROTEIN_SHARE_MAX,
    monopole_floor: float = MONOPOLE_HARD_SHARE_FLOOR,
) -> dict[str, Any]:
    """Frozen seed-2 distribution floors on the committed subset (max-p≥0.60).

    Soft corpus balance is *not* sufficient. Pass requires:
    - protein spread of the committed population (≥8/12, no protein >35% of committed)
    - corpus ``committed_hard_share_max`` < monopole lower band (0.56)
    - ``per_structure_committed_hard_max`` < 0.56 (no single protein's committed
      set is itself a local monopole)
    """
    n_with = int(report.get("n_proteins_with_committed") or 0)
    n_total = int(report.get("n_proteins_total") or n_proteins_expected)
    prot_share = float(report.get("committed_protein_share_max", float("nan")))
    commit_hard_max = float(report.get("committed_hard_share_max", float("nan")))
    per_struct_commit_hard = float(
        report.get("per_structure_committed_hard_max", float("nan"))
    )
    n_committed = int(report.get("n_committed") or 0)

    protein_spread_pass = bool(
        n_committed > 0
        and n_with >= min_proteins_with_committed
        and prot_share == prot_share
        and prot_share <= max_protein_share
    )
    corpus_expert_pass = bool(
        n_committed > 0
        and commit_hard_max == commit_hard_max
        and commit_hard_max < monopole_floor
    )
    per_struct_expert_pass = bool(
        n_committed > 0
        and per_struct_commit_hard == per_struct_commit_hard
        and per_struct_commit_hard < monopole_floor
    )
    passes = protein_spread_pass and corpus_expert_pass and per_struct_expert_pass
    fails: list[str] = []
    if n_committed <= 0:
        fails.append("COMMIT_EMPTY")
    else:
        if not protein_spread_pass:
            fails.append("COMMITTED_PROTEIN_CONCENTRATED")
        if not corpus_expert_pass:
            fails.append("COMMITTED_HARD_MONOPOLE")
        if not per_struct_expert_pass:
            fails.append("PER_STRUCTURE_COMMITTED_HARD_MONOPOLE")

    if not n_committed:
        verdict = "COMMIT_EMPTY"
    elif passes:
        verdict = "COMMITTED_DISTRIBUTION_PASS"
    elif "COMMITTED_HARD_MONOPOLE" in fails or "PER_STRUCTURE_COMMITTED_HARD_MONOPOLE" in fails:
        verdict = "COMMITTED_HARD_CONCENTRATED"
    elif "COMMITTED_PROTEIN_CONCENTRATED" in fails:
        verdict = "COMMITTED_PROTEIN_CONCENTRATED"
    else:
        verdict = "AMBIGUOUS"

    return {
        "verdict": verdict,
        "passes": passes,
        "fails": fails,
        "protein_spread_pass": protein_spread_pass,
        "corpus_expert_pass": corpus_expert_pass,
        "per_struct_expert_pass": per_struct_expert_pass,
        "floors": {
            "min_proteins_with_committed": min_proteins_with_committed,
            "n_proteins_expected": n_proteins_expected,
            "max_committed_protein_share": max_protein_share,
            "monopole_hard_share_floor": monopole_floor,
            "clear_if_committed_hard_share_max_lt": monopole_floor,
            "clear_if_per_structure_committed_hard_max_lt": monopole_floor,
        },
        "n_proteins_with_committed": n_with,
        "n_proteins_total": n_total,
        "committed_protein_share_max": prot_share,
        "committed_hard_share_max": commit_hard_max,
        "per_structure_committed_hard_max": per_struct_commit_hard,
    }


def committed_majority_partition(
    weights: np.ndarray,
    dehydron: np.ndarray | None = None,
    *,
    commit_thr: float = 0.60,
    n_experts: int = 4,
) -> dict[str, Any]:
    """Mechanism-agnostic committed majority/minority split (hard assignment).

    Majority = expert with the largest share of residues with max-p≥commit_thr.
    Does **not** condition on dehydron; dehydron rates are optional monitors.
    """
    w = np.asarray(weights, dtype=float)
    if w.ndim != 2 or w.shape[0] == 0:
        return {
            "n_committed": 0,
            "majority_expert": None,
            "majority_share": float("nan"),
            "n_majority": 0,
            "n_minority": 0,
            "majority_dehydron_rate": float("nan"),
            "minority_dehydron_rate": float("nan"),
        }
    hard = w.argmax(axis=1)
    commit = w.max(axis=1) >= commit_thr
    n_c = int(commit.sum())
    if n_c == 0:
        return {
            "n_committed": 0,
            "majority_expert": None,
            "majority_share": float("nan"),
            "n_majority": 0,
            "n_minority": 0,
            "majority_dehydron_rate": float("nan"),
            "minority_dehydron_rate": float("nan"),
        }
    counts = np.bincount(hard[commit], minlength=n_experts).astype(float)
    maj_e = int(counts.argmax())
    maj_share = float(counts[maj_e] / n_c)
    maj_mask = commit & (hard == maj_e)
    min_mask = commit & (hard != maj_e)
    maj_dh = float("nan")
    min_dh = float("nan")
    if dehydron is not None:
        dh = np.asarray(dehydron, dtype=float).reshape(-1)[: w.shape[0]]
        if maj_mask.any():
            maj_dh = float(dh[maj_mask].mean())
        if min_mask.any():
            min_dh = float(dh[min_mask].mean())
    return {
        "n_committed": n_c,
        "majority_expert": maj_e,
        "majority_share": maj_share,
        "n_majority": int(maj_mask.sum()),
        "n_minority": int(min_mask.sum()),
        "majority_dehydron_rate": maj_dh,
        "minority_dehydron_rate": min_dh,
    }


def score_dehydron_partition_purity(
    per_structure: list[dict[str, Any]],
    *,
    maj_rate_max: float = MAJORITY_DEHYDRON_RATE_MAX,
    min_rate_min: float = MINORITY_DEHYDRON_RATE_MIN,
    min_committed: int = PURITY_MIN_COMMITTED,
    min_minority: int = PURITY_MIN_MINORITY,
    floor_mode: str = PURITY_FLOOR_MODE,
) -> dict[str, Any]:
    """Monitor: dehydron purity of committed maj/min partitions.

    Standing mode ``relative`` (2026-07-15): per structure, committed majority
    must match the structure's corpus majority class (core vs dehydron), and
    minority the other — not a fixed ``maj_dh ≤ 0.05``.

    Legacy ``fixed_core_majority`` keeps the old absolute floors for audit replay.
    """
    eligible: list[dict[str, Any]] = []
    fails: list[str] = []
    blurred: list[str] = []
    mode = str(floor_mode or "relative")
    for row in per_structure:
        n_c = int(row.get("n_committed") or 0)
        n_min = int(row.get("n_minority") or 0)
        maj_r = row.get("majority_dehydron_rate")
        min_r = row.get("minority_dehydron_rate")
        if n_c < min_committed or n_min < min_minority:
            continue
        if maj_r is None or min_r is None:
            continue
        maj_r_f = float(maj_r)
        min_r_f = float(min_r)
        if maj_r_f != maj_r_f or min_r_f != min_r_f:
            continue
        pdb = str(row.get("pdb_id") or "?")
        corpus_frac = row.get("corpus_dehydron_frac")
        corpus_frac_f = (
            float(corpus_frac)
            if corpus_frac is not None and float(corpus_frac) == float(corpus_frac)
            else float("nan")
        )
        entry: dict[str, Any] = {
            "pdb_id": pdb,
            "n_committed": n_c,
            "n_minority": n_min,
            "majority_dehydron_rate": maj_r_f,
            "minority_dehydron_rate": min_r_f,
            "corpus_dehydron_frac": (
                corpus_frac_f if corpus_frac_f == corpus_frac_f else None
            ),
        }
        if mode == "relative":
            if corpus_frac_f != corpus_frac_f:
                continue
            ok, expected = _relative_purity_ok(
                maj_r_f,
                min_r_f,
                corpus_frac_f,
                maj_rate_max=maj_rate_max,
                min_rate_min=min_rate_min,
            )
            entry["expected_majority_class"] = expected
            entry["relative_pass"] = bool(ok)
            eligible.append(entry)
            if not ok:
                blurred.append(pdb)
        else:
            entry["expected_majority_class"] = "core"
            eligible.append(entry)
            if maj_r_f > maj_rate_max or min_r_f < min_rate_min:
                blurred.append(pdb)
    if blurred:
        fails.append("DEHYDRON_PARTITION_BLURRED")
    if not eligible:
        verdict = "PURITY_NOT_APPLICABLE"
        passes = True
    elif not fails:
        verdict = "DEHYDRON_PARTITION_PURE"
        passes = True
    else:
        verdict = "DEHYDRON_PARTITION_BLURRED"
        passes = False
    return {
        "verdict": verdict,
        "passes": passes,
        "fails": fails,
        "n_eligible_structures": len(eligible),
        "blurred_pdb_ids": blurred,
        "floor_mode": mode,
        "floors": {
            "mode": mode,
            "majority_dehydron_rate_max": maj_rate_max,
            "minority_dehydron_rate_min": min_rate_min,
            "min_committed": min_committed,
            "min_minority": min_minority,
            "note": (
                "relative: committed maj matches corpus majority class; "
                "fixed_core_majority: legacy maj_dh≤max / min_dh≥min"
            ),
        },
        "per_structure_eligible": eligible,
    }


def score_majority_conditional_ladder(
    report: dict[str, Any],
    *,
    monopole_floor: float = MAJORITY_SHARE_HINGE_TAU,
) -> dict[str, Any]:
    """Outcome buckets for the majority-conditional hinge pre-reg."""
    dist = report.get("committed_distribution") or {}
    purity = report.get("dehydron_partition_purity") or {}
    frac_mp = float(report.get("frac_max_p_ge_0_60", float("nan")))
    per_struct = float(report.get("per_structure_committed_hard_max", float("nan")))
    tau_contrast = float(report.get("best_tau_mean_contrast", float("nan")))

    dist_pass = bool(dist.get("passes"))
    purity_pass = bool(purity.get("passes", True))
    commit_held = frac_mp == frac_mp and frac_mp >= 0.15
    axis_held = tau_contrast == tau_contrast and tau_contrast >= 0.40
    local_cleared = (
        per_struct == per_struct and per_struct < monopole_floor and dist_pass
    )

    fails: list[str] = []
    if not purity_pass:
        fails.append("DEHYDRON_PARTITION_BLURRED")
    if not commit_held:
        fails.append("COMMIT_REGRESSED")
    if not axis_held:
        fails.append("AXIS_COLLAPSE")
    if commit_held and not local_cleared and purity_pass:
        fails.append("LOCAL_MONOPOLE_PERSISTS")

    if "DEHYDRON_PARTITION_BLURRED" in fails:
        verdict = "AXIS_SCRAMBLED_BY_DIVERSITY"
    elif "COMMIT_REGRESSED" in fails:
        verdict = "COMMIT_KILLED_BY_DIVERSITY"
    elif local_cleared and purity_pass and commit_held and axis_held:
        verdict = "MAJORITY_SPLIT_WIN"
    elif "LOCAL_MONOPOLE_PERSISTS" in fails:
        # Ambiguous: wrong λ vs wrong lever — see SSOT coeff policy.
        verdict = "MAJORITY_SPLIT_COEFF_INCONCLUSIVE"
    else:
        verdict = "AMBIGUOUS"

    return {
        "verdict": verdict,
        "passes": verdict == "MAJORITY_SPLIT_WIN",
        "fails": fails,
        "local_monopole_cleared": local_cleared,
        "dehydron_purity_held": purity_pass,
        "commit_l2_held": commit_held,
        "axis_held": axis_held,
        "floors": {
            "per_structure_committed_hard_max_lt": monopole_floor,
            "frac_max_p_ge_0_60_min": 0.15,
            "best_tau_mean_contrast_min": 0.40,
            "majority_dehydron_rate_max": MAJORITY_DEHYDRON_RATE_MAX,
            "minority_dehydron_rate_min": MINORITY_DEHYDRON_RATE_MIN,
            "majority_share_hinge_tau": MAJORITY_SHARE_HINGE_TAU,
            "eligible_min_committed": PURITY_MIN_COMMITTED,
            "eligible_min_minority": PURITY_MIN_MINORITY,
        },
    }


def score_core_majority_conditional_ladder(
    report: dict[str, Any],
    *,
    monopole_floor: float = MAJORITY_SHARE_HINGE_TAU,
) -> dict[str, Any]:
    """Outcome buckets for the core-only (dehydron=0) majority hinge pre-reg.

    Same floors as the agnostic majority ladder; verdict names use CORE_* prefix.
    Eligible purity structures: n_committed≥20 and n_minority≥5 (inlined in floors).
    """
    base = score_majority_conditional_ladder(report, monopole_floor=monopole_floor)
    rename = {
        "MAJORITY_SPLIT_WIN": "CORE_MAJORITY_SPLIT_WIN",
        "MAJORITY_SPLIT_COEFF_INCONCLUSIVE": "CORE_MAJORITY_SPLIT_COEFF_INCONCLUSIVE",
        "AXIS_SCRAMBLED_BY_DIVERSITY": "AXIS_SCRAMBLED_BY_DIVERSITY",
        "COMMIT_KILLED_BY_DIVERSITY": "COMMIT_KILLED_BY_DIVERSITY",
        "AMBIGUOUS": "AMBIGUOUS",
    }
    verdict = rename.get(str(base["verdict"]), str(base["verdict"]))
    return {
        **base,
        "verdict": verdict,
        "passes": verdict == "CORE_MAJORITY_SPLIT_WIN",
        "guarantee_note": (
            "Direct pressure on dh=1 residues excluded; purity floors still "
            "monitor indirect dilution onto minority-held experts."
        ),
    }


def score_core_quota_ladder(
    report: dict[str, Any],
    *,
    monopole_floor: float = MAJORITY_SHARE_HINGE_TAU,
    unplaced_max: float = 0.05,
) -> dict[str, Any]:
    """Outcome buckets for core capacity quotas (pre-reg CORE_QUOTA_*)."""
    base = score_majority_conditional_ladder(report, monopole_floor=monopole_floor)
    quota = report.get("core_quota") or {}
    unplaced_frac = float(quota.get("mean_unplaced_frac") or 0.0)
    forced = int(quota.get("forced_cross_axis_total") or 0)
    commit_ok = bool(base.get("commit_l2_held"))
    axis_ok = bool(base.get("axis_held"))
    local_ok = bool(base.get("local_monopole_cleared"))
    purity_ok = bool(base.get("dehydron_purity_held"))
    quota_ok = unplaced_frac <= float(unplaced_max) and forced == 0

    if not commit_ok:
        verdict = "COMMIT_KILLED_BY_QUOTA"
    elif not purity_ok:
        verdict = "AXIS_SCRAMBLED_BY_QUOTA"
    elif not quota_ok:
        verdict = "QUOTA_STARVE"
    elif not (local_ok and axis_ok):
        verdict = "CORE_QUOTA_NO_MOVE"
    elif commit_ok and axis_ok and local_ok and purity_ok and quota_ok:
        verdict = "CORE_QUOTA_WIN"
    else:
        verdict = "AMBIGUOUS"

    return {
        **base,
        "verdict": verdict,
        "passes": verdict == "CORE_QUOTA_WIN",
        "quota_integrity_held": quota_ok,
        "mean_unplaced_frac": unplaced_frac,
        "forced_cross_axis_total": forced,
        "floors": {
            **(base.get("floors") or {}),
            "quota_unplaced_frac_max": float(unplaced_max),
            "quota_forced_cross_axis_max": 0,
        },
    }


def routing_balance_from_weight_rows(
    weight_rows: list[np.ndarray],
    *,
    pdb_ids: list[str] | None = None,
    commit_thr: float = 0.60,
    n_experts: int = 4,
) -> dict[str, Any]:
    """Corpus soft/hard/committed shares + per-structure max companions.

    Soft load stays protein-mean-of-means (matches standing jsonl). Hard and
    committed shares are residue-weighted — the concentration metrics soft
    averages can hide. Per-structure soft max/min match ``inference_routing``
    style (max/min expert soft fraction within each protein, then max/min
    across proteins).
    """
    empty = {
        "soft_load": [0.0] * n_experts,
        "soft_load_max": 0.0,
        "effective_experts_soft": float("nan"),
        "hard_share": [0.0] * n_experts,
        "hard_share_max": 0.0,
        "effective_experts_hard": float("nan"),
        "committed_hard_share": [0.0] * n_experts,
        "committed_hard_share_max": float("nan"),
        "effective_experts_committed": float("nan"),
        "n_committed": 0,
        "n_proteins_total": 0,
        "n_proteins_with_committed": 0,
        "committed_protein_share_max": float("nan"),
        "committed_protein_share_max_id": None,
        "per_structure_soft_max": float("nan"),
        "per_structure_soft_min": float("nan"),
        "per_structure_hard_max": float("nan"),
        "per_structure_committed_hard_max": float("nan"),
        "per_structure": [],
    }
    if not weight_rows:
        return empty

    loads: list[np.ndarray] = []
    hard_counts = np.zeros(n_experts, dtype=float)
    commit_counts = np.zeros(n_experts, dtype=float)
    n_total = 0
    n_committed = 0
    per_struct_soft_maxes: list[float] = []
    per_struct_soft_mins: list[float] = []
    per_struct_hard_maxes: list[float] = []
    per_struct_commit_maxes: list[float] = []
    per_structure: list[dict[str, Any]] = []
    committed_by_protein: list[tuple[str, int]] = []

    for i, w in enumerate(weight_rows):
        w = np.asarray(w, dtype=float)
        if w.ndim != 2 or w.shape[1] < n_experts:
            raise ValueError(f"expected (N,{n_experts}+) weights, got {w.shape}")
        w = w[:, :n_experts]
        n = int(w.shape[0])
        soft = w.mean(axis=0)
        loads.append(soft)
        hard = w.argmax(axis=1)
        max_p = w.max(axis=1)
        commit_mask = max_p >= commit_thr
        hard_c = np.bincount(hard, minlength=n_experts).astype(float)
        commit_c = np.bincount(hard[commit_mask], minlength=n_experts).astype(float)
        hard_counts += hard_c
        commit_counts += commit_c
        n_total += n
        n_c = int(commit_mask.sum())
        n_committed += n_c

        hard_share_p = hard_c / max(n, 1)
        commit_share_p = commit_c / max(n_c, 1) if n_c else np.zeros(n_experts)
        soft_max_p = float(soft.max())
        soft_min_p = float(soft.min())
        hard_max_p = float(hard_share_p.max())
        commit_max_p = float(commit_share_p.max()) if n_c else float("nan")
        per_struct_soft_maxes.append(soft_max_p)
        per_struct_soft_mins.append(soft_min_p)
        per_struct_hard_maxes.append(hard_max_p)
        if n_c:
            per_struct_commit_maxes.append(commit_max_p)

        pdb = "?"
        if pdb_ids is not None and i < len(pdb_ids):
            pdb = str(pdb_ids[i])
        if n_c:
            committed_by_protein.append((pdb, n_c))
        per_structure.append(
            {
                "pdb_id": pdb,
                "n_residues": n,
                "n_committed": n_c,
                "soft_load": [float(x) for x in soft.tolist()],
                "soft_load_max": soft_max_p,
                "soft_load_min": soft_min_p,
                "hard_share": [float(x) for x in hard_share_p.tolist()],
                "hard_share_max": hard_max_p,
                "committed_hard_share": [float(x) for x in commit_share_p.tolist()],
                "committed_hard_share_max": None if n_c == 0 else commit_max_p,
            }
        )

    mean_load = np.mean(np.stack(loads), axis=0)
    hard_share = hard_counts / max(n_total, 1)
    committed_hard_share = (
        commit_counts / max(n_committed, 1)
        if n_committed
        else np.zeros(n_experts, dtype=float)
    )
    if committed_by_protein and n_committed:
        shares = {pdb: c / n_committed for pdb, c in committed_by_protein}
        max_id = max(shares, key=shares.get)
        committed_protein_share_max = float(shares[max_id])
        committed_protein_share_max_id = max_id
    else:
        committed_protein_share_max = float("nan")
        committed_protein_share_max_id = None

    return {
        "soft_load": [float(x) for x in mean_load.tolist()],
        "soft_load_max": float(np.max(mean_load)),
        "effective_experts_soft": _effective_experts_from_shares(mean_load),
        "hard_share": [float(x) for x in hard_share.tolist()],
        "hard_share_max": float(np.max(hard_share)),
        "effective_experts_hard": _effective_experts_from_shares(hard_share),
        "committed_hard_share": [float(x) for x in committed_hard_share.tolist()],
        "committed_hard_share_max": (
            float(np.max(committed_hard_share)) if n_committed else float("nan")
        ),
        "effective_experts_committed": (
            _effective_experts_from_shares(committed_hard_share)
            if n_committed
            else float("nan")
        ),
        "n_committed": int(n_committed),
        "n_proteins_total": int(len(weight_rows)),
        "n_proteins_with_committed": int(len(committed_by_protein)),
        "committed_protein_share_max": committed_protein_share_max,
        "committed_protein_share_max_id": committed_protein_share_max_id,
        "per_structure_soft_max": float(max(per_struct_soft_maxes)),
        "per_structure_soft_min": float(min(per_struct_soft_mins)),
        "per_structure_hard_max": float(max(per_struct_hard_maxes)),
        "per_structure_committed_hard_max": (
            float(max(per_struct_commit_maxes)) if per_struct_commit_maxes else float("nan")
        ),
        "per_structure": per_structure,
    }


def measure_routing_companions(
    model: torch.nn.Module,
    proteins: list[dict[str, Any]],
    device: str,
) -> dict[str, Any]:
    """Soft-load / hard-share / max-p / residue→proto dist_range companions."""
    import torch.nn.functional as F

    max_ps: list[float] = []
    Hs: list[float] = []
    weight_rows: list[np.ndarray] = []
    pdb_ids: list[str] = []
    dehydron_rows: list[np.ndarray | None] = []
    dist_ranges: list[float] = []
    dist_cvs: list[float] = []
    tau_by_expert: dict[int, list[float]] = {i: [] for i in range(4)}
    quota_rows: list[dict[str, Any]] = []
    model.eval()
    gate = model.gate
    with torch.no_grad():
        for prot in proteins:
            data = prepare_training_batch(model, prot, device)
            out = model(data)
            w = out["expert_weights"].float()
            max_p = w.max(dim=-1).values
            H = -(w.clamp_min(1e-8) * w.clamp_min(1e-8).log()).sum(dim=-1)
            max_ps.extend(max_p.cpu().tolist())
            Hs.extend(H.cpu().tolist())
            w_np = w.cpu().numpy()
            weight_rows.append(w_np)
            pdb_ids.append(str(prot.get("pdb_id") or prot.get("id") or "?").upper())
            cq = out.get("core_quota")
            if cq:
                quota_rows.append(dict(cq))
            dh = None
            if prot.get("target_dehydron") is not None:
                dh = np.asarray(prot["target_dehydron"], dtype=float).reshape(-1)[
                    : w_np.shape[0]
                ]
            elif data.x.size(1) > 1:
                # topology_three_vector: channel 1 tracks dehydron/τ in this lineage
                dh = data.x[:, 1].float().cpu().numpy()
            dehydron_rows.append(dh)
            pre = getattr(gate, "_last_pre_softmax", None) or {}
            dists = pre.get("dists")
            if dists is not None:
                d = dists.float()
                d_range = (d.max(dim=-1).values - d.min(dim=-1).values).cpu().numpy()
                d_mean = d.mean(dim=-1).clamp_min(1e-8)
                d_cv = (d.std(dim=-1) / d_mean).cpu().numpy()
                dist_ranges.extend(d_range.tolist())
                dist_cvs.extend(d_cv.tolist())
            # τ flag for axis contrast (data.x[:,1] when present)
            if data.x.size(1) > 1:
                tau = data.x[:, 1].float().cpu().numpy()
                hard = w.argmax(dim=-1).cpu().numpy()
                for i, e in enumerate(hard):
                    tau_by_expert[int(e)].append(float(tau[i]))

    max_ps_a = np.asarray(max_ps, dtype=float)
    Hs_a = np.asarray(Hs, dtype=float)
    balance = routing_balance_from_weight_rows(weight_rows, pdb_ids=pdb_ids)
    # Attach majority/minority dehydron rates (monitor for majority-conditional lever).
    for i, row in enumerate(balance.get("per_structure") or []):
        dh_row = dehydron_rows[i] if i < len(dehydron_rows) else None
        part = committed_majority_partition(
            weight_rows[i],
            dh_row,
        )
        row["majority_expert"] = part["majority_expert"]
        row["n_majority"] = part["n_majority"]
        row["n_minority"] = part["n_minority"]
        row["majority_dehydron_rate"] = part["majority_dehydron_rate"]
        row["minority_dehydron_rate"] = part["minority_dehydron_rate"]
        if dh_row is not None and len(dh_row) > 0:
            row["corpus_dehydron_frac"] = float(np.mean(np.asarray(dh_row) > 0.5))
        else:
            row["corpus_dehydron_frac"] = float("nan")
    purity = score_dehydron_partition_purity(balance.get("per_structure") or [])
    # Best pairwise underwrap contrast proxy: max |mean_τ(e_i) − mean_τ(e_j)|
    expert_tau_means = {
        e: float(np.mean(v)) if v else float("nan") for e, v in tau_by_expert.items()
    }
    contrasts = []
    for i in range(4):
        for j in range(i + 1, 4):
            a, b = expert_tau_means[i], expert_tau_means[j]
            if a == a and b == b:
                contrasts.append(abs(a - b))
    softplus = float(F.softplus(gate.logit_scale.detach().float()).cpu())
    floor = getattr(gate, "logit_softplus_floor", None)
    effective = max(softplus, float(floor)) if floor is not None else softplus
    core_quota_summary: dict[str, Any] | None = None
    if quota_rows:
        unplaced_fracs = [
            float(r.get("quota_unplaced_frac") or 0.0) for r in quota_rows
        ]
        core_quota_summary = {
            "n_structures": len(quota_rows),
            "mean_unplaced_frac": float(np.mean(unplaced_fracs)),
            "max_unplaced_frac": float(np.max(unplaced_fracs)),
            "unplaced_core_total": int(
                sum(int(r.get("quota_unplaced_core") or 0) for r in quota_rows)
            ),
            "forced_cross_axis_total": int(
                sum(int(r.get("quota_forced_cross_axis") or 0) for r in quota_rows)
            ),
            "per_structure": quota_rows,
        }
    return {
        "frac_max_p_ge_0_60": float(np.mean(max_ps_a >= 0.60)) if len(max_ps_a) else float("nan"),
        "frac_H_lt_0_5": float(np.mean(Hs_a < 0.5)) if len(Hs_a) else float("nan"),
        "median_H_i": float(np.median(Hs_a)) if len(Hs_a) else float("nan"),
        "mean_max_p": float(np.mean(max_ps_a)) if len(max_ps_a) else float("nan"),
        "max_max_p": float(np.max(max_ps_a)) if len(max_ps_a) else float("nan"),
        **balance,
        "dehydron_partition_purity": purity,
        "core_quota": core_quota_summary,
        "mean_dist_range": float(np.mean(dist_ranges)) if dist_ranges else float("nan"),
        "mean_dist_cv": float(np.mean(dist_cvs)) if dist_cvs else float("nan"),
        "softplus_scale": softplus,
        "logit_softplus_floor": None if floor is None else float(floor),
        "effective_softplus": effective,
        "best_tau_mean_contrast": float(max(contrasts)) if contrasts else float("nan"),
        "expert_tau_means": expert_tau_means,
    }


def measure_prototype_repulsion_epoch(
    model: torch.nn.Module,
    proteins: list[dict[str, Any]],
    device: str,
    *,
    twin: tuple[int, int] = HISTORICAL_TWIN,
) -> dict[str, Any]:
    """Full standing-metric pack for one scored epoch."""
    geom = measure_prototype_pairwise(model, twin=twin)
    sens = measure_degree_rival_sensitivity(model, proteins, device, twin=twin)
    route = measure_routing_companions(model, proteins, device)
    return {**geom, **sens, **route}


def score_proto_sep_ladder(
    report: dict[str, Any],
    *,
    sasa_frac_max_p_ge_060: float = 0.0,
) -> dict[str, Any]:
    """Score against frozen PROTO_SEP L1/L2/L3 floors + fail conditions."""
    twin_d = float(report.get("historical_twin_hyp_dist", float("nan")))
    nearest_d = float(report.get("nearest_pair_hyp_dist", float("nan")))
    rival_soft = float(report.get("mean_soft_on_rival_twin", float("nan")))
    deg_gap = float(report.get("degree_swap_mean_abs_delta_logit_gap", float("nan")))
    other_min = float(report.get("other_hyp_dist_min", float("nan")))
    twin_ratio = float(report.get("twin_over_other_mean", float("nan")))
    frac_mp = float(report.get("frac_max_p_ge_0_60", float("nan")))
    load_max = float(report.get("soft_load_max", float("nan")))
    eff = float(report.get("effective_experts_soft", float("nan")))

    fails: list[str] = []
    if nearest_d < 0.10 and twin_d < 0.10:
        fails.append("REPULSION_NO_MOVE")
    if other_min < 0.05:
        fails.append("COLLATERAL_NEW_COLLINEAR")
    if load_max > 0.45 or (eff == eff and eff < 3.0):
        fails.append("LOAD_COLLAPSE")
    if nearest_d >= 0.15 and deg_gap < 0.03:
        fails.append("SENSITIVITY_STILL_DEAD")

    # L1(a): twin ≥0.15 OR (if twin not nearest) nearest ≥0.15
    nearest_is_twin = sorted(report.get("nearest_pair") or []) == sorted(list(HISTORICAL_TWIN))
    l1_dist = (twin_d >= 0.15) or ((not nearest_is_twin) and nearest_d >= 0.15)
    l1_sens = deg_gap >= 0.05
    l1_rival = rival_soft < 0.30
    l1_collateral = other_min >= 0.05
    l1 = bool(l1_dist and l1_sens and l1_rival and l1_collateral)

    l2_dist = nearest_d >= 0.25
    l2_rival = rival_soft <= 0.20
    l2_func = frac_mp > sasa_frac_max_p_ge_060
    l2 = bool(l1 and l2_dist and l2_rival and l2_func)

    l3 = bool(l2 and twin_ratio >= 0.50)

    if "REPULSION_NO_MOVE" in fails:
        verdict = "REPULSION_NO_MOVE"
    elif any(
        f in fails
        for f in ("COLLATERAL_NEW_COLLINEAR", "LOAD_COLLAPSE", "DEHYDRON_CONTRAST_LOST")
    ):
        verdict = "COLLATERAL_DAMAGE"
    elif "SENSITIVITY_STILL_DEAD" in fails:
        verdict = "DISTANCE_WITHOUT_SENSITIVITY"
    elif l3:
        verdict = "PROTO_SEP_L3"
    elif l2:
        verdict = "PROTO_SEP_L2"
    elif l1:
        verdict = "PROTO_SEP_L1"
    else:
        verdict = "AMBIGUOUS"

    return {
        "verdict": verdict,
        "fails": fails,
        "l1": l1,
        "l2": l2,
        "l3": l3,
        "checks": {
            "l1_dist": l1_dist,
            "l1_sens": l1_sens,
            "l1_rival": l1_rival,
            "l1_collateral": l1_collateral,
            "l2_dist": l2_dist,
            "l2_rival": l2_rival,
            "l2_func_frac_max_p": l2_func,
            "twin_ratio": twin_ratio,
        },
    }


def score_stack_ladder(report: dict[str, Any]) -> dict[str, Any]:
    """Score repulsion × elevated-scale stack pre-reg (frozen 2026-07-14)."""
    nearest_d = float(report.get("nearest_pair_hyp_dist", float("nan")))
    deg_gap = float(report.get("degree_swap_mean_abs_delta_logit_gap", float("nan")))
    rival_soft = float(report.get("mean_soft_on_rival_twin", float("nan")))
    other_min = float(report.get("other_hyp_dist_min", float("nan")))
    frac_mp = float(report.get("frac_max_p_ge_0_60", float("nan")))
    softplus = float(report.get("effective_softplus", report.get("softplus_scale", float("nan"))))
    dist_range = float(report.get("mean_dist_range", float("nan")))
    dist_cv = float(report.get("mean_dist_cv", float("nan")))
    tau_c = float(report.get("best_tau_mean_contrast", float("nan")))

    fails: list[str] = []
    if softplus == softplus and softplus < 5.0:
        fails.append("SCALE_LOST")
    if nearest_d < 0.15 or deg_gap < 0.03:
        fails.append("REPULSION_REGRESSED")
    if frac_mp == 0.0:
        fails.append("COMMIT_STILL_DEAD")
    # Concrete DIST_RANGE_KILLS_SCALE trigger (every scored epoch)
    if (
        softplus >= 5.0
        and frac_mp < 0.02
        and (
            (dist_range == dist_range and dist_range <= 0.15)
            or (dist_cv == dist_cv and dist_cv <= 0.025)
        )
    ):
        fails.append("DIST_RANGE_KILLS_SCALE")
    if tau_c == tau_c and tau_c < 0.40:
        fails.append("AXIS_COLLAPSE")
    if other_min < 0.05:
        fails.append("COLLATERAL_NEW_COLLINEAR")

    sens_held = nearest_d >= 0.25 and deg_gap >= 0.05
    scale_held = softplus >= 5.0
    commit_l1 = frac_mp >= 0.05
    commit_l2 = frac_mp >= 0.15 and rival_soft <= 0.20
    axis_held = not (tau_c == tau_c and tau_c < 0.40)

    win_l1 = bool(
        sens_held
        and scale_held
        and commit_l1
        and axis_held
        and "DIST_RANGE_KILLS_SCALE" not in fails
        and "COLLATERAL_NEW_COLLINEAR" not in fails
    )
    win_l2 = bool(win_l1 and commit_l2)

    if "DIST_RANGE_KILLS_SCALE" in fails:
        verdict = "DIST_RANGE_NULLIFIES"
    elif "SCALE_LOST" in fails:
        verdict = "AMBIGUOUS"
    elif commit_l1 and scale_held and not sens_held:
        verdict = "SCALE_WINS_REPULSION_LOST"
    elif sens_held and scale_held and ("COMMIT_STILL_DEAD" in fails or not commit_l1):
        verdict = "SENS_WINS_COMMIT_DEAD"
    elif win_l2:
        verdict = "STACK_WIN_L2"
    elif win_l1:
        verdict = "STACK_WIN_L1"
    else:
        verdict = "AMBIGUOUS"

    return {
        "verdict": verdict,
        "fails": fails,
        "STACK_SENSITIVITY_HELD": sens_held,
        "STACK_SCALE_HELD": scale_held,
        "STACK_COMMIT_L1": commit_l1,
        "STACK_COMMIT_L2": commit_l2,
        "STACK_AXIS_HELD": axis_held,
        "checks": {
            "nearest_d": nearest_d,
            "deg_gap": deg_gap,
            "softplus": softplus,
            "frac_max_p_ge_0_60": frac_mp,
            "mean_dist_range": dist_range,
            "mean_dist_cv": dist_cv,
            "rival_soft": rival_soft,
            "best_tau_mean_contrast": tau_c,
        },
    }


def score_gram_cond_ladder(report: dict[str, Any]) -> dict[str, Any]:
    """Score full-bank Gram logdet hinge pre-reg (frozen 2026-07-15).

    Monopole clearing is out of scope — logged but not required.
    """
    frac_mp = float(report.get("frac_max_p_ge_0_60", float("nan")))
    tau_c = float(report.get("best_tau_mean_contrast", float("nan")))
    eig_min = float(report.get("gram_eig_min", float("nan")))
    cond = float(report.get("gram_condition", float("nan")))
    logdet = float(report.get("gram_logdet", float("nan")))
    nearest_d = float(report.get("nearest_pair_hyp_dist", float("nan")))
    purity = report.get("dehydron_partition_purity") or {}
    n_eligible = int(purity.get("n_eligible_structures") or 0)
    blurred = purity.get("blurred_pdb_ids") or []
    n_pass = max(0, n_eligible - len(blurred)) if n_eligible else 0
    if purity.get("passes") is True and n_eligible > 0:
        n_pass = n_eligible

    commit_held = frac_mp >= 0.15
    axis_held = not (tau_c == tau_c and tau_c < 0.40)
    bank_held = (
        eig_min == eig_min
        and eig_min >= 0.15
        and cond == cond
        and cond <= 15.0
        and logdet == logdet
        and logdet >= -1.15
    )
    # Per-seed purity floors are judged outside (seed1 ≥10/12, seed2 12/12);
    # ladder reports pass count for the run under test.
    purity_held = n_eligible > 0 and n_pass >= n_eligible  # strict for single-run view

    fails: list[str] = []
    if eig_min == eig_min and eig_min < 0.08:
        fails.append("GRAM_NO_MOVE")
    if frac_mp == frac_mp and frac_mp < 0.15:
        fails.append("COMMIT_KILLED_BY_GRAM")
    if nearest_d == nearest_d and nearest_d < 0.10:
        fails.append("PAIRWISE_COLLATERAL")
    if n_eligible > 0 and n_pass == 0 and bank_held:
        fails.append("AXIS_SCRAMBLED_BY_GRAM")

    if "COMMIT_KILLED_BY_GRAM" in fails:
        verdict = "COMMIT_KILLED_BY_GRAM"
    elif "PAIRWISE_COLLATERAL" in fails:
        verdict = "PAIRWISE_COLLATERAL"
    elif "AXIS_SCRAMBLED_BY_GRAM" in fails:
        verdict = "AXIS_SCRAMBLED_BY_GRAM"
    elif commit_held and axis_held and bank_held and purity_held:
        verdict = "GRAM_COND_WIN"
    elif commit_held and axis_held and ("GRAM_NO_MOVE" in fails or not bank_held):
        verdict = "GRAM_COND_NO_MOVE"
    elif bank_held and commit_held and axis_held and not purity_held:
        verdict = "GRAM_COND_PARTIAL"
    else:
        verdict = "AMBIGUOUS"

    return {
        "verdict": verdict,
        "fails": fails,
        "GRAM_COMMIT_HELD": commit_held,
        "GRAM_AXIS_HELD": axis_held,
        "GRAM_BANK_CONDITIONED": bank_held,
        "GRAM_PURITY_HELD": purity_held,
        "checks": {
            "frac_max_p_ge_0_60": frac_mp,
            "best_tau_mean_contrast": tau_c,
            "gram_eig_min": eig_min,
            "gram_condition": cond,
            "gram_logdet": logdet,
            "nearest_pair_hyp_dist": nearest_d,
            "n_eligible": n_eligible,
            "n_pass": n_pass,
            "per_structure_committed_hard_max": report.get(
                "per_structure_committed_hard_max"
            ),
        },
    }
