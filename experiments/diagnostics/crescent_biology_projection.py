"""
crescent_biology_projection.py
==============================
Read-only biology overlays on the Poincaré disc (no training, no gradients).

Checkpoint: lever_a_clean_slate_v1/v6_best_disc.pt (validated baseline — NOT
failed Phase 4 checkpoints).

Tiers:
  Tier 1 (trained, expected): ρ wrapping density — radial organization
  Tier 2 (emergence — GATED): dehydron angular KS after partialling disc_r
  Tier 2 is NOT declared on raw KS alone (crescent radial–angular coupling confound).

Outputs per structure:
  - {sid}_rho.png, {sid}_dehydron.png, {sid}_pharmacophore.png, {sid}_composite.png
  - angular_distribution_stats.json (depth-partialed KS gate for Tier 2)

Usage:
  python -m experiments.diagnostics.crescent_biology_projection --run
  make project-crescent-biology
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

from experiments.diagnostics.epistemic_decoupling_baseline import _ols_residual
from experiments.diagnostics.embedding_occupancy_audit import (
    _forward_audit,
    load_audit_model,
)
from experiments.training.v6._data import TAU, load_protein_graph
from experiments.training.v6.train_loop import attach_v6_features

DEFAULT_CHECKPOINT = Path(
    "checkpoints/v6/runs/lever_a_clean_slate_v1/v6_best_disc.pt"
)
DEFAULT_PDB_DIR = Path("/tmp/dtie_pdb_cache")
DEFAULT_STRUCTURES = ["11QE", "4OBE", "4DSO", "1IVO", "4MNE"]
DEFAULT_OUTPUT = Path("checkpoints/v6/diagnostics/crescent_projections")

# Validated pharmacophore / site residues (PDB numbering, chain A).
# KRAS three-bond lock: GOSP physics (not model-discovered). SPOP: docs/findings.
KRAS_THREE_BOND_LOCK: list[dict[str, str | int]] = [
    {"label": "Asp12", "resnum": 12, "note": "G12D switch-I (GOSP-validated)"},
    {"label": "Tyr32", "resnum": 32, "note": "three-bond lock (GOSP)"},
    {"label": "Gly60", "resnum": 60, "note": "switch-II (GOSP)"},
    {"label": "Gln61", "resnum": 61, "note": "three-bond lock (GOSP)"},
]

PHARMACOPHORE_SITES: dict[str, list[dict[str, str | int]]] = {
    "11QE": [
        *KRAS_THREE_BOND_LOCK,
        {"label": "I55E", "resnum": 55, "note": "Suppressor second-site (structural)"},
    ],
    "4OBE": [
        {"label": "Gly12", "resnum": 12, "note": "KRAS WT switch-I (GOSP)"},
        {"label": "Tyr32", "resnum": 32, "note": "three-bond lock (GOSP)"},
        {"label": "Gly60", "resnum": 60, "note": "switch-II (GOSP)"},
        {"label": "Gln61", "resnum": 61, "note": "three-bond lock (GOSP)"},
    ],
    "4DSO": list(KRAS_THREE_BOND_LOCK),
    "1IVO": [
        {"label": "DFG motif", "resnum": 855, "note": "EGFR kinase — verify PDB numbering"},
    ],
    "4MNE": [
        {"label": "Active site", "resnum": 41, "note": "Placeholder — calibrate for figure"},
    ],
    "3IVV": [
        {"label": "Ser33", "resnum": 33, "note": "SPOP AURKA phospho site"},
        {"label": "Phe57", "resnum": 57, "note": "SPOP allosteric bottleneck"},
    ],
}

# 9EST elastase / 1FLE elafin interface — chain-residue pairs for PeSTo setup.
# Populate when structures are onboarded; empty skips interface layer.
INTERFACE_SITES: dict[str, list[dict[str, str | int]]] = {
    "9EST": [],
    "1FLE": [],
}


@dataclass
class ResidueBiology:
    structure_id: str
    chain: str
    n_residues: int
    disc_xy: list[list[float]]
    disc_r: list[float]
    disc_theta_deg: list[float]
    rho: list[float]
    dehydron: list[bool]
    res_ids: list[str]
    cone_depth: list[float]


@dataclass
class QuartileKs:
    quartile: int
    disc_r_min: float
    disc_r_max: float
    n_dehydron: int
    n_non_dehydron: int
    ks_statistic: float
    ks_pvalue: float
    within_quartile_perm_p: float = float("nan")


@dataclass
class AngularStats:
    structure_id: str
    n_dehydron: int
    n_non_dehydron: int
    mean_theta_dehydron_deg: float
    mean_theta_non_dehydron_deg: float
    # Raw KS — confounded by crescent radial–angular geometry; do not use for Tier 2
    ks_statistic_raw: float
    ks_pvalue_raw: float
    # Depth-partialed gate (θ residual after OLS on disc_r)
    ks_statistic_theta_residual: float
    ks_pvalue_theta_residual: float
    quartile_ks: list[QuartileKs]
    quartile_ks_significant_count: int
    label_shuffle_ks_empirical_p: float
    # Nonlinear confound gate: shuffle dehydron labels within disc_r quartile only
    within_quartile_max_ks_observed: float
    within_quartile_max_ks_perm_p: float
    within_quartile_perm_n: int
    within_quartile_significant_quartile_count: int
    corr_rho_disc_r: float
    corr_rho_cone_depth: float
    corr_disc_r_cone_depth: float
    corr_disc_r_cone_depth_spearman: float
    radial_depth_note: str
    tier2_gate_status: str
    tier2_interpretation: str
    tier2_sbir_framing: str
    tier1_note: str


MIN_GROUP = 5
LABEL_SHUFFLE_N = 499
WITHIN_QUARTILE_PERM_N = 5000


def _ks_two_groups(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if len(a) < MIN_GROUP or len(b) < MIN_GROUP:
        return float("nan"), float("nan")
    stat, p = stats.ks_2samp(a, b)
    return float(stat), float(p)


def _label_shuffle_empirical_p(
    theta: np.ndarray,
    dehyd: np.ndarray,
    ks_obs: float,
    *,
    n_perm: int = LABEL_SHUFFLE_N,
    seed: int = 42,
) -> float:
    """How often does a random dehydron label shuffle match raw KS? (geometry confound probe)."""
    if not np.isfinite(ks_obs) or dehyd.sum() < MIN_GROUP or (~dehyd).sum() < MIN_GROUP:
        return float("nan")
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_perm):
        shuf = rng.permutation(dehyd)
        ks_s, _ = _ks_two_groups(theta[shuf], theta[~shuf])
        if np.isfinite(ks_s) and ks_s >= ks_obs:
            exceed += 1
    return float((exceed + 1) / (n_perm + 1))


def _quartile_masks(disc_r: np.ndarray) -> list[np.ndarray]:
    if len(disc_r) < 4:
        return []
    edges = np.quantile(disc_r, [0.0, 0.25, 0.5, 0.75, 1.0])
    masks: list[np.ndarray] = []
    for q in range(4):
        lo, hi = float(edges[q]), float(edges[q + 1])
        if q < 3:
            masks.append((disc_r >= lo) & (disc_r < hi))
        else:
            masks.append((disc_r >= lo) & (disc_r <= hi))
    return masks


def _quartile_ks(theta: np.ndarray, dehyd: np.ndarray, disc_r: np.ndarray) -> list[QuartileKs]:
    masks = _quartile_masks(disc_r)
    if not masks:
        return []
    edges = np.quantile(disc_r, [0.0, 0.25, 0.5, 0.75, 1.0])
    out: list[QuartileKs] = []
    for q, mask in enumerate(masks):
        lo, hi = float(edges[q]), float(edges[q + 1])
        ks_stat, ks_p = _ks_two_groups(theta[dehyd & mask], theta[~dehyd & mask])
        out.append(
            QuartileKs(
                quartile=q + 1,
                disc_r_min=lo,
                disc_r_max=hi,
                n_dehydron=int((dehyd & mask).sum()),
                n_non_dehydron=int((~dehyd & mask).sum()),
                ks_statistic=ks_stat,
                ks_pvalue=ks_p,
            )
        )
    return out


def _within_quartile_permutation(
    theta: np.ndarray,
    dehyd: np.ndarray,
    disc_r: np.ndarray,
    *,
    n_perm: int = WITHIN_QUARTILE_PERM_N,
    seed: int = 42,
) -> tuple[float, float, list[float]]:
    """Shuffle dehydron labels within each disc_r quartile; test max-bin KS.

    Returns (max_observed_ks, max_bin_empirical_p, per_quartile_empirical_p).
    """
    masks = _quartile_masks(disc_r)
    if not masks:
        return float("nan"), float("nan"), []

    obs_per_q: list[float] = []
    for mask in masks:
        ks_s, _ = _ks_two_groups(theta[dehyd & mask], theta[~dehyd & mask])
        obs_per_q.append(ks_s)

    finite_obs = [s for s in obs_per_q if np.isfinite(s)]
    if not finite_obs:
        return float("nan"), float("nan"), [float("nan")] * len(masks)

    obs_max = float(max(finite_obs))
    rng = np.random.default_rng(seed)
    exceed_max = 0
    exceed_per_q = [0] * len(masks)

    for _ in range(n_perm):
        shuf = dehyd.copy()
        for mask in masks:
            idx = np.where(mask)[0]
            if len(idx) < 2:
                continue
            labels = shuf[idx].copy()
            rng.shuffle(labels)
            shuf[idx] = labels

        perm_per_q: list[float] = []
        for mask in masks:
            ks_s, _ = _ks_two_groups(theta[shuf & mask], theta[~shuf & mask])
            perm_per_q.append(ks_s)

        perm_finite = [s for s in perm_per_q if np.isfinite(s)]
        if perm_finite:
            perm_max = max(perm_finite)
            if perm_max >= obs_max:
                exceed_max += 1

        for q in range(len(masks)):
            if np.isfinite(obs_per_q[q]) and np.isfinite(perm_per_q[q]):
                if perm_per_q[q] >= obs_per_q[q]:
                    exceed_per_q[q] += 1

    max_p = float((exceed_max + 1) / (n_perm + 1))
    per_q_p = []
    for q in range(len(masks)):
        if not np.isfinite(obs_per_q[q]):
            per_q_p.append(float("nan"))
        else:
            per_q_p.append(float((exceed_per_q[q] + 1) / (n_perm + 1)))
    return obs_max, max_p, per_q_p


def _attach_quartile_perm_p(
    quartile_results: list[QuartileKs],
    per_q_perm_p: list[float],
) -> list[QuartileKs]:
    out: list[QuartileKs] = []
    for i, q in enumerate(quartile_results):
        perm_p = per_q_perm_p[i] if i < len(per_q_perm_p) else float("nan")
        out.append(
            QuartileKs(
                quartile=q.quartile,
                disc_r_min=q.disc_r_min,
                disc_r_max=q.disc_r_max,
                n_dehydron=q.n_dehydron,
                n_non_dehydron=q.n_non_dehydron,
                ks_statistic=q.ks_statistic,
                ks_pvalue=q.ks_pvalue,
                within_quartile_perm_p=perm_p,
            )
        )
    return out


TIER2_SBIR_FRAMING = (
    "Dehydron residues preferentially occupy distinct angular regions in a space "
    "where the angular axis encodes local chemical environment and domain topology — "
    "a relationship not directly supervised on dehydron labels, consistent with "
    "under-wrapped backbone hydrogen bonds occupying structurally heterogeneous "
    "microenvironments."
)


def _tier2_verdict(
    ks_residual_p: float,
    quartile_results: list[QuartileKs],
    *,
    within_quartile_max_perm_p: float,
) -> tuple[str, str]:
    sig_perm_q = sum(
        1
        for q in quartile_results
        if np.isfinite(q.within_quartile_perm_p) and q.within_quartile_perm_p < 0.05
    )
    surviving_q = [
        q.quartile
        for q in quartile_results
        if np.isfinite(q.within_quartile_perm_p) and q.within_quartile_perm_p < 0.05
    ]

    if not np.isfinite(ks_residual_p):
        return (
            "inconclusive",
            "insufficient data for depth-partialed KS — Tier 2 gate not closed",
        )
    if ks_residual_p >= 0.05:
        return (
            "residual_fail",
            "depth-partialed KS not significant — Tier 2 not supported for this structure",
        )

    if not np.isfinite(within_quartile_max_perm_p):
        return (
            "residual_only",
            "residual KS significant but within-quartile permutation inconclusive — "
            "Tier 2 not declared",
        )

    if within_quartile_max_perm_p < 0.05 or sig_perm_q >= 2:
        return (
            "gate_closed_pass",
            "residual + within-quartile permutation support angular clustering beyond "
            "crescent geometry — structure eligible for Tier 2 (cohort not yet declared)",
        )

    if sig_perm_q == 1:
        qn = surviving_q[0]
        return (
            "gate_closed_partial",
            f"angular signal localized to depth quartile Q{qn} after within-quartile "
            f"permutation (p<0.05) — weaker per-structure claim; review before Tier 2",
        )

    return (
        "gate_closed_fail",
        "within-quartile permutation does not support angular clustering beyond "
        "crescent geometry — Tier 2 not supported for this structure",
    )


def _parse_structure(spec: str) -> tuple[str, str]:
    parts = spec.strip().upper().split(":")
    return parts[0], parts[1] if len(parts) > 1 else "A"


def _parse_structures(raw: str) -> list[str]:
    return [_parse_structure(s)[0] for s in raw.split(",") if s.strip()]


def _resnum_from_id(res_id: str) -> int | None:
    # Format from _data.py: "A:12:"
    parts = res_id.split(":")
    if len(parts) >= 2 and parts[1].isdigit():
        return int(parts[1])
    return None


def _load_biology_arrays(
    structure_id: str,
    chain: str,
    model: torch.nn.Module,
    pdb_dir: Path,
    device: str,
) -> ResidueBiology:
    prot = load_protein_graph(structure_id, chain, pdb_dir)
    if prot is None:
        raise RuntimeError(f"Could not load {structure_id}:{chain}")

    out = _forward_audit(model, "v6", prot, device)

    data = attach_v6_features(prot["data"])
    xy = out["hyp_projections_2d"].detach().cpu().numpy()
    depth = out["cone_depth"].detach().cpu().numpy().reshape(-1)
    x = data.x.cpu().numpy()
    rho = x[:, 0]
    dehydron = rho < TAU

    res_ids = prot.get("residue_ids") or prot.get("res_ids")
    if res_ids is None:
        res_ids = [f"{chain}:{i}:" for i in range(len(rho))]
    elif isinstance(res_ids, torch.Tensor):
        res_ids = [str(r) for r in res_ids]
    else:
        res_ids = list(res_ids)

    r = np.linalg.norm(xy, axis=1)
    theta = np.degrees(np.arctan2(xy[:, 1], xy[:, 0]))

    return ResidueBiology(
        structure_id=structure_id,
        chain=chain,
        n_residues=int(len(rho)),
        disc_xy=xy.tolist(),
        disc_r=r.tolist(),
        disc_theta_deg=theta.tolist(),
        rho=rho.tolist(),
        dehydron=dehydron.astype(bool).tolist(),
        res_ids=res_ids,
        cone_depth=depth.tolist(),
    )


def _compute_angular_stats(bio: ResidueBiology) -> AngularStats:
    theta = np.array(bio.disc_theta_deg, dtype=float)
    rho = np.array(bio.rho, dtype=float)
    r = np.array(bio.disc_r, dtype=float)
    depth = np.array(bio.cone_depth, dtype=float)
    dehyd = np.array(bio.dehydron, dtype=bool)

    d_theta = theta[dehyd]
    nd_theta = theta[~dehyd]
    ks_raw, p_raw = _ks_two_groups(d_theta, nd_theta)

    theta_resid = _ols_residual(theta, r)
    ks_res, p_res = _ks_two_groups(theta_resid[dehyd], theta_resid[~dehyd])

    q_ks = _quartile_ks(theta, dehyd, r)
    sig_q = sum(1 for q in q_ks if np.isfinite(q.ks_pvalue) and q.ks_pvalue < 0.05)
    shuffle_p = _label_shuffle_empirical_p(theta, dehyd, ks_raw)

    obs_max_q, max_perm_p, per_q_perm_p = _within_quartile_permutation(theta, dehyd, r)
    q_ks = _attach_quartile_perm_p(q_ks, per_q_perm_p)
    sig_perm_q = sum(
        1 for q in q_ks if np.isfinite(q.within_quartile_perm_p) and q.within_quartile_perm_p < 0.05
    )

    corr_rho_r = float(np.corrcoef(rho, r)[0, 1]) if len(rho) > 2 else float("nan")
    corr_rho_d = float(np.corrcoef(rho, depth)[0, 1]) if len(rho) > 2 else float("nan")
    corr_rd = float(np.corrcoef(r, depth)[0, 1]) if len(r) > 2 else float("nan")
    corr_rd_sp = float(stats.spearmanr(r, depth).statistic) if len(r) > 2 else float("nan")

    gate_status, interp = _tier2_verdict(
        p_res, q_ks, within_quartile_max_perm_p=max_perm_p
    )

    radial_note = (
        "lever_a uses disc_radial_source=radial_depth: disc_r (=||disc_2d||) is set by "
        "apply_disc_radial_override from radial_depth, not independently learned. "
        "cone_depth = hyperbolic dist0(x_hyp). High corr(disc_r, cone_depth) means radial "
        "disc position and burial depth are one monotonic family — state once in SBIR text."
    )

    return AngularStats(
        structure_id=bio.structure_id,
        n_dehydron=int(dehyd.sum()),
        n_non_dehydron=int((~dehyd).sum()),
        mean_theta_dehydron_deg=float(np.mean(d_theta)) if len(d_theta) else float("nan"),
        mean_theta_non_dehydron_deg=float(np.mean(nd_theta)) if len(nd_theta) else float("nan"),
        ks_statistic_raw=ks_raw,
        ks_pvalue_raw=p_raw,
        ks_statistic_theta_residual=ks_res,
        ks_pvalue_theta_residual=p_res,
        quartile_ks=q_ks,
        quartile_ks_significant_count=sig_q,
        label_shuffle_ks_empirical_p=shuffle_p,
        within_quartile_max_ks_observed=obs_max_q,
        within_quartile_max_ks_perm_p=max_perm_p,
        within_quartile_perm_n=WITHIN_QUARTILE_PERM_N,
        within_quartile_significant_quartile_count=sig_perm_q,
        corr_rho_disc_r=corr_rho_r,
        corr_rho_cone_depth=corr_rho_d,
        corr_disc_r_cone_depth=corr_rd,
        corr_disc_r_cone_depth_spearman=corr_rd_sp,
        radial_depth_note=radial_note,
        tier2_gate_status=gate_status,
        tier2_interpretation=interp,
        tier2_sbir_framing=TIER2_SBIR_FRAMING,
        tier1_note=(
            "ρ vs disc_r reflects trained radial organization (ρ ∈ data.x[:,0]); "
            "confirms physics encoded — not an emergence claim"
        ),
    )


def _angular_stats_to_dict(ang: AngularStats) -> dict[str, Any]:
    d = asdict(ang)
    d["quartile_ks"] = [asdict(q) for q in ang.quartile_ks]
    return d


def _site_mask(bio: ResidueBiology, sites: list[dict[str, str | int]]) -> np.ndarray:
    mask = np.zeros(bio.n_residues, dtype=bool)
    resnums = [_resnum_from_id(rid) for rid in bio.res_ids]
    for site in sites:
        rn = int(site["resnum"])
        for i, num in enumerate(resnums):
            if num == rn:
                mask[i] = True
    return mask


def _draw_disc_base(ax, curvature: float) -> None:
    r_ball = 1.0 / math.sqrt(curvature) if curvature > 0 else 1.0
    circle = plt.Circle((0, 0), r_ball, fill=False, color="#666", linestyle="--", linewidth=0.8)
    ax.add_patch(circle)
    ax.set_aspect("equal")
    ax.axhline(0, color="#ccc", linewidth=0.4, zorder=0)
    ax.axvline(0, color="#ccc", linewidth=0.4, zorder=0)


def _plot_rho(
    bio: ResidueBiology,
    output: Path,
    curvature: float,
    *,
    title_suffix: str = "",
) -> None:
    import matplotlib.pyplot as plt

    xy = np.array(bio.disc_xy)
    rho = np.array(bio.rho)
    fig, ax = plt.subplots(figsize=(6, 6))
    _draw_disc_base(ax, curvature)
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=rho, cmap="viridis", s=14, alpha=0.85, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="ρ wrapping density", shrink=0.8)
    title = f"{bio.structure_id} — Tier 1: ρ (trained radial signal)"
    if title_suffix:
        title = f"{title} — {title_suffix}"
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("disc x")
    ax.set_ylabel("disc y")
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_dehydron(
    bio: ResidueBiology,
    output: Path,
    curvature: float,
    ang: AngularStats,
) -> None:
    import matplotlib.pyplot as plt

    xy = np.array(bio.disc_xy)
    dehyd = np.array(bio.dehydron)
    fig, ax = plt.subplots(figsize=(6, 6))
    _draw_disc_base(ax, curvature)
    ax.scatter(
        xy[~dehyd, 0],
        xy[~dehyd, 1],
        s=10,
        c="#cccccc",
        alpha=0.5,
        label="non-dehydron",
        edgecolors="none",
    )
    ax.scatter(
        xy[dehyd, 0],
        xy[dehyd, 1],
        s=28,
        c="#e74c3c",
        alpha=0.9,
        marker="*",
        label=f"τ<{TAU} dehydron",
        edgecolors="k",
        linewidths=0.3,
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(
        f"{bio.structure_id} — dehydrons "
        f"(resid p={ang.ks_pvalue_theta_residual:.4f}; "
        f"within-Q perm p={ang.within_quartile_max_ks_perm_p:.4f})",
        fontsize=10,
    )
    ax.set_xlabel("disc x")
    ax.set_ylabel("disc y")
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_pharmacophore(
    bio: ResidueBiology,
    sites: list[dict[str, str | int]],
    output: Path,
    curvature: float,
) -> None:
    import matplotlib.pyplot as plt

    if not sites:
        return
    xy = np.array(bio.disc_xy)
    mask = _site_mask(bio, sites)
    fig, ax = plt.subplots(figsize=(6, 6))
    _draw_disc_base(ax, curvature)
    ax.scatter(xy[:, 0], xy[:, 1], s=10, c="#eca53a", alpha=0.6, edgecolors="none")
    if mask.any():
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=80,
            facecolors="none",
            edgecolors="#2ecc71",
            linewidths=2,
            label="validated site",
        )
        for i, on in enumerate(mask):
            if on:
                for site in sites:
                    if _resnum_from_id(bio.res_ids[i]) == int(site["resnum"]):
                        ax.annotate(
                            str(site["label"]),
                            (xy[i, 0], xy[i, 1]),
                            fontsize=7,
                            color="#1a5c2e",
                            xytext=(4, 4),
                            textcoords="offset points",
                        )
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"{bio.structure_id} — Tier 2: pharmacophore sites", fontsize=10)
    ax.set_xlabel("disc x")
    ax.set_ylabel("disc y")
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_composite(
    bio: ResidueBiology,
    sites: list[dict[str, str | int]],
    output: Path,
    curvature: float,
    ang: AngularStats,
) -> None:
    import matplotlib.pyplot as plt

    xy = np.array(bio.disc_xy)
    rho = np.array(bio.rho)
    dehyd = np.array(bio.dehydron)
    site_mask = _site_mask(bio, sites) if sites else np.zeros(bio.n_residues, dtype=bool)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax in axes:
        _draw_disc_base(ax, curvature)

    sc0 = axes[0].scatter(xy[:, 0], xy[:, 1], c=rho, cmap="viridis", s=10, alpha=0.85)
    axes[0].set_title("Tier 1: ρ")
    plt.colorbar(sc0, ax=axes[0], shrink=0.7)

    axes[1].scatter(xy[~dehyd, 0], xy[~dehyd, 1], s=8, c="#ccc", alpha=0.5)
    axes[1].scatter(xy[dehyd, 0], xy[dehyd, 1], s=20, c="#e74c3c", marker="*", alpha=0.9)
    axes[1].set_title(
        f"dehydrons resid p={ang.ks_pvalue_theta_residual:.3f} | "
        f"within-Q perm p={ang.within_quartile_max_ks_perm_p:.3f} [{ang.tier2_gate_status}]"
    )

    axes[2].scatter(xy[:, 0], xy[:, 1], s=8, c="#eca53a", alpha=0.6)
    if site_mask.any():
        axes[2].scatter(
            xy[site_mask, 0],
            xy[site_mask, 1],
            s=60,
            facecolors="none",
            edgecolors="#2ecc71",
            linewidths=2,
        )
    axes[2].set_title("pharmacophore (downstream of Tier 2 gate)")

    fig.suptitle(f"{bio.structure_id} crescent biology — lever_a", fontsize=12)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def project_biology_onto_disc(
    checkpoint_path: Path,
    structure_ids: list[str],
    pdb_dir: Path,
    output_dir: Path,
    *,
    chain: str = "A",
    device: str = "cpu",
    layers: list[str] | None = None,
) -> dict[str, Any]:
    layers = layers or ["rho", "dehydron", "pharmacophore", "interface", "composite"]
    output_dir.mkdir(parents=True, exist_ok=True)

    model, version = load_audit_model(checkpoint_path, device)
    if version != "v6":
        raise RuntimeError("crescent_biology_projection requires v6 checkpoint")
    curvature = float(model.curvature.detach().cpu())

    per_structure: list[dict[str, Any]] = []
    angular_stats: list[dict[str, Any]] = []

    for sid in structure_ids:
        bio = _load_biology_arrays(sid, chain, model, pdb_dir, device)
        ang = _compute_angular_stats(bio)
        angular_stats.append(_angular_stats_to_dict(ang))

        sites = PHARMACOPHORE_SITES.get(sid.upper(), [])
        paths: dict[str, str] = {}

        if "rho" in layers:
            p = output_dir / f"{sid}_rho.png"
            _plot_rho(bio, p, curvature, title_suffix="lever_a")
            paths["rho"] = str(p)

        if "dehydron" in layers:
            p = output_dir / f"{sid}_dehydron.png"
            _plot_dehydron(bio, p, curvature, ang)
            paths["dehydron"] = str(p)

        if "pharmacophore" in layers and sites:
            p = output_dir / f"{sid}_pharmacophore.png"
            _plot_pharmacophore(bio, sites, p, curvature)
            paths["pharmacophore"] = str(p)

        if "interface" in layers:
            iface = INTERFACE_SITES.get(sid.upper(), [])
            if iface:
                p = output_dir / f"{sid}_interface.png"
                _plot_pharmacophore(bio, iface, p, curvature)
                paths["interface"] = str(p)

        if "composite" in layers:
            p = output_dir / f"{sid}_composite.png"
            _plot_composite(bio, sites, p, curvature, ang)
            paths["composite"] = str(p)

        per_structure.append(
            {
                "structure_id": sid,
                "n_residues": bio.n_residues,
                "figures": paths,
                "angular_stats": _angular_stats_to_dict(ang),
            }
        )

    report = {
        "checkpoint": str(checkpoint_path),
        "pdb_dir": str(pdb_dir),
        "chain": chain,
        "structures": structure_ids,
        "layers": layers,
        "tier1_claim": (
            "DTIE's hyperbolic embedding geometrically organizes wrapping density "
            "onto the radial axis, making the physical basis of dehydron prediction "
            "spatially interpretable (trained behavior — ρ ∈ data.x[:,0])"
        ),
        "tier2_gate": (
            "Tier 2 requires: (1) depth-partialed residual KS p<0.05, then "
            "(2) within-quartile label permutation (nonlinear crescent confound). "
            "gate_closed_pass/partial = structure eligible; cohort Tier 2 not auto-declared."
        ),
        "tier2_sbir_framing": TIER2_SBIR_FRAMING,
        "per_structure": per_structure,
        "angular_distribution_stats": angular_stats,
    }
    stats_path = output_dir / "angular_distribution_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--pdb-dir", type=Path, default=DEFAULT_PDB_DIR)
    ap.add_argument("--structures", default=",".join(DEFAULT_STRUCTURES))
    ap.add_argument("--chain", default="A")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument(
        "--layers",
        default="rho,dehydron,pharmacophore,composite",
        help="Comma-separated: rho,dehydron,pharmacophore,interface,composite",
    )
    args = ap.parse_args()
    if not args.run and not args.structures:
        ap.print_help()
        return 0

    structure_ids = [s.strip().upper() for s in args.structures.split(",") if s.strip()]
    layers = [x.strip() for x in args.layers.split(",") if x.strip()]

    report = project_biology_onto_disc(
        args.checkpoint,
        structure_ids,
        args.pdb_dir,
        args.output_dir,
        chain=args.chain,
        device=args.device,
        layers=layers,
    )
    print(json.dumps(report["angular_distribution_stats"], indent=2))
    print(f"Wrote → {args.output_dir / 'angular_distribution_stats.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
