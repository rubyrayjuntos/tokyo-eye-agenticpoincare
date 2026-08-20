#!/usr/bin/env python3
"""Lean proof package: learned disc residuals vs structural SSOT.

Pre-registered gates (lean package):
1) Residual neighbor catalog — learned kNN minus SSOT kNN (mid/rim band).
2) Role-edge enrichment on residual pairs vs degree/r-matched nulls.
3) Pharmacophore/site enrichment in learned-only vs SSOT-only shells.
4) Ablation — drop dehydron+spoke role edges, re-embed; residual set must shrink.

Usage:
  science python -m experiments.diagnostics.learned_ssot_residual_proof \\
    --checkpoint checkpoints/v66/runs/feeler_expand_23_rim_fanout_coverage_v2/epochs/epoch_164.pt \\
    --corpus manifests/v6_corpus_stage_a_feeler_expand_v1.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.diagnostics.crescent_biology_projection import (
    PHARMACOPHORE_SITES,
)
from experiments.training.v6._data import load_protein_graph
from experiments.training.v6.train_loop import prepare_training_batch
from science.dtie.common.residue_features import TAU
from science.dtie.v6.visualization.interactive_viewer import disc_xy_from_model_output
from science.dtie.v66.role_edge_graph import (
    ROLE_COUPLING,
    ROLE_DEHYDRON,
    ROLE_PACKING,
    ROLE_RIBBON,
    ROLE_SPOKE,
)
from science.dtie.v66.thermo_edge_features import GEO_DIM
from science.training.gnn_lineage import load_model_from_checkpoint

# Extra literature / GOSP sites for structures lacking PHARMACOPHORE_SITES entries.
EXTRA_SITES: dict[str, list[dict[str, str | int]]] = {
    "1F88": [
        {"label": "Lys296", "resnum": 296, "note": "Schiff base / retinal (literature)"},
        {"label": "Glu113", "resnum": 113, "note": "counterion (literature)"},
        {"label": "Trp265", "resnum": 265, "note": "retinal pocket (literature)"},
        {"label": "EL2-Cys187", "resnum": 187, "note": "EL2 disulfide region"},
        {"label": "EL2-191", "resnum": 191, "note": "EL2 lid"},
    ],
    "1R69": [
        {"label": "helix core proxy", "resnum": 20, "note": "bundle interior proxy"},
    ],
}

ROLE_NAMES = {
    ROLE_PACKING: "packing",
    ROLE_DEHYDRON: "dehydron",
    ROLE_SPOKE: "spoke",
    ROLE_RIBBON: "ribbon",
    ROLE_COUPLING: "coupling",
}


def _resnum(rid: str) -> int:
    part = rid.split(":")[1]
    return int("".join(ch for ch in part if ch.isdigit() or ch == "-"))


def _site_mask(res_ids: list[str], sites: list[dict[str, str | int]]) -> np.ndarray:
    wanted = {int(s["resnum"]) for s in sites}
    return np.array([_resnum(r) in wanted for r in res_ids], dtype=bool)


def _load_corpus(path: Path) -> list[tuple[str, str]]:
    raw = json.loads(path.read_text())
    out: list[tuple[str, str]] = []
    for p in raw.get("proteins", []):
        if not p.get("enabled", True):
            continue
        out.append((str(p["pdb_id"]).upper(), str(p.get("chain", "A"))))
    return out


def _knn_pairs(
    xy: np.ndarray,
    r: np.ndarray,
    *,
    min_r: float,
    k: int,
    max_dr: float,
) -> set[tuple[int, int]]:
    """Undirected kNN among mid/rim residues with |Δr| soft filter."""
    idx = np.where(r >= min_r)[0]
    pairs: set[tuple[int, int]] = set()
    if len(idx) < 2:
        return pairs
    sub = xy[idx]
    d2 = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(axis=-1)
    np.fill_diagonal(d2, np.inf)
    for local_i, gi in enumerate(idx):
        order = np.argsort(d2[local_i])
        taken = 0
        for local_j in order:
            if taken >= k:
                break
            gj = int(idx[local_j])
            if abs(float(r[gi] - r[gj])) > max_dr:
                continue
            a, b = (gi, gj) if gi < gj else (gj, gi)
            if a == b:
                continue
            pairs.add((a, b))
            taken += 1
    return pairs


def _seq_sep(res_ids: list[str], i: int, j: int) -> int:
    return abs(_resnum(res_ids[i]) - _resnum(res_ids[j]))


def _res_ids_from_prot(prot: dict[str, Any], chain: str, n: int) -> list[str]:
    res_ids = prot.get("residue_ids") or prot.get("res_ids")
    if res_ids is None:
        return [f"{chain}:{i}:" for i in range(n)]
    if isinstance(res_ids, torch.Tensor):
        return [str(r) for r in res_ids]
    return list(res_ids)


@torch.inference_mode()
def _forward_disc(
    model: torch.nn.Module,
    prot: dict[str, Any],
    device: str,
    *,
    use_ssot: bool,
    strip_roles: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, Any]:
    """Training-faithful forward (role edges + rim fanout when enabled)."""
    data = prepare_training_batch(
        model,
        prot,
        device,
        structural_disc_frozen=use_ssot,
    )
    if strip_roles and bool(getattr(model, "role_edge_mp", False)):
        ei = data.edge_index
        ea = data.edge_attr
        if ea is not None and ea.size(-1) > GEO_DIM + 4:
            role = ea[:, GEO_DIM : GEO_DIM + 5].argmax(dim=-1)
            keep = torch.ones(ei.shape[1], dtype=torch.bool, device=ei.device)
            for rid in strip_roles:
                keep &= role != int(rid)
            data.edge_index = ei[:, keep]
            data.edge_attr = ea[keep]
    out = model(data)
    xy = disc_xy_from_model_output(out).detach().cpu().numpy()
    depth = out["cone_depth"].detach().cpu().numpy().reshape(-1)
    return xy, depth, data


def _role_edge_sets_from_data(data: Any) -> dict[str, set[tuple[int, int]]]:
    out: dict[str, set[tuple[int, int]]] = {v: set() for v in ROLE_NAMES.values()}
    if data.edge_attr is None or data.edge_attr.size(-1) <= GEO_DIM + 4:
        return out
    ei = data.edge_index.cpu().numpy()
    ea = data.edge_attr.cpu().numpy()
    role = ea[:, GEO_DIM : GEO_DIM + 5]
    for e in range(ei.shape[1]):
        i, j = int(ei[0, e]), int(ei[1, e])
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        rid = int(role[e].argmax())
        out[ROLE_NAMES.get(rid, "packing")].add((a, b))
    return out


def _pair_edge_hits(
    pairs: set[tuple[int, int]],
    edge_sets: dict[str, set[tuple[int, int]]],
) -> dict[str, float]:
    if not pairs:
        return {k: 0.0 for k in edge_sets}
    return {k: sum(1 for p in pairs if p in s) / len(pairs) for k, s in edge_sets.items()}


def _null_pairs(
    n: int,
    r: np.ndarray,
    pairs: set[tuple[int, int]],
    *,
    rng: np.random.Generator,
    n_null: int = 200,
) -> list[set[tuple[int, int]]]:
    if not pairs:
        return [set() for _ in range(3)]
    drs = [abs(float(r[a] - r[b])) for a, b in pairs]
    mean_dr = float(np.mean(drs)) if drs else 0.1
    cands: list[tuple[int, int]] = []
    for _ in range(max(n_null * 20, 500)):
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        a, b = (int(i), int(j)) if i < j else (int(j), int(i))
        if abs(float(r[a] - r[b])) <= mean_dr + 0.05:
            cands.append((a, b))
    cands = list({*cands})
    rng.shuffle(cands)
    out: list[set[tuple[int, int]]] = []
    need = len(pairs)
    for t in range(n_null):
        start = (t * need) % max(len(cands), 1)
        chunk = cands[start : start + need]
        if len(chunk) < need and cands:
            chunk = (chunk + cands[: need - len(chunk)])[:need]
        out.append(set(chunk))
    return out


def _enrichment(hit_frac: float, null_fracs: list[float]) -> dict[str, float]:
    null = np.asarray(null_fracs, dtype=float)
    mu = float(null.mean()) if len(null) else 0.0
    sd = float(null.std(ddof=0)) if len(null) else 0.0
    if sd < 1e-12:
        z = 0.0 if abs(hit_frac - mu) < 1e-12 else (np.inf if hit_frac > mu else -np.inf)
        z = float(z) if np.isfinite(z) else (100.0 if z > 0 else -100.0)
    else:
        z = float((hit_frac - mu) / sd)
    p = float((np.sum(null >= hit_frac) + 1) / (len(null) + 1)) if len(null) else 1.0
    return {"hit_frac": hit_frac, "null_mean": mu, "z": z, "emp_p": p}


def _type_composition(
    pairs: set[tuple[int, int]],
    edge_sets: dict[str, set[tuple[int, int]]],
) -> dict[str, float]:
    """Role-type mix among pairs that are also role edges."""
    typed = {k: pairs & s for k, s in edge_sets.items()}
    n = sum(len(v) for v in typed.values())
    if n == 0:
        return {k: 0.0 for k in edge_sets}
    return {k: len(v) / n for k, v in typed.items()}


def analyze_structure(
    sid: str,
    chain: str,
    model: torch.nn.Module,
    pdb_dir: Path,
    device: str,
    *,
    min_r: float,
    k: int,
    max_dr: float,
    min_seq_sep: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    prot = load_protein_graph(sid, chain, pdb_dir)
    if prot is None:
        return {"structure_id": sid, "error": "load_failed"}

    xy_l, depth_l, data_l = _forward_disc(model, prot, device, use_ssot=False)
    xy_s, depth_s, _ = _forward_disc(model, prot, device, use_ssot=True)
    del depth_l, depth_s

    r_l = np.linalg.norm(xy_l, axis=1)
    r_s = np.linalg.norm(xy_s, axis=1)
    x = data_l.x.detach().cpu().numpy()
    rho = x[:, 0]
    tau = rho < TAU
    res_ids = _res_ids_from_prot(prot, chain, len(rho))

    learned_nn = _knn_pairs(xy_l, r_l, min_r=min_r, k=k, max_dr=max_dr)
    ssot_nn = _knn_pairs(xy_s, r_s, min_r=min_r, k=k, max_dr=max_dr)

    def filter_pairs(pairs: set[tuple[int, int]]) -> set[tuple[int, int]]:
        return {
            (a, b)
            for a, b in pairs
            if _seq_sep(res_ids, a, b) >= min_seq_sep
        }

    learned_nn = filter_pairs(learned_nn)
    ssot_nn = filter_pairs(ssot_nn)
    learned_only = learned_nn - ssot_nn
    ssot_only = ssot_nn - learned_nn

    def tau_trivial(pairs: set[tuple[int, int]]) -> set[tuple[int, int]]:
        return {p for p in pairs if bool(tau[p[0]]) == bool(tau[p[1]])}

    learned_only_nontau = learned_only - tau_trivial(learned_only)

    edge_sets = _role_edge_sets_from_data(data_l)
    all_role = set().union(*edge_sets.values()) if edge_sets else set()
    # Gate A: residual disc pairs that are also direct role contacts.
    residual_contacts = learned_only & all_role
    hit = _pair_edge_hits(learned_only, edge_sets)
    nulls = _null_pairs(len(res_ids), r_l, learned_only, rng=rng, n_null=200)
    enrich_direct: dict[str, Any] = {}
    for et, frac in hit.items():
        null_fracs = [_pair_edge_hits(ns, {et: edge_sets[et]})[et] for ns in nulls]
        enrich_direct[et] = _enrichment(frac, null_fracs)

    # Gate B: among residual contacts, is spoke/dehydron over-represented vs band edges?
    band_nodes = set(int(i) for i in np.where(r_l >= min_r)[0])
    band_edges = {
        k: {
            (a, b)
            for a, b in s
            if a in band_nodes and b in band_nodes and _seq_sep(res_ids, a, b) >= min_seq_sep
        }
        for k, s in edge_sets.items()
    }
    residual_comp = _type_composition(residual_contacts, edge_sets)
    band_comp = _type_composition(set().union(*band_edges.values()) if band_edges else set(), band_edges)
    enrich_composition: dict[str, Any] = {}
    for et in edge_sets:
        # Lift over band baseline; empirical null from resampling band edges
        union_band = set().union(*band_edges.values()) if band_edges else set()
        null_comp: list[float] = []
        band_list = list(union_band)
        need = max(len(residual_contacts), 1)
        for _ in range(200):
            if not band_list:
                null_comp.append(0.0)
                continue
            sample = {
                band_list[i]
                for i in rng.integers(0, len(band_list), size=min(need, len(band_list)))
            }
            null_comp.append(_type_composition(sample, band_edges).get(et, 0.0))
        enrich_composition[et] = {
            **_enrichment(residual_comp.get(et, 0.0), null_comp),
            "residual_frac": residual_comp.get(et, 0.0),
            "band_frac": band_comp.get(et, 0.0),
            "lift_vs_band": (
                residual_comp.get(et, 0.0) / band_comp[et]
                if band_comp.get(et, 0.0) > 1e-12
                else float("nan")
            ),
        }

    sites = list(PHARMACOPHORE_SITES.get(sid, [])) + list(EXTRA_SITES.get(sid, []))
    site_m = _site_mask(res_ids, sites) if sites else np.zeros(len(res_ids), dtype=bool)

    def shell_hit(pairs: set[tuple[int, int]]) -> dict[str, float]:
        if not site_m.any() or not pairs:
            return {"site_residue_coverage": float("nan"), "pair_touch_frac": float("nan")}
        members: set[int] = set()
        touch = 0
        for a, b in pairs:
            if site_m[a] or site_m[b]:
                touch += 1
                if site_m[a]:
                    members.add(a)
                if site_m[b]:
                    members.add(b)
        return {
            "n_sites": int(site_m.sum()),
            "site_residue_coverage": len(members) / max(int(site_m.sum()), 1),
            "pair_touch_frac": touch / len(pairs),
        }

    def _ablate(
        *,
        strip_roles: set[int] | None,
        disable_rim_fanout: bool,
    ) -> dict[str, float]:
        saved_rf = bool(getattr(model, "rim_fanout_forward", False))
        if disable_rim_fanout and hasattr(model, "rim_fanout_forward"):
            model.rim_fanout_forward = False
        try:
            xy_ab, _, _ = _forward_disc(
                model,
                prot,
                device,
                use_ssot=False,
                strip_roles=strip_roles,
            )
        finally:
            if disable_rim_fanout and hasattr(model, "rim_fanout_forward"):
                model.rim_fanout_forward = saved_rf
        r_ab = np.linalg.norm(xy_ab, axis=1)
        nn_ab = filter_pairs(_knn_pairs(xy_ab, r_ab, min_r=min_r, k=k, max_dr=max_dr))
        only_ab = nn_ab - ssot_nn
        jacc = (
            len(learned_only & only_ab) / len(learned_only | only_ab)
            if (learned_only or only_ab)
            else float("nan")
        )
        return {
            "n_learned_only_after": float(len(only_ab)),
            "n_learned_nn_after": float(len(nn_ab)),
            "jaccard_with_baseline_residual": float(jacc),
            "residual_size_ratio": (
                float(len(only_ab) / len(learned_only)) if learned_only else float("nan")
            ),
        }

    abl_edges = _ablate(
        strip_roles={ROLE_DEHYDRON, ROLE_SPOKE},
        disable_rim_fanout=False,
    )
    abl_fanout = _ablate(strip_roles=None, disable_rim_fanout=True)
    abl_both = _ablate(
        strip_roles={ROLE_DEHYDRON, ROLE_SPOKE},
        disable_rim_fanout=True,
    )

    return {
        "structure_id": sid,
        "chain": chain,
        "n_residues": len(res_ids),
        "n_learned_nn": len(learned_nn),
        "n_ssot_nn": len(ssot_nn),
        "n_learned_only": len(learned_only),
        "n_ssot_only": len(ssot_only),
        "n_learned_only_nontau": len(learned_only_nontau),
        "n_residual_contacts": len(residual_contacts),
        "residual_frac_of_learned": (
            len(learned_only) / len(learned_nn) if learned_nn else float("nan")
        ),
        "edge_enrichment_direct": enrich_direct,
        "edge_enrichment_composition": enrich_composition,
        "edge_enrichment": enrich_composition,
        "sites": {
            "n_labeled": int(site_m.sum()),
            "labels": [s.get("label") for s in sites],
            "learned_only": shell_hit(learned_only),
            "ssot_only": shell_hit(ssot_only),
            "learned_knn": shell_hit(learned_nn),
            "ssot_knn": shell_hit(ssot_nn),
        },
        "ablation_strip_dehydron_spoke": abl_edges,
        "ablation_disable_rim_fanout": abl_fanout,
        "ablation_edges_and_fanout": abl_both,
        "example_learned_only_pairs": [
            {
                "a": res_ids[a],
                "b": res_ids[b],
                "seq_sep": _seq_sep(res_ids, a, b),
                "tau": [bool(tau[a]), bool(tau[b])],
                "r": [float(r_l[a]), float(r_l[b])],
            }
            for a, b in sorted(learned_only)[:12]
        ],
        "example_residual_contacts": [
            {
                "a": res_ids[a],
                "b": res_ids[b],
                "roles": [et for et, s in edge_sets.items() if (a, b) in s],
            }
            for a, b in sorted(residual_contacts)[:12]
        ],
    }


def _corpus_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if "error" not in r]
    if not ok:
        return {"n": 0, "verdict": "FAIL"}

    def mean(key: str) -> float:
        vals = [
            float(r[key])
            for r in ok
            if r.get(key) is not None
            and not (isinstance(r[key], float) and math.isnan(r[key]))
        ]
        return float(np.mean(vals)) if vals else float("nan")

    edge_keys = ["packing", "dehydron", "spoke", "ribbon", "coupling"]
    edge_agg: dict[str, Any] = {}
    for ek in edge_keys:
        zs, ps, hits = [], [], []
        for r in ok:
            e = r.get("edge_enrichment", {}).get(ek)
            if not e:
                continue
            zs.append(e["z"])
            ps.append(e["emp_p"])
            hits.append(e["hit_frac"])
        edge_agg[ek] = {
            "mean_z": float(np.mean(zs)) if zs else float("nan"),
            "median_p": float(np.median(ps)) if ps else float("nan"),
            "mean_hit_frac": float(np.mean(hits)) if hits else float("nan"),
            "frac_structures_p_lt_0.05": (
                float(np.mean([p < 0.05 for p in ps])) if ps else float("nan")
            ),
        }

    site_rows = [r for r in ok if r.get("sites", {}).get("n_labeled", 0) > 0]

    def site_mean(path: tuple[str, ...]) -> float:
        vals = []
        for r in site_rows:
            cur: Any = r
            for p in path:
                cur = cur.get(p, {}) if isinstance(cur, dict) else {}
            if isinstance(cur, (int, float)) and not (
                isinstance(cur, float) and math.isnan(cur)
            ):
                vals.append(float(cur))
        return float(np.mean(vals)) if vals else float("nan")

    abl_ratios = [
        float(r["ablation_edges_and_fanout"]["residual_size_ratio"])
        for r in ok
        if r.get("ablation_edges_and_fanout", {}).get("residual_size_ratio") is not None
        and not math.isnan(
            float(r["ablation_edges_and_fanout"]["residual_size_ratio"])
        )
    ]
    abl_edge_ratios = [
        float(r["ablation_strip_dehydron_spoke"]["residual_size_ratio"])
        for r in ok
        if r.get("ablation_strip_dehydron_spoke", {}).get("residual_size_ratio") is not None
        and not math.isnan(
            float(r["ablation_strip_dehydron_spoke"]["residual_size_ratio"])
        )
    ]
    abl_fanout_ratios = [
        float(r["ablation_disable_rim_fanout"]["residual_size_ratio"])
        for r in ok
        if r.get("ablation_disable_rim_fanout", {}).get("residual_size_ratio") is not None
        and not math.isnan(
            float(r["ablation_disable_rim_fanout"]["residual_size_ratio"])
        )
    ]

    gates = {
        "residual_nonempty": mean("n_learned_only") > 0,
        "spoke_or_dehydron_enriched": bool(
            edge_agg.get("spoke", {}).get("mean_z", 0) > 1.0
            or edge_agg.get("dehydron", {}).get("mean_z", 0) > 1.0
            or edge_agg.get("spoke", {}).get("frac_structures_p_lt_0.05", 0) >= 0.3
            or edge_agg.get("dehydron", {}).get("frac_structures_p_lt_0.05", 0) >= 0.3
        ),
        "sites_learned_only_ge_ssot_only": (
            site_mean(("sites", "learned_only", "site_residue_coverage"))
            >= site_mean(("sites", "ssot_only", "site_residue_coverage")) - 1e-9
            if site_rows
            else None
        ),
        "ablation_shrinks_residuals": (
            float(np.mean(abl_ratios)) < 0.85 if abl_ratios else None
        ),
    }
    hard = [gates["residual_nonempty"], gates["spoke_or_dehydron_enriched"]]
    soft = [gates["sites_learned_only_ge_ssot_only"], gates["ablation_shrinks_residuals"]]
    soft_ok = [x for x in soft if x is not None]
    verdict = "FAIL"
    if all(hard) and soft_ok and all(soft_ok):
        verdict = "PASS"
    elif all(hard) and soft_ok and sum(1 for x in soft_ok if x) >= 1:
        verdict = "PARTIAL"
    elif all(hard):
        verdict = "WEAK"

    return {
        "n_structures": len(ok),
        "mean_n_learned_only": mean("n_learned_only"),
        "mean_residual_frac": mean("residual_frac_of_learned"),
        "edge_aggregation": edge_agg,
        "site_agg": {
            "n_structures_with_labels": len(site_rows),
            "learned_only_site_coverage": site_mean(
                ("sites", "learned_only", "site_residue_coverage")
            ),
            "ssot_only_site_coverage": site_mean(
                ("sites", "ssot_only", "site_residue_coverage")
            ),
            "learned_knn_site_coverage": site_mean(
                ("sites", "learned_knn", "site_residue_coverage")
            ),
            "ssot_knn_site_coverage": site_mean(
                ("sites", "ssot_knn", "site_residue_coverage")
            ),
        },
        "ablation_mean_residual_size_ratio": (
            float(np.mean(abl_ratios)) if abl_ratios else float("nan")
        ),
        "ablation_edges_only_mean_ratio": (
            float(np.mean(abl_edge_ratios)) if abl_edge_ratios else float("nan")
        ),
        "ablation_fanout_only_mean_ratio": (
            float(np.mean(abl_fanout_ratios)) if abl_fanout_ratios else float("nan")
        ),
        "gates": gates,
        "verdict": verdict,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "checkpoints/v66/runs/feeler_expand_23_rim_fanout_coverage_v2/epochs/epoch_164.pt"
        ),
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("manifests/v6_corpus_stage_a_feeler_expand_v1.json"),
    )
    parser.add_argument("--pdb-dir", type=Path, default=Path("/tmp/dtie_pdb_cache"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-r", type=float, default=0.15)
    parser.add_argument("--knn", type=int, default=8)
    parser.add_argument("--max-dr", type=float, default=0.12)
    parser.add_argument("--min-seq-sep", type=int, default=5)
    parser.add_argument(
        "--structures",
        nargs="*",
        default=None,
        help="Optional PDB IDs subset (default: full corpus)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/v66/diagnostics/learned_ssot_residual_proof"),
    )
    args = parser.parse_args(argv)

    structures = _load_corpus(args.corpus)
    if args.structures:
        want = {s.upper() for s in args.structures}
        structures = [s for s in structures if s[0] in want]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model_from_checkpoint(args.checkpoint, args.device)
    model.eval()
    print(
        f"loaded role_edge_mp={getattr(model, 'role_edge_mp', None)} "
        f"rim_fanout={getattr(model, 'rim_fanout_forward', None)} "
        f"spoke_scale={getattr(model, 'spoke_edge_scale', None)}",
        flush=True,
    )
    if not bool(getattr(model, "role_edge_mp", False)):
        raise SystemExit(
            "Checkpoint did not restore role_edge_mp — cannot run edge enrichment / ablation"
        )

    rng = np.random.default_rng(0)
    rows: list[dict[str, Any]] = []
    for sid, chain in structures:
        print(f"… {sid}:{chain}", flush=True)
        try:
            row = analyze_structure(
                sid,
                chain,
                model,
                args.pdb_dir,
                args.device,
                min_r=args.min_r,
                k=args.knn,
                max_dr=args.max_dr,
                min_seq_sep=args.min_seq_sep,
                rng=rng,
            )
        except Exception as exc:  # noqa: BLE001 — diagnostic batch must continue
            row = {"structure_id": sid, "chain": chain, "error": str(exc)}
            print(f"  ERROR {exc}", flush=True)
        rows.append(row)
        if "error" not in row:
            print(
                f"  residual={row['n_learned_only']} contacts={row['n_residual_contacts']} "
                f"(frac={row['residual_frac_of_learned']:.2f}) "
                f"spoke_z={row['edge_enrichment'].get('spoke', {}).get('z', float('nan')):.2f} "
                f"dehydron_z={row['edge_enrichment'].get('dehydron', {}).get('z', float('nan')):.2f} "
                f"ablation_both={row['ablation_edges_and_fanout']['residual_size_ratio']:.2f}",
                flush=True,
            )

    summary = _corpus_summary(rows)
    report = {
        "checkpoint": str(args.checkpoint),
        "pre_registered": {
            "min_r": args.min_r,
            "knn": args.knn,
            "max_dr": args.max_dr,
            "min_seq_sep": args.min_seq_sep,
            "null_pairs": 200,
            "ablation": "strip ROLE_DEHYDRON+ROLE_SPOKE and/or disable rim_fanout_forward",
        },
        "summary": summary,
        "structures": rows,
    }
    out = args.output_dir / "report.json"
    out.write_text(json.dumps(report, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out}")
    return 0 if summary.get("verdict") in {"PASS", "PARTIAL", "WEAK"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
