from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Tuple

import numpy as np

from hyperbolic_graph_utils import build_rips_adjacency, hyperbolic_pairwise_distance

logger_p3_v4 = logging.getLogger("DTIE_Phase3_v4")


def summarize_robust_interval(samples: Iterable[float]) -> Dict[str, float]:
    vals = np.asarray(list(samples), dtype=np.float64)
    if vals.size == 0:
        return {"min": 0.0, "median": 0.0, "max": 0.0}
    return {
        "min": float(np.min(vals)),
        "median": float(np.median(vals)),
        "max": float(np.max(vals)),
    }


def _safe_norm01(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.size == 0:
        return v
    v_min = float(np.min(v))
    v_max = float(np.max(v))
    span = v_max - v_min
    if span <= 1e-12:
        return np.zeros_like(v)
    return (v - v_min) / span


def _connected_components(
    adj: np.ndarray, active: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    n = adj.shape[0]
    labels = np.full(n, -1, dtype=np.int64)
    sizes: List[int] = []
    comp_id = 0

    for start in np.where(active)[0]:
        if labels[start] != -1:
            continue
        stack = [int(start)]
        labels[start] = comp_id
        size = 0
        while stack:
            u = stack.pop()
            size += 1
            nbrs = np.where(adj[u] > 0)[0]
            for v in nbrs:
                if not active[v] or labels[v] != -1:
                    continue
                labels[v] = comp_id
                stack.append(int(v))
        sizes.append(size)
        comp_id += 1

    if not sizes:
        return labels, np.zeros(0, dtype=np.int64)
    return labels, np.asarray(sizes, dtype=np.int64)


def _radius_clip(points: np.ndarray, c: float = 1.0, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(points, dtype=np.float64)
    max_norm = (1.0 / np.sqrt(c)) - eps
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    scale = np.ones_like(norms)
    mask = norms > max_norm
    scale[mask] = max_norm / np.clip(norms[mask], 1e-12, None)
    return x * scale


def _build_thresholds(depth_norm: np.ndarray, n_levels: int = 9) -> np.ndarray:
    if depth_norm.size == 0:
        return np.zeros(0, dtype=np.float64)
    q = np.linspace(0.1, 0.95, n_levels)
    thr = np.quantile(depth_norm, q)
    thr = np.unique(np.clip(thr, 0.0, 1.0))
    if thr.size == 0:
        return np.asarray([0.0], dtype=np.float64)
    return thr.astype(np.float64)


def _node_interval_from_filtration(
    node_idx: int,
    thresholds: np.ndarray,
    depth_norm: np.ndarray,
    adjacency: np.ndarray,
    min_component_size: int,
) -> Tuple[float, float, float, float, float]:
    active_flags: List[bool] = []
    comp_fracs: List[float] = []
    degree_vals: List[float] = []

    n = adjacency.shape[0]
    for thr in thresholds:
        active = depth_norm >= thr
        if not active[node_idx]:
            active_flags.append(False)
            comp_fracs.append(0.0)
            degree_vals.append(0.0)
            continue

        adj_thr = adjacency.copy()
        off = ~active
        adj_thr[off, :] = 0
        adj_thr[:, off] = 0

        labels, sizes = _connected_components(adj_thr, active)
        deg = float(np.sum(adj_thr[node_idx] > 0))
        degree_vals.append(deg)

        if labels[node_idx] < 0 or sizes.size == 0:
            active_flags.append(False)
            comp_fracs.append(0.0)
            continue

        comp_size = int(sizes[labels[node_idx]])
        support = comp_size >= int(max(1, min_component_size))
        active_flags.append(bool(support))
        comp_fracs.append(float(comp_size) / float(max(1, np.sum(active))))

    if not any(active_flags):
        return 0.0, 0.0, 0.0, 0.0, 0.0

    idx = np.where(np.asarray(active_flags, dtype=bool))[0]
    birth = float(thresholds[idx[0]])
    death = float(thresholds[idx[-1]])
    span = max(0.0, death - birth)
    comp_support = float(np.median(np.asarray(comp_fracs, dtype=np.float64)))
    degree_support = float(np.median(np.asarray(degree_vals, dtype=np.float64)))
    return birth, death, span, comp_support, degree_support


def _compute_adaptive_epsilon(dist_matrix: np.ndarray, quantile: float = 0.18) -> float:
    n = dist_matrix.shape[0]
    if n <= 1:
        return 0.0
    tri = dist_matrix[np.triu_indices(n, k=1)]
    tri = tri[np.isfinite(tri)]
    if tri.size == 0:
        return 0.0
    return float(np.quantile(tri, quantile))


def _row_entropy(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 2 or w.shape[1] <= 1:
        return (
            np.zeros(w.shape[0], dtype=np.float64)
            if w.ndim == 2
            else np.zeros(0, dtype=np.float64)
        )
    w = np.clip(w, 1e-12, None)
    w = w / np.clip(np.sum(w, axis=1, keepdims=True), 1e-12, None)
    h = -np.sum(w * np.log(w), axis=1)
    return h / np.log(w.shape[1])


def _doorway_index_map(phase2_result: Dict) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for d in phase2_result.get("doorways", []):
        out[str(d.get("id"))] = d
    return out


def _build_depth_map(condition: Dict) -> Dict[str, float]:
    ids = [str(r) for r in condition.get("residue_ids", [])]
    depth = np.asarray(condition.get("cone_depth", []), dtype=np.float64).reshape(-1)
    if len(ids) != int(depth.size):
        return {}
    depth_n = _safe_norm01(depth)
    return {rid: float(depth_n[i]) for i, rid in enumerate(ids)}


def _build_depth_raw_map(condition: Dict) -> Dict[str, float]:
    ids = [str(r) for r in condition.get("residue_ids", [])]
    depth = np.asarray(condition.get("cone_depth", []), dtype=np.float64).reshape(-1)
    if len(ids) != int(depth.size):
        return {}
    return {rid: float(depth[i]) for i, rid in enumerate(ids)}


def _normalize_maps_together(maps: List[Dict[str, float]]) -> List[Dict[str, float]]:
    all_vals: List[float] = []
    for m in maps:
        all_vals.extend(list(m.values()))
    if not all_vals:
        return maps

    v = np.asarray(all_vals, dtype=np.float64)
    v_min = float(np.min(v))
    v_max = float(np.max(v))
    span = v_max - v_min
    if span <= 1e-12:
        return [{k: 0.0 for k in m.keys()} for m in maps]

    out: List[Dict[str, float]] = []
    for m in maps:
        out.append({k: float((val - v_min) / span) for k, val in m.items()})
    return out


def _detect_contrast_tier(conditions: Dict[str, Dict]) -> str:
    keys = set(conditions.keys())
    if {"wt_inactive", "wt_active", "mut_inactive", "mut_active"}.issubset(keys):
        return "A"
    if {"wt", "mut"}.issubset(keys):
        return "B"
    return "C"


def _compute_contrast_features(
    conditions: Dict[str, Dict],
    anchor_residue_ids: List[str],
) -> Dict[str, np.ndarray]:
    tier = _detect_contrast_tier(conditions)

    if "gdp" in conditions and "gtp" in conditions:
        gdp_map = _build_depth_map(conditions["gdp"])
        gtp_map = _build_depth_map(conditions["gtp"])
        state_stability_c = np.asarray(
            [
                float(
                    np.clip(
                        1.0 - abs(gdp_map.get(rid, 0.0) - gtp_map.get(rid, 0.0)),
                        0.0,
                        1.0,
                    )
                )
                for rid in anchor_residue_ids
            ],
            dtype=np.float64,
        )
    else:
        state_stability_c = np.full(len(anchor_residue_ids), 0.5, dtype=np.float64)

    if tier == "A":
        wt_i_raw = _build_depth_raw_map(conditions["wt_inactive"])
        wt_a_raw = _build_depth_raw_map(conditions["wt_active"])
        mut_i_raw = _build_depth_raw_map(conditions["mut_inactive"])
        mut_a_raw = _build_depth_raw_map(conditions["mut_active"])
        wt_i, wt_a, mut_i, mut_a = _normalize_maps_together(
            [wt_i_raw, wt_a_raw, mut_i_raw, mut_a_raw]
        )

        gp_vals = []
        st_vals = []
        for rid in anchor_residue_ids:
            d_wi = wt_i.get(rid, 0.0)
            d_wa = wt_a.get(rid, 0.0)
            d_mi = mut_i.get(rid, 0.0)
            d_ma = mut_a.get(rid, 0.0)

            delta_i = d_mi - d_wi
            delta_a = d_ma - d_wa
            mean_delta = 0.5 * (delta_i + delta_a)
            consistency = float(np.clip(1.0 - abs(delta_i - delta_a), 0.0, 1.0))

            gp = float(np.clip(0.5 + 0.5 * mean_delta, 0.0, 1.0))
            gp = float(np.clip(0.75 * gp + 0.25 * consistency, 0.0, 1.0))
            st = float(np.clip(1.0 - abs(d_ma - d_mi), 0.0, 1.0))

            gp_vals.append(gp)
            st_vals.append(st)

        return {
            "tier": np.asarray([tier] * len(anchor_residue_ids), dtype=object),
            "genotype_persistence": np.asarray(gp_vals, dtype=np.float64),
            "state_stability": np.asarray(st_vals, dtype=np.float64),
        }

    if tier == "B":
        wt_raw = _build_depth_raw_map(conditions["wt"])
        mut_raw = _build_depth_raw_map(conditions["mut"])
        wt, mut = _normalize_maps_together([wt_raw, mut_raw])

        gp_vals = []
        st_vals = []
        for i, rid in enumerate(anchor_residue_ids):
            d_w = wt.get(rid, 0.0)
            d_m = mut.get(rid, 0.0)
            gp = float(np.clip(0.5 + 0.5 * (d_m - d_w), 0.0, 1.0))
            gp_vals.append(gp)
            st_vals.append(float(state_stability_c[i]))

        return {
            "tier": np.asarray([tier] * len(anchor_residue_ids), dtype=object),
            "genotype_persistence": np.asarray(gp_vals, dtype=np.float64),
            "state_stability": np.asarray(st_vals, dtype=np.float64),
        }

    return {
        "tier": np.asarray([tier] * len(anchor_residue_ids), dtype=object),
        "genotype_persistence": np.full(len(anchor_residue_ids), 0.5, dtype=np.float64),
        "state_stability": state_stability_c,
    }


def execute_phase_3_v4_cone_persistence(
    phase1_v4_result: Dict,
    phase2_result: Dict,
) -> Dict:
    """Cone-space shell filtration with uncertainty-halo robustness.

    - Builds a residue-level hyperbolic Rips graph (GDP condition) as base graph.
    - Sweeps outer-shell depth thresholds and tracks doorway persistence support.
    - Runs local uncertainty perturbations for high-epistemic residues to estimate
      robust persistence intervals and fragility.
    """
    conditions = phase1_v4_result["conditions"]
    gdp = conditions["gdp"]
    if "gtp" not in conditions:
        raise ValueError("Phase 3 v4 requires conditions.gtp")
    gtp = conditions["gtp"]

    residue_ids: List[str] = list(gdp["residue_ids"])
    ca_coords = np.asarray(gdp["ca_coords"], dtype=np.float64)
    if "projections" not in gdp:
        raise ValueError(
            "Phase 3 v4 requires gdp.projections; projection fallback is disabled"
        )
    projections = np.asarray(gdp["projections"], dtype=np.float64)
    required_gdp = [
        "cone_depth",
        "cone_width",
        "expert_weights",
        "audit_trail",
        "epistemic",
        "total_uncertainty",
    ]
    missing_gdp = [k for k in required_gdp if k not in gdp]
    if missing_gdp:
        raise ValueError(f"Phase 3 v4 requires gdp fields: {missing_gdp}")

    depth = np.asarray(gdp["cone_depth"], dtype=np.float64).reshape(-1)
    depth_gtp = np.asarray(gtp["cone_depth"], dtype=np.float64).reshape(-1)
    cone_width = np.asarray(gdp["cone_width"], dtype=np.float64).reshape(-1)
    expert_weights = np.asarray(gdp["expert_weights"], dtype=np.float64)
    audit = gdp["audit_trail"]
    if not isinstance(audit, dict):
        raise ValueError("Phase 3 v4 requires gdp.audit_trail to be a dict")
    epistemic = np.asarray(gdp["epistemic"], dtype=np.float64).reshape(-1)
    total_unc = np.asarray(gdp["total_uncertainty"], dtype=np.float64).reshape(-1)

    if projections.shape[0] != len(residue_ids):
        raise ValueError("Phase 3 v4: projection/residue size mismatch")

    id_to_idx = {rid: i for i, rid in enumerate(residue_ids)}
    depth_norm = _safe_norm01(depth)
    depth_gtp_norm = (
        _safe_norm01(depth_gtp)
        if depth_gtp.size == depth.size
        else np.zeros_like(depth_norm)
    )
    epi_norm = _safe_norm01(epistemic)
    unc_norm = _safe_norm01(total_unc)
    cone_width_norm = (
        _safe_norm01(cone_width)
        if cone_width.size == depth.size
        else np.zeros_like(depth_norm)
    )
    routing_entropy = _row_entropy(expert_weights)
    routing_entropy_median = (
        float(np.median(routing_entropy)) if routing_entropy.size else 0.0
    )

    proj_applied_fraction = float(audit.get("projection_applied_fraction", 0.0) or 0.0)
    projection_depth_conditioning = bool(audit.get("depth_conditioning_enabled", False))

    epsilon_quantile = float(
        np.clip(0.14 + 0.12 * float(np.median(cone_width_norm)), 0.12, 0.30)
    )
    max_neighbors = int(
        np.clip(round(6 + 8 * float(np.median(cone_width_norm))), 6, 14)
    )
    halo_quantile = float(
        np.clip(0.75 + 0.15 * (1.0 - routing_entropy_median), 0.75, 0.95)
    )
    n_replicates = int(np.clip(round(8 + 8 * proj_applied_fraction), 8, 20))
    base_scale = 0.01 + 0.06 * float(
        np.median(cone_width_norm * np.clip(unc_norm, 0.05, 1.0))
    )
    perturb_scale = float(np.clip(base_scale, 0.01, 0.08))
    if projection_depth_conditioning:
        perturb_scale = float(np.clip(perturb_scale * 1.15, 0.01, 0.08))

    contrast = _compute_contrast_features(conditions, residue_ids)
    genotype_persistence = contrast["genotype_persistence"]
    state_stability = contrast["state_stability"]
    contrast_tier = str(contrast["tier"][0]) if contrast["tier"].size else "C"

    dist_matrix = hyperbolic_pairwise_distance(projections)
    epsilon = _compute_adaptive_epsilon(dist_matrix, quantile=epsilon_quantile)
    base_adj = build_rips_adjacency(
        dist_matrix, epsilon=epsilon, max_neighbors=max_neighbors
    )

    thresholds = _build_thresholds(depth_norm, n_levels=9)
    doorway_meta = _doorway_index_map(phase2_result)

    # Halo nodes: only these are perturbed during robust recomputation.
    halo_mask = epi_norm >= float(np.quantile(epi_norm, halo_quantile))
    halo_idx = np.where(halo_mask)[0]

    rng = np.random.default_rng(42)

    # Precompute replicate adjacency matrices.
    replicate_adjs: List[np.ndarray] = []
    for _ in range(n_replicates):
        noisy_proj = projections.copy()
        if halo_idx.size > 0:
            noise = rng.normal(size=(halo_idx.size, projections.shape[1]))
            noise = noise * (
                perturb_scale * np.clip(unc_norm[halo_idx][:, None], 0.05, 1.0)
            )
            noisy_proj[halo_idx] = noisy_proj[halo_idx] + noise
            noisy_proj = _radius_clip(noisy_proj, c=1.0)
        d_rep = hyperbolic_pairwise_distance(noisy_proj)
        eps_rep = _compute_adaptive_epsilon(d_rep, quantile=epsilon_quantile)
        replicate_adjs.append(
            build_rips_adjacency(d_rep, epsilon=eps_rep, max_neighbors=max_neighbors)
        )

    lifted_sites: List[Dict] = []

    for doorway in phase2_result.get("doorways", []):
        rid = str(doorway.get("id"))
        idx = id_to_idx.get(rid)
        if idx is None:
            continue

        birth, death, span, comp_support, degree_support = (
            _node_interval_from_filtration(
                node_idx=idx,
                thresholds=thresholds,
                depth_norm=depth_norm,
                adjacency=base_adj,
                min_component_size=2,
            )
        )

        span_samples = [span]
        for adj_rep in replicate_adjs:
            _, _, span_rep, _, _ = _node_interval_from_filtration(
                node_idx=idx,
                thresholds=thresholds,
                depth_norm=depth_norm,
                adjacency=adj_rep,
                min_component_size=2,
            )
            span_samples.append(float(span_rep))

        robust = summarize_robust_interval(span_samples)
        fragility = float(np.clip(robust["max"] - robust["min"], 0.0, 1.0))
        uncertainty_robustness = float(1.0 - fragility)

        rho = doorway_meta.get(rid, {}).get("rho")
        if rho is None:
            leak_intensity = float(depth_norm[idx])
        else:
            leak_intensity = float(np.clip((13.0 - float(rho)) / 13.0, 0.0, 1.0))

        lifted_sites.append(
            {
                "residue_id": rid,
                "barycenter_xyz": ca_coords[idx],
                "birth": birth,
                "death": death,
                "persistence_span": span,
                "depth_norm": float(depth_norm[idx]),
                "depth_norm_gtp": (
                    float(depth_gtp_norm[idx]) if idx < depth_gtp_norm.size else 0.0
                ),
                "state_depth_delta": (
                    float(depth_norm[idx] - depth_gtp_norm[idx])
                    if idx < depth_gtp_norm.size
                    else 0.0
                ),
                "epistemic_norm": float(epi_norm[idx]),
                "uncertainty_norm": float(unc_norm[idx]),
                "genotype_persistence": float(genotype_persistence[idx]),
                "state_stability": float(state_stability[idx]),
                "contrast_tier": contrast_tier,
                "robust_interval": robust,
                "fragility_index": fragility,
                "uncertainty_robustness": uncertainty_robustness,
                "component_support": comp_support,
                "median_degree": degree_support,
                "leak_intensity": leak_intensity,
                "rho": rho,
                "num_vertices": 1,
            }
        )

    logger_p3_v4.info(
        "Phase 3 v4: "
        f"{len(lifted_sites)} source sites | epsilon={epsilon:.4f} | "
        f"halo_nodes={int(halo_idx.size)}"
    )
    return {
        "lifted_sites": lifted_sites,
        "method": "cone_depth_rips_filtration_uncertainty_halo",
        "contrast_tier": contrast_tier,
        "graph_summary": {
            "epsilon": epsilon,
            "epsilon_quantile": epsilon_quantile,
            "max_neighbors": max_neighbors,
            "n_nodes": int(len(residue_ids)),
            "n_edges": int(np.sum(base_adj) // 2),
            "thresholds": thresholds.tolist(),
            "halo_fraction": float(np.mean(halo_mask)) if halo_mask.size else 0.0,
            "halo_quantile": halo_quantile,
            "n_replicates": n_replicates,
            "perturb_scale": perturb_scale,
            "routing_entropy_median": routing_entropy_median,
            "projection_applied_fraction": proj_applied_fraction,
            "depth_conditioning_enabled": projection_depth_conditioning,
            "residue_ids": residue_ids,
            "reference_adjacency": base_adj,
        },
    }
