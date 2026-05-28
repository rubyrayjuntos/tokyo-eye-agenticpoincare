from __future__ import annotations

import logging
from collections import deque
from typing import Dict, List, Optional

import numpy as np

from .contracts import Phase4Output
from .source_leak_scoring import (
    candidate_priority,
    propagation_score,
    source_leak_score,
)

logger_p4_v4 = logging.getLogger("DTIE_Phase4_v4")


def _clamp01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _choose_effector_indices(
    effector_sites: List[int],
    residue_ids: List[str],
) -> List[tuple[int, int]]:
    if not residue_ids:
        raise ValueError("Phase 4 v4 requires non-empty residue_ids")
    n = len(residue_ids)
    idx: List[tuple[int, int]] = []
    for e in effector_sites:
        try:
            j = int(e)
        except Exception:
            raise ValueError(f"Invalid effector index value: {e}")
        if not (0 <= j < n):
            raise ValueError(f"Effector index out of range: {j} for n={n}")
        idx.append((j, j))
    if not idx:
        raise ValueError("At least one effector index is required")
    return idx


def _shortest_path_unweighted(
    adj: np.ndarray, start: int, goal: int
) -> Optional[List[int]]:
    if start == goal:
        return [start]

    n = adj.shape[0]
    prev = np.full(n, -1, dtype=np.int64)
    seen = np.zeros(n, dtype=bool)
    q: deque[int] = deque([start])
    seen[start] = True

    while q:
        u = q.popleft()
        for v in np.where(adj[u] > 0)[0]:
            if seen[v]:
                continue
            seen[v] = True
            prev[v] = u
            if int(v) == goal:
                path = [goal]
                cur = goal
                while prev[cur] != -1:
                    cur = int(prev[cur])
                    path.append(cur)
                path.reverse()
                return path
            q.append(int(v))
    return None


def _path_coverage(path_idx: Optional[List[int]], adj: np.ndarray) -> float:
    if not path_idx:
        return 0.0
    n = max(1, adj.shape[0])
    return _clamp01(len(path_idx) / n)


def _path_conductance(path_idx: Optional[List[int]], component_support: float) -> float:
    if not path_idx or len(path_idx) <= 1:
        return 0.0
    hop_conductance = 1.0 / float(len(path_idx) - 1)
    return _clamp01(0.7 * hop_conductance + 0.3 * _clamp01(component_support))


def execute_phase_4_v4_leak_flow(
    phase3_v4_result: Dict,
    effector_sites: List[int],
) -> List[Phase4Output]:
    """Source-first leak flow mapping using robust cone persistence features."""
    if "graph_summary" not in phase3_v4_result:
        raise ValueError("Phase 4 v4 requires graph_summary")
    if "lifted_sites" not in phase3_v4_result:
        raise ValueError("Phase 4 v4 requires lifted_sites")

    graph_summary = phase3_v4_result["graph_summary"]
    if "residue_ids" not in graph_summary:
        raise ValueError("Phase 4 v4 requires graph_summary.residue_ids")
    residue_ids = list(graph_summary["residue_ids"])
    id_to_idx = {rid: i for i, rid in enumerate(residue_ids)}

    reference_adjacency = graph_summary.get("reference_adjacency")
    if reference_adjacency is None:
        raise ValueError("Phase 4 v4 requires graph_summary.reference_adjacency")
    else:
        reference_adjacency = np.asarray(reference_adjacency, dtype=np.int8)

    if not effector_sites:
        raise ValueError("Phase 4 v4 requires explicit effector_sites")
    effector_idx = _choose_effector_indices(effector_sites, residue_ids)

    records: List[Phase4Output] = []
    for site in phase3_v4_result["lifted_sites"]:
        required_site = [
            "residue_id",
            "genotype_persistence",
            "state_stability",
            "persistence_span",
            "uncertainty_robustness",
            "leak_intensity",
            "component_support",
            "barycenter_xyz",
            "fragility_index",
            "contrast_tier",
        ]
        missing_site = [k for k in required_site if k not in site]
        if missing_site:
            raise ValueError(f"Phase 4 v4 site missing fields: {missing_site}")

        rid = site["residue_id"]
        idx = id_to_idx.get(rid)

        genotype_persistence = _clamp01(site["genotype_persistence"])
        state_stability = _clamp01(site["state_stability"])
        depth_persistence = _clamp01(site["persistence_span"])
        uncertainty_robustness = _clamp01(site["uncertainty_robustness"])
        leak_intensity = _clamp01(site["leak_intensity"])

        src = source_leak_score(
            genotype_persistence=genotype_persistence,
            state_stability=state_stability,
            depth_persistence=depth_persistence,
            uncertainty_robustness=uncertainty_robustness,
            leak_intensity=leak_intensity,
        )

        for target_idx, target in effector_idx:

            path_idx = None
            if idx is not None and 0 <= idx < reference_adjacency.shape[0]:
                path_idx = _shortest_path_unweighted(
                    reference_adjacency, idx, target_idx
                )

            component_support = _clamp01(site["component_support"])
            conductance = _path_conductance(path_idx, component_support)
            path_cov = _path_coverage(path_idx, reference_adjacency)
            prop = propagation_score(
                conductance=conductance,
                source_support=src,
                path_coverage=path_cov,
                uncertainty_robustness=uncertainty_robustness,
            )
            prio = candidate_priority(src, prop)

            if path_idx:
                pathway = [residue_ids[i] for i in path_idx]
            else:
                pathway = [rid, f"effector_{target}"]

            records.append(
                Phase4Output(
                    doorway_node=rid,
                    target=target,
                    coupling_strength=prio,
                    r_eff=1.0 / max(prio, 1e-6),
                    pathway=pathway,
                    doorway_xyz=(
                        site["barycenter_xyz"].tolist()
                        if hasattr(site["barycenter_xyz"], "tolist")
                        else site["barycenter_xyz"]
                    ),
                    source_score=src,
                    propagation_score=prop,
                    candidate_priority=prio,
                    fragility_index=site["fragility_index"],
                    uncertainty_robustness=uncertainty_robustness,
                    persistence_span=depth_persistence,
                    genotype_persistence=genotype_persistence,
                    state_stability=state_stability,
                    contrast_tier=site["contrast_tier"],
                )
            )

    records.sort(key=lambda x: -x.candidate_priority)
    logger_p4_v4.info(f"Phase 4 v4: {len(records)} source-flow pathways ranked")
    return records
