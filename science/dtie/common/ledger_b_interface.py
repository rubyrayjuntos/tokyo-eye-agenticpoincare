"""Ledger B — interface alignment scoring (pre-registered sets only)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def expand_conditional_ranges(
    conditional: Mapping[str, Any] | None,
    present_resseqs: set[int],
) -> list[int]:
    """Expand SH2/SH3-style ranges only when endpoints exist in the deposit."""
    out: list[int] = []
    if not conditional:
        return out
    for _name, spec in conditional.items():
        if isinstance(spec, dict) and "range" in spec:
            lo, hi = int(spec["range"][0]), int(spec["range"][1])
            if lo in present_resseqs or hi in present_resseqs:
                out.extend(r for r in range(lo, hi + 1) if r in present_resseqs)
        elif isinstance(spec, list):
            out.extend(int(r) for r in spec if int(r) in present_resseqs)
    return out


def _apply_auth_offset(resseqs: Sequence[int], offset: int) -> list[int]:
    return [int(r) + int(offset) for r in resseqs]


def resolve_interface_set(
    structure_spec: Mapping[str, Any],
    present_resseqs: set[int],
) -> dict[str, Any]:
    """Apply mapping rule: literature→auth offset, drop missing, append conditionals."""
    offset = int(structure_spec.get("literature_to_auth_offset") or 0)
    primary_lit = [int(r) for r in structure_spec.get("primary_resseqs") or []]
    primary = _apply_auth_offset(primary_lit, offset)
    missing = sorted(r for r in primary if r not in present_resseqs)
    present_primary = [r for r in primary if r in present_resseqs]

    # Conditionals: shift ranges/lists by the same offset, then expand if present.
    cond_raw = structure_spec.get("conditional_resseqs_if_present") or {}
    cond_shifted: dict[str, Any] = {}
    for name, spec in cond_raw.items():
        if isinstance(spec, dict) and "range" in spec:
            lo, hi = int(spec["range"][0]), int(spec["range"][1])
            cond_shifted[name] = {"range": [lo + offset, hi + offset]}
        elif isinstance(spec, list):
            cond_shifted[name] = [int(r) + offset for r in spec]
    conditional = expand_conditional_ranges(cond_shifted, present_resseqs)

    interface = sorted(set(present_primary) | set(conditional))
    secondary_lit = [
        int(r) for r in structure_spec.get("secondary_resseqs_report_only") or []
    ]
    secondary = [
        r
        for r in _apply_auth_offset(secondary_lit, offset)
        if r in present_resseqs
    ]
    return {
        "interface_resseqs": interface,
        "missing_from_deposit": missing,
        "conditional_included": conditional,
        "secondary_resseqs": secondary,
        "n_interface": len(interface),
        "n_primary_requested": len(primary),
        "literature_to_auth_offset": offset,
        "primary_literature_resseqs": primary_lit,
    }


def top_k_mask_by_score(scores: Sequence[float], *, k_frac: float = 0.10) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    n = s.size
    k = max(1, int(np.ceil(float(k_frac) * n)))
    order = np.argsort(-np.nan_to_num(s, nan=-np.inf))
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask


def ranked_hub_inventory(
    out_effect: Sequence[float],
    resseq_to_index: Mapping[int, int],
    *,
    k_frac: float = 0.10,
) -> list[dict[str, Any]]:
    """Top-k% residues by out_effect, ranked 1..k (auth_seq keys)."""
    oe = np.asarray(out_effect, dtype=np.float64).reshape(-1)
    index_to_resseq = {i: rs for rs, i in resseq_to_index.items() if 0 <= i < oe.size}
    k = max(1, int(np.ceil(float(k_frac) * oe.size)))
    order = np.argsort(-np.nan_to_num(oe, nan=-np.inf))
    rows: list[dict[str, Any]] = []
    for rank, i in enumerate(order[:k], start=1):
        i = int(i)
        rows.append(
            {
                "rank": rank,
                "graph_index": i,
                "auth_resseq": int(index_to_resseq[i]) if i in index_to_resseq else None,
                "out_effect": float(oe[i]),
            }
        )
    return rows


def interface_alignment_score(
    out_effect: Sequence[float],
    resseq_to_index: Mapping[int, int],
    interface_resseqs: Sequence[int],
    *,
    k_frac: float = 0.10,
    recall_threshold: float = 0.25,
    enrichment_threshold: float = 1.25,
) -> dict[str, Any]:
    """Recall@top-k% flow on pre-registered interface; enrichment secondary."""
    oe = np.asarray(out_effect, dtype=np.float64).reshape(-1)
    hubs = ranked_hub_inventory(oe, resseq_to_index, k_frac=k_frac)
    hub_resseqs = {h["auth_resseq"] for h in hubs if h["auth_resseq"] is not None}
    hub_resseqs_sorted = [
        int(h["auth_resseq"]) for h in hubs if h["auth_resseq"] is not None
    ]

    iface = [int(r) for r in interface_resseqs if int(r) in resseq_to_index]
    if not iface:
        return {
            "pass": False,
            "recall_at_top_k": float("nan"),
            "enrichment": float("nan"),
            "n_interface": 0,
            "n_hub_overlap": 0,
            "hub_overlap_resseqs": [],
            "hub_resseqs": hub_resseqs_sorted,
            "hub_outside_interface": hub_resseqs_sorted,
            "hubs_ranked": hubs,
            "reason": "empty_interface_after_mapping",
        }

    overlap = sorted(set(iface) & hub_resseqs)
    outside = [r for r in hub_resseqs_sorted if r not in set(iface)]
    recall = len(overlap) / float(len(iface))

    iface_idx = [resseq_to_index[r] for r in iface]
    iface_oe = oe[iface_idx]
    complement_mask = np.ones(oe.size, dtype=bool)
    for i in iface_idx:
        complement_mask[i] = False
    comp_oe = oe[complement_mask]
    mean_i = float(np.nanmean(iface_oe)) if iface_oe.size else float("nan")
    mean_c = float(np.nanmean(comp_oe)) if comp_oe.size else float("nan")
    enrichment = (
        mean_i / mean_c
        if np.isfinite(mean_i) and np.isfinite(mean_c) and mean_c > 1e-12
        else float("nan")
    )

    primary_ok = bool(np.isfinite(recall) and recall >= float(recall_threshold))
    return {
        "pass": primary_ok,
        "recall_at_top_k": float(recall),
        "recall_threshold": float(recall_threshold),
        "enrichment": enrichment,
        "enrichment_threshold": float(enrichment_threshold),
        "enrichment_pass_secondary": bool(
            np.isfinite(enrichment) and enrichment > float(enrichment_threshold)
        ),
        "k_frac": float(k_frac),
        "n_interface": len(iface),
        "n_hubs": len(hub_resseqs_sorted),
        "n_hub_overlap": len(overlap),
        "hub_overlap_resseqs": overlap,
        "hub_resseqs": hub_resseqs_sorted,
        "hub_outside_interface": outside,
        "hubs_ranked": hubs,
        "mean_out_effect_interface": mean_i,
        "mean_out_effect_complement": mean_c,
    }


__all__ = [
    "expand_conditional_ranges",
    "interface_alignment_score",
    "ranked_hub_inventory",
    "resolve_interface_set",
    "top_k_mask_by_score",
]
