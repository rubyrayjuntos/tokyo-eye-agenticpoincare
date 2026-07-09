"""Residue-first aleatoric diagnostics — site decisions vs corpus health monitors.

Aleatoric uncertainty from the NIG head is a **per-residue** (per-dehydron-site)
quantity: local probability mass that remains accessible in the observed ensemble.
Corpus-level means or global ``std(ale)`` gates smear rim-localized signal across
stable core residues and are **not** valid site-certification criteria.

Use:
- **Local triage** — high aleatoric + rim-localized (high disc_r) + low graph clustering
- **Global summaries** — dataset health, calibration shape, active-learning triage only

See ``docs/audit/GNNV7_SUCCESS_CRITERIA.md`` § Residue-first aleatoric doctrine.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

# Target band for fraction of residues with ale > t_ale within a protein (validation).
HIGH_ALE_FRACTION_VALIDATION_BAND = (0.12, 0.20)
# Legacy absolute floor (training-health scale ~0.05); use only with explicit --t-ale.
DEFAULT_T_ALE_ABSOLUTE = 0.05
# Default: corpus-relative percentile — ~10% of residues above threshold at corpus level.
DEFAULT_T_ALE_PERCENTILE = 90.0
# Back-compat alias for callers passing an absolute threshold explicitly.
DEFAULT_T_ALE = DEFAULT_T_ALE_ABSOLUTE


def resolve_t_ale(
    rows: Sequence[dict[str, Any]],
    *,
    t_ale: float | None = None,
    t_ale_percentile: float = DEFAULT_T_ALE_PERCENTILE,
) -> tuple[float, dict[str, Any]]:
    """Resolve high-aleatoric threshold — percentile by default, absolute when ``t_ale`` set."""
    if t_ale is not None:
        return float(t_ale), {
            "mode": "absolute",
            "value": float(t_ale),
            "percentile": None,
        }
    if not rows:
        return DEFAULT_T_ALE_ABSOLUTE, {
            "mode": "percentile",
            "percentile": t_ale_percentile,
            "value": DEFAULT_T_ALE_ABSOLUTE,
            "reason": "empty_rows_fallback",
        }
    ale = _col(rows, "aleatoric")
    finite = ale[np.isfinite(ale)]
    if finite.size == 0:
        return DEFAULT_T_ALE_ABSOLUTE, {
            "mode": "percentile",
            "percentile": t_ale_percentile,
            "value": DEFAULT_T_ALE_ABSOLUTE,
            "reason": "no_finite_aleatoric_fallback",
        }
    value = float(np.percentile(finite, t_ale_percentile))
    return value, {
        "mode": "percentile",
        "percentile": t_ale_percentile,
        "value": value,
    }


def _rows_by_protein(
    rows: Sequence[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        sid = str(row.get("structure_id", "")).lower()
        chain = str(row.get("chain", "A"))
        grouped.setdefault((sid, chain), []).append(row)
    return grouped


def _col(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in rows], dtype=np.float64)


def _finite_col(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray | None:
    if not rows or key not in rows[0]:
        return None
    vals = _col(rows, key)
    if not np.any(np.isfinite(vals)):
        return None
    return vals


def corpus_aleatoric_histogram(
    rows: Sequence[dict[str, Any]],
    *,
    n_bins: int = 40,
    t_ale: float | None = None,
    t_ale_percentile: float = DEFAULT_T_ALE_PERCENTILE,
) -> dict[str, Any]:
    """Residue-level aleatoric distribution — expect right-skew when calibrated."""
    if not rows:
        return {"ok": False, "reason": "empty"}

    ale = _col(rows, "aleatoric")
    finite = np.isfinite(ale)
    ale = ale[finite]
    if ale.size == 0:
        return {"ok": False, "reason": "no_finite_aleatoric"}

    resolved_t_ale, t_ale_meta = resolve_t_ale(
        rows,
        t_ale=t_ale,
        t_ale_percentile=t_ale_percentile,
    )

    mean = float(np.mean(ale))
    std = float(np.std(ale))
    p50, p75, p90, p95, p99 = [float(np.percentile(ale, q)) for q in (50, 75, 90, 95, 99)]
    skew = float(_skewness(ale))
    # Healthy calibration: bulk low, long right tail (positive skew).
    right_skew_ok = skew > 0.25 and p90 > mean + 0.5 * std

    counts, edges = np.histogram(ale, bins=n_bins)
    hist = {
        "bin_edges": [float(x) for x in edges.tolist()],
        "counts": [int(x) for x in counts.tolist()],
    }

    return {
        "ok": True,
        "n_residues": int(ale.size),
        "mean": mean,
        "std": std,
        "min": float(np.min(ale)),
        "max": float(np.max(ale)),
        "p50": p50,
        "p75": p75,
        "p90": p90,
        "p95": p95,
        "p99": p99,
        "skewness": skew,
        "right_skew_ok": right_skew_ok,
        "fraction_above_t_ale": float(np.mean(ale > resolved_t_ale)),
        "t_ale": resolved_t_ale,
        "t_ale_resolution": t_ale_meta,
        "histogram": hist,
        "role": "dataset_health_monitor",
    }


def per_protein_aleatoric_summary(
    rows: Sequence[dict[str, Any]],
    *,
    structure_id: str,
    chain: str,
    t_ale: float = DEFAULT_T_ALE,
    gene: str | None = None,
    fold_id: str | None = None,
) -> dict[str, Any]:
    """Per-protein aggregates for active learning and dehydron burden tables."""
    if not rows:
        return {
            "structure_id": structure_id.lower(),
            "chain": chain,
            "n_residues": 0,
            "ok": False,
        }

    ale = _col(rows, "aleatoric")
    high_mask = ale > t_ale
    n = len(rows)
    n_high = int(high_mask.sum())

    dehyd_mask = np.array(
        [float(r.get("tau_flag", 0.0)) >= 0.5 for r in rows],
        dtype=bool,
    )
    dehyd_high = int((high_mask & dehyd_mask).sum()) if dehyd_mask.any() else 0
    dehyd_n = int(dehyd_mask.sum())

    summary: dict[str, Any] = {
        "structure_id": structure_id.lower(),
        "chain": chain,
        "gene": gene,
        "fold_id": fold_id,
        "n_residues": n,
        "t_ale": t_ale,
        "aleatoric_mean": float(np.mean(ale)),
        "aleatoric_std": float(np.std(ale)),
        "aleatoric_variance": float(np.var(ale)),
        "aleatoric_max": float(np.max(ale)),
        "aleatoric_p95": float(np.percentile(ale, 95)),
        "high_aleatoric_count": n_high,
        "high_aleatoric_fraction": float(n_high / n) if n else 0.0,
        "dehydron_high_aleatoric_count": dehyd_high,
        "dehydron_high_aleatoric_fraction": (
            float(dehyd_high / dehyd_n) if dehyd_n else float("nan")
        ),
        "n_dehydron_flagged": dehyd_n,
        "fraction_in_validation_band": _in_band(
            float(n_high / n) if n else 0.0,
            HIGH_ALE_FRACTION_VALIDATION_BAND,
        ),
        "ok": True,
        "role": "active_learning_triage",
    }
    return summary


def rank_proteins_for_active_learning(
    protein_summaries: Sequence[dict[str, Any]],
    *,
    primary_key: str = "aleatoric_max",
) -> list[dict[str, Any]]:
    """Rank structures by max residue aleatoric — uncertainty-guided acquisition list."""
    valid = [p for p in protein_summaries if p.get("ok") and p.get("n_residues", 0) > 0]
    ranked = sorted(
        valid,
        key=lambda p: float(p.get(primary_key, 0.0)),
        reverse=True,
    )
    out: list[dict[str, Any]] = []
    for rank, item in enumerate(ranked, start=1):
        out.append({**item, "active_learning_rank": rank})
    return out


def investigation_site_thresholds(
    rows: Sequence[dict[str, Any]],
    *,
    t_ale: float = DEFAULT_T_ALE,
    disc_r_percentile: float = 75.0,
    clustering_percentile: float = 25.0,
    routing_max_percentile: float = 25.0,
) -> dict[str, float]:
    """Corpus-relative thresholds for local site triage."""
    disc = _finite_col(rows, "disc_r")
    clust = _finite_col(rows, "clustering")
    rmax = _finite_col(rows, "expert_routing_max")

    thresholds: dict[str, float] = {"t_ale": t_ale}
    if disc is not None:
        thresholds["disc_r_min"] = float(np.percentile(disc, disc_r_percentile))
    if clust is not None:
        thresholds["clustering_max"] = float(np.percentile(clust, clustering_percentile))
    if rmax is not None:
        thresholds["routing_max_max"] = float(np.percentile(rmax, routing_max_percentile))
    return thresholds


def flag_investigation_sites(
    rows: Sequence[dict[str, Any]],
    *,
    t_ale: float = DEFAULT_T_ALE,
    thresholds: Mapping[str, float] | None = None,
    require_disc_r: bool = True,
    require_clustering: bool = True,
) -> list[dict[str, Any]]:
    """Local druggability / conformational triage — residue-first decision rule.

    Actionable sites: high aleatoric + rim-localized (high disc_r) + low graph clustering.
    ``low p`` in the doctrine maps to rim localization (under-wrapped, high-entropy disc
  regions), not a corpus mean. Optional ``expert_routing_max`` supports low-confidence
    routing as a secondary ambiguity signal.
    """
    if not rows:
        return []

    thr = dict(thresholds or investigation_site_thresholds(rows, t_ale=t_ale))
    flagged: list[dict[str, Any]] = []

    for row in rows:
        ale = float(row["aleatoric"])
        high_ale = ale > thr.get("t_ale", t_ale)

        disc_r = row.get("disc_r")
        rim_like = True
        if require_disc_r and disc_r is not None and math.isfinite(float(disc_r)):
            rim_like = float(disc_r) >= thr.get("disc_r_min", float("-inf"))

        clustering = row.get("clustering")
        low_clustering = True
        if require_clustering and clustering is not None and math.isfinite(float(clustering)):
            low_clustering = float(clustering) <= thr.get("clustering_max", float("inf"))

        routing_max = row.get("expert_routing_max")
        low_routing_conf = True
        if routing_max is not None and math.isfinite(float(routing_max)):
            cap = thr.get("routing_max_max")
            if cap is not None:
                low_routing_conf = float(routing_max) <= cap

        investigate = high_ale and rim_like and low_clustering and low_routing_conf
        if investigate:
            flagged.append(
                {
                    **row,
                    "investigate": True,
                    "triage_reason": "high_ale_rim_low_clustering",
                    "t_ale": thr.get("t_ale", t_ale),
                }
            )
    flagged.sort(key=lambda r: float(r["aleatoric"]), reverse=True)
    return flagged


def aleatoric_dataset_health_report(
    rows: Sequence[dict[str, Any]],
    *,
    t_ale: float | None = None,
    t_ale_percentile: float = DEFAULT_T_ALE_PERCENTILE,
    gene_by_structure: Mapping[str, str] | None = None,
    fold_by_structure: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Corpus health monitors — **not** site pass/fail certification."""
    resolved_t_ale, t_ale_meta = resolve_t_ale(
        rows,
        t_ale=t_ale,
        t_ale_percentile=t_ale_percentile,
    )
    hist = corpus_aleatoric_histogram(
        rows,
        t_ale=resolved_t_ale,
        t_ale_percentile=t_ale_percentile,
    )
    hist["t_ale_resolution"] = t_ale_meta
    grouped = _rows_by_protein(rows)

    protein_summaries: list[dict[str, Any]] = []
    for (sid, chain), prot_rows in grouped.items():
        protein_summaries.append(
            per_protein_aleatoric_summary(
                prot_rows,
                structure_id=sid,
                chain=chain,
                t_ale=resolved_t_ale,
                gene=(gene_by_structure or {}).get(sid),
                fold_id=(fold_by_structure or {}).get(sid),
            )
        )

    ranked = rank_proteins_for_active_learning(protein_summaries)
    thresholds = investigation_site_thresholds(rows, t_ale=resolved_t_ale)
    sites = flag_investigation_sites(rows, t_ale=resolved_t_ale, thresholds=thresholds)

    frac_band = HIGH_ALE_FRACTION_VALIDATION_BAND
    fractions = [
        float(p["high_aleatoric_fraction"])
        for p in protein_summaries
        if p.get("ok") and p.get("n_residues", 0) > 0
    ]
    n_in_band = sum(1 for f in fractions if _in_band(f, frac_band))

    return {
        "version": "aleatoric_residue_diagnostics_v1",
        "doctrine": "residue_first_global_monitors_only",
        "corpus_histogram": hist,
        "global_aleatoric_std": hist.get("std"),
        "global_aleatoric_std_role": (
            "dataset_health_monitor_only — not a site certification gate"
        ),
        "high_ale_fraction_validation_band": list(frac_band),
        "proteins_in_high_ale_fraction_band": n_in_band,
        "n_proteins": len(protein_summaries),
        "per_protein": protein_summaries,
        "active_learning_ranking": ranked,
        "investigation_thresholds": thresholds,
        "investigation_sites": sites,
        "n_investigation_sites": len(sites),
        "t_ale": resolved_t_ale,
        "t_ale_resolution": t_ale_meta,
    }


def global_aleatoric_health_monitor(
    rows: Sequence[dict[str, Any]],
    *,
    informative_std_floor: float = 0.05,
) -> dict[str, Any]:
    """Explicitly monitoring-only wrapper around corpus std(ale).

    Replaces using ``global_ale_std`` as a pass/fail site gate. Training save gates
    may still consult spread; interpret those as collapse detectors, not biology.
    """
    if not rows:
        return {"ok": False, "reason": "empty", "role": "dataset_health_monitor"}
    ale = _col(rows, "aleatoric")
    std = float(np.std(ale))
    collapsed = std < informative_std_floor
    return {
        "ok": not collapsed,
        "global_ale_std": std,
        "informative_std_floor": informative_std_floor,
        "collapsed": collapsed,
        "role": "dataset_health_monitor",
        "not_a_site_gate": True,
        "interpretation": (
            "corpus std(ale) detects head collapse or narrow calibration — "
            "does not certify per-residue ambiguity at druggable sites"
        ),
    }


def _skewness(x: np.ndarray) -> float:
    if x.size < 3:
        return float("nan")
    m = float(np.mean(x))
    s = float(np.std(x))
    if s < 1e-12:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))


def _in_band(value: float, band: tuple[float, float]) -> bool:
    if not math.isfinite(value):
        return False
    return band[0] <= value <= band[1]
