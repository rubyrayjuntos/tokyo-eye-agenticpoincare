"""Generic flow–centrality concordance (platform probe; no residue-ID gates).

Spec intent: measure Spearman concordance between forward-knockout
``out_effect`` and classical betweenness on arbitrary PDBs. KRAS landmarks
are never used in Pass/Fail.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

DEFAULT_TOP_K_FRAC = 0.10
DEFAULT_PASS_RHO = 0.50
DEFAULT_STABILITY_DRHO_MAX = 0.05


def top_k_hub_mask(betweenness: Sequence[float], *, k_frac: float = DEFAULT_TOP_K_FRAC) -> np.ndarray:
    """Boolean mask for top ``k_frac`` residues by classical betweenness."""
    b = np.asarray(betweenness, dtype=np.float64).reshape(-1)
    n = b.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    k = max(1, int(np.ceil(float(k_frac) * n)))
    order = np.argsort(-np.nan_to_num(b, nan=-np.inf))
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    b = np.asarray(y, dtype=np.float64).reshape(-1)
    if a.size != b.size or a.size < 3:
        return float("nan")
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return float("nan")
    aa, bb = a[mask], b[mask]
    ra = aa.argsort().argsort().astype(np.float64)
    rb = bb.argsort().argsort().astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    if denom < 1e-12:
        return float("nan")
    return float((ra * rb).sum() / denom)


def concordance_report(
    out_effect: Sequence[float],
    betweenness: Sequence[float],
    *,
    k_frac: float = DEFAULT_TOP_K_FRAC,
    pass_rho: float = DEFAULT_PASS_RHO,
) -> dict[str, Any]:
    """Full-structure ρ (primary) + top-k% diagnostics (report-only)."""
    oe = np.asarray(out_effect, dtype=np.float64).reshape(-1)
    btw = np.asarray(betweenness, dtype=np.float64).reshape(-1)
    rho_full = spearman_rho(oe, btw)
    hub = top_k_hub_mask(btw, k_frac=k_frac)
    rho_hub = spearman_rho(oe[hub], btw[hub]) if hub.sum() >= 3 else float("nan")
    oe_hub = float(np.nanmean(oe[hub])) if hub.any() else float("nan")
    oe_non = float(np.nanmean(oe[~hub])) if (~hub).any() else float("nan")
    enrichment = (
        oe_hub / oe_non if np.isfinite(oe_hub) and np.isfinite(oe_non) and oe_non > 1e-12 else float("nan")
    )
    ok = bool(np.isfinite(rho_full) and rho_full > float(pass_rho))
    return {
        "spearman_full": rho_full,
        "spearman_top_k": rho_hub,
        "k_frac": float(k_frac),
        "n_hubs": int(hub.sum()),
        "n_residues": int(oe.size),
        "hub_out_effect_mean": oe_hub,
        "nonhub_out_effect_mean": oe_non,
        "hub_enrichment": enrichment,
        "pass_rho": float(pass_rho),
        "pass": ok,
        # Never include residue sequence numbers in Pass logic.
    }


def stability_ok(
    rho_primary: float,
    rho_perturbed: float,
    *,
    max_abs_drho: float = DEFAULT_STABILITY_DRHO_MAX,
) -> bool:
    if not (np.isfinite(rho_primary) and np.isfinite(rho_perturbed)):
        return False
    return abs(float(rho_primary) - float(rho_perturbed)) <= float(max_abs_drho)


def panel_verdict(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Panel Pass if every structure passes concordance (+ stability when present)."""
    if not rows:
        return {"pass": False, "outcome": "Fail", "n": 0, "n_pass": 0}
    n_pass = 0
    for r in rows:
        ok = bool(r.get("pass"))
        if "stability_pass" in r:
            ok = ok and bool(r["stability_pass"])
        if ok:
            n_pass += 1
    all_ok = n_pass == len(rows)
    return {
        "pass": all_ok,
        "outcome": "Pass" if all_ok else "Fail",
        "n": len(rows),
        "n_pass": n_pass,
        "median_spearman_full": float(
            np.median([float(r["spearman_full"]) for r in rows if np.isfinite(r.get("spearman_full", np.nan))])
        )
        if rows
        else float("nan"),
    }


__all__ = [
    "DEFAULT_PASS_RHO",
    "DEFAULT_STABILITY_DRHO_MAX",
    "DEFAULT_TOP_K_FRAC",
    "concordance_report",
    "panel_verdict",
    "spearman_rho",
    "stability_ok",
    "top_k_hub_mask",
]
