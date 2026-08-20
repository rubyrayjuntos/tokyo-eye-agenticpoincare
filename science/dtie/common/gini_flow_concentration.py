"""Full-chain knockout flow concentration metrics (structural monopoly).

Gini and mass-share are computed on the **entire** ``out_effect`` vector —
never on the top-k% hub slice *H* alone (that conditioning is tautological).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def gini_coefficient(values: Sequence[float] | np.ndarray) -> float:
    """Gini of non-negative mass. 0 = equal; →1 as mass concentrates.

    Returns NaN for empty or all-zero (undefined inequality).
    """
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    x = np.clip(x, 0.0, None)
    if x.size == 0:
        return float("nan")
    total = float(x.sum())
    if total <= 0.0:
        return float("nan")
    xs = np.sort(x)
    n = int(xs.size)
    i = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(i * xs) / (n * total)) - (n + 1) / n)


def concentration_metrics(
    out_effect: Sequence[float] | np.ndarray,
) -> dict[str, Any]:
    """Full-chain concentration suite for one knockout ``out_effect`` array."""
    oe = np.asarray(out_effect, dtype=np.float64).reshape(-1)
    oe = oe[np.isfinite(oe)]
    oe = np.clip(oe, 0.0, None)
    n = int(oe.size)
    if n == 0:
        return {
            "n": 0,
            "gini": float("nan"),
            "max": float("nan"),
            "median": float("nan"),
            "max_over_median": float("nan"),
            "top1pct_mass_share": float("nan"),
            "top10pct_mass_share": float("nan"),
            "cv": float("nan"),
            "sum": 0.0,
        }

    total = float(oe.sum())
    med = float(np.median(oe))
    mx = float(oe.max())
    mean = float(oe.mean()) if total > 0 else float("nan")
    cv = float(oe.std() / mean) if mean and mean > 0 else float("nan")

    def _top_frac_mass(frac: float) -> float:
        if total <= 0.0:
            return float("nan")
        k = max(1, int(np.ceil(frac * n)))
        return float(np.sort(oe)[::-1][:k].sum() / total)

    return {
        "n": n,
        "gini": gini_coefficient(oe),
        "max": mx,
        "median": med,
        "max_over_median": (mx / med) if med > 0 else float("nan"),
        "top1pct_mass_share": _top_frac_mass(0.01),
        "top10pct_mass_share": _top_frac_mass(0.10),
        "cv": cv,
        "sum": total,
    }


def compare_concentration(
    baseline_out_effect: Sequence[float] | np.ndarray,
    champion_out_effect: Sequence[float] | np.ndarray,
    *,
    archive_arrays: bool = False,
) -> dict[str, Any]:
    """ΔG = G_base − G_champion; positive ⇒ structural monopoly reduced."""
    base_m = concentration_metrics(baseline_out_effect)
    champ_m = concentration_metrics(champion_out_effect)
    g_b = float(base_m["gini"])
    g_c = float(champ_m["gini"])
    if np.isnan(g_b) or np.isnan(g_c):
        delta = float("nan")
        reduced = False
    else:
        delta = g_b - g_c
        reduced = bool(delta > 0.0)

    out: dict[str, Any] = {
        "baseline": dict(base_m),
        "champion": dict(champ_m),
        "delta_gini": delta,
        "delta_top1pct_mass_share": (
            float(base_m["top1pct_mass_share"]) - float(champ_m["top1pct_mass_share"])
            if np.isfinite(base_m["top1pct_mass_share"])
            and np.isfinite(champ_m["top1pct_mass_share"])
            else float("nan")
        ),
        "delta_top10pct_mass_share": (
            float(base_m["top10pct_mass_share"]) - float(champ_m["top10pct_mass_share"])
            if np.isfinite(base_m["top10pct_mass_share"])
            and np.isfinite(champ_m["top10pct_mass_share"])
            else float("nan")
        ),
        "delta_max_over_median": (
            float(base_m["max_over_median"]) - float(champ_m["max_over_median"])
            if np.isfinite(base_m["max_over_median"])
            and np.isfinite(champ_m["max_over_median"])
            else float("nan")
        ),
        "monopoly_reduced": reduced,
    }
    if archive_arrays:
        out["baseline"]["out_effect"] = (
            np.asarray(baseline_out_effect, dtype=np.float64).reshape(-1).tolist()
        )
        out["champion"]["out_effect"] = (
            np.asarray(champion_out_effect, dtype=np.float64).reshape(-1).tolist()
        )
    return out
