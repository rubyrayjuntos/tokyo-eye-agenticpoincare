"""
Wrapper / Stabilizer Design Service

Provides simple heuristic wrapper suggestions for glueable sites.
"""

from __future__ import annotations

from typing import List, Dict

from gosp.models.data_models import GlueableSite, WrapperSuggestion


WRAPPER_GAINS = {
    "trehalose": 12,
    "proline": 5,
    "arginine": 4,
}


def design_wrappers(site: GlueableSite) -> List[Dict[str, int]]:
    """Design wrapper suggestions for a glueable site."""
    total_deficit = max(0, int(19 - site.avg_rho)) * max(1, len(site.dehydron_ids))

    wrappers: List[Dict[str, int]] = []
    trehalose_needed = min(total_deficit // WRAPPER_GAINS["trehalose"], len(site.dehydron_ids) // 2)
    for _ in range(trehalose_needed):
        wrappers.append({"type": "trehalose", "gain": WRAPPER_GAINS["trehalose"]})

    remaining = total_deficit - trehalose_needed * WRAPPER_GAINS["trehalose"]
    proline_needed = min(remaining // WRAPPER_GAINS["proline"], len(site.dehydron_ids))
    for _ in range(proline_needed):
        wrappers.append({"type": "proline", "gain": WRAPPER_GAINS["proline"]})

    remaining -= proline_needed * WRAPPER_GAINS["proline"]
    arginine_needed = min(remaining // WRAPPER_GAINS["arginine"], len(site.dehydron_ids))
    for _ in range(arginine_needed):
        wrappers.append({"type": "arginine", "gain": WRAPPER_GAINS["arginine"]})

    return wrappers


def predict_ddg(wrappers: List[Dict[str, int]]) -> float:
    """Predict delta delta G from wrapper placements."""
    total_gain = sum(wrapper["gain"] for wrapper in wrappers)
    wrapping_energy = 1.25 * total_gain
    desolv_penalty = 2.0 * len(wrappers)
    return wrapping_energy - desolv_penalty


def build_wrapper_suggestion(site: GlueableSite) -> WrapperSuggestion:
    """Create a wrapper suggestion for a site."""
    wrappers = design_wrappers(site)
    ddg = predict_ddg(wrappers)
    return WrapperSuggestion(site_id=site.site_id, wrappers=wrappers, predicted_ddg=ddg)
