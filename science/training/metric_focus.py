"""Training focus assessment — which metrics need work vs which matter."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

from science.training.checkpoint_score import (
    PROJ_FRAC_MAX,
    ROUTING_ENTROPY_PROMOTE_MAX,
    ROUTING_ENTROPY_SAVE_MAX,
)

Priority = Literal["critical", "important", "monitor", "healthy"]

# Stretch targets (beyond minimum gates) for mature P2 / theory-test runs
STRETCH_DEPTH_SASA = 0.80
STRETCH_EPI_SASA = 0.70
STRETCH_PROJ_DEPTH = 0.95
STRETCH_DISC_SASA = 0.70
STRETCH_DISC_R_STD = 0.10
STRETCH_CONE_RANGE = 0.85
STRETCH_ROUTING_H = ROUTING_ENTROPY_PROMOTE_MAX  # 1.2
UNIFORM_LOAD = 0.25  # per expert if perfectly balanced


@dataclass(frozen=True)
class FocusItem:
    metric: str
    value: float | None
    priority: Priority
    matters: bool
    status: Literal["needs_work", "watch", "healthy"]
    target: str
    note: str


def _finite(value: object) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _expert_load_imbalance(losses: dict[str, Any]) -> float | None:
    loads = []
    for i in range(4):
        v = _finite(losses.get(f"expert_load_{i}"))
        if v is not None:
            loads.append(v)
    if len(loads) < 2:
        el = losses.get("expert_load")
        if el is not None and hasattr(el, "numel") and el.numel() >= 2:
            loads = [float(el[i]) for i in range(min(4, int(el.numel())))]
    if len(loads) < 2:
        return None
    return max(loads) - min(loads)


def assess_training_focus(
    health: dict[str, float],
    losses: dict[str, float],
    *,
    phase: int = 2,
    routing_save_max: float | None = None,
    min_probe_r_depth_sasa_save: float | None = None,
    checkpoint_eligible: bool | None = None,
) -> dict[str, Any]:
    """
    Classify metrics into: needs work (and how much it matters), watch, healthy.

    Returns a summary suitable for MLflow tags/artifacts and human review.
    """
    items: list[FocusItem] = []
    route_max = routing_save_max if routing_save_max is not None else ROUTING_ENTROPY_SAVE_MAX
    save_floor = min_probe_r_depth_sasa_save if min_probe_r_depth_sasa_save is not None else 0.65

    route_h = _finite(losses.get("routing_entropy"))
    if route_h is not None and phase >= 2:
        if route_h > route_max:
            items.append(
                FocusItem(
                    metric="routing_entropy",
                    value=route_h,
                    priority="critical",
                    matters=True,
                    status="needs_work",
                    target=f"≤ {route_max:.2f} (save ceiling this epoch)",
                    note="Blocks v6_best saves; MoE still too uniform vs current gate.",
                )
            )
        elif route_h > STRETCH_ROUTING_H:
            items.append(
                FocusItem(
                    metric="routing_entropy",
                    value=route_h,
                    priority="important",
                    matters=True,
                    status="watch",
                    target=f"≤ {STRETCH_ROUTING_H:.2f} (promotion stretch)",
                    note="Eligible saves possible but routing not fully specialized.",
                )
            )
        else:
            items.append(
                FocusItem(
                    metric="routing_entropy",
                    value=route_h,
                    priority="healthy",
                    matters=True,
                    status="healthy",
                    target=f"≤ {STRETCH_ROUTING_H:.2f}",
                    note="MoE specialization in good range.",
                )
            )

    starve = _finite(losses.get("expert_starvation_count"))
    if starve is not None and starve >= 1:
        items.append(
            FocusItem(
                metric="expert_starvation_count",
                value=starve,
                priority="critical" if starve >= 2 else "important",
                matters=True,
                status="needs_work",
                target="0",
                note="Experts dropping out; ineligible if ≥ 2 in P2.",
            )
        )

    pf = _finite(health.get("proj_frac_mean"))
    if pf is not None:
        if pf > PROJ_FRAC_MAX:
            items.append(
                FocusItem(
                    metric="proj_frac_mean",
                    value=pf,
                    priority="critical",
                    matters=True,
                    status="needs_work",
                    target=f"≈ 0 (max {PROJ_FRAC_MAX})",
                    note="Hyperbolic boundary collapse — train P1-style recovery.",
                )
            )
        elif pf > 0.05:
            items.append(
                FocusItem(
                    metric="proj_frac_mean",
                    value=pf,
                    priority="important",
                    matters=True,
                    status="watch",
                    target="≈ 0",
                    note="Some boundary clipping; usually fix before P2 MoE pressure.",
                )
            )
        else:
            items.append(
                FocusItem(
                    metric="proj_frac_mean",
                    value=pf,
                    priority="healthy",
                    matters=True,
                    status="healthy",
                    target="≈ 0",
                    note="Saturated — do not optimize further.",
                )
            )

    r_ds = _finite(health.get("probe_r_depth_sasa"))
    if r_ds is not None:
        if r_ds < save_floor:
            items.append(
                FocusItem(
                    metric="probe_r_depth_sasa",
                    value=r_ds,
                    priority="critical",
                    matters=True,
                    status="needs_work",
                    target=f"≥ {save_floor:.2f} (save gate)",
                    note="Core shell signal weak — depth not aligned with SASA.",
                )
            )
        elif r_ds < STRETCH_DEPTH_SASA:
            items.append(
                FocusItem(
                    metric="probe_r_depth_sasa",
                    value=r_ds,
                    priority="important",
                    matters=True,
                    status="watch",
                    target=f"≥ {STRETCH_DEPTH_SASA:.2f} (strong shell)",
                    note="Passes save gate but below production stretch.",
                )
            )
        else:
            items.append(
                FocusItem(
                    metric="probe_r_depth_sasa",
                    value=r_ds,
                    priority="healthy",
                    matters=True,
                    status="healthy",
                    target=f"≥ {STRETCH_DEPTH_SASA:.2f}",
                    note="Shell geometry production-grade — protect during MoE training.",
                )
            )

    r_pd = _finite(health.get("probe_r_proj_depth"))
    if r_pd is not None:
        if r_pd < 0.30:
            items.append(
                FocusItem(
                    metric="probe_r_proj_depth",
                    value=r_pd,
                    priority="critical",
                    matters=True,
                    status="needs_work",
                    target="≥ 0.30 (minimum)",
                    note="Disc collapsed or inverted vs depth.",
                )
            )
        elif r_pd < STRETCH_PROJ_DEPTH:
            items.append(
                FocusItem(
                    metric="probe_r_proj_depth",
                    value=r_pd,
                    priority="important",
                    matters=True,
                    status="watch",
                    target=f"≥ {STRETCH_PROJ_DEPTH:.2f}",
                    note="Disc–depth coupling could be tighter.",
                )
            )
        else:
            items.append(
                FocusItem(
                    metric="probe_r_proj_depth",
                    value=r_pd,
                    priority="healthy",
                    matters=True,
                    status="healthy",
                    target=f"≥ {STRETCH_PROJ_DEPTH:.2f}",
                    note="Disc tracks burial — saturated.",
                )
            )

    r_es = _finite(health.get("probe_r_epi_sasa"))
    if r_es is not None:
        if r_es < 0.20:
            items.append(
                FocusItem(
                    metric="probe_r_epi_sasa",
                    value=r_es,
                    priority="critical",
                    matters=True,
                    status="needs_work",
                    target="≥ 0.20 (alive)",
                    note="Uncertainty shell signal absent.",
                )
            )
        elif r_es < STRETCH_EPI_SASA:
            items.append(
                FocusItem(
                    metric="probe_r_epi_sasa",
                    value=r_es,
                    priority="monitor",
                    matters=False,
                    status="watch",
                    target=f"≥ {STRETCH_EPI_SASA:.2f}",
                    note="Secondary shell probe; often already strong in your runs.",
                )
            )
        else:
            items.append(
                FocusItem(
                    metric="probe_r_epi_sasa",
                    value=r_es,
                    priority="healthy",
                    matters=False,
                    status="healthy",
                    target=f"≥ {STRETCH_EPI_SASA:.2f}",
                    note="Saturated for ingest quality.",
                )
            )

    cr = _finite(health.get("cone_range_mean"))
    if cr is not None and cr < STRETCH_CONE_RANGE:
        items.append(
            FocusItem(
                metric="cone_range_mean",
                value=cr,
                priority="important" if cr < 0.04 else "monitor",
                matters=cr < 0.04,
                status="needs_work" if cr < 0.04 else "watch",
                target=f"≥ {STRETCH_CONE_RANGE:.2f}",
                note="Depth spread across structure.",
            )
        )

    load_spread = _expert_load_imbalance(losses)
    if load_spread is not None and route_h is not None and route_h > STRETCH_ROUTING_H:
        if load_spread < 0.08:
            items.append(
                FocusItem(
                    metric="expert_load_spread",
                    value=load_spread,
                    priority="important",
                    matters=True,
                    status="needs_work",
                    target="≥ 0.08 max−min load",
                    note="Experts still sharing work evenly — increase MoE pressure gradually.",
                )
            )

    dr_std = _finite(health.get("disc_r_std_mean"))
    if dr_std is not None and dr_std < STRETCH_DISC_R_STD and phase >= 2:
        items.append(
            FocusItem(
                metric="disc_r_std_mean",
                value=dr_std,
                priority="monitor",
                matters=False,
                status="watch",
                target=f"≥ {STRETCH_DISC_R_STD:.2f}",
                note="Disc spread; usually addressed in P1d — low leverage once shell strong.",
            )
        )

    # Per-expert shell weak spots (theory test monitoring)
    for e in range(4):
        r_e = _finite(health.get(f"expert_{e}_r_depth_sasa"))
        if r_e is not None and r_e < 0.45:
            items.append(
                FocusItem(
                    metric=f"expert_{e}_r_depth_sasa",
                    value=r_e,
                    priority="monitor",
                    matters=False,
                    status="watch",
                    target="≥ 0.45 per expert",
                    note=f"Expert {e} residues weak on depth↔SASA (often surface specialist).",
                )
            )

    total = _finite(losses.get("total"))
    if total is not None:
        items.append(
            FocusItem(
                metric="total",
                value=total,
                priority="monitor",
                matters=False,
                status="watch",
                target="↓ slowly",
                note="Optimization loss — do not chase alone; use tier-1 probes instead.",
            )
        )

    needs_work = [i for i in items if i.status == "needs_work"]
    watch = [i for i in items if i.status == "watch"]
    healthy = [i for i in items if i.status == "healthy"]
    critical = [i for i in items if i.priority == "critical"]
    matters_needs_work = [i for i in needs_work if i.matters]

    primary_focus = [i.metric for i in matters_needs_work[:5]]
    if not primary_focus and watch:
        primary_focus = [i.metric for i in watch if i.matters][:3]

    return {
        "checkpoint_eligible": checkpoint_eligible,
        "needs_work": [asdict(i) for i in needs_work],
        "watch": [asdict(i) for i in watch],
        "healthy": [asdict(i) for i in healthy],
        "primary_focus": primary_focus,
        "primary_focus_str": ", ".join(primary_focus) if primary_focus else "none",
        "counts": {
            "critical": len(critical),
            "needs_work": len(needs_work),
            "watch": len(watch),
            "healthy": len(healthy),
            "matters_needs_work": len(matters_needs_work),
        },
        "recommendation": _recommendation(needs_work, watch, healthy, checkpoint_eligible),
    }


def _recommendation(
    needs_work: list[FocusItem],
    watch: list[FocusItem],
    healthy: list[FocusItem],
    eligible: bool | None,
) -> str:
    if eligible:
        return "Eligible epoch — geometry and MoE gates passed; compare score to prior best."
    critical = [i for i in needs_work if i.priority == "critical" and i.matters]
    if critical:
        names = ", ".join(i.metric for i in critical[:3])
        return f"Train more on: {names}. These block saves or threaten shell/MoE quality."
    if needs_work:
        names = ", ".join(i.metric for i in needs_work if i.matters)[:3]
        return f"Secondary focus: {names}. Shell likely OK; tune MoE or stretch targets."
    if watch:
        return "Most gates passed; watch stretch metrics before adding pressure."
    return "Metrics saturated on tracked focus areas — extend epochs or tighten gates carefully."


def focus_mlflow_metrics(summary: dict[str, Any]) -> dict[str, float]:
    """Compact scalars for MLflow charts."""
    counts = summary.get("counts", {})
    out: dict[str, float] = {
        "focus_critical_count": float(counts.get("critical", 0)),
        "focus_needs_work_count": float(counts.get("needs_work", 0)),
        "focus_watch_count": float(counts.get("watch", 0)),
        "focus_healthy_count": float(counts.get("healthy", 0)),
        "focus_matters_needs_work_count": float(counts.get("matters_needs_work", 0)),
    }
    for item in summary.get("needs_work", []):
        if item.get("matters"):
            key = f"focus_flag_{item['metric']}"
            out[key] = 1.0
    return out
