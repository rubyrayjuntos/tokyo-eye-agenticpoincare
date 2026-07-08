"""Composite checkpoint scoring — aligned with assess promotion gate."""

from __future__ import annotations

import math
from dataclasses import dataclass

from science.dtie.v6.gnn.model import uses_disc_radial_override
from science.training.disc_occupancy import (
    DISC_SIGMA2_SIGMA1_PROMOTE_MIN,
    DISC_R_STD_PROMOTE_MIN,
    disc_occupancy_ineligibility_reasons,
)

# Shared with experiments.training.v6.assess_checkpoint promotion_gate
ROUTING_ENTROPY_PROMOTE_MAX = 1.2
ROUTING_ENTROPY_SAVE_MAX = 1.21  # slightly looser during training selection
PROJ_FRAC_MAX = 0.95
CONE_RANGE_MIN_P1 = 0.02
CONE_RANGE_MIN = 0.04

# Shell probe floors — aligned with experiments/diagnostics/shell_signal_diagnostics.py
SHELL_DEPTH_SASA_MIN = 0.0  # reject inversion (b2-style negative r)
SHELL_EPI_SASA_WEAK = 0.20  # probe 1 "signal diluted" floor
SHELL_DEPTH_SASA_WEAK = 0.20
SHELL_PROJ_DEPTH_MIN = 0.30  # probe 4 collapsed floor

# Biological tier-1 gates when disc radius is authoritative (Lever A).
RADIAL_DEPTH_SHELL_SAVE_MIN = 0.55
RADIAL_DEPTH_THICKNESS_SAVE_MIN = 0.025


@dataclass(frozen=True)
class CheckpointScoreResult:
    score: float
    eligible: bool
    reasons: list[str]


def _finite_probe(value: object) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def line_thickness_for_save_gate(health: dict[str, float]) -> float | None:
    """Prefer pre-routing thickness when the model emits it."""
    pre = _finite_probe(health.get("disc_line_thickness_pre_mean"))
    if pre is not None:
        return pre
    return _finite_probe(health.get("disc_line_thickness_rms_mean"))


def radial_depth_biological_ineligibility_reasons(
    health: dict[str, float],
    *,
    min_probe_r_depth_sasa: float = RADIAL_DEPTH_SHELL_SAVE_MIN,
    min_line_thickness: float = RADIAL_DEPTH_THICKNESS_SAVE_MIN,
    topology_depth: bool = False,
) -> list[str]:
    """
    Tier-1 save gates for authoritative-radius disc modes (Lever A).

    When radius is biology-supervised, eff_rank and σ₂/σ₁ are poor proxies;
    thickness + depth×SASA alignment are the correct promotion criteria.
    """
    reasons: list[str] = []
    if not topology_depth:
        r_depth_sasa = _finite_probe(health.get("probe_r_depth_sasa"))
        if r_depth_sasa is not None and r_depth_sasa < min_probe_r_depth_sasa:
            reasons.append(
                f"probe_r_depth_sasa={r_depth_sasa:.3f}<{min_probe_r_depth_sasa} "
                "(radial_depth shell gate)"
            )
    thickness = line_thickness_for_save_gate(health)
    if thickness is not None and thickness < min_line_thickness:
        reasons.append(
            f"disc_line_thickness={thickness:.5f}<{min_line_thickness} "
            "(pre-routing visual gate)"
        )
    return reasons


def shell_probe_ineligibility_reasons(
    health: dict[str, float],
    *,
    topology_depth: bool = False,
) -> list[str]:
    """
    Skipped when probe keys are absent (legacy callers / unit tests).
    Topology lineage: SASA probes ignored — depth is τ/ρ supervised only.
    """
    if topology_depth:
        r_proj_depth = _finite_probe(health.get("probe_r_proj_depth"))
        if r_proj_depth is None:
            return []
        reasons: list[str] = []
        if r_proj_depth < 0:
            reasons.append(f"probe_r_proj_depth={r_proj_depth:.3f}<0 (inverted disc gradient)")
        elif r_proj_depth < SHELL_PROJ_DEPTH_MIN:
            reasons.append(
                f"probe_r_proj_depth={r_proj_depth:.3f}<{SHELL_PROJ_DEPTH_MIN} (collapsed)"
            )
        return reasons

    r_depth_sasa = _finite_probe(health.get("probe_r_depth_sasa"))
    r_epi_sasa = _finite_probe(health.get("probe_r_epi_sasa"))
    r_proj_depth = _finite_probe(health.get("probe_r_proj_depth"))

    if r_depth_sasa is None and r_epi_sasa is None and r_proj_depth is None:
        return []

    reasons: list[str] = []

    if r_depth_sasa is not None and r_depth_sasa < SHELL_DEPTH_SASA_MIN:
        reasons.append(
            f"probe_r_depth_sasa={r_depth_sasa:.3f}<{SHELL_DEPTH_SASA_MIN} (inverted shell)"
        )

    if r_proj_depth is not None:
        if r_proj_depth < 0:
            reasons.append(f"probe_r_proj_depth={r_proj_depth:.3f}<0 (inverted disc gradient)")
        elif r_proj_depth < SHELL_PROJ_DEPTH_MIN:
            reasons.append(
                f"probe_r_proj_depth={r_proj_depth:.3f}<{SHELL_PROJ_DEPTH_MIN} (collapsed)"
            )

    if r_depth_sasa is not None and r_epi_sasa is not None:
        depth_alive = r_depth_sasa >= SHELL_DEPTH_SASA_WEAK
        epi_alive = r_epi_sasa >= SHELL_EPI_SASA_WEAK
        if not depth_alive and not epi_alive:
            reasons.append(
                "shell signal absent "
                f"(r_depth_sasa={r_depth_sasa:.3f}, r_epi_sasa={r_epi_sasa:.3f})"
            )

    return reasons


def uncertainty_save_ineligibility_reasons(
    health: dict[str, float],
    *,
    max_probe_r_epi_ale: float | None = None,
    max_probe_r_epi_sasa: float | None = None,
    min_epistemic_std: float | None = None,
    min_aleatoric_std: float | None = None,
    require_tau_ale_elevation: bool = False,
    min_tau_ale_lift: float = 0.0,
) -> list[str]:
    """Uncertainty calibration save gates (Phase 4 uncertainty).

    When ``require_tau_ale_elevation`` is True (S6 joint), low r(epi,ale) alone is
    insufficient — P8 τ-boundary aleatoric lift must also pass.
    """
    reasons: list[str] = []
    r_epi_ale = _finite_probe(health.get("probe_r_epi_ale"))
    r_epi_sasa = _finite_probe(health.get("probe_r_epi_sasa"))
    epi_std = _finite_probe(health.get("epistemic_std_mean"))
    ale_std = _finite_probe(health.get("aleatoric_std_mean"))

    if max_probe_r_epi_ale is not None and r_epi_ale is not None and r_epi_ale > max_probe_r_epi_ale:
        reasons.append(
            f"probe_r_epi_ale={r_epi_ale:.3f}>{max_probe_r_epi_ale} (epi/ale coupled)"
        )
    if max_probe_r_epi_sasa is not None and r_epi_sasa is not None and r_epi_sasa > max_probe_r_epi_sasa:
        reasons.append(
            f"probe_r_epi_sasa={r_epi_sasa:.3f}>{max_probe_r_epi_sasa} (epi/sasa coupled)"
        )
    if min_epistemic_std is not None and epi_std is not None and epi_std < min_epistemic_std:
        reasons.append(
            f"epistemic_std={epi_std:.4f}<{min_epistemic_std} (epi collapsed)"
        )
    if min_aleatoric_std is not None and ale_std is not None and ale_std < min_aleatoric_std:
        reasons.append(
            f"aleatoric_std={ale_std:.4f}<{min_aleatoric_std} (ale collapsed)"
        )
    if require_tau_ale_elevation:
        lift = _finite_probe(health.get("node_aleatoric_tau_lift"))
        elevated = health.get("uncertainty_tau_ale_elevated")
        if elevated is not None and float(elevated) < 1.0:
            reasons.append("tau_ale_elevation=0 (P8 fail; S6 joint gate)")
        elif lift is not None and lift <= min_tau_ale_lift:
            reasons.append(
                f"node_aleatoric_tau_lift={lift:.4f}<={min_tau_ale_lift} (P8 fail; S6 joint)"
            )
    return reasons


def disc_save_ineligibility_reasons(
    health: dict[str, float],
    *,
    disc_radial_source: str = "mobius",
    min_sigma_ratio: float | None = None,
    min_disc_r_std: float | None = None,
    min_effective_rank: float | None = None,
    min_line_thickness: float | None = None,
    min_probe_r_depth_sasa: float | None = None,
    topology_depth: bool = False,
) -> list[str]:
    """Disc occupancy / visual gates — mode-aware for Lever A override sources."""
    if uses_disc_radial_override(disc_radial_source):
        shell_floor = (
            min_probe_r_depth_sasa
            if min_probe_r_depth_sasa is not None
            else RADIAL_DEPTH_SHELL_SAVE_MIN
        )
        thick_floor = (
            min_line_thickness
            if min_line_thickness is not None
            else RADIAL_DEPTH_THICKNESS_SAVE_MIN
        )
        return radial_depth_biological_ineligibility_reasons(
            health,
            min_probe_r_depth_sasa=shell_floor,
            min_line_thickness=thick_floor,
            topology_depth=topology_depth,
        )

    return disc_occupancy_ineligibility_reasons(
        health,
        min_sigma_ratio=min_sigma_ratio if min_sigma_ratio is not None else DISC_SIGMA2_SIGMA1_PROMOTE_MIN,
        min_effective_rank=min_effective_rank,
        min_disc_r_std=min_disc_r_std,
        min_line_thickness=min_line_thickness,
    )


def score_checkpoint(
    health: dict[str, float],
    losses: dict[str, float],
    *,
    phase: int = 1,
    routing_save_max: float | None = None,
    min_probe_r_depth_sasa_save: float | None = None,
    min_disc_sigma2_sigma1_save: float | None = None,
    min_disc_r_std_save: float | None = None,
    min_disc_effective_rank_save: float | None = None,
    min_disc_line_thickness_save: float | None = None,
    disc_radial_source: str = "mobius",
    max_probe_r_epi_ale_save: float | None = None,
    max_probe_r_epi_sasa_save: float | None = None,
    min_epistemic_std_save: float | None = None,
    min_aleatoric_std_save: float | None = None,
    require_tau_ale_elevation_save: bool = False,
    topology_depth: bool = False,
) -> CheckpointScoreResult:
    """
    Rank checkpoints for v6_best.pt selection.

    ``eligible`` checkpoints satisfy geometry + MoE + shell probe gates for the phase.
    ``score`` is higher-is-better for ranking among eligible (and fallback) epochs.

    When ``disc_radial_source`` is ``radial_depth`` or ``dist0_x_hyp``, tier-1 save
    keys on pre-routing ``line_thickness`` + ``probe_r_depth_sasa``; eff_rank and
    σ₂/σ₁ gates are bypassed. Uniform MoE routing (high entropy) is also tolerated
    until Lever C — routing entropy is not a save blocker in override mode.
    """
    pf = float(health.get("proj_frac_mean") or 0.0)
    cr = float(health.get("cone_range_mean") or 0.0)
    route_h = float(losses.get("routing_entropy", 1.386))
    starve = int(losses.get("expert_starvation_count", 0))
    total_loss = float(losses.get("total", 0.0))
    radial_override = uses_disc_radial_override(disc_radial_source)

    cone_min = CONE_RANGE_MIN_P1 if phase == 1 else CONE_RANGE_MIN
    if phase >= 2 and not radial_override:
        route_max = (
            routing_save_max if routing_save_max is not None else ROUTING_ENTROPY_SAVE_MAX
        )
    else:
        route_max = ROUTING_ENTROPY_PROMOTE_MAX + 0.2

    reasons: list[str] = []
    for label, val in (
        ("total_loss", total_loss),
        ("cone_range", cr),
        ("routing_H", route_h),
        ("proj_frac", pf),
    ):
        if math.isnan(val) or math.isinf(val):
            reasons.append(f"{label}=nonfinite")
    if pf > PROJ_FRAC_MAX:
        reasons.append(f"proj_frac={pf:.3f}")
    if phase >= 2 and not radial_override and route_h > route_max:
        reasons.append(f"routing_H={route_h:.3f}")
    if starve >= 2 and phase >= 2:
        reasons.append(f"starvation={starve}")
    if cr < cone_min:
        reasons.append(f"cone_range={cr:.4f}<{cone_min}")

    reasons.extend(shell_probe_ineligibility_reasons(health, topology_depth=topology_depth))

    disc_floor = min_disc_sigma2_sigma1_save
    if disc_floor is None and phase >= 2 and not radial_override:
        disc_floor = DISC_SIGMA2_SIGMA1_PROMOTE_MIN
    disc_r_floor = min_disc_r_std_save
    if disc_r_floor is None and phase >= 2 and not radial_override:
        disc_r_floor = DISC_R_STD_PROMOTE_MIN
    disc_eff_floor = min_disc_effective_rank_save if not radial_override else None

    if radial_override or disc_floor is not None:
        reasons.extend(
            disc_save_ineligibility_reasons(
                health,
                disc_radial_source=disc_radial_source,
                min_sigma_ratio=disc_floor,
                min_effective_rank=disc_eff_floor,
                min_disc_r_std=disc_r_floor if not radial_override else None,
                min_line_thickness=min_disc_line_thickness_save,
                min_probe_r_depth_sasa=min_probe_r_depth_sasa_save,
                topology_depth=topology_depth,
            )
            )

    reasons.extend(
        uncertainty_save_ineligibility_reasons(
            health,
            max_probe_r_epi_ale=max_probe_r_epi_ale_save,
            max_probe_r_epi_sasa=max_probe_r_epi_sasa_save,
            min_epistemic_std=min_epistemic_std_save,
            min_aleatoric_std=min_aleatoric_std_save,
            require_tau_ale_elevation=require_tau_ale_elevation_save,
        )
    )

    if (
        not topology_depth
        and not radial_override
        and min_probe_r_depth_sasa_save is not None
    ):
        r_depth_sasa = _finite_probe(health.get("probe_r_depth_sasa"))
        if (
            r_depth_sasa is not None
            and r_depth_sasa < min_probe_r_depth_sasa_save
        ):
            reasons.append(
                f"probe_r_depth_sasa={r_depth_sasa:.3f}<{min_probe_r_depth_sasa_save} "
                "(shell save gate)"
            )

    eligible = len(reasons) == 0

    # Composite: reward spread + low boundary clip + specialized routing + shell alignment
    score = cr + (1.0 - min(pf, 1.0))
    if route_h <= ROUTING_ENTROPY_PROMOTE_MAX:
        score += 0.5 * (ROUTING_ENTROPY_PROMOTE_MAX - route_h)
    else:
        score -= 1.0 * (route_h - ROUTING_ENTROPY_PROMOTE_MAX)

    r_depth_tau = _finite_probe(health.get("probe_r_depth_tau"))
    r_depth_rho = _finite_probe(health.get("probe_r_depth_rho"))
    if topology_depth:
        if r_depth_tau is not None and r_depth_tau > 0:
            score += 0.4 * min(r_depth_tau, 0.8)
        if r_depth_rho is not None and r_depth_rho < 0:
            score += 0.2 * min(abs(r_depth_rho), 0.8)
    else:
        r_depth_sasa = _finite_probe(health.get("probe_r_depth_sasa"))
        r_epi_sasa = _finite_probe(health.get("probe_r_epi_sasa"))
        if r_depth_sasa is not None and r_depth_sasa > 0:
            score += 0.4 * min(r_depth_sasa, 0.5)
        if r_epi_sasa is not None and r_epi_sasa > 0:
            score += 0.3 * min(r_epi_sasa, 0.5)

    disc_ratio = _finite_probe(health.get("disc_sigma2_sigma1_mean"))
    if disc_ratio is not None and disc_ratio > 0:
        score += 0.5 * min(disc_ratio, 0.5)

    thickness = line_thickness_for_save_gate(health)
    if thickness is not None and thickness > 0:
        score += 0.6 * min(thickness / 0.05, 1.0)

    return CheckpointScoreResult(score=score, eligible=eligible, reasons=reasons)


def score_route_checkpoint(
    health: dict[str, float],
    losses: dict[str, float],
    *,
    routing_save_max: float | None = None,
    min_routing_fraction: float = 0.08,
    topology_depth: bool = False,
    inference_routing: dict[str, float] | None = None,
) -> CheckpointScoreResult:
    """Rank checkpoints for v6_best_route.pt — routing-first, geometry guardrails."""
    route_h = float(losses.get("routing_entropy", 1.386))
    route_max = routing_save_max if routing_save_max is not None else ROUTING_ENTROPY_SAVE_MAX
    min_r = float(
        (inference_routing or {}).get("min_routing_fraction")
        or losses.get("min_routing_fraction")
        or 0.0
    )
    loads = [losses.get(f"expert_load_{i}") for i in range(8)]
    loads = [float(x) for x in loads if x is not None]
    spread = (max(loads) - min(loads)) if loads else 0.0
    r_dt = _finite_probe(health.get("probe_r_depth_tau"))

    reasons: list[str] = []
    if route_h > route_max:
        reasons.append(f"routing_H={route_h:.3f}")
    if min_r < min_routing_fraction:
        reasons.append(f"min_r={min_r:.3f}<{min_routing_fraction}")
    if topology_depth and r_dt is not None and r_dt < 0.45:
        reasons.append(f"probe_r_depth_tau={r_dt:.3f}<0.45")
    eligible = len(reasons) == 0

    # Higher is better: low entropy + load spread + min_r floor + τ alignment
    score = -route_h + 0.6 * spread + 0.4 * min_r
    if r_dt is not None and r_dt > 0:
        score += 0.25 * min(r_dt, 1.0)
    return CheckpointScoreResult(score=score, eligible=eligible, reasons=reasons)
