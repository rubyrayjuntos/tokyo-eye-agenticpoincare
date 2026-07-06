"""P_DEHYDRON_CONE_01 — dehydron-rim cone alignment precondition (MASTER cold lineage).

After Phase 1, cone depth must correlate with dehydron flag τ (not SASA burial).
Blocks Phase 2 entry when the radial head still encodes a burial shell.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

GATE_NAME = "P_DEHYDRON_CONE_01"
MIN_R_CONE_DEPTH_TAU = 0.15


@dataclass(frozen=True)
class DehydronConeGateVerdict:
    passed: bool
    r_cone_depth_tau: float
    r_cone_depth_sasa: float
    r_cone_depth_rho: float
    sasa_dominates_tau: bool
    reason: str


def dehydron_cone_gate_verdict(health: dict[str, float]) -> DehydronConeGateVerdict:
    """Evaluate P_DEHYDRON_CONE_01 from geometry health probes."""
    r_tau = float(health.get("probe_r_depth_tau", float("nan")))
    r_sasa = float(health.get("probe_r_depth_sasa", float("nan")))
    r_rho = float(health.get("probe_r_depth_rho", float("nan")))

    if not math.isfinite(r_tau):
        return DehydronConeGateVerdict(
            passed=False,
            r_cone_depth_tau=r_tau,
            r_cone_depth_sasa=r_sasa,
            r_cone_depth_rho=r_rho,
            sasa_dominates_tau=False,
            reason="probe_r_depth_tau missing or non-finite",
        )

    sasa_wins = (
        math.isfinite(r_sasa)
        and abs(r_sasa) > abs(r_tau)
    )
    tau_ok = r_tau >= MIN_R_CONE_DEPTH_TAU

    if tau_ok and not sasa_wins:
        return DehydronConeGateVerdict(
            passed=True,
            r_cone_depth_tau=r_tau,
            r_cone_depth_sasa=r_sasa,
            r_cone_depth_rho=r_rho,
            sasa_dominates_tau=False,
            reason=(
                f"r(cone_depth,τ)={r_tau:.3f} >= {MIN_R_CONE_DEPTH_TAU} "
                f"and |r(depth,SASA)|={abs(r_sasa):.3f} <= |r(depth,τ)|"
            ),
        )

    parts: list[str] = []
    if not tau_ok:
        parts.append(
            f"r(cone_depth,τ)={r_tau:.3f} < {MIN_R_CONE_DEPTH_TAU} (dehydron rim not learned)"
        )
    if sasa_wins:
        parts.append(
            f"|r(depth,SASA)|={abs(r_sasa):.3f} > |r(depth,τ)|={abs(r_tau):.3f} (SASA dominates)"
        )
    return DehydronConeGateVerdict(
        passed=False,
        r_cone_depth_tau=r_tau,
        r_cone_depth_sasa=r_sasa,
        r_cone_depth_rho=r_rho,
        sasa_dominates_tau=sasa_wins,
        reason="; ".join(parts),
    )


def dehydron_cone_gate_passed(health: dict[str, float]) -> bool:
    return dehydron_cone_gate_verdict(health).passed
