"""P_DEHYDRON_CONE_01 — dehydron-rim cone alignment precondition (MASTER cold lineage).

After Phase 1, hyperbolic ``cone_depth`` (dist0) must correlate with dehydron flag τ.
SASA is not part of this gate — it is Euclidean exposure, not Poincaré topology.
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
    r_cone_depth_rho: float
    reason: str


def dehydron_cone_gate_verdict(health: dict[str, float]) -> DehydronConeGateVerdict:
    """Evaluate P_DEHYDRON_CONE_01 from geometry health probes (τ only)."""
    r_tau = float(health.get("probe_r_depth_tau", float("nan")))
    r_rho = float(health.get("probe_r_depth_rho", float("nan")))

    if not math.isfinite(r_tau):
        return DehydronConeGateVerdict(
            passed=False,
            r_cone_depth_tau=r_tau,
            r_cone_depth_rho=r_rho,
            reason="probe_r_depth_tau missing or non-finite",
        )

    if r_tau >= MIN_R_CONE_DEPTH_TAU:
        return DehydronConeGateVerdict(
            passed=True,
            r_cone_depth_tau=r_tau,
            r_cone_depth_rho=r_rho,
            reason=f"r(cone_depth,τ)={r_tau:.3f} >= {MIN_R_CONE_DEPTH_TAU}",
        )

    return DehydronConeGateVerdict(
        passed=False,
        r_cone_depth_tau=r_tau,
        r_cone_depth_rho=r_rho,
        reason=(
            f"r(cone_depth,τ)={r_tau:.3f} < {MIN_R_CONE_DEPTH_TAU} (dehydron rim not learned)"
        ),
    )


def dehydron_cone_gate_passed(health: dict[str, float]) -> bool:
    return dehydron_cone_gate_verdict(health).passed
