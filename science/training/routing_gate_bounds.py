"""N-aware routing gate bounds — scale Stage A bands with ``num_experts``.

Reference calibration at N=4:
  - effective_experts ∈ [3.0, 4.5], floor 2.5
  - routing entropy save ceiling H = 1.21 nats  →  exp(H)/N ≈ 0.838 of uniform

For N≠4, entropy ceilings shift by ``ln(N/N_ref)`` so the same effective-expert
fraction is required. Compare specialization across N via ``exp(H)``, not raw H.
"""

from __future__ import annotations

import math

import torch.nn as nn

from science.training.checkpoint_score import ROUTING_ENTROPY_SAVE_MAX

# N=4 SSOT (mlflow_governance §5)
_REF_NUM_EXPERTS = 4
_REF_EFF_MIN = 3.0
_REF_EFF_MAX = 4.5
_REF_EFF_FLOOR = 2.5

# Pre-registered A-vs-C discriminator for N>4 capacity experiments.
CAPACITY_OUTCOME_A_VS_C = (
    "Outcome A (capacity confirmed) only if joint-feasible region opens AND "
    "expert_load_spread shows genuine differentiation AND per-expert r_depth_sasa "
    "diverge with live grad probes. Region opens with uniform/flat experts = Outcome C "
    "(over-provisioning), not capacity relief — do not ship N on gate alone."
)


def routing_entropy_offset(num_experts: int) -> float:
    """Add to N=4-calibrated entropy thresholds for ``num_experts``."""
    n = max(1, int(num_experts))
    return math.log(n / _REF_NUM_EXPERTS)


def scale_routing_entropy_ceiling(ceiling: float, num_experts: int) -> float:
    """Scale an N=4-calibrated entropy threshold to ``num_experts``."""
    return float(ceiling) + routing_entropy_offset(num_experts)


def model_num_experts(model: nn.Module, *, default: int = _REF_NUM_EXPERTS) -> int:
    if hasattr(model, "experts"):
        return max(1, len(model.experts))
    gate = getattr(model, "gate", None)
    if gate is not None and hasattr(gate, "num_experts"):
        return max(1, int(gate.num_experts))
    return default


def stage_a_effective_experts_bounds(
    num_experts: int,
) -> tuple[float, float, float]:
    """Return (min, max, floor) for ``stage_a_gate_passed`` routing band."""
    n = max(1, int(num_experts))
    scale = n / _REF_NUM_EXPERTS
    return (
        _REF_EFF_MIN * scale,
        _REF_EFF_MAX * scale,
        _REF_EFF_FLOOR * scale,
    )


def routing_entropy_save_ceiling(*, num_experts: int | None = None) -> float:
    """Training / P3-gate entropy ceiling in nats, scaled for ``num_experts``."""
    n = max(1, int(num_experts or _REF_NUM_EXPERTS))
    return scale_routing_entropy_ceiling(ROUTING_ENTROPY_SAVE_MAX, n)


def effective_experts_fraction_at_ceiling(num_experts: int) -> float:
    """exp(H_ceiling)/N — comparable specialization bar across expert counts."""
    n = max(1, int(num_experts))
    return math.exp(routing_entropy_save_ceiling(num_experts=n)) / n


def uniform_routing_entropy(num_experts: int) -> float:
    """Maximum routing entropy (uniform over N experts)."""
    return math.log(max(1, int(num_experts)))
