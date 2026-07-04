"""Stage A P2 inference-mode stop enforcement (P_STOP_ENFORCEMENT).

Pre-registered stop-and-diagnose criteria use inference routing (dropout off),
not training-mode min_routing_fraction (contaminated by expert_dropout_p).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from science.training.corpus_governance import is_locked_stage_a_manifest_path
from science.training.mlflow_governance import (
    STAGE_A_EFFECTIVE_EXPERTS_FLOOR,
    STAGE_A_MIN_ROUTING_FRACTION,
)

STAGE_A_STOP_CANARY_PDB = "1PGB"
STAGE_A_ROUTING_LOAD_FLOOR_COEFF = 10.0
STAGE_A_ROUTING_LOAD_FLOOR_MIN = 0.05
_PER_STRUCTURE_PREFIX = "eval_min_routing_fraction."


@dataclass(frozen=True)
class StageAStopVerdict:
    """Result of inference-mode stop check for one epoch."""

    tripped: bool
    reasons: tuple[str, ...]

    @property
    def summary(self) -> str:
        return "; ".join(self.reasons)


def _worst_structure_key(inference_routing: dict[str, float]) -> str | None:
    worst_pdb: str | None = None
    worst_mr = float("inf")
    for key, value in inference_routing.items():
        if not key.startswith(_PER_STRUCTURE_PREFIX):
            continue
        mr = float(value)
        if math.isfinite(mr) and mr < worst_mr:
            worst_mr = mr
            worst_pdb = key[len(_PER_STRUCTURE_PREFIX) :]
    return worst_pdb


def check_stage_a_inference_stop(inference_routing: dict[str, float]) -> StageAStopVerdict:
    """Evaluate pre-registered P2 stop criteria on inference-mode routing metrics."""
    reasons: list[str] = []

    min_frac = float(inference_routing.get("min_routing_fraction", float("nan")))
    eff_min = float(inference_routing.get("effective_experts_min", float("nan")))
    canary_key = f"{_PER_STRUCTURE_PREFIX}{STAGE_A_STOP_CANARY_PDB}"
    canary_mr = float(inference_routing.get(canary_key, float("nan")))

    if math.isfinite(min_frac) and min_frac < STAGE_A_MIN_ROUTING_FRACTION:
        worst = _worst_structure_key(inference_routing) or "?"
        reasons.append(
            f"inference min_routing_fraction={min_frac:.4f} < {STAGE_A_MIN_ROUTING_FRACTION} "
            f"(worst={worst})"
        )

    if math.isfinite(canary_mr) and canary_mr < STAGE_A_MIN_ROUTING_FRACTION:
        reasons.append(
            f"canary {STAGE_A_STOP_CANARY_PDB} inference min_r={canary_mr:.4f} "
            f"< {STAGE_A_MIN_ROUTING_FRACTION}"
        )

    if math.isfinite(eff_min) and eff_min <= STAGE_A_EFFECTIVE_EXPERTS_FLOOR:
        reasons.append(
            f"inference effective_experts_min={eff_min:.3f} "
            f"<= {STAGE_A_EFFECTIVE_EXPERTS_FLOOR}"
        )

    return StageAStopVerdict(tripped=bool(reasons), reasons=tuple(reasons))


def should_enforce_stage_a_stop(config: Any, phase: int) -> bool:
    """Enforce stop only on locked Stage A corpus, phase 2, when enabled."""
    if not getattr(config, "enforce_stage_a_stop", True):
        return False
    if phase != 2:
        return False
    manifest = Path(getattr(config, "corpus_manifest", ""))
    return is_locked_stage_a_manifest_path(manifest)


def format_stop_message(verdict: StageAStopVerdict) -> str:
    return f"Stage A stop enforced (inference routing): {verdict.summary}"
