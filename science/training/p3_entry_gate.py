"""P3_ENTRY_GATE — routing-stability precondition before Phase 3.

Phase 3 freezes the gate and fine-tunes experts. It must not start until train-mode
routing entropy has settled *below* the save ceiling for N consecutive P2 epochs.
A single limit-cycle trough (e.g. one epoch grazing 1.21) must not qualify.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from science.training.checkpoint_score import ROUTING_ENTROPY_SAVE_MAX

GATE_NAME = "P3_ENTRY_GATE"
DEFAULT_CONSECUTIVE_EPOCHS = 5


@dataclass(frozen=True)
class P3EntryGateVerdict:
    passed: bool
    max_consecutive_below_ceiling: int
    required_consecutive: int
    ceiling: float
    n_p2_epochs: int
    reason: str


def max_consecutive_below(
    values: Sequence[float],
    ceiling: float,
) -> int:
    """Longest run of values strictly below ``ceiling``."""
    best = 0
    streak = 0
    for raw in values:
        try:
            v = float(raw)
        except (TypeError, ValueError):
            streak = 0
            continue
        if v < ceiling:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return best


def p3_entry_gate_verdict(
    p2_routing_entropies: Sequence[float],
    *,
    ceiling: float = ROUTING_ENTROPY_SAVE_MAX,
    consecutive_epochs: int = DEFAULT_CONSECUTIVE_EPOCHS,
) -> P3EntryGateVerdict:
    """
    Two-sided gate:
      - Stable routing (≥N consecutive below ceiling) → pass → P3 may start.
      - Oscillating / graze-only P2 (this master-cold run) → fail → P3 blocked.
    """
    required = max(1, int(consecutive_epochs))
    streak = max_consecutive_below(p2_routing_entropies, ceiling)
    n = len(p2_routing_entropies)

    if n == 0:
        return P3EntryGateVerdict(
            passed=False,
            max_consecutive_below_ceiling=0,
            required_consecutive=required,
            ceiling=ceiling,
            n_p2_epochs=0,
            reason="no P2 routing_entropy history",
        )

    if streak >= required:
        return P3EntryGateVerdict(
            passed=True,
            max_consecutive_below_ceiling=streak,
            required_consecutive=required,
            ceiling=ceiling,
            n_p2_epochs=n,
            reason=(
                f"routing_entropy below {ceiling:.2f} for {streak} consecutive P2 epochs "
                f"(required {required})"
            ),
        )

    return P3EntryGateVerdict(
        passed=False,
        max_consecutive_below_ceiling=streak,
        required_consecutive=required,
        ceiling=ceiling,
        n_p2_epochs=n,
        reason=(
            f"routing stability not met: max consecutive below {ceiling:.2f} is {streak} "
            f"(required {required}); limit-cycle graze does not qualify"
        ),
    )


def p3_entry_gate_passed(
    p2_routing_entropies: Sequence[float],
    *,
    ceiling: float = ROUTING_ENTROPY_SAVE_MAX,
    consecutive_epochs: int = DEFAULT_CONSECUTIVE_EPOCHS,
) -> bool:
    return p3_entry_gate_verdict(
        p2_routing_entropies,
        ceiling=ceiling,
        consecutive_epochs=consecutive_epochs,
    ).passed
