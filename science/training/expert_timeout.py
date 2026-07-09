"""Hard expert timeout — ban dominant experts from the routing pool for N epochs.

When an expert's soft (or eval) share exceeds ``max_share``, it is masked out of
gate logits for ``ban_epochs`` training epochs, then re-enabled with a short
cooldown so the same expert cannot be banned again immediately.

Timing (end-of-epoch tick):
  1. Decrement existing bans (experts that trained under the mask this epoch).
  2. Observe soft loads and optionally ban a new dominant expert at full
     ``ban_epochs`` (not decremented until the next end-of-epoch).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging

import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class ExpertTimeoutController:
    """Epoch-level expert ban state for MoE-alive training."""

    num_experts: int
    max_share: float = 0.50
    ban_epochs: int = 1
    cooldown_epochs: int = 1
    # If set, only these expert ids may be banned (e.g. [2] for generalist-only).
    eligible_experts: list[int] | None = None
    # expert_id → epochs remaining in ban (0 = active)
    ban_remaining: dict[int, int] = field(default_factory=dict)
    # expert_id → epochs remaining before eligible for another ban
    cooldown_remaining: dict[int, int] = field(default_factory=dict)

    def active_bans(self) -> list[int]:
        return sorted(i for i, n in self.ban_remaining.items() if n > 0)

    def apply_to_gate(self, model: nn.Module) -> None:
        """Push current ban mask onto the gate (train-time logit mask)."""
        gate = getattr(model, "gate", None)
        if gate is None:
            return
        banned = self.active_bans()
        gate.expert_timeout_banned = banned

    def clear_gate(self, model: nn.Module) -> None:
        gate = getattr(model, "gate", None)
        if gate is not None:
            gate.expert_timeout_banned = []

    def tick_end_of_epoch(self) -> list[int]:
        """Decrement ban/cooldown after a training epoch; return still-banned ids."""
        just_released: list[int] = []
        for eid in list(self.ban_remaining):
            if self.ban_remaining[eid] > 0:
                self.ban_remaining[eid] -= 1
                if self.ban_remaining[eid] == 0:
                    just_released.append(eid)
                    self.cooldown_remaining[eid] = self.cooldown_epochs
                    logger.info(
                        "  Expert timeout: e%d re-enters pool (cooldown=%d)",
                        eid,
                        self.cooldown_epochs,
                    )
        # Do not burn cooldown on the same tick it was granted.
        for eid in list(self.cooldown_remaining):
            if eid in just_released:
                continue
            if self.cooldown_remaining[eid] > 0 and self.ban_remaining.get(eid, 0) == 0:
                self.cooldown_remaining[eid] -= 1
        return self.active_bans()

    def observe_loads(
        self,
        soft_loads: list[float] | tuple[float, ...],
        *,
        epoch: int,
    ) -> list[int]:
        """After an epoch, ban any non-cooldown expert whose share exceeds max_share.

        Returns newly banned expert ids. Call *after* ``tick_end_of_epoch`` so a
        fresh ban starts at full ``ban_epochs``.
        """
        newly: list[int] = []
        if len(soft_loads) < self.num_experts:
            return newly
        eligible = (
            set(int(i) for i in self.eligible_experts)
            if self.eligible_experts is not None
            else None
        )
        # Only ban the single worst offender per epoch to avoid emptying the pool.
        ranked = sorted(
            enumerate(float(x) for x in soft_loads[: self.num_experts]),
            key=lambda t: t[1],
            reverse=True,
        )
        for eid, share in ranked:
            if share <= self.max_share:
                break
            if eligible is not None and eid not in eligible:
                continue
            if self.ban_remaining.get(eid, 0) > 0:
                continue
            if self.cooldown_remaining.get(eid, 0) > 0:
                continue
            # Keep at least 2 experts available after this ban.
            active = [
                i
                for i in range(self.num_experts)
                if self.ban_remaining.get(i, 0) == 0 and i != eid
            ]
            if len(active) < 2:
                logger.info(
                    "  Expert timeout: skip ban e%d (share=%.3f) — would leave <2 experts",
                    eid,
                    share,
                )
                break
            self.ban_remaining[eid] = self.ban_epochs
            newly.append(eid)
            logger.info(
                "  Expert timeout: e%d share=%.3f>%.2f → banned for %d epochs (ep=%d)",
                eid,
                share,
                self.max_share,
                self.ban_epochs,
                epoch,
            )
            break  # one ban per epoch
        return newly
