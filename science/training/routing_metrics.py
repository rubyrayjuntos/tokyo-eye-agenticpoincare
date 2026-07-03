"""MoE routing collapse metrics — entropy-derived effective expert count.

The mean per-expert routing fraction is always 1/N by construction and cannot
detect collapse. Gates must use effective_experts = exp(H(p)) and/or
min_routing_fraction = min(p_i).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


def entropy_nats(probs: torch.Tensor | np.ndarray) -> float:
    """Shannon entropy H(p) in nats for a probability vector summing to ~1."""
    if isinstance(probs, torch.Tensor):
        p = probs.detach().float().cpu().numpy()
    else:
        p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def effective_experts(probs: torch.Tensor | np.ndarray) -> float:
    """Effective number of experts: exp(H(p)). Uniform over N → N; collapse → 1."""
    return float(math.exp(entropy_nats(probs)))


def min_routing_fraction(probs: torch.Tensor | np.ndarray) -> float:
    """Smallest per-expert routing share — collapse tell on the fraction scale."""
    if isinstance(probs, torch.Tensor):
        p = probs.detach().float().cpu().numpy()
    else:
        p = np.asarray(probs, dtype=np.float64)
    if p.size == 0:
        return float("nan")
    return float(np.min(p))


def collapse_metrics_from_epoch_losses(losses: dict[str, Any]) -> dict[str, float]:
    """Derive governance routing metrics from one train_epoch aggregate."""
    if "effective_experts" in losses:
        eff_mean = float(
            losses["effective_experts"].item()
            if torch.is_tensor(losses["effective_experts"])
            else losses["effective_experts"]
        )
    else:
        routing_h = losses.get("routing_entropy")
        if routing_h is not None:
            h = float(routing_h.item() if torch.is_tensor(routing_h) else routing_h)
            eff_mean = float(math.exp(h))
        else:
            eff_mean = float("nan")

    if "effective_experts_min" in losses:
        eff_min_val = float(
            losses["effective_experts_min"].item()
            if torch.is_tensor(losses["effective_experts_min"])
            else losses["effective_experts_min"]
        )
    else:
        eff_min_val = eff_mean

    if "min_routing_fraction" in losses:
        min_frac_val = float(
            losses["min_routing_fraction"].item()
            if torch.is_tensor(losses["min_routing_fraction"])
            else losses["min_routing_fraction"]
        )
    else:
        loads = [
            float(losses[k])
            for k in losses
            if k.startswith("expert_load_") and k[len("expert_load_") :].isdigit()
        ]
        min_frac_val = float(min(loads)) if loads else float("nan")

    return {
        "effective_experts": eff_mean,
        "effective_experts_min": eff_min_val,
        "min_routing_fraction": min_frac_val,
    }
