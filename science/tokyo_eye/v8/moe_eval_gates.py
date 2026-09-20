"""Eval-mode MoE utilization gates (tokyo_eye_equ_moe_eval_util)."""
from __future__ import annotations

import math
from typing import Any

import torch


def eval_moe_utilization(
    routing_argmax_onehot: torch.Tensor,
    *,
    min_load: float = 0.10,
    alive_floor: float = 0.05,
    n_alive_required: int = 4,
    entropy_norm_floor: float = 0.85,
) -> dict[str, Any]:
    """Seal MoE from eval argmax routing ``[N, E]`` one-hot rows."""
    if routing_argmax_onehot.ndim != 2:
        raise ValueError("routing must be [N, E]")
    load = routing_argmax_onehot.float().mean(dim=0)
    e = int(load.numel())
    probs = load.clamp_min(1e-12)
    entropy = float(-(probs * probs.log()).sum().item())
    ent_norm = entropy / math.log(max(e, 2))
    n_alive = int((load >= float(alive_floor)).sum().item())
    min_f = float(load.min().item())
    gates = {
        "moe_eval_min_load": {
            "pass": min_f >= float(min_load),
            "value": min_f,
            "threshold": float(min_load),
        },
        "moe_eval_n_alive": {
            "pass": n_alive >= int(n_alive_required),
            "value": float(n_alive),
            "threshold": float(n_alive_required),
        },
        "moe_eval_entropy_norm": {
            "pass": ent_norm >= float(entropy_norm_floor),
            "value": ent_norm,
            "threshold": float(entropy_norm_floor),
        },
    }
    return {
        "load": [float(x) for x in load.tolist()],
        "gates": gates,
        "all_pass": all(g["pass"] for g in gates.values()),
    }


__all__ = ["eval_moe_utilization"]
