"""Gradient reachability helpers for z_hyp G_fit (card tokyo_eye_equ_wrap1_zhyp_g_fit).

Distinguishes:
  NONE  — parameter not in the autograd graph (grad is None)
  ZERO  — in graph, gradient exactly 0
  NZ    — in graph, nonzero

Card decision 5: under SDRP-live (sole task loss), spine buckets must be NZ;
disabled dehydron/margin paths must be omitted from the loss expression so
mechanism_head stays NONE (not merely ZERO from ``0.0 * bce``).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from science.tokyo_eye.v8.heads import (
    mechanism_margin_loss_v2,
    sdrp_cross_entropy,
)
from science.tokyo_eye.v8.model import TokyoEyesHyperbolicV8

# Buckets that must receive nonzero gradient from SDRP CE alone (card lock).
SPINE_NZ_BUCKETS: tuple[str, ...] = (
    "attn_layers",
    "projector",
    "moe",
    "sdrp_head",
)

# Under SDRP-only backprop these must be entirely NONE (not in graph).
SDRP_ONLY_NONE_BUCKETS: tuple[str, ...] = (
    "mechanism_head",
    "evidential_head",
    "_log_c",
)


def param_bucket(name: str) -> str:
    """Map ``named_parameters`` keys to card buckets."""
    root = name.split(".", 1)[0]
    if root == "attn_layers":
        return "attn_layers"
    if root == "moe":
        return "moe"
    return root


def bucket_grad_stats(module: nn.Module) -> dict[str, dict[str, Any]]:
    """Per-bucket counts after a backward (call with set_to_none=True before)."""
    per: dict[str, dict[str, Any]] = {}
    for name, p in module.named_parameters():
        b = param_bucket(name)
        d = per.setdefault(
            b, {"g2": 0.0, "none": 0, "zero": 0, "nz": 0, "n": 0, "names_none": []}
        )
        d["n"] += 1
        if p.grad is None:
            d["none"] += 1
            d["names_none"].append(name)
        else:
            s = float(p.grad.detach().pow(2).sum().item())
            d["g2"] += s
            if s == 0.0:
                d["zero"] += 1
            else:
                d["nz"] += 1
    for d in per.values():
        d["grad_l2"] = float(d["g2"] ** 0.5)
        d.pop("names_none", None)  # drop bulky list from default returns
    return per


def bucket_status(d: dict[str, Any]) -> str:
    if d["nz"] == 0 and d["zero"] == 0:
        return "NONE"
    if d["nz"] == 0:
        return "ZERO"
    return "NZ"


def make_toy_spine(
    *,
    seed: int = 0,
    n: int = 8,
    scalar_dim: int = 8,
    vector_dim: int = 3,
    hidden_dim: int = 8,
    num_sdrp_classes: int = 5,
    device: torch.device | None = None,
) -> tuple[TokyoEyesHyperbolicV8, dict[str, torch.Tensor]]:
    """Sprint-scale synthetic batch — no PDB / graph cache."""
    device = device or torch.device("cpu")
    torch.manual_seed(seed)
    model = TokyoEyesHyperbolicV8(
        scalar_dim=scalar_dim,
        vector_dim=vector_dim,
        hidden_dim=hidden_dim,
        num_attn_layers=2,
        num_sdrp_classes=num_sdrp_classes,
        num_relations=6,
        gate_hidden=8,
    ).to(device)
    model.set_moe_mode("ablated")
    model.train()
    model.set_moe_temperature(1.0)
    model.set_moe_explore_epsilon(0.0)

    s = torch.randn(n, scalar_dim, device=device)
    v = torch.randn(n, vector_dim, device=device) * 0.1
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7], [1, 0, 3, 2, 5, 4, 7, 6]],
        dtype=torch.long,
        device=device,
    )
    edge_type = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long, device=device)
    batch = {
        "s": s,
        "v": v,
        "edge_index": edge_index,
        "edge_type": edge_type,
        "sdrp_target": torch.randint(0, num_sdrp_classes, (n,), device=device),
        "dehydron_labels": torch.randint(0, 2, (n,), device=device).float(),
        "mechanism_pos": torch.rand(n, device=device) * 0.5 + 0.4,
        "mechanism_neg": torch.rand(n, device=device) * 0.3 + 0.1,
    }
    return model, batch


def forward_losses(
    model: TokyoEyesHyperbolicV8,
    batch: dict[str, torch.Tensor],
    *,
    tau_ceiling: float = 0.85,
    sdrp_coeff: float = 0.1,
    dehydron_coeff: float = 0.0,
    margin_coeff: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Card-coeff losses. Disabled terms are omitted (not multiplied by 0.0)."""
    out = model(
        batch["s"],
        batch["v"],
        batch["edge_index"],
        batch["edge_type"],
        tau_ceiling=tau_ceiling,
    )
    losses: dict[str, torch.Tensor] = {
        "SDRP_CE": sdrp_cross_entropy(out["sdrp_logits"], batch["sdrp_target"]),
        "BCE_dehydron": F.binary_cross_entropy_with_logits(
            out["mechanism_score"], batch["dehydron_labels"]
        ),
        "margin": mechanism_margin_loss_v2(
            out["mechanism_score"],
            batch["mechanism_pos"],
            batch["mechanism_neg"],
        ),
    }
    # Compose task loss — omit zero-coeff terms so they stay out of the graph.
    parts: list[torch.Tensor] = []
    if float(sdrp_coeff) != 0.0:
        parts.append(float(sdrp_coeff) * losses["SDRP_CE"])
    if float(dehydron_coeff) != 0.0:
        parts.append(float(dehydron_coeff) * losses["BCE_dehydron"])
    if float(margin_coeff) != 0.0:
        parts.append(float(margin_coeff) * losses["margin"])
    if not parts:
        raise ValueError("all task coeffs are zero — no loss graph")
    losses["CARD_TOTAL"] = parts[0] if len(parts) == 1 else sum(parts[1:], parts[0])
    return losses


def backward_bucket_table(
    model: nn.Module,
    loss: torch.Tensor,
) -> dict[str, dict[str, Any]]:
    model.zero_grad(set_to_none=True)
    loss.backward()
    return bucket_grad_stats(model)


def assert_sdrp_live_spine_reachable(
    stats: dict[str, dict[str, Any]],
    *,
    context: str = "",
) -> None:
    """Card G_grad_spine: spine NZ under SDRP; mechanism/evidential/_log_c NONE."""
    prefix = f"{context}: " if context else ""
    for b in SPINE_NZ_BUCKETS:
        assert b in stats, f"{prefix}missing bucket {b}"
        st = bucket_status(stats[b])
        assert st == "NZ", (
            f"{prefix}expected {b} status NZ under SDRP-live, got {st} "
            f"(none={stats[b]['none']} zero={stats[b]['zero']} nz={stats[b]['nz']} "
            f"l2={stats[b]['grad_l2']:.4g})"
        )
        assert stats[b]["grad_l2"] > 0.0
    for b in SDRP_ONLY_NONE_BUCKETS:
        assert b in stats, f"{prefix}missing bucket {b}"
        st = bucket_status(stats[b])
        assert st == "NONE", (
            f"{prefix}expected {b} status NONE under SDRP-only (grad is None), "
            f"got {st}. If ZERO, a disabled term was likely left in the loss as "
            f"0.0*tensor — omit it instead."
        )
        assert stats[b]["none"] == stats[b]["n"]
        assert stats[b]["nz"] == 0 and stats[b]["zero"] == 0


def assert_dehydron_alone_spine_unreachable(
    stats: dict[str, dict[str, Any]],
    *,
    context: str = "",
) -> None:
    """Negative control: BCE dehydron alone must NOT light hyperbolic spine."""
    prefix = f"{context}: " if context else ""
    for b in ("attn_layers", "projector", "moe", "sdrp_head"):
        assert b in stats, f"{prefix}missing bucket {b}"
        st = bucket_status(stats[b])
        assert st == "NONE", (
            f"{prefix}expected {b} NONE under dehydron-BCE alone (euc_skip wiring), "
            f"got {st}"
        )


__all__ = [
    "SPINE_NZ_BUCKETS",
    "SDRP_ONLY_NONE_BUCKETS",
    "param_bucket",
    "bucket_grad_stats",
    "bucket_status",
    "make_toy_spine",
    "forward_losses",
    "backward_bucket_table",
    "assert_sdrp_live_spine_reachable",
    "assert_dehydron_alone_spine_unreachable",
]
