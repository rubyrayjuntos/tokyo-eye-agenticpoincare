"""Biology loss gradient attribution: hyperbolic state vs Euclidean skip.

Architecture SSOT: before any claim-bearing biology run, log how much of the
biology loss gradient flows through ``z_hyp`` (post-attention / post-MoE)
versus the Euclidean skip ``h_euc``. Do not assume DualSpaceSDRPHead fusion
is balanced credit; dehydron BCE via MechanismScoreHead(h_euc) is euc-only.

Standing rule: prose without a failing test always loses to a convenient default.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn.functional as F


def _tensor_grad_norm(t: torch.Tensor | None) -> float:
    if t is None or t.grad is None:
        return 0.0
    return float(t.grad.detach().float().norm().item())


def biology_grad_by_source(
    *,
    z_hyp: torch.Tensor,
    h_euc: torch.Tensor,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    retain_graph: bool = False,
) -> dict[str, float]:
    """Backprop ``loss_fn(z_hyp, h_euc)`` and report grad norms into each input.

    Callers must ensure ``z_hyp`` and ``h_euc`` require grad (typically via
    ``retain_grad`` on intermediate activations from a live forward).
    """
    if not z_hyp.requires_grad:
        z_hyp = z_hyp.detach().requires_grad_(True)
    if not h_euc.requires_grad:
        h_euc = h_euc.detach().requires_grad_(True)
    if z_hyp.grad is not None:
        z_hyp.grad = None
    if h_euc.grad is not None:
        h_euc.grad = None

    loss = loss_fn(z_hyp, h_euc)
    if not torch.is_tensor(loss):
        raise TypeError("loss_fn must return a torch.Tensor")
    loss.backward(retain_graph=retain_graph)

    g_hyp = _tensor_grad_norm(z_hyp)
    g_euc = _tensor_grad_norm(h_euc)
    total = g_hyp + g_euc
    return {
        "biology_grad_hyp": g_hyp,
        "biology_grad_euc_skip": g_euc,
        "biology_grad_hyp_share": (g_hyp / total) if total > 0.0 else 0.0,
        "biology_grad_euc_share": (g_euc / total) if total > 0.0 else 0.0,
        "biology_grad_total": total,
        "biology_loss": float(loss.detach().float().item()),
    }


def dehydron_grad_by_source(
    mechanism_head: Any,
    *,
    z_hyp: torch.Tensor,
    h_euc: torch.Tensor,
    dehydron_labels: torch.Tensor,
    retain_graph: bool = False,
) -> dict[str, float]:
    """Dehydron BCE through MechanismScoreHead — typically euc-skip only today."""

    def _loss(_z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        score = mechanism_head(h)
        return F.binary_cross_entropy_with_logits(score, dehydron_labels)

    out = biology_grad_by_source(
        z_hyp=z_hyp, h_euc=h_euc, loss_fn=_loss, retain_graph=retain_graph
    )
    out["biology_loss_name"] = 1.0  # marker: dehydron path (float for metrics)
    out["biology_path"] = 0.0  # 0=dehydron, 1=sdrp (numeric log-friendly)
    return out


def sdrp_grad_by_source(
    sdrp_head: Any,
    *,
    z_hyp: torch.Tensor,
    h_euc: torch.Tensor,
    sdrp_target: torch.Tensor,
    tau_ceiling: float = 0.70,
    retain_graph: bool = False,
) -> dict[str, float]:
    """SDRP CE through DualSpaceSDRPHead — fused hyp + euc_skip."""

    def _loss(z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        logits = sdrp_head(z, h, tau_ceiling=float(tau_ceiling))
        return F.cross_entropy(logits, sdrp_target)

    out = biology_grad_by_source(
        z_hyp=z_hyp, h_euc=h_euc, loss_fn=_loss, retain_graph=retain_graph
    )
    out["biology_path"] = 1.0
    return out


__all__ = [
    "biology_grad_by_source",
    "dehydron_grad_by_source",
    "sdrp_grad_by_source",
]
