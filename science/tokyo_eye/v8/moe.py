"""Topology-aware hard-commitment MoE guilds (TokyoEye-v8 Sprint 3).

Isolated under ``science.tokyo_eye.v8``. Post-transport specialization into
E0–E3 with train-time ``gumbel_softmax(..., hard=True)`` (STE) and eval argmax.

FROZEN: ``docs/superpowers/specs/2026-07-22-tokyo-eye-v8-equiformer-hyp-design.md`` §8.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter

from science.tokyo_eye.v8.attention import (
    GyroOrthogonalMap,
    gyroscalar_mul,
    mobius_add,
    project_to_ball,
    tau_relative_residual_step,
)

# E0–E3 guild labels are DESIGN INTENT (wrapped/rim/interface/coil).
# Eval utilization not guaranteed until tokyo_eye_equ_moe_eval_util Pass.
NUM_EXPERTS = 4
CHEM_GATE_DIM = 7  # ρ, τ, ss_H, ss_E, ss_C, degree, SASA


def topology_gate_features(
    z: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    chem: torch.Tensor | None = None,
    c: float = 1.0,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Gate vector ``[ρ, τ, ss_H, ss_E, ss_C, degree, SASA, r, marker]``.

    Chemistry columns come from the R0–R5 loader. ``r`` is a read-only
    Poincaré radius (diagnostics), not a Linear on ball coordinates.

    Trailing ``marker=2.0`` is deliberate: ``pure_hyp_pass._is_open_ball_batch``
    is a numeric heuristic; the marker pushes the feature-row norm above
    ``_BALL_MAX`` so gate ``nn.Linear`` calls are not false-flagged as
    ``linear_on_ball``. New gate/head authors must keep an equivalent dodge
    (see ``pure_hyp_pass`` module docstring).
    """
    if z.ndim != 2:
        raise ValueError("z must be [N, d]")
    n = z.shape[0]
    z = project_to_ball(z, c=c, eps=eps)
    radius = torch.linalg.vector_norm(z, dim=-1, keepdim=True)

    if chem is None:
        deg = torch.zeros(n, 1, device=z.device, dtype=z.dtype)
        if edge_index.numel():
            dst = edge_index[1].long()
            deg = scatter(
                torch.ones(dst.shape[0], device=z.device, dtype=z.dtype),
                dst,
                dim=0,
                dim_size=n,
                reduce="sum",
            ).unsqueeze(-1)
        chem = torch.zeros(n, CHEM_GATE_DIM, device=z.device, dtype=z.dtype)
        chem[:, 5:6] = deg
    elif chem.ndim != 2 or chem.shape[0] != n or chem.shape[1] != CHEM_GATE_DIM:
        raise ValueError(f"chem must be [N, {CHEM_GATE_DIM}]")
    marker = torch.full((n, 1), 2.0, device=z.device, dtype=z.dtype)
    return torch.cat([chem.to(dtype=z.dtype), radius, marker], dim=-1)


class GyroExpert(nn.Module):
    """On-manifold expert: gyro maps + **damped** Möbius residual.

    Undamped ``z ⊕ R₂(R₁(z))`` inflates every radius past ``τ`` (same failure
    mode as attention's undamped ``self ⊕ attended``); clamp then pins the
    post-MoE ball. Residual is ``z ⊕ (t ⊗ R₂(R₁(z)))`` with ``t ≈ 0.25``.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.R1 = GyroOrthogonalMap(dim)
        self.R2 = GyroOrthogonalMap(dim)
        # sigmoid(-1.0986) ≈ 0.25 — matches HyperbolicGraphAttention.
        self.residual_logit = nn.Parameter(torch.tensor(-1.0986122886681098))

    def forward(self, z: torch.Tensor, *, c: float, eps: float, tau_ceiling: float = 0.70) -> torch.Tensor:
        h = self.R1(z, c=c, eps=eps)
        h = self.R2(h, c=c, eps=eps)
        r = torch.linalg.vector_norm(z, dim=-1)
        step = tau_relative_residual_step(
            self.residual_logit, tau_ceiling=float(tau_ceiling), radius=r
        )
        h_step = gyroscalar_mul(step, h, c=c, eps=eps)
        return mobius_add(z, h_step, c=c, eps=eps)


def cv_load_balance_loss(routing: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Coefficient-of-variation penalty over mean expert load fractions.

    ``routing``: ``[N, E]`` one-hot (or soft) assignment rows.
    Returns **unscaled** CV — harness applies ``cv_coeff``.
    """
    # Fraction of nodes assigned to each expert
    load = routing.mean(dim=0)  # [E]
    mean = load.mean()
    std = load.std(unbiased=False)
    return std / (mean.abs() + eps)




def eval_proxy_routing(logits: torch.Tensor, *, temperature: float = 0.25) -> torch.Tensor:
    """Differentiable stand-in for eval argmax: softmax(logits / T), T < 1.

    ``T=0.05`` (Smoke B) matches argmax when margins are tiny, but saturates to
    a dead ceiling (loss→E, ``‖∇‖→0``) once mean max−2nd logit margin is
    ≳0.5 — reproducing the masking problem the proxy was built to fix.
    Default ``T=0.25`` still tracks monopoly (load ≳0.9 at margin≈0.8) with
    usable gate gradients.
    """
    t = max(float(temperature), 1e-4)
    return F.softmax(logits / t, dim=-1)


def eval_proxy_load_balance_loss(
    logits: torch.Tensor, *, temperature: float = 0.25
) -> torch.Tensor:
    """Switch-style LB on sharp clean-logit routing (eval-argmax proxy)."""
    return switch_load_balance_loss(eval_proxy_routing(logits, temperature=temperature))

def switch_load_balance_loss(soft: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Switch Transformer–style load balance on soft gate probabilities.

    ``soft``: ``[N, E]`` softmax(logits) (not hard STE one-hots).
    ``loss = E * Σ_e f_e · P_e`` with ``f_e = mean_i soft_{i,e}``, ``P_e = f_e``
    when importance equals load (standard uniform-expert target). Unscaled —
    harness applies ``switch_lb_coeff``.
    """
    if soft.ndim != 2:
        raise ValueError("soft must be [N, E]")
    n_exp = soft.shape[1]
    f = soft.mean(dim=0)  # [E] fraction of soft mass
    # Importance = mean gate probability (same as f for dense softmax over all tokens)
    p = f
    return float(n_exp) * torch.sum(f * p)


def soft_and_hard_quota_losses(
    soft: torch.Tensor,
    hard: torch.Tensor,
    *,
    floor: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quota hinge on soft probs (primary) and hard STE (secondary)."""
    return (
        min_load_quota_loss(soft, floor=floor),
        min_load_quota_loss(hard, floor=floor),
    )

def majority_hinge_loss(
    routing: torch.Tensor,
    *,
    cap: float = 0.50,
) -> torch.Tensor:
    """In-structure majority hinge: ``ReLU(max_e load_e − cap)²``."""
    load = routing.mean(dim=0)
    return F.relu(load.max() - float(cap)) ** 2


def min_load_quota_loss(
    routing: torch.Tensor,
    *,
    floor: float = 0.05,
) -> torch.Tensor:
    """Soft minimum-load hinge: ``Σ_e ReLU(floor − load_e)²``.

    ``routing``: ``[N, E]`` hard STE (train) or one-hot (eval) rows.
    Returns **unscaled** quota — harness applies ``moe_quota_coeff``.
    """
    if floor < 0.0 or floor > 1.0:
        raise ValueError("floor must be in [0, 1]")
    load = routing.mean(dim=0)  # [E]
    return torch.sum(F.relu(float(floor) - load) ** 2)


def epsilon_greedy_hard_override(
    routing: torch.Tensor,
    *,
    epsilon: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """§2.6: per-node Bernoulli swap to a fresh hard one-hot (train only).

    Discrete replace — never a convex mix. Returns ``(routing_out, override_mask)``
    where ``override_mask`` is ``[N]`` float {0,1}.
    """
    if routing.ndim != 2:
        raise ValueError("routing must be [N, E]")
    n, e = routing.shape
    eps = float(epsilon)
    if eps <= 0.0:
        return routing, routing.new_zeros(n)
    if eps > 1.0:
        raise ValueError("epsilon must be in [0, 1]")
    probs = routing.new_full((n,), eps)
    if generator is None:
        bern = torch.bernoulli(probs)
        rand_idx = torch.randint(0, e, (n,), device=routing.device)
    else:
        bern = torch.bernoulli(probs, generator=generator)
        rand_idx = torch.randint(
            0, e, (n,), device=routing.device, generator=generator
        )
    rand_oh = F.one_hot(rand_idx, num_classes=e).to(
        dtype=routing.dtype, device=routing.device
    )
    mask = bern.unsqueeze(-1)
    out = torch.where(mask > 0.5, rand_oh, routing)
    if not torch.allclose(
        out.sum(dim=-1),
        torch.ones(n, device=out.device, dtype=out.dtype),
        atol=1e-5,
    ):
        raise RuntimeError("epsilon override left non-one-hot rows")
    return out, bern


class TopologyAwareHardMoE(nn.Module):
    """E0–E3 guilds with topology gate + hard Gumbel (train) / argmax (eval)."""

    def __init__(
        self,
        dim: int,
        *,
        num_experts: int = NUM_EXPERTS,
        gate_hidden: int = 16,
        temperature: float = 1.0,
        c: float = 1.0,
        eps: float = 1e-5,
        cv_coeff: float = 1.0,
        moe_quota_floor: float = 0.05,
        eval_proxy_temperature: float = 0.25,
        explore_epsilon: float = 0.0,
    ) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError("dim must be >= 1")
        if num_experts < 2:
            raise ValueError("num_experts must be >= 2")
        self.dim = int(dim)
        self.num_experts = int(num_experts)
        self.gate_hidden = int(gate_hidden)
        self.temperature = float(temperature)
        self.c = float(c)
        self.eps = float(eps)
        # Deprecated: scaling is harness SSOT (Sprint 9). Kept for ctor compat.
        self.cv_coeff = float(cv_coeff)
        self.moe_quota_floor = float(moe_quota_floor)
        self.eval_proxy_temperature = float(eval_proxy_temperature)
        self.explore_epsilon = float(explore_epsilon)
        # live = hard MoE (default). ablated = fixed expert-0 one-hot; no gate/experts 1-3.
        self.moe_mode = "live"

        gate_in = CHEM_GATE_DIM + 2  # chemistry + radius + non-ball marker
        self.gate_fc1 = nn.Linear(gate_in, gate_hidden)
        self.gate_fc2 = nn.Linear(gate_hidden + 1, num_experts)
        self.experts = nn.ModuleList([GyroExpert(dim) for _ in range(num_experts)])

    def set_temperature(self, tau: float) -> None:
        """Curriculum cool-down for Gumbel logits (tau > 0)."""
        if tau <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = float(tau)

    def set_explore_epsilon(self, epsilon: float) -> None:
        """§2.6 curriculum: per-node hard-override probability in train."""
        e = float(epsilon)
        if e < 0.0 or e > 1.0:
            raise ValueError("explore_epsilon must be in [0, 1]")
        self.explore_epsilon = e

    def set_moe_mode(self, mode: str) -> None:
        """``live`` (default) or ``ablated`` (fixed expert 0; gate frozen)."""
        m = str(mode).strip().lower()
        if m not in ("live", "ablated"):
            raise ValueError("moe_mode must be 'live' or 'ablated'")
        self.moe_mode = m
        train_gate = m == "live"
        for mod in (self.gate_fc1, self.gate_fc2):
            for p in mod.parameters():
                p.requires_grad = train_gate

    def _route(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``([N, E] hard assignment, [N] override mask)``.

        Train: Gumbel ``hard=True`` STE, then optional §2.6 epsilon swap.
        Eval: argmax only (epsilon ignored).
        """
        if self.training:
            # Frozen contract: hard=True required — soft mush forbidden.
            routing = F.gumbel_softmax(
                logits, tau=self.temperature, hard=True, dim=-1
            )
            routing, override = epsilon_greedy_hard_override(
                routing, epsilon=self.explore_epsilon
            )
            return routing, override
        idx = logits.argmax(dim=-1)
        routing = F.one_hot(idx, num_classes=self.num_experts).to(
            dtype=logits.dtype
        )
        return routing, routing.new_zeros(routing.shape[0])

    def forward(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        h: torch.Tensor | None = None,
        chem: torch.Tensor | None = None,
        tau_ceiling: float = 0.70,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Route post-transport states through a single hard-selected expert.

        Args:
            z: ``[N, d]`` manifold node states after hyp attention.
            edge_index: ``[2, E]`` sparse R0–R5 graph.
            h: unused Euclidean skip (compat); chemistry comes from ``chem``.
            chem: ``[N, 7]`` ρ/τ/SS/deg/SASA from the loader.
            tau_ceiling: curriculum radius ceiling — scales expert residual damp.

        Returns:
            ``(out, aux)`` with STE ``routing`` as the logged load tensor.
        """
        del h
        if z.ndim != 2 or z.shape[1] != self.dim:
            raise ValueError(f"z must be [N, {self.dim}]")

        z = project_to_ball(z, c=self.c, eps=self.eps)

        if self.moe_mode == "ablated":
            # Fixed expert-0 one-hot. Do not run gate or experts 1..E-1.
            # MoE balance / eps terms are exact zeros (not MoE evidence).
            n = int(z.shape[0])
            routing = z.new_zeros(n, self.num_experts)
            routing[:, 0] = 1.0
            out = self.experts[0](
                z, c=self.c, eps=self.eps, tau_ceiling=float(tau_ceiling)
            )
            out = project_to_ball(out, c=self.c, eps=self.eps)
            zero = z.new_zeros(())
            load = routing.mean(dim=0)
            soft = routing
            aux: dict[str, Any] = {
                "routing": routing,
                "soft": soft,
                "logits": routing,  # placeholder; unused when ablated
                "load": load,
                "soft_load": load,
                "cv_loss": zero,
                "quota_loss": zero,
                "majority_hinge_loss": zero,
                "switch_lb_loss": zero,
                "soft_quota_loss": zero,
                "eval_proxy": soft,
                "eval_proxy_load": load,
                "eval_proxy_lb_loss": zero,
                "eval_proxy_quota_loss": zero,
                "temperature": self.temperature,
                "explore_epsilon": 0.0,
                "explore_override_frac": 0.0,
                "hard": "ablated",
                "moe_mode": "ablated",
            }
            return out, aux

        feats = topology_gate_features(
            z, edge_index, chem=chem, c=self.c, eps=self.eps
        )
        hidden = F.silu(self.gate_fc1(feats))
        marker = torch.full(
            (hidden.shape[0], 1), 2.0, device=hidden.device, dtype=hidden.dtype
        )
        logits = self.gate_fc2(torch.cat([hidden, marker], dim=-1))
        routing, override_mask = self._route(logits)  # [N, E] one-hot — logged load

        stacked = torch.stack(
            [
                expert(z, c=self.c, eps=self.eps, tau_ceiling=float(tau_ceiling))
                for expert in self.experts
            ],
            dim=1,
        )
        out = (routing.unsqueeze(-1) * stacked).sum(dim=1)
        out = project_to_ball(out, c=self.c, eps=self.eps)

        # LOAD-BEARING: soft LB MUST use raw pre-Gumbel softmax(logits).
        # Never softmax(gumbel_noised) or hard STE rows — those mask collapse
        # the same way train moe_load_* did. See tokyo_eye_equ_moe_eval_util.
        soft = F.softmax(logits, dim=-1)
        load = routing.mean(dim=0)
        soft_load = soft.mean(dim=0)
        cv_loss = cv_load_balance_loss(routing)
        quota_loss = min_load_quota_loss(routing, floor=self.moe_quota_floor)
        majority_hinge = majority_hinge_loss(routing)
        switch_lb_loss = switch_load_balance_loss(soft)  # T=1 softmax — insufficient for argmax
        soft_quota_loss = min_load_quota_loss(soft, floor=self.moe_quota_floor)
        eval_proxy = eval_proxy_routing(
            logits, temperature=self.eval_proxy_temperature
        )
        eval_proxy_lb_loss = switch_load_balance_loss(eval_proxy)
        eval_proxy_quota_loss = min_load_quota_loss(
            eval_proxy, floor=self.moe_quota_floor
        )
        aux = {
            "routing": routing,
            "soft": soft,
            "logits": logits,
            "load": load,
            "soft_load": soft_load,
            "cv_loss": cv_loss,
            "quota_loss": quota_loss,
            "majority_hinge_loss": majority_hinge,
            "switch_lb_loss": switch_lb_loss,
            "soft_quota_loss": soft_quota_loss,
            "eval_proxy": eval_proxy,
            "eval_proxy_load": eval_proxy.mean(dim=0),
            "eval_proxy_lb_loss": eval_proxy_lb_loss,
            "eval_proxy_quota_loss": eval_proxy_quota_loss,
            "temperature": self.temperature,
            "explore_epsilon": self.explore_epsilon,
            "explore_override_frac": float(override_mask.mean().detach())
            if override_mask.numel()
            else 0.0,
            "hard": True if self.training else "argmax",
            "moe_mode": "live",
        }
        return out, aux


__all__ = [
    "NUM_EXPERTS",
    "TopologyAwareHardMoE",
    "CHEM_GATE_DIM",
    "GyroExpert",
    "cv_load_balance_loss",
    "majority_hinge_loss",
    "min_load_quota_loss",
    "switch_load_balance_loss",
    "eval_proxy_routing",
    "eval_proxy_load_balance_loss",
    "soft_and_hard_quota_losses",
    "epsilon_greedy_hard_override",
    "topology_gate_features",
]
