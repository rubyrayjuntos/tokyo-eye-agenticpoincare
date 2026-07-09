"""Hyperbolic MoE primitives — prototype gate and manifold-aware mixing."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from geoopt.manifolds.stereographic import math as pmath

from science.dtie.v5.gnn.model import MobiusLinear
from science.training.routing_metrics import routing_load_floor_penalty


class HyperbolicPrototypeBank(nn.Module):
    """Learnable expert prototypes as tangent params mapped with dynamic curvature."""

    def __init__(self, num_prototypes: int, dim: int, init_scale: float = 1e-3) -> None:
        super().__init__()
        self.prototype_tangent = nn.Parameter(init_scale * torch.randn(num_prototypes, dim))

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        proto_hyp = pmath.expmap0(self.prototype_tangent, k=k)
        return pmath.project(proto_hyp, k=k)


def geodesic_interp(x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Geodesic interpolation between x and y at t (broadcastable [N,1] or scalar)."""
    v = pmath.logmap(x, y, k=k)
    if t.dim() == 1:
        t = t.unsqueeze(-1)
    return pmath.expmap(x, t * v, k=k)


def mobius_weighted_combine(
    points: torch.Tensor,
    weights: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    """
    Approximate hyperbolic weighted barycenter via sequential geodesic steps.

    Args:
        points: [N, E, D] on the Poincaré ball
        weights: [N, E] routing probabilities (non-negative)
        k: curvature tensor (negative c)
    """
    w = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    order = w.argsort(dim=-1, descending=True)
    idx0 = order[:, 0]
    combined = points[torch.arange(points.size(0), device=points.device), idx0]
    mass = w.gather(1, idx0.unsqueeze(1)).squeeze(1)
    for j in range(1, points.size(1)):
        idx = order[:, j]
        pt = points[torch.arange(points.size(0), device=points.device), idx]
        wj = w.gather(1, idx.unsqueeze(1)).squeeze(1)
        alpha = (wj / (mass + wj + 1e-8)).unsqueeze(-1)
        combined = geodesic_interp(combined, pt, alpha, k=k)
        mass = mass + wj
    return pmath.project(combined, k=k)


def project_ball(
    hyp: torch.Tensor,
    k: torch.Tensor,
    *,
    softness: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project onto Poincaré ball with soft radial ceiling.

    ``softness`` blends toward identity (1.0 = only ``pmath.project``; 0.0 = hard 0.99 ceiling).
    """
    hyp = pmath.project(hyp, k=k)
    r = hyp.norm(dim=-1, keepdim=True)
    r_ball = torch.rsqrt(torch.clamp(-k, min=1e-8))
    ceiling = 0.99 * r_ball
    hard_scale = torch.clamp(ceiling / (r + 1e-8), max=1.0)
    blend = max(0.0, min(1.0, float(softness)))
    scale = blend + (1.0 - blend) * hard_scale
    hyp = hyp * scale
    return hyp, r


def project_disc_2d(
    hyp_proj_2d: torch.Tensor,
    k: torch.Tensor,
    *,
    softness: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clamp 2D Poincaré disc coords and return (xy, radial norm)."""
    return project_ball(hyp_proj_2d, k=k, softness=softness)


def project_disc_2d_legacy(
    hyp_proj_2d: torch.Tensor,
    k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-arch-fix projection: post-routing head + hard 0.99 ball ceiling."""
    return project_ball(hyp_proj_2d, k=k, softness=0.0)


class HyperbolicPrototypeGate(nn.Module):
    """
    MoE gate with Möbius trunk on x_hyp and distance-to-prototype logits.

    Routing: logit_e = -scale * d_H(fused_hyp, prototype_e) + bias_e
    Optional pre-routing disc readout (gate_disc_proj) feeds xy + |p| into topo encoder.
    """

    TOPO_DIM = 8
    DISC_DIM = 3  # disc_x, disc_y, disc_r

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int = 4,
        capacity_threshold: float = 0.4,
        expert_dropout_p: float = 0.15,
        min_usage: float = 0.05,
        topology_only: bool = False,
        use_disc_position: bool = True,
        disc_feature_scale: float = 1.0,
        use_gumbel: bool = False,
        gumbel_temperature: float = 1.0,
        deep_gate: bool = False,
        structure_gate: bool = False,
        init_scale: float = 1e-3,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.capacity_threshold = capacity_threshold
        self.expert_dropout_p = expert_dropout_p
        self.min_usage = min_usage
        self.topology_only = topology_only
        self.use_disc_position = use_disc_position
        self.disc_feature_scale = disc_feature_scale
        self.use_gumbel = use_gumbel
        self.gumbel_temperature = gumbel_temperature
        self.deep_gate = deep_gate
        self.detach_gate_input = False
        # Epoch-level hard bans (train-only); set by ExpertTimeoutController.
        self.expert_timeout_banned: list[int] = []

        if not topology_only:
            self.mobius1 = MobiusLinear(hidden_dim, hidden_dim)
            self.mobius2 = MobiusLinear(hidden_dim, hidden_dim)
            if deep_gate:
                self.mobius3 = MobiusLinear(hidden_dim, hidden_dim)

        if use_disc_position:
            self.gate_disc_proj = MobiusLinear(hidden_dim, 2)

        topo_in = self.TOPO_DIM + (self.DISC_DIM if use_disc_position else 0)
        self.topo_encoder = nn.Sequential(
            nn.Linear(topo_in, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.prototype_bank = HyperbolicPrototypeBank(num_experts, hidden_dim, init_scale=init_scale)
        self.expert_bias = nn.Parameter(torch.zeros(num_experts))
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.structure_gate = structure_gate
        if structure_gate:
            self.structure_proj = nn.Linear(hidden_dim, num_experts, bias=False)
            nn.init.zeros_(self.structure_proj.weight)

        self.register_buffer("degree_mean", torch.tensor(0.0))
        self.register_buffer("degree_var", torch.tensor(1.0))
        self.register_buffer("rho_mean", torch.tensor(0.0))
        self.register_buffer("rho_var", torch.tensor(1.0))
        self.register_buffer("num_updates", torch.tensor(0))

    def _update_running_stats(self, log_degree: torch.Tensor, rho: torch.Tensor) -> None:
        momentum = 0.1
        with torch.no_grad():
            d_mean = log_degree.mean()
            d_var = log_degree.var(unbiased=False)
            r_mean = rho.mean()
            r_var = rho.var(unbiased=False)
            if self.num_updates == 0:
                self.degree_mean.copy_(d_mean)
                self.degree_var.copy_(d_var)
                self.rho_mean.copy_(r_mean)
                self.rho_var.copy_(r_var)
            else:
                self.degree_mean.mul_(1 - momentum).add_(momentum * d_mean)
                self.degree_var.mul_(1 - momentum).add_(momentum * d_var)
                self.rho_mean.mul_(1 - momentum).add_(momentum * r_mean)
                self.rho_var.mul_(1 - momentum).add_(momentum * r_var)
            self.num_updates.add_(1)

    def _topo_features(
        self,
        clustering: torch.Tensor,
        cone_depth: torch.Tensor,
        degree: torch.Tensor,
        rho: torch.Tensor,
        ss_onehot: torch.Tensor,
        tau_flag: torch.Tensor | None = None,
        disc_xy: torch.Tensor | None = None,
        disc_r: torch.Tensor | None = None,
    ) -> torch.Tensor:
        log_degree = torch.log1p(degree.float())
        if self.training:
            self._update_running_stats(log_degree, rho)
        norm_degree = (log_degree - self.degree_mean) / (self.degree_var.sqrt() + 1e-8)
        norm_rho = (rho - self.rho_mean) / (self.rho_var.sqrt() + 1e-8)
        if tau_flag is None:
            tau_feat = torch.zeros_like(clustering).unsqueeze(-1)
        else:
            tau_feat = tau_flag.unsqueeze(-1).float()
        parts = [
            clustering.unsqueeze(-1),
            cone_depth.detach(),
            norm_degree.unsqueeze(-1),
            norm_rho.unsqueeze(-1),
            tau_feat,
            ss_onehot,
        ]
        if self.use_disc_position and disc_xy is not None and disc_r is not None:
            scale = self.disc_feature_scale
            parts.extend([(scale * disc_xy).detach(), (scale * disc_r).detach()])
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        x_hyp: torch.Tensor,
        k: torch.Tensor,
        clustering: torch.Tensor,
        cone_depth: torch.Tensor,
        degree: torch.Tensor,
        rho: torch.Tensor,
        ss_onehot: torch.Tensor,
        *,
        tau_flag: torch.Tensor | None = None,
        disc_xy: torch.Tensor | None = None,
        disc_r: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        c = -k
        if self.use_disc_position:
            if disc_xy is None or disc_r is None:
                gate_x_disc = x_hyp.detach() if self.detach_gate_input else x_hyp
                disc_raw = self.gate_disc_proj(gate_x_disc, c=c)
                disc_xy, disc_r = project_disc_2d(disc_raw, k=k)

        topo = self._topo_features(
            clustering,
            cone_depth,
            degree,
            rho,
            ss_onehot,
            tau_flag=tau_flag,
            disc_xy=disc_xy,
            disc_r=disc_r,
        )
        topo_tangent = self.topo_encoder(topo)
        topo_hyp = pmath.project(pmath.expmap0(topo_tangent, k=k), k=k)

        if self.topology_only:
            fused_hyp = topo_hyp
        else:
            gate_x = x_hyp.detach() if self.detach_gate_input else x_hyp
            z_hyp = pmath.project(self.mobius1(gate_x, c=c), k=k)
            z_hyp = pmath.project(self.mobius2(z_hyp, c=c), k=k)
            if self.deep_gate:
                z_hyp = pmath.project(self.mobius3(z_hyp, c=c), k=k)
            fused_hyp = pmath.project(pmath.mobius_add(z_hyp, topo_hyp, k=k), k=k)

        proto_hyp = self.prototype_bank(k)
        fused_expand = fused_hyp.unsqueeze(1).expand(-1, self.num_experts, -1)
        proto_expand = proto_hyp.unsqueeze(0).expand(fused_hyp.size(0), -1, -1)
        dists = pmath.dist(fused_expand, proto_expand, k=k)
        raw_logits = -F.softplus(self.logit_scale) * dists + self.expert_bias
        if self.structure_gate:
            # Per-structure pooling: one protein per forward pass in training.
            struct_pool = topo_tangent.mean(dim=0, keepdim=True)
            raw_logits = raw_logits + self.structure_proj(struct_pool).expand_as(raw_logits)

        scores_initial = F.softmax(raw_logits, dim=-1)
        expected_load = scores_initial.mean(dim=0)
        overload_penalty = torch.relu(expected_load - self.capacity_threshold) ** 2
        adjusted_logits = raw_logits - 2.0 * overload_penalty.unsqueeze(0)

        if self.training and torch.rand(1).item() < self.expert_dropout_p:
            drop_idx = torch.randint(0, self.num_experts, (1,)).item()
            adjusted_logits[:, drop_idx] = -1e9

        if self.training and self.expert_timeout_banned:
            for ban_idx in self.expert_timeout_banned:
                if 0 <= int(ban_idx) < self.num_experts:
                    adjusted_logits[:, int(ban_idx)] = -1e9

        if self.use_gumbel and self.training:
            scores = F.gumbel_softmax(
                adjusted_logits, tau=self.gumbel_temperature, hard=True, dim=-1
            )
        elif self.use_gumbel and not self.training:
            scores = F.one_hot(adjusted_logits.argmax(dim=-1), self.num_experts).float()
        else:
            scores = F.softmax(adjusted_logits, dim=-1)
        soft_scores = F.softmax(adjusted_logits, dim=-1)
        f_soft = soft_scores.mean(dim=0)
        capacity_loss = torch.relu(self.min_usage - f_soft).pow(2).sum()
        routing_load_floor = routing_load_floor_penalty(f_soft, self.min_usage)

        audit: dict[str, Any] = {
            "gate_space": "hyperbolic",
            "gate_readout": "distance_to_expert_prototypes",
            "prototype_norm_mean": float(proto_hyp.norm(dim=-1).mean().detach()),
            "topology_only": self.topology_only,
            "gate_disc_input": self.use_disc_position,
            "deep_gate": self.deep_gate,
        }
        if disc_r is not None:
            audit["gate_disc_r_mean"] = float(disc_r.mean().detach())
            audit["gate_disc_r_std"] = float(disc_r.std().detach())
            audit["gate_disc_external"] = disc_xy is not None
        audit["routing_load_floor"] = routing_load_floor
        return scores, capacity_loss, audit
