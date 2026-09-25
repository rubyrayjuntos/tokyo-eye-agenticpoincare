"""Sparse relation-aware Hyperbolic Graph Attention (TokyoEye-v8 / EQU correct_start).

FROZEN contract:
``docs/superpowers/specs/2026-07-22-tokyo-eye-v8-equiformer-hyp-design.md`` §7
+ pure-hyp amendment ``tokyo_eye_equ_pure_hyp_v1``.

* Sparse ``edge_index`` / ``edge_type`` only — no dense ``N×N`` logit board.
* Logits: ``(-d_H(Q_i, K_j) * γ_R + β_R) / sqrt(d_h)`` on each directed edge.
  With ``relation_value_path=True`` β_R is removed and relation typing moves to the
  value path (per-relation SO(d) rotation + gyroscalar on each edge message).
* Q/K/V/output via origin-centered **orthogonal gyro-maps** (ambient SO(d) on
  Poincaré coords) — **not** ``exp₀(W · log₀(z))``.
* Aggregation: Lorentz-weighted Einstein midpoint in Klein coordinates.
* Residual mix: Möbius addition of self with a **gyroscalar-damped**
  attended message (``self ⊕ (t ⊗ attended)``). Full-strength ``self ⊕
  attended`` inflates every radius past ``τ`` within two layers.
* Option A isolates: self gyro-map only — no empty barycenter.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.utils import scatter, softmax

# R0–R5
NUM_RELATIONS_DEFAULT = 6


def clamp_ball_radius(
    x: torch.Tensor, *, max_r: float, eps: float = 1e-5
) -> torch.Tensor:
    """Radial clamp toward the origin; used by ``project_to_ball`` and τ-ceiling."""
    x = torch.nan_to_num(x, nan=0.0, posinf=max_r, neginf=-max_r)
    r = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(eps)
    scale = torch.clamp(float(max_r) / r, max=1.0)
    return x * scale


def project_to_ball(x: torch.Tensor, *, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    """Clamp points strictly inside the open ball of radius ``1/sqrt(c)``."""
    max_r = (1.0 / math.sqrt(c)) - eps
    return clamp_ball_radius(x, max_r=max_r, eps=eps)


def poincare_dist(
    x: torch.Tensor,
    y: torch.Tensor,
    c: float = 1.0,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Strict Poincaré geodesic distance with stable gradients near x≈y."""
    x = project_to_ball(x, c=c, eps=eps)
    y = project_to_ball(y, c=c, eps=eps)
    sq_norm_x = torch.sum(x * x, dim=-1).clamp(max=1.0 / c - eps)
    sq_norm_y = torch.sum(y * y, dim=-1).clamp(max=1.0 / c - eps)
    sq_dist = torch.sum((x - y) * (x - y), dim=-1).clamp_min(eps)
    denom = (1.0 - c * sq_norm_x) * (1.0 - c * sq_norm_y)
    arg = 1.0 + 2.0 * c * sq_dist / denom.clamp(min=eps)
    arg = arg.clamp(min=1.0 + 1e-4, max=1.0e6)
    return torch.acosh(arg) / math.sqrt(c)


def lorentz_factor(x: torch.Tensor, *, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    """γ = 1 / sqrt(1 - c ||x||²) for Poincaré (or Klein) points with ||·||² < 1/c."""
    sq = torch.sum(x * x, dim=-1, keepdim=True).clamp(max=1.0 / c - eps)
    return 1.0 / torch.sqrt((1.0 - c * sq).clamp(min=eps))


def poincare_to_klein(x: torch.Tensor, *, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    """V_K = 2 V_H / (1 + c ||V_H||²) (plan §3.2)."""
    x = project_to_ball(x, c=c, eps=eps)
    sq = torch.sum(x * x, dim=-1, keepdim=True)
    return (2.0 * x) / (1.0 + c * sq).clamp(min=eps)


def klein_to_poincare(k: torch.Tensor, *, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    """V_H = V_K / (1 + sqrt(1 - c ||V_K||²)) (plan §3.2)."""
    sq = torch.sum(k * k, dim=-1, keepdim=True).clamp(max=1.0 / c - eps)
    denom = 1.0 + torch.sqrt((1.0 - c * sq).clamp(min=eps))
    x = k / denom.clamp(min=eps)
    return project_to_ball(x, c=c, eps=eps)


def exp_map_zero(v: torch.Tensor, *, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    """Exponential map at the origin of the Poincaré ball (lift / projector only)."""
    sqrt_c = math.sqrt(c)
    v_norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(eps)
    return project_to_ball(
        torch.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm),
        c=c,
        eps=eps,
    )


def log_map_zero(x: torch.Tensor, *, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    """Logarithmic map at the origin (readout / diagnostics — not QKV geometry)."""
    x = project_to_ball(x, c=c, eps=eps)
    sqrt_c = math.sqrt(c)
    x_norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(eps)
    u = (sqrt_c * x_norm).clamp(max=1.0 - 1e-4)
    return (torch.atanh(u) / (sqrt_c * x_norm)) * x


def mobius_add(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    c: float = 1.0,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Poincaré Möbius addition ⊕_c (on-manifold residual mix)."""
    x = project_to_ball(x, c=c, eps=eps)
    y = project_to_ball(y, c=c, eps=eps)
    xy = torch.sum(x * y, dim=-1, keepdim=True)
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    num = (1.0 + 2.0 * c * xy + c * y2) * x + (1.0 - c * x2) * y
    denom = 1.0 + 2.0 * c * xy + (c * c) * x2 * y2
    return project_to_ball(num / denom.clamp(min=eps), c=c, eps=eps)


def gyroscalar_mul(
    t: float | torch.Tensor,
    x: torch.Tensor,
    *,
    c: float = 1.0,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Gyrovector scalar multiply ``t ⊗_c x`` (on-manifold; ``t=1`` is identity).

    Full-strength ``self ⊕ attended`` stacks hyperbolic translations and drives
    radii past ``τ`` within two layers; damp the message with ``t ∈ (0, 1)``
    before Möbius residual mix.
    """
    x = project_to_ball(x, c=c, eps=eps)
    if not torch.is_tensor(t):
        t = x.new_tensor(float(t))
    else:
        t = t.to(device=x.device, dtype=x.dtype)
    while t.ndim < x.ndim:
        t = t.unsqueeze(-1)
    sqrt_c = math.sqrt(c)
    r = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(eps)
    u = (sqrt_c * r).clamp(max=1.0 - 1e-4)
    new_r = torch.tanh(t * torch.atanh(u)) / sqrt_c
    return project_to_ball(x * (new_r / r), c=c, eps=eps)


# Curriculum-start / v3 dual-seal bar — absolute damp was calibrated here.
RESIDUAL_TAU_REF = 0.70


def tau_relative_residual_step(
    residual_logit: torch.Tensor,
    *,
    tau_ceiling: float,
    tau_ref: float = RESIDUAL_TAU_REF,
    radius: torch.Tensor | None = None,
) -> torch.Tensor:
    """Learned gyroscalar step, scaled by curriculum headroom.

    Fixed ``t≈0.25`` is calibrated at ``τ_ref=0.70``. As the curriculum ceiling
    rises, the same absolute residual re-pins every node to the moving ``τ``
    (attn + post-MoE). Two scalings:

    1. ``τ_ref / τ`` (global) — shrink base strength above the calibration ref.
    2. Per-node headroom ``(τ − r)_+/τ`` when ``radius`` is provided — nodes
       already near the ceiling get almost no residual push.

    Returns a scalar or ``[N, 1]`` tensor broadcastable for ``gyroscalar_mul``.
    """
    t = torch.sigmoid(residual_logit)
    tau = max(float(tau_ceiling), 1e-3)
    ref = max(float(tau_ref), 1e-3)
    # Stronger than linear: mild (τ_ref/τ) was insufficient under training.
    scale = min(1.0, (ref / tau) ** 2)
    t = t * scale
    if radius is None:
        return t
    r = radius.reshape(-1, 1).to(device=t.device, dtype=t.dtype)
    headroom = ((tau - r) / tau).clamp(min=0.0, max=1.0)
    return t * headroom


def einstein_midpoint(
    points: torch.Tensor,
    weights: torch.Tensor,
    *,
    c: float = 1.0,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Weighted Einstein midpoint via Klein + Lorentz γ.

    Args:
        points: ``[N, d]`` Poincaré points (or already-indexed rows).
        weights: ``[N]`` or ``[N, 1]`` non-negative weights (need not sum to 1).
    """
    if weights.ndim == 1:
        weights = weights.unsqueeze(-1)
    k = poincare_to_klein(points, c=c, eps=eps)
    gamma = lorentz_factor(k, c=c, eps=eps)
    w_g = weights * gamma
    numer = torch.sum(w_g * k, dim=0, keepdim=True)
    denom = torch.sum(w_g, dim=0, keepdim=True).clamp(min=eps)
    bar_k = numer / denom
    return klein_to_poincare(bar_k, c=c, eps=eps).squeeze(0)


def sparse_einstein_klein_aggregate(
    v: torch.Tensor,
    attn: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    *,
    n: int,
    c: float = 1.0,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Sparse Einstein midpoint: Lorentz-weighted Klein mean over ``dst`` neighborhoods."""
    v_k = poincare_to_klein(v, c=c, eps=eps)
    gamma_k = lorentz_factor(v_k, c=c, eps=eps)
    msg = attn.unsqueeze(-1) * gamma_k[src] * v_k[src]
    wsum = scatter(
        attn.unsqueeze(-1) * gamma_k[src], dst, dim=0, dim_size=n, reduce="sum"
    )
    agg_k = scatter(msg, dst, dim=0, dim_size=n, reduce="sum")
    agg_k = agg_k / wsum.clamp(min=eps)
    return klein_to_poincare(agg_k, c=c, eps=eps)


class GyroOrthogonalMap(nn.Module):
    """Origin-centered gyro-rotation: ambient orthogonal action on ball coords.

    Not a renamed ``nn.Linear``. The live map is ``z ↦ z R`` after a **fresh**
    QR of the raw Parameter into ``SO(d)`` on every forward (``R`` is not
    cached). ``RᵀR = I``, ``det R = +1``. Radii from 0 are preserved.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self.dim = int(dim)
        # Near-identity free factor; QR yields the live SO(d) matrix.
        eye = torch.eye(dim)
        self.weight = nn.Parameter(eye.clone())
        with torch.no_grad():
            self.weight.add_(0.01 * torch.randn(dim, dim))

    def rotation(self) -> torch.Tensor:
        q, r = torch.linalg.qr(self.weight)
        # Absorb sign of R diag so Q has consistent orientation
        signs = torch.sign(torch.diagonal(r))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        q = q * signs.unsqueeze(0)
        # Force det = +1 (SO(d), not O(d))
        if self.dim >= 1:
            det = torch.det(q)
            if det < 0:
                q = q.clone()
                q[:, 0] = -q[:, 0]
        return q

    def forward(self, z: torch.Tensor, *, c: float, eps: float) -> torch.Tensor:
        # Fresh QR from the raw Parameter every call — R is never cached.
        r = self.rotation().to(dtype=z.dtype, device=z.device)
        return project_to_ball(z @ r, c=c, eps=eps)


class HyperbolicGraphAttention(nn.Module):
    """Relation-aware sparse hyp attention with Einstein midpoint aggregation."""

    def __init__(
        self,
        dim: int,
        *,
        num_relations: int = NUM_RELATIONS_DEFAULT,
        c: float = 1.0,
        eps: float = 1e-5,
        relation_value_path: bool = False,
    ) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError("dim must be >= 1")
        if num_relations < 1:
            raise ValueError("num_relations must be >= 1")
        self.dim = int(dim)
        self.num_relations = int(num_relations)
        self.c = float(c)
        self.eps = float(eps)

        self.R_q = GyroOrthogonalMap(dim)
        self.R_k = GyroOrthogonalMap(dim)
        self.R_v = GyroOrthogonalMap(dim)
        self.R_o = GyroOrthogonalMap(dim)

        self.relation_value_path = bool(relation_value_path)
        self.gamma = nn.Embedding(num_relations, 1)
        nn.init.ones_(self.gamma.weight)
        if self.relation_value_path:
            # Relation identity acts on message CONTENT, not on softmax weights:
            # the additive beta_R is a constant shift inside a destination-segmented
            # softmax and cancels wherever a neighborhood is single-relation.
            self.beta = None
            self.R_v_rel = nn.ModuleList(
                [GyroOrthogonalMap(dim) for _ in range(num_relations)]
            )
            self.v_scale = nn.Parameter(torch.ones(num_relations))
        else:
            self.beta = nn.Embedding(num_relations, 1)
            nn.init.zeros_(self.beta.weight)
        # sigmoid(-1.0986) ≈ 0.25 — full-strength Möbius residual pins τ.
        self.residual_logit = nn.Parameter(torch.tensor(-1.0986122886681098))
        self.last_attn: torch.Tensor | None = None
        self.last_edge_index: torch.Tensor | None = None
        self.last_edge_type: torch.Tensor | None = None

    def _self_transport(self, z: torch.Tensor) -> torch.Tensor:
        """Option A isolate / self path: on-manifold orthogonal gyro-map."""
        return self.R_o(z, c=self.c, eps=self.eps)

    def _relation_edge_values(
        self, v: torch.Tensor, src: torch.Tensor, rel: torch.Tensor
    ) -> torch.Tensor:
        """``v_ij = t_type(ij) ⊗ (v_j · R_type(ij))`` — on-manifold, per directed edge."""
        v_e = v[src]
        rotated = v_e.clone()
        for r in torch.unique(rel).tolist():
            m = rel == int(r)
            rotated[m] = self.R_v_rel[int(r)](v_e[m], c=self.c, eps=self.eps)
        t = self.v_scale[rel].clamp_min(0.05)
        return gyroscalar_mul(t, rotated, c=self.c, eps=self.eps)

    def forward(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        *,
        tau_ceiling: float = RESIDUAL_TAU_REF,
    ) -> torch.Tensor:
        """Sparse hyp attention.

        Args:
            z: ``[N, d]`` Poincaré ball features.
            edge_index: ``[2, E]`` directed sparse edges.
            edge_type: ``[E]`` relation ids in ``{0..num_relations-1}``.
            edge_attr: unused (Sprint-2 attrs reserved for later bias).
            tau_ceiling: curriculum radius ceiling — scales residual damp.
        """
        del edge_attr
        if z.ndim != 2:
            raise ValueError(f"z must be [N, d], got {tuple(z.shape)}")
        n, d = z.shape
        if d != self.dim:
            raise ValueError(f"z dim {d} != layer dim {self.dim}")

        z = project_to_ball(z, c=self.c, eps=self.eps)
        self_out = self._self_transport(z)

        if edge_index.numel() == 0:
            return self_out

        if edge_index.shape[0] != 2:
            raise ValueError("edge_index must be [2, E]")
        e = int(edge_index.shape[1])
        if edge_type.numel() != e:
            raise ValueError("edge_type length must match E")
        if int(edge_type.min()) < 0 or int(edge_type.max()) >= self.num_relations:
            raise ValueError(
                f"edge_type out of range for num_relations={self.num_relations}"
            )

        src = edge_index[0].long()
        dst = edge_index[1].long()
        rel = edge_type.long()

        # On-manifold Q/K/V (plan §3.1): R ⊗_c z via ambient SO(d) at origin
        q = self.R_q(z, c=self.c, eps=self.eps)
        k = self.R_k(z, c=self.c, eps=self.eps)
        v = self.R_v(z, c=self.c, eps=self.eps)

        d_ij = poincare_dist(q[src], k[dst], c=self.c, eps=self.eps)
        gamma_r = self.gamma(rel).squeeze(-1)
        if self.relation_value_path:
            logits = (-d_ij * gamma_r) / math.sqrt(self.dim)
        else:
            beta_r = self.beta(rel).squeeze(-1)
            logits = (-d_ij * gamma_r + beta_r) / math.sqrt(self.dim)

        # Freeze §7.1: segmented softmax over destination neighborhoods.
        attn = softmax(logits, dst, num_nodes=n)
        self.last_attn = attn
        self.last_edge_index = edge_index
        self.last_edge_type = edge_type

        # Einstein midpoint in Klein with Lorentz γ (plan §3.2)
        if self.relation_value_path:
            v_edge = self._relation_edge_values(v, src, rel)
            # Per-edge message points: reuse the whitelisted midpoint with src=arange(E)
            # (identical math; keeps the pure-hyp gate's manifold-formula list unchanged).
            agg_h = sparse_einstein_klein_aggregate(
                v_edge,
                attn,
                torch.arange(v_edge.shape[0], device=v_edge.device),
                dst,
                n=n,
                c=self.c,
                eps=self.eps,
            )
        else:
            agg_h = sparse_einstein_klein_aggregate(
                v, attn, src, dst, n=n, c=self.c, eps=self.eps
            )
        attended = self.R_o(agg_h, c=self.c, eps=self.eps)

        deg = scatter(
            torch.ones(e, device=z.device, dtype=z.dtype),
            dst,
            dim=0,
            dim_size=n,
            reduce="sum",
        )
        isolate = deg <= 0
        # On-manifold mix (forbidden: tangent residual then exp₀).
        # τ-relative + headroom damp — fixed t re-pins to moving curriculum ceiling.
        r_self = torch.linalg.vector_norm(self_out, dim=-1)
        step = tau_relative_residual_step(
            self.residual_logit,
            tau_ceiling=float(tau_ceiling),
            radius=r_self,
        )
        attended_step = gyroscalar_mul(step, attended, c=self.c, eps=self.eps)
        mixed = mobius_add(self_out, attended_step, c=self.c, eps=self.eps)
        out = torch.where(isolate.unsqueeze(-1), self_out, mixed)
        return project_to_ball(out, c=self.c, eps=self.eps)


__all__ = [
    "NUM_RELATIONS_DEFAULT",
    "GyroOrthogonalMap",
    "HyperbolicGraphAttention",
    "einstein_midpoint",
    "sparse_einstein_klein_aggregate",
    "exp_map_zero",
    "klein_to_poincare",
    "log_map_zero",
    "lorentz_factor",
    "mobius_add",
    "gyroscalar_mul",
    "tau_relative_residual_step",
    "RESIDUAL_TAU_REF",
    "poincare_dist",
    "poincare_to_klein",
    "project_to_ball",
    "clamp_ball_radius",
]
