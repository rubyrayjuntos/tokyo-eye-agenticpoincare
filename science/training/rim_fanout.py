"""Rim-conditional disc fan-out — model forward spread + training losses."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _unit_disc_directions(hyp_proj_2d: torch.Tensor) -> torch.Tensor:
    r = hyp_proj_2d.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return hyp_proj_2d / r


def rim_angular_repulsion_loss(
    hyp_proj_2d: torch.Tensor,
    ca_coords: torch.Tensor,
    *,
    min_r_rim: float = 0.35,
    min_angular_sep: float = 0.12,
    spatial_exempt_cutoff: float = 8.0,
    scale: float = 4.0,
    subsample_pairs: int = 1024,
) -> dict[str, torch.Tensor]:
    """Repel non-neighbor rim pairs that collapse to the same hyperbolic ray.

    Only pairs with both disc radii >= ``min_r_rim`` and Cα distance >= cutoff
    (not local 3D neighbors) contribute. Penalizes small chord distance between
    unit disc directions — orthogonal spread at fixed rim radius.
    """
    xy = hyp_proj_2d
    z = torch.zeros((), device=xy.device, dtype=xy.dtype)
    n = xy.shape[0]
    if n < 4 or ca_coords is None or ca_coords.shape[0] != n:
        return {"rim_angular_repulsion": z, "rim_angular_pairs": z}

    r = xy.norm(dim=-1)
    rim = r >= float(min_r_rim)
    rim_idx = torch.nonzero(rim, as_tuple=False).squeeze(-1)
    if rim_idx.numel() < 2:
        return {"rim_angular_repulsion": z, "rim_angular_pairs": z}

    dirs = _unit_disc_directions(xy[rim_idx])
    ca = ca_coords[rim_idx].float()
    ca_d = torch.cdist(ca, ca)
    ang_d = torch.cdist(dirs, dirs)
    m = rim_idx.numel()
    tri = torch.triu(torch.ones(m, m, device=xy.device, dtype=torch.bool), diagonal=1)
    eligible = tri & (ca_d >= float(spatial_exempt_cutoff))
    if not bool(eligible.any()):
        return {"rim_angular_repulsion": z, "rim_angular_pairs": z}

    sep = ang_d[eligible]
    rim_w = ((r[rim_idx].unsqueeze(0) + r[rim_idx].unsqueeze(1)) / 2.0)[eligible]
    penalty = torch.relu(float(min_angular_sep) - sep) * rim_w
    if penalty.numel() > subsample_pairs:
        idx = torch.randperm(penalty.numel(), device=xy.device)[:subsample_pairs]
        penalty = penalty[idx]
    loss = penalty.pow(2).mean() * float(scale)
    return {
        "rim_angular_repulsion": loss,
        "rim_angular_pairs": torch.tensor(float(penalty.numel()), device=xy.device),
    }


def rim_pc2_floor_loss(
    hyp_proj_2d: torch.Tensor,
    *,
    min_r_rim: float = 0.35,
    min_pc2_std: float = 0.06,
    scale: float = 3.0,
) -> dict[str, torch.Tensor]:
    """PC2 spread floor on rim-band residues only (not corpus-wide occupancy)."""
    xy = hyp_proj_2d
    z = torch.zeros((), device=xy.device, dtype=xy.dtype)
    r = xy.norm(dim=-1)
    rim_xy = xy[r >= float(min_r_rim)]
    if rim_xy.shape[0] < 3:
        return {"rim_pc2_floor": z, "rim_pc2_std": z}

    X = rim_xy.float() - rim_xy.float().mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(X, full_matrices=False)
    pc2 = X @ vh[1]
    pc2_std = pc2.std()
    loss = torch.relu(float(min_pc2_std) - pc2_std) * float(scale)
    return {"rim_pc2_floor": loss, "rim_pc2_std": pc2_std.detach()}


def apply_rim_fanout_spread(
    hyp_proj_2d: torch.Tensor,
    ca_coords: torch.Tensor | None = None,
    *,
    min_r_rim: float = 0.35,
    min_angular_sep: float = 0.12,
    spatial_exempt_cutoff: float = 8.0,
    spread_strength: float | torch.Tensor = 0.12,
) -> torch.Tensor:
    """Model forward: preserve rim radii; rotate crowded rim directions apart."""
    xy = hyp_proj_2d
    n = xy.shape[0]
    if n < 2:
        return xy

    r = xy.norm(dim=-1).clamp(min=1e-8)
    rim = r >= float(min_r_rim)
    rim_idx = torch.nonzero(rim, as_tuple=False).squeeze(-1)
    if rim_idx.numel() < 2:
        return xy

    dirs = xy[rim_idx] / r[rim_idx].unsqueeze(-1).clamp(min=1e-8)
    rim_r = r[rim_idx].unsqueeze(-1)
    m = int(rim_idx.numel())

    ang_d = torch.cdist(dirs, dirs)
    weight = torch.relu(float(min_angular_sep) - ang_d)
    weight = weight * (1.0 - torch.eye(m, device=xy.device, dtype=xy.dtype))

    if ca_coords is not None and ca_coords.shape[0] == n:
        ca_rim = ca_coords[rim_idx].float()
        spatial_ok = torch.cdist(ca_rim, ca_rim) >= float(spatial_exempt_cutoff)
        weight = weight * spatial_ok.to(dtype=weight.dtype)

    perp = torch.stack([-dirs[:, 1], dirs[:, 0]], dim=-1)
    cross = (
        dirs[:, 0].unsqueeze(1) * dirs[:, 1].unsqueeze(0)
        - dirs[:, 1].unsqueeze(1) * dirs[:, 0].unsqueeze(0)
    )
    idx = torch.arange(m, device=xy.device)
    alt = torch.where(idx.unsqueeze(0) < idx.unsqueeze(1), 1.0, -1.0)
    signed = torch.where(cross.abs() < 1e-6, alt, torch.sign(cross))
    force = signed.unsqueeze(-1) * perp.unsqueeze(1)
    push = torch.einsum("ij,ijc->ic", weight, force)
    denom = weight.sum(dim=1, keepdim=True).clamp(min=1e-6)
    push = push / denom

    strength = spread_strength
    if not isinstance(strength, torch.Tensor):
        strength = torch.tensor(float(strength), device=xy.device, dtype=xy.dtype)
    new_dirs = F.normalize(dirs + strength * push, p=2, dim=-1, eps=1e-8)
    out = xy.clone()
    out[rim_idx] = new_dirs * rim_r
    return out


class RimFanoutSpread(nn.Module):
    """Differentiable rim angular spreading applied in the GNN forward pass."""

    def __init__(
        self,
        *,
        min_r_rim: float = 0.35,
        min_angular_sep: float = 0.12,
        spatial_exempt_cutoff: float = 8.0,
        spread_strength: float = 0.12,
        learnable_strength: bool = True,
        max_strength: float = 0.35,
    ):
        super().__init__()
        self.min_r_rim = float(min_r_rim)
        self.min_angular_sep = float(min_angular_sep)
        self.spatial_exempt_cutoff = float(spatial_exempt_cutoff)
        self.max_strength = float(max_strength)
        init = max(float(spread_strength), 1e-4)
        if learnable_strength:
            self.log_strength = nn.Parameter(torch.tensor(init).log())
        else:
            self.register_buffer("log_strength", torch.tensor(init).log())

    @property
    def strength(self) -> torch.Tensor:
        return self.log_strength.exp().clamp(max=self.max_strength)

    def forward(
        self,
        hyp_proj_2d: torch.Tensor,
        ca_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return apply_rim_fanout_spread(
            hyp_proj_2d,
            ca_coords,
            min_r_rim=self.min_r_rim,
            min_angular_sep=self.min_angular_sep,
            spatial_exempt_cutoff=self.spatial_exempt_cutoff,
            spread_strength=self.strength,
        )


__all__ = [
    "RimFanoutSpread",
    "apply_rim_fanout_spread",
    "rim_angular_repulsion_loss",
    "rim_pc2_floor_loss",
]
