"""Expert–θ geometry: untie crest/barrier specialists from shared rays."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def expert_angular_diversity_loss(
    hyp_proj_2d: torch.Tensor,
    expert_weights: torch.Tensor,
    *,
    min_r: float = 0.12,
    max_resultant_length: float = 0.55,
    min_mean_sep: float = 0.55,
    concentration_scale: float = 3.0,
    separation_scale: float = 2.0,
    r_softness: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Penalize ray-like expert angular collapse + reward expert mean separation.

    Soft-weights residues by gate mass × mid/rim radius. High circular resultant
    length R means an expert is pinned to a crest/barrier ray (e.g. 1F88 e1).
    """
    xy = hyp_proj_2d
    w_exp = expert_weights
    z = torch.zeros((), device=xy.device, dtype=xy.dtype)
    n, e = w_exp.shape[0], w_exp.shape[1]
    if n < 4 or e < 2 or xy.shape[0] != n:
        return {
            "expert_angular_diversity": z,
            "expert_angular_concentration": z,
            "expert_angular_separation": z,
            "expert_angular_mean_R": z,
            "expert_angular_min_sep": z,
        }

    r = xy.norm(dim=-1).clamp(min=1e-8)
    theta = torch.atan2(xy[:, 1], xy[:, 0])
    rim_w = torch.sigmoid((r - float(min_r)) / max(float(r_softness), 1e-4))

    mean_dirs = []
    r_lengths = []
    for ei in range(e):
        ww = w_exp[:, ei] * rim_w
        mass = ww.sum().clamp(min=1e-6)
        c = (ww * torch.cos(theta)).sum() / mass
        s = (ww * torch.sin(theta)).sum() / mass
        R = torch.sqrt(c * c + s * s + 1e-12)
        mean_dirs.append(torch.stack([c, s]))
        r_lengths.append(R)

    R_stack = torch.stack(r_lengths)
    dirs = F.normalize(torch.stack(mean_dirs), p=2, dim=-1, eps=1e-8)

    conc = torch.relu(R_stack - float(max_resultant_length)).pow(2).mean() * float(
        concentration_scale
    )

    # Pairwise chord distance between expert circular means.
    if e >= 2:
        d = torch.cdist(dirs, dirs)
        tri = torch.triu(torch.ones(e, e, device=xy.device, dtype=torch.bool), diagonal=1)
        seps = d[tri]
        sep_loss = torch.relu(float(min_mean_sep) - seps).pow(2).mean() * float(
            separation_scale
        )
        min_sep = seps.min().detach()
    else:
        sep_loss = z
        min_sep = z

    return {
        "expert_angular_diversity": conc + sep_loss,
        "expert_angular_concentration": conc.detach(),
        "expert_angular_separation": sep_loss.detach(),
        "expert_angular_mean_R": R_stack.mean().detach(),
        "expert_angular_min_sep": min_sep,
    }


def expert_sector_recruit_loss(
    hyp_proj_2d: torch.Tensor,
    expert_weights: torch.Tensor,
    *,
    min_r: float = 0.12,
    n_bins: int = 12,
    temperature: float = 0.20,
    global_min_bin_frac: float = 0.35,
    expert_min_bin_frac: float = 0.25,
    scale: float = 3.0,
    r_softness: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Recruit every expert into globally underfilled θ bins (anti-barrier fill).

    Global soft occupancy finds empty sectors; each expert's own soft histogram
    is floored on those bins so crest specialists cannot monopolize the walls
    while leaving the wedge to nobody.
    """
    xy = hyp_proj_2d
    w_exp = expert_weights
    z = torch.zeros((), device=xy.device, dtype=xy.dtype)
    n, e = w_exp.shape[0], w_exp.shape[1]
    n_bins = int(n_bins)
    if n < 4 or e < 2 or n_bins < 4 or xy.shape[0] != n:
        return {
            "expert_sector_recruit": z,
            "expert_sector_empty_bins": z,
        }

    r = xy.norm(dim=-1).clamp(min=1e-8)
    dirs = xy / r.unsqueeze(-1)
    rim_w = torch.sigmoid((r - float(min_r)) / max(float(r_softness), 1e-4))
    if float(rim_w.detach().sum()) < 1e-4:
        return {"expert_sector_recruit": z, "expert_sector_empty_bins": z}

    angles = torch.linspace(
        0.0, 2.0 * math.pi, n_bins + 1, device=xy.device, dtype=xy.dtype
    )[:-1]
    centers = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
    soft = torch.softmax((dirs @ centers.T) / max(float(temperature), 1e-4), dim=-1)

    # Global mid/rim occupancy [B]
    global_mass = (soft * rim_w.unsqueeze(-1)).sum(dim=0)
    global_mass = global_mass / global_mass.sum().clamp(min=1e-8)
    g_floor = float(global_min_bin_frac) / float(n_bins)
    underfilled = global_mass < g_floor
    if not bool(underfilled.any()):
        return {
            "expert_sector_recruit": z,
            "expert_sector_empty_bins": underfilled.float().sum().detach(),
        }

    # Per-expert soft histograms [E, B], rim-weighted.
    # mass[e,b] = sum_i w_exp[i,e] * rim_w[i] * soft[i,b]
    weighted = w_exp * rim_w.unsqueeze(-1)  # [N, E]
    expert_mass = torch.einsum("ne,nb->eb", weighted, soft)
    expert_mass = expert_mass / expert_mass.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    e_floor = float(expert_min_bin_frac) / float(n_bins)
    deficits = torch.relu(e_floor - expert_mass[:, underfilled])
    loss = deficits.pow(2).mean() * float(scale)

    return {
        "expert_sector_recruit": loss,
        "expert_sector_empty_bins": underfilled.float().sum().detach(),
    }


__all__ = [
    "expert_angular_diversity_loss",
    "expert_sector_recruit_loss",
]
