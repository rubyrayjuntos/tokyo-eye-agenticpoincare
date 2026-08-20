"""Expert angular anti-barrier losses."""

from __future__ import annotations

import torch

from science.training.expert_angular import (
    expert_angular_diversity_loss,
    expert_sector_recruit_loss,
)


def test_expert_angular_diversity_penalizes_shared_ray() -> None:
    # Two experts pinned to +x; two opposite — high R / low sep vs spread experts.
    n = 24
    xy = torch.zeros(n, 2)
    xy[:12, 0] = 0.40
    xy[12:, 0] = -0.40
    # Soft gates: e0/e1 own +x, e2/e3 own -x but also collapse e0+e1 together
    w_ray = torch.zeros(n, 4)
    w_ray[:12, 0] = 0.7
    w_ray[:12, 1] = 0.3
    w_ray[12:, 2] = 0.7
    w_ray[12:, 3] = 0.3
    # Better: experts at 90° spacing
    xy_spread = torch.tensor(
        [[0.40, 0.0], [0.0, 0.40], [-0.40, 0.0], [0.0, -0.40]] * 6,
        dtype=torch.float32,
    )
    w_spread = torch.zeros(24, 4)
    for i in range(24):
        w_spread[i, i % 4] = 1.0

    ray = expert_angular_diversity_loss(
        xy, w_ray, min_r=0.12, max_resultant_length=0.55, min_mean_sep=0.55
    )
    spread = expert_angular_diversity_loss(
        xy_spread, w_spread, min_r=0.12, max_resultant_length=0.55, min_mean_sep=0.55
    )
    assert ray["expert_angular_diversity"] > spread["expert_angular_diversity"]
    assert ray["expert_angular_mean_R"] >= spread["expert_angular_mean_R"]


def test_expert_sector_recruit_penalizes_missing_expert_in_empty_bin() -> None:
    # Mass only in +x half; e3 has zero weight there and nowhere else → needs recruit.
    n = 16
    angles = torch.linspace(-0.3, 0.3, n)
    xy = torch.stack([0.40 * torch.cos(angles), 0.40 * torch.sin(angles)], dim=-1)
    w = torch.zeros(n, 4)
    w[:, :3] = 1.0 / 3.0  # e3 absent
    bad = expert_sector_recruit_loss(xy, w, min_r=0.12, n_bins=8, expert_min_bin_frac=0.5)
    w_ok = torch.ones(n, 4) / 4.0
    # Full ring so no underfilled global bins
    ring = torch.tensor(
        [
            [0.4, 0.0],
            [0.28, 0.28],
            [0.0, 0.4],
            [-0.28, 0.28],
            [-0.4, 0.0],
            [-0.28, -0.28],
            [0.0, -0.4],
            [0.28, -0.28],
        ]
        * 2,
        dtype=torch.float32,
    )
    w_ring = torch.ones(16, 4) / 4.0
    good = expert_sector_recruit_loss(
        ring, w_ring, min_r=0.12, n_bins=8, expert_min_bin_frac=0.5
    )
    assert bad["expert_sector_recruit"] > good["expert_sector_recruit"]
    assert bad["expert_sector_empty_bins"] >= 1


def test_expert_angular_losses_differentiable() -> None:
    xy = torch.randn(20, 2) * 0.2 + torch.tensor([0.25, 0.0])
    xy = xy.clone().requires_grad_(True)
    w = torch.softmax(torch.randn(20, 4), dim=-1)
    loss = expert_angular_diversity_loss(xy, w)["expert_angular_diversity"]
    loss = loss + expert_sector_recruit_loss(xy, w)["expert_sector_recruit"]
    loss.backward()
    assert xy.grad is not None
