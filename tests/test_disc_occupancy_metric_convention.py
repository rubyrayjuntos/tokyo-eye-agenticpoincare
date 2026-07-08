"""σ₂/σ₁ direction: high ratio = healthy spread, low = rank-1 streak."""

import numpy as np

from science.training.disc_occupancy import disc_occupancy_from_numpy


def test_sigma2_over_sigma1_high_on_2d_disk() -> None:
    """Uniform angular spread at multiple radii → σ₂/σ₁ near 1."""
    n = 80
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radii = 0.2 + 0.5 * np.linspace(0, 1, n)
    xy = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
    occ = disc_occupancy_from_numpy(xy)
    assert occ["disc_sigma2_sigma1"] > 0.7


def test_sigma2_over_sigma1_low_on_rank1_streak() -> None:
    """Collinear points → σ₂/σ₁ near 0."""
    t = np.linspace(-1.0, 1.0, 60)
    xy = np.stack([t, 0.01 * t], axis=1)
    occ = disc_occupancy_from_numpy(xy)
    assert occ["disc_sigma2_sigma1"] < 0.15
