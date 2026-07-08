"""
hyperbolic_lorentz_ops.py

Generalized, curvature-parameterized, N-DIMENSIONAL hyperbolic operations,
ported from kras-shared-pca-differential-controller-safe.html and
corrected/verified.

The source HTML implements these hardcoded to curvature -1 (unit disc) AND
hardcoded to 2 dimensions (it's driving a 2D canvas visualization). This
module generalizes both axes -- arbitrary curvature c (GNNv6's
0.6054342985153198, TS-002's 0.6808, or any other) AND arbitrary embedding
dimension n (2 for the dashboard's use case, 128 for GNNv6's actual
embeddings) -- and is unit-tested at both n=2 (to confirm it still matches
the original 2D complex-arithmetic derivation exactly) and n=128 (to
confirm it's actually valid on real GNNv6-scale data, not just the toy
case it was ported from).

All points are plain numpy arrays of shape (n,), not (x, y) tuples.

Source review summary (see tests/test_hyperbolic_lorentz_ops.py for the checks):
  - lorentz_from_disc / disc_from_lorentz: source HTML's versions were
    correct (verified exact mutual inverses, verified on-hyperboloid).
    Generalized from 2D/c=1 to n-D/arbitrary c.
  - lorentzian_barycenter: source HTML's `barycenter()` was a correct,
    legitimate closed-form approximation to a hyperbolic centroid (weighted
    sum in Lorentz coordinates, renormalized via the Lorentz inner
    product) -- NOT a naive Euclidean average. Generalized.
  - mobius_recenter: source HTML's `mobius(z, a)` is NOT a hyperbolic
    isometry, despite the name. It satisfies mobius(a,a)=0 and stays
    inside the disc, but does not preserve hyperbolic distances --
    verified up to 85% of sampled point pairs showing >5% distance
    distortion, growing to ~4x near the boundary. Root cause: it computes
    (z-a)/(1-a*conj(z)) -- conjugating the *variable* z instead of the
    fixed point a -- non-holomorphic, not a genuine disc automorphism.
    This module implements the corrected, dimension-general version via
    the standard Mobius gyrovector addition formula (Ungar; Ganea et al.
    2018), verified isometric to ~1e-15 at n=2 (matching the original
    complex-number derivation exactly) and at n=8/n=128.
"""

from __future__ import annotations

import numpy as np


def _as_vec(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def lorentz_from_disc(z, c: float) -> np.ndarray:
    """Poincare ball (curvature -c, any dimension n) -> hyperboloid
    (Lorentz) model. Returns an (n+1,)-vector (t, x_1, ..., x_n)."""
    z = _as_vec(z)
    u = np.sqrt(c) * z
    r2 = np.dot(u, u)
    denom = max(1e-12, 1.0 - r2)
    T = (1.0 + r2) / denom
    X = 2.0 * u / denom
    inv_sqrt_c = 1.0 / np.sqrt(c)
    return np.concatenate(([T * inv_sqrt_c], X * inv_sqrt_c))


def disc_from_lorentz(L, c: float) -> np.ndarray:
    """Hyperboloid (Lorentz) model -> Poincare ball (curvature -c).
    L is an (n+1,)-vector (t, x_1, ..., x_n); returns an (n,)-vector."""
    L = _as_vec(L)
    sqrt_c = np.sqrt(c)
    T = sqrt_c * L[0]
    X = sqrt_c * L[1:]
    denom = max(1e-12, T + 1.0)
    u = X / denom
    return u / sqrt_c


def lorentzian_barycenter(points, weights, c: float) -> np.ndarray:
    """Closed-form (non-iterative) hyperbolic centroid approximation:
    weighted sum in Lorentz coordinates, renormalized onto the curvature-c
    hyperboloid via the Lorentz inner product, then mapped back to the
    disc. This is the Lorentzian centroid -- a fast, well-known stand-in
    for the true Frechet/Karcher mean (which requires iterative
    optimization). Not exact, but not a Euclidean-average cheat either.
    """
    S = None
    for z, w in zip(points, weights):
        L = lorentz_from_disc(z, c)
        S = w * L if S is None else S + w * L
    lorentz_norm_sq = S[0] ** 2 - np.dot(S[1:], S[1:])
    k = np.sqrt(max(1e-12, c * lorentz_norm_sq))
    return disc_from_lorentz(S / k, c)


def mobius_add(z, y, c: float) -> np.ndarray:
    """Standard n-dimensional Mobius gyrovector addition (Ungar; Ganea et
    al. 2018), curvature -c:
        z (+)_c y = [(1+2c<z,y>+c|y|^2) z + (1-c|z|^2) y] / [1+2c<z,y>+c^2|z|^2|y|^2]
    Verified to match the original 2D complex-arithmetic Mobius formula
    exactly at n=2 (diff ~3e-16)."""
    z, y = _as_vec(z), _as_vec(y)
    zy, zz, yy = np.dot(z, y), np.dot(z, z), np.dot(y, y)
    num = (1 + 2 * c * zy + c * yy) * z + (1 - c * zz) * y
    den = 1 + 2 * c * zy + c * c * zz * yy
    return num / max(den, 1e-15)


def mobius_recenter(z, a, c: float) -> np.ndarray:
    """Genuine hyperbolic isometry sending `a` to the origin, any
    dimension. Equivalent to (-a) (+)_c z. Corrected replacement for the
    source HTML's broken mobius() -- see module docstring."""
    return mobius_add(-_as_vec(a), _as_vec(z), c)


def poincare_distance(z1, z2, c: float) -> float:
    """Exact closed-form geodesic distance, curvature -c, any dimension.
    Same formula family verified in discover_hyperbolic_motifs.py's
    poincare_distance_matrix."""
    z1, z2 = _as_vec(z1), _as_vec(z2)
    sqdist = np.sum((z1 - z2) ** 2)
    denom = (1 - c * np.dot(z1, z1)) * (1 - c * np.dot(z2, z2))
    arg = 1.0 + 2.0 * c * sqdist / max(denom, 1e-15)
    return float(np.arccosh(max(arg, 1.0)) / np.sqrt(c))


def hyperbolic_distance_cap(c: float, *, max_frac: float = 0.92) -> float:
    """Max geodesic distance from origin when Euclidean |z| = max_frac * ball_radius(c)."""
    import math

    frac = min(float(max_frac), 1.0 - 1e-6)
    return (2.0 / math.sqrt(c)) * math.atanh(frac)


def expmap0_tangent_at_origin(v, c: float) -> np.ndarray:
    """Map tangent vector at origin into the Poincaré ball (curvature -c).

    ``||v||`` equals the hyperbolic geodesic distance from the origin — use this
    for structural placement, not Euclidean ``r * [cos φ, sin φ]``.
    """
    return _exp0(v, c)


def logmap0_at_origin(z, c: float) -> np.ndarray:
    """Tangent vector at origin with norm equal to hyperbolic distance from origin."""
    return _log0(z, c)


def hyperbolic_distance_from_origin(z, c: float) -> float:
    z = _as_vec(z)
    return poincare_distance(np.zeros_like(z), z, c)


def _exp0(v, c: float) -> np.ndarray:
    v = _as_vec(v)
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        return np.zeros_like(v)
    scale = (1.0 / np.sqrt(c)) * np.tanh(np.sqrt(c) * nv / 2.0) / nv
    return v * scale


def _log0(x, c: float) -> np.ndarray:
    x = _as_vec(x)
    nx = np.linalg.norm(x)
    if nx < 1e-12:
        return np.zeros_like(x)
    d = poincare_distance(np.zeros_like(x), x, c)
    return x * d / nx


def log_p(x, p, c: float) -> np.ndarray:
    """Map ball point x to a Euclidean tangent vector at reference point p.
    ||log_p(x)|| equals the true geodesic distance from p to x, by
    construction. Tier B middleware -- approximate for spread-out data."""
    return _log0(mobius_recenter(x, p, c), c)


def exp_p(v, p, c: float) -> np.ndarray:
    """Inverse of log_p: map a Euclidean tangent vector at p back into the ball."""
    p = _as_vec(p)
    return mobius_recenter(_exp0(v, c), -p, c)


def ball_radius(c: float) -> float:
    """Maximum Euclidean norm for points strictly inside the curvature-c ball."""
    return 1.0 / np.sqrt(c)


def clamp_to_ball(z: np.ndarray, c: float, *, margin: float = 1e-4) -> np.ndarray:
    """Scale z inward if it lies on or outside the model ball boundary."""
    z = _as_vec(z)
    r_max = ball_radius(c) * (1.0 - margin)
    n = np.linalg.norm(z)
    if n <= r_max or n < 1e-15:
        return z
    return z * (r_max / n)
