"""Quantitative Poincaré disc/ball property checks for embeddings."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from data.db_helpers.vector_queries import poincare_ball_distance
from science.dtie.common.hyperbolic_utils import hyperbolic_dist0, hyperbolic_pairwise_distance
from science.dtie.common.poincare_conventions import (
    ConventionProbe,
    estimate_clamp_applied_fraction,
    mobius_recenter_2d,
    model_clamp_ceiling,
    probe_disc_convention,
)


def mobius_transform_xy(
    x: np.ndarray,
    y: np.ndarray,
    ax: float,
    ay: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Match visualizer/frontend/src/lib/poincareViewportMath.ts mobiusTransform (c=1 unit disc)."""
    num_re = x - ax
    num_im = y - ay
    den_re = 1.0 - (ax * x + ay * y)
    den_im = -(ax * y - ay * x)
    den_mag_sq = den_re**2 + den_im**2 + 1e-8
    return (
        (num_re * den_re + num_im * den_im) / den_mag_sq,
        (num_im * den_re - num_re * den_im) / den_mag_sq,
    )


def ball_radius(c: float) -> float:
    if c <= 0:
        raise ValueError("curvature c must be positive")
    return 1.0 / math.sqrt(c)


def norm_ratio_vs_r_ball(points: np.ndarray, c: float) -> np.ndarray:
    r_ball = ball_radius(c)
    return np.linalg.norm(points, axis=1) / r_ball


def norm_ratio_vs_clamp_ceiling(points: np.ndarray, ceiling: float) -> np.ndarray:
    return np.linalg.norm(points, axis=1) / ceiling


def pairwise_ball_distances(points: np.ndarray, c: float) -> np.ndarray:
    n = len(points)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = poincare_ball_distance(points[i], points[j], c)
            dist[i, j] = d
            dist[j, i] = d
    return dist


def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(matrix.shape[0], k=1)
    return matrix[iu]


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 2 or b.size < 2:
        return float("nan")
    if np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


@dataclass
class NormDistributionReport:
    label: str
    n: int
    max_raw_norm: float
    clamp_ceiling: float
    ratio_vs_ceiling_mean: float
    ratio_vs_ceiling_std: float
    ratio_vs_ceiling_p50: float
    ratio_vs_ceiling_p90: float
    ratio_vs_ceiling_p99: float
    fraction_above_0_9_of_ceiling: float
    fraction_above_0_99_of_ceiling: float
    ratio_vs_r_ball_mean: float
    ratio_vs_r_ball_p50: float
    ratio_vs_r_ball_p90: float
    ratio_vs_r_ball_p99: float
    fraction_above_0_8_r_ball: float
    clamp_applied_fraction: float | None
    histogram_edges: list[float]
    histogram_counts: list[int]


@dataclass
class BoundaryMarginReport:
    label: str
    r_ball: float
    clamp_ceiling: float
    epsilon: float
    max_ratio_vs_r_ball: float
    max_ratio_vs_clamp_ceiling: float
    n_violations_geoopt: int
    n_within_epsilon_of_clamp_ceiling: int
    fraction_within_epsilon_of_clamp: float
    pass_strict_interior: bool


@dataclass
class BallDiscConsistencyReport:
    disc_dim: int
    ball_dim: int
    pearson_distances: float
    max_abs_dist_diff: float
    mean_abs_dist_diff: float
    note: str
    skipped: bool = False
    skip_reason: str | None = None


@dataclass
class MobiusIsometryReport:
    mobius_center: tuple[float, float]
    max_distance_drift_geodesic: float
    mean_distance_drift_geodesic: float
    pass_isometry_geodesic: bool
    max_distance_drift_viewer: float
    mean_distance_drift_viewer: float
    pass_isometry_viewer: bool
    tolerance: float
    note: str


@dataclass
class PhysicsFidelityReport:
    cone_depth_std: float
    radial_vs_cone_depth_r: float
    radial_vs_rho_r: float
    radial_vs_sasa_r: float | None
    angular_epistemic_r: float | None
    note: str
    skipped_depth_correlation: bool = False


@dataclass
class ViewerClipReport:
    mobius_center: tuple[float, float]
    rscale: float
    clip_radius: float
    viewer_clip_fraction: float
    max_render_norm: float
    geodesic_mobius_pass: bool
    diagnosis: str
    note: str


@dataclass
class PropertySuiteResult:
    structure_id: str
    curvature: float
    source: str
    convention: ConventionProbe
    norm_disc: NormDistributionReport
    norm_ball3d: NormDistributionReport | None
    norm_routed_highd: NormDistributionReport | None
    boundary_disc: BoundaryMarginReport
    ball_disc: BallDiscConsistencyReport | None
    mobius: MobiusIsometryReport
    physics: PhysicsFidelityReport
    viewer_clip: ViewerClipReport


def _histogram(ratios: np.ndarray, bins: int = 20) -> tuple[list[float], list[int]]:
    hi = min(1.0, float(ratios.max()) * 1.05) if len(ratios) else 1.0
    counts, edges = np.histogram(ratios, bins=bins, range=(0.0, max(hi, 1.0)))
    return edges.astype(float).tolist(), counts.astype(int).tolist()


def report_norm_distribution(
    points: np.ndarray,
    c: float,
    label: str,
    *,
    clamp_ceiling: float | None = None,
    legacy_unit_clamp: bool = False,
) -> NormDistributionReport:
    norms = np.linalg.norm(points, axis=1)
    ceiling = clamp_ceiling if clamp_ceiling is not None else model_clamp_ceiling(c)
    vs_ceiling = norm_ratio_vs_clamp_ceiling(points, ceiling)
    vs_r = norm_ratio_vs_r_ball(points, c)
    edges, counts = _histogram(vs_ceiling)
    clamp_frac = None
    if label == "disc_2d":
        clamp_frac = estimate_clamp_applied_fraction(norms, c, legacy_unit_clamp=legacy_unit_clamp)
    return NormDistributionReport(
        label=label,
        n=int(len(vs_ceiling)),
        max_raw_norm=float(norms.max()) if len(norms) else 0.0,
        clamp_ceiling=ceiling,
        ratio_vs_ceiling_mean=float(vs_ceiling.mean()),
        ratio_vs_ceiling_std=float(vs_ceiling.std()),
        ratio_vs_ceiling_p50=float(np.percentile(vs_ceiling, 50)),
        ratio_vs_ceiling_p90=float(np.percentile(vs_ceiling, 90)),
        ratio_vs_ceiling_p99=float(np.percentile(vs_ceiling, 99)),
        fraction_above_0_9_of_ceiling=float((vs_ceiling > 0.9).mean()),
        fraction_above_0_99_of_ceiling=float((vs_ceiling > 0.99).mean()),
        ratio_vs_r_ball_mean=float(vs_r.mean()),
        ratio_vs_r_ball_p50=float(np.percentile(vs_r, 50)),
        ratio_vs_r_ball_p90=float(np.percentile(vs_r, 90)),
        ratio_vs_r_ball_p99=float(np.percentile(vs_r, 99)),
        fraction_above_0_8_r_ball=float((vs_r > 0.8).mean()),
        clamp_applied_fraction=clamp_frac,
        histogram_edges=edges,
        histogram_counts=counts,
    )


def report_boundary_margin(
    points: np.ndarray,
    c: float,
    label: str,
    *,
    clamp_ceiling: float,
    epsilon: float = 1e-3,
) -> BoundaryMarginReport:
    r_ball = ball_radius(c)
    vs_r = norm_ratio_vs_r_ball(points, c)
    vs_ceiling = norm_ratio_vs_clamp_ceiling(points, clamp_ceiling)
    norms = np.linalg.norm(points, axis=1)
    violations = int(np.sum(norms >= r_ball))
    near = int(np.sum(vs_ceiling >= 1.0 - epsilon))
    return BoundaryMarginReport(
        label=label,
        r_ball=r_ball,
        clamp_ceiling=clamp_ceiling,
        epsilon=epsilon,
        max_ratio_vs_r_ball=float(vs_r.max()) if len(vs_r) else 0.0,
        max_ratio_vs_clamp_ceiling=float(vs_ceiling.max()) if len(vs_ceiling) else 0.0,
        n_violations_geoopt=violations,
        n_within_epsilon_of_clamp_ceiling=near,
        fraction_within_epsilon_of_clamp=float(near / len(vs_ceiling)) if len(vs_ceiling) else 0.0,
        pass_strict_interior=violations == 0,
    )


def report_ball_disc_consistency(
    disc: np.ndarray,
    ball3d: np.ndarray,
    c: float,
) -> BallDiscConsistencyReport:
    d2 = hyperbolic_pairwise_distance(disc, c=c)
    d3 = pairwise_ball_distances(ball3d, c)
    u2 = upper_triangle_values(d2)
    u3 = upper_triangle_values(d3)
    diff = np.abs(u2 - u3)
    return BallDiscConsistencyReport(
        disc_dim=int(disc.shape[1]),
        ball_dim=int(ball3d.shape[1]),
        pearson_distances=pearson(u2, u3),
        max_abs_dist_diff=float(diff.max()) if diff.size else 0.0,
        mean_abs_dist_diff=float(diff.mean()) if diff.size else 0.0,
        note=(
            "DIAGNOSTIC (not pass/fail): separate MobiusLinear heads from x_routed_hyp. "
            "Reconcile graph builders before interpreting Pearson."
        ),
    )


def report_mobius_isometry(
    disc: np.ndarray,
    c: float,
    mobius_center: tuple[float, float] = (0.35, 0.12),
    tolerance: float = 1e-5,
) -> MobiusIsometryReport:
    d_before = hyperbolic_pairwise_distance(disc, c=c)

    # Geodesic (curvature-correct) Möbius recenter — same ball as geoopt project/clamp.
    transformed_geo = mobius_recenter_2d(disc, mobius_center, c)
    d_after_geo = hyperbolic_pairwise_distance(transformed_geo, c=c)
    drift_geo = np.abs(upper_triangle_values(d_before) - upper_triangle_values(d_after_geo))
    max_geo = float(drift_geo.max()) if drift_geo.size else 0.0

    # Canvas path (post-fix): c-aware Möbius recenter — matches PoincareDiscCanvas.
    d_after_canvas = hyperbolic_pairwise_distance(transformed_geo, c=c)
    drift_canvas = drift_geo
    max_canvas = max_geo

    # Legacy canvas: c=1 complex automorphism (pre-fix); clamp to unit disc before d(·,c=1).
    tx, ty = mobius_transform_xy(disc[:, 0], disc[:, 1], mobius_center[0], mobius_center[1])
    transformed_legacy = np.column_stack([tx, ty])
    leg_norms = np.linalg.norm(transformed_legacy, axis=1, keepdims=True)
    transformed_legacy = transformed_legacy * np.minimum(1.0, 0.99 / (leg_norms + 1e-8))
    d_after_legacy = hyperbolic_pairwise_distance(transformed_legacy, c=1.0)
    drift_legacy = np.abs(upper_triangle_values(d_before) - upper_triangle_values(d_after_legacy))
    max_legacy = float(drift_legacy.max()) if drift_legacy.size else 0.0

    return MobiusIsometryReport(
        mobius_center=mobius_center,
        max_distance_drift_geodesic=max_geo,
        mean_distance_drift_geodesic=float(drift_geo.mean()) if drift_geo.size else 0.0,
        pass_isometry_geodesic=max_geo <= tolerance,
        max_distance_drift_viewer=max_legacy,
        mean_distance_drift_viewer=float(drift_legacy.mean()) if drift_legacy.size else 0.0,
        pass_isometry_viewer=max_legacy <= tolerance,
        tolerance=tolerance,
        note=(
            "geodesic/canvas: c-aware Möbius recenter + hyperbolic_pairwise_distance(c). "
            "viewer (legacy): c=1 mobiusTransform + d(·,c=1) on clamped coords."
        ),
    )


def report_viewer_clip(
    disc: np.ndarray,
    c: float,
    mobius_center: tuple[float, float],
    *,
    geodesic_mobius_pass: bool,
    clamp_applied_fraction: float | None = None,
    legacy_unit_clamp: bool = False,
    clip_radius: float = 0.965,
    rscale_exponent: float = 0.6,
) -> ViewerClipReport:
    """Simulate PoincareDiscCanvas render path (Möbius → rscale → clip)."""
    transformed = mobius_recenter_2d(disc, mobius_center, c)
    rscale = float(c**rscale_exponent)
    wx = transformed[:, 0] * rscale
    wy = transformed[:, 1] * rscale
    rr = np.hypot(wx, wy)
    clip_mask = rr > clip_radius
    clip_frac = float(clip_mask.mean()) if len(rr) else 0.0
    max_norm = float(rr.max()) if len(rr) else 0.0

    if geodesic_mobius_pass and clip_frac >= 0.02:
        diagnosis = "viewer_clip_likely"
        note = (
            "Geodesic Möbius PASS + render clip active → left-stripe / boundary band "
            "is likely the 0.965 canvas clip, not embedding geometry."
        )
    elif geodesic_mobius_pass and legacy_unit_clamp and (clamp_applied_fraction or 0) >= 0.03:
        diagnosis = "model_clamp_band"
        note = (
            "Geodesic Möbius PASS + legacy 0.99 model clamp saturation → radial band is "
            "pre-fix clamp geometry; viewer clip not required to explain stripe."
        )
    elif not geodesic_mobius_pass:
        diagnosis = "geometry_distortion"
        note = "Geodesic Möbius FAIL — investigate ball convention / clamp before blaming viewer."
    elif clip_frac < 0.02:
        diagnosis = "no_clip_signal"
        note = "Few points hit viewer clip at default Möbius offset; stripe may need non-zero pan."
    else:
        diagnosis = "inconclusive"
        note = "Mixed signals — compare with non-zero mobius pan in dashboard."

    return ViewerClipReport(
        mobius_center=mobius_center,
        rscale=rscale,
        clip_radius=clip_radius,
        viewer_clip_fraction=clip_frac,
        max_render_norm=max_norm,
        geodesic_mobius_pass=geodesic_mobius_pass,
        diagnosis=diagnosis,
        note=note,
    )


def report_physics_fidelity(
    disc: np.ndarray,
    cone_depth: np.ndarray,
    rho: np.ndarray | None,
    sasa: np.ndarray | None,
    epistemic: np.ndarray | None,
    c: float,
) -> PhysicsFidelityReport:
    depth_std = float(np.std(cone_depth))
    radial_hyp = hyperbolic_dist0(disc, c=c)
    skipped = depth_std <= 1e-12
    if skipped:
        depth_r = float("nan")
        note = (
            f"SKIPPED depth correlation: cone_depth std={depth_std:.6e} (flat constant). "
            "Re-run on a structure with burial variance."
        )
    else:
        depth_r = pearson(radial_hyp, cone_depth)
        note = (
            "v5/v6 intent: radial tracks burial/depth; angular separates domains. "
            "Weak depth correlation may indicate boundary saturation or viewer clip."
        )
    rho_r = pearson(radial_hyp, rho) if rho is not None else float("nan")
    sasa_r = pearson(radial_hyp, sasa) if sasa is not None else None
    ang = np.degrees(np.arctan2(disc[:, 1], disc[:, 0])) % 360.0
    ang_r = pearson(ang, epistemic) if epistemic is not None else None
    return PhysicsFidelityReport(
        cone_depth_std=depth_std,
        radial_vs_cone_depth_r=depth_r,
        radial_vs_rho_r=rho_r,
        radial_vs_sasa_r=sasa_r,
        angular_epistemic_r=ang_r,
        note=note,
        skipped_depth_correlation=skipped,
    )


def run_property_suite(
    *,
    structure_id: str,
    curvature: float,
    source: str,
    disc_2d: np.ndarray,
    cone_depth: np.ndarray,
    ball_3d: np.ndarray | None = None,
    x_routed_hyp: np.ndarray | None = None,
    x_hyp_highd: np.ndarray | None = None,
    rho: np.ndarray | None = None,
    sasa: np.ndarray | None = None,
    epistemic: np.ndarray | None = None,
    epsilon: float = 1e-3,
    mobius_center: tuple[float, float] = (0.35, 0.12),
) -> PropertySuiteResult:
    c = float(curvature)
    convention = probe_disc_convention(disc_2d, c)
    ceiling = convention.effective_norm_boundary
    legacy = convention.legacy_unit_clamp_detected

    ball_disc: BallDiscConsistencyReport | None = None
    if x_routed_hyp is not None and ball_3d is not None and len(ball_3d) == len(disc_2d):
        ball_disc = report_ball_disc_consistency(disc_2d, ball_3d, c)
    elif x_hyp_highd is not None and x_routed_hyp is None:
        ball_disc = BallDiscConsistencyReport(
            disc_dim=int(disc_2d.shape[1]),
            ball_dim=int(x_hyp_highd.shape[1]) if len(x_hyp_highd) else 0,
            pearson_distances=float("nan"),
            max_abs_dist_diff=float("nan"),
            mean_abs_dist_diff=float("nan"),
            note="INFERENCE-ONLY: disc is hyp_proj_head_2d(x_routed_hyp); DB embedding_double is pre-MoE x_hyp.",
            skipped=True,
            skip_reason="db_x_hyp_vs_post_route_disc",
        )

    norm_routed = None
    if x_routed_hyp is not None:
        norm_routed = report_norm_distribution(
            x_routed_hyp, c, "x_routed_hyp", clamp_ceiling=ceiling, legacy_unit_clamp=legacy
        )

    mobius = report_mobius_isometry(disc_2d, c, mobius_center=mobius_center)
    norm_disc = report_norm_distribution(
        disc_2d, c, "disc_2d", clamp_ceiling=ceiling, legacy_unit_clamp=legacy
    )

    return PropertySuiteResult(
        structure_id=structure_id,
        curvature=c,
        source=source,
        convention=convention,
        norm_disc=norm_disc,
        norm_ball3d=(
            report_norm_distribution(ball_3d, c, "ball_3d", clamp_ceiling=ceiling, legacy_unit_clamp=legacy)
            if ball_3d is not None
            else None
        ),
        norm_routed_highd=norm_routed,
        boundary_disc=report_boundary_margin(
            disc_2d, c, "disc_2d", clamp_ceiling=ceiling, epsilon=epsilon
        ),
        ball_disc=ball_disc,
        mobius=mobius,
        physics=report_physics_fidelity(disc_2d, cone_depth, rho, sasa, epistemic, c),
        viewer_clip=report_viewer_clip(
            disc_2d,
            c,
            mobius.mobius_center,
            geodesic_mobius_pass=mobius.pass_isometry_geodesic,
            clamp_applied_fraction=norm_disc.clamp_applied_fraction,
            legacy_unit_clamp=legacy,
        ),
    )
