"""Property tests for hyperbolic_lorentz_ops.py at GNNv6 curvature and n=2/8/128."""

import numpy as np
import pytest

from science.dtie.common.hyperbolic_lorentz_ops import (
    disc_from_lorentz,
    exp_p,
    log_p,
    lorentz_from_disc,
    lorentzian_barycenter,
    mobius_add,
    mobius_recenter,
    poincare_distance,
)

C = 0.6054342985153198  # GNNv6
RADIUS = 1.0 / np.sqrt(C)
RNG = np.random.default_rng(11)
DIMS_TO_TEST = [2, 8, 128]


def _rand_in_ball(dim, frac=0.7):
    v = RNG.normal(size=dim)
    v /= np.linalg.norm(v)
    return v * RNG.uniform(0, frac) * RADIUS


@pytest.mark.parametrize("dim", DIMS_TO_TEST)
def test_lorentz_disc_round_trip(dim):
    for _ in range(300):
        z = _rand_in_ball(dim)
        z2 = disc_from_lorentz(lorentz_from_disc(z, C), C)
        assert np.max(np.abs(z - z2)) < 1e-9


@pytest.mark.parametrize("dim", DIMS_TO_TEST)
def test_lorentz_from_disc_on_hyperboloid(dim):
    for _ in range(300):
        z = _rand_in_ball(dim)
        L = lorentz_from_disc(z, C)
        lorentz_norm_sq = L[0] ** 2 - np.dot(L[1:], L[1:])
        assert abs(lorentz_norm_sq - 1.0 / C) < 1e-8


@pytest.mark.parametrize("dim", DIMS_TO_TEST)
def test_mobius_recenter_is_true_isometry(dim):
    n_trials = 500 if dim < 128 else 150
    for _ in range(n_trials):
        z1, z2, a = _rand_in_ball(dim), _rand_in_ball(dim), _rand_in_ball(dim)
        d0 = poincare_distance(z1, z2, C)
        m1, m2 = mobius_recenter(z1, a, C), mobius_recenter(z2, a, C)
        d1 = poincare_distance(m1, m2, C)
        assert abs(d0 - d1) < 1e-8


@pytest.mark.parametrize("dim", DIMS_TO_TEST)
def test_mobius_recenter_sends_focus_to_origin(dim):
    for _ in range(100):
        a = _rand_in_ball(dim)
        out = mobius_recenter(a, a, C)
        assert np.max(np.abs(out)) < 1e-9


@pytest.mark.parametrize("dim", DIMS_TO_TEST)
def test_mobius_recenter_preserves_ball(dim):
    for _ in range(500):
        z, a = _rand_in_ball(dim, frac=0.95), _rand_in_ball(dim, frac=0.9)
        out = mobius_recenter(z, a, C)
        assert np.linalg.norm(out) < RADIUS + 1e-9


def test_mobius_recenter_matches_original_2d_complex_derivation():
    def complex_version(z, a, c):
        sqrt_c = np.sqrt(c)
        u = complex(sqrt_c * z[0], sqrt_c * z[1])
        b = complex(sqrt_c * a[0], sqrt_c * a[1])
        r = (u - b) / (1.0 - b.conjugate() * u)
        return np.array([r.real, r.imag]) / sqrt_c

    for _ in range(300):
        z, a = _rand_in_ball(2), _rand_in_ball(2)
        assert np.max(np.abs(mobius_recenter(z, a, C) - complex_version(z, a, C))) < 1e-10


@pytest.mark.parametrize("dim", DIMS_TO_TEST)
def test_lorentzian_barycenter_symmetric_case_gives_origin(dim):
    e = np.eye(dim) * 0.3 * RADIUS
    pts = list(e) + list(-e)
    bc = lorentzian_barycenter(pts, [1.0] * len(pts), C)
    assert np.max(np.abs(bc)) < 1e-8


@pytest.mark.parametrize("dim", DIMS_TO_TEST)
def test_lorentzian_barycenter_pulled_toward_heavier_point(dim):
    p0 = np.zeros(dim)
    p1 = np.zeros(dim)
    p1[0] = 0.6 * RADIUS
    bc = lorentzian_barycenter([p0, p1], [1, 5], C)
    assert 0.0 < bc[0] < p1[0]
    assert np.max(np.abs(bc[1:])) < 1e-8


@pytest.mark.parametrize("dim", DIMS_TO_TEST)
def test_lorentzian_barycenter_stays_in_ball(dim):
    for _ in range(100):
        pts = [_rand_in_ball(dim, frac=0.9) for _ in range(5)]
        weights = list(RNG.uniform(0.1, 5, 5))
        bc = lorentzian_barycenter(pts, weights, C)
        assert np.linalg.norm(bc) < RADIUS


def _broken_mobius_from_source_html(z, a):
    zr, zi, ar, ai = z[0], z[1], a[0], a[1]
    num = (zr - ar, zi - ai)
    den = (1 - (ar * zr + ai * zi), -(ar * zi - ai * zr))
    d2 = den[0] ** 2 + den[1] ** 2
    return np.array([(num[0] * den[0] - num[1] * den[1]) / d2, (num[1] * den[0] + num[0] * den[1]) / d2])


def test_source_html_mobius_is_confirmed_non_isometric():
    z1, z2, a = np.array([0.3, 0.0]), np.array([-0.2, 0.1]), np.array([0.4, 0.1])
    d0 = poincare_distance(z1, z2, 1.0)
    m1 = _broken_mobius_from_source_html(z1, a)
    m2 = _broken_mobius_from_source_html(z2, a)
    d1 = poincare_distance(m1, m2, 1.0)
    assert abs(d0 - d1) > 0.01


@pytest.mark.parametrize("dim", DIMS_TO_TEST)
def test_log_p_exp_p_round_trip(dim):
    for _ in range(300):
        p, x = _rand_in_ball(dim, frac=0.5), _rand_in_ball(dim, frac=0.5)
        v = log_p(x, p, C)
        x2 = exp_p(v, p, C)
        assert np.max(np.abs(x - x2)) < 1e-8


@pytest.mark.parametrize("dim", DIMS_TO_TEST)
def test_log_p_magnitude_equals_true_distance(dim):
    for _ in range(300):
        p, x = _rand_in_ball(dim, frac=0.5), _rand_in_ball(dim, frac=0.5)
        v = log_p(x, p, C)
        assert abs(np.linalg.norm(v) - poincare_distance(p, x, C)) < 1e-8


def test_log_p_distortion_grows_with_distance_from_reference():
    dim = 2
    p = np.zeros(dim)
    rel_errors_by_offset = {}
    for offset_frac in [0.05, 0.4, 0.95]:
        diffs = []
        for _ in range(150):
            ang = RNG.uniform(0, 2 * np.pi)
            r = offset_frac * RADIUS
            x1 = np.array([r * np.cos(ang), r * np.sin(ang)])
            x2 = exp_p(RNG.normal(scale=0.15, size=dim), x1, C)
            true_d = poincare_distance(x1, x2, C)
            v1, v2 = log_p(x1, p, C), log_p(x2, p, C)
            euclid_d = np.linalg.norm(v1 - v2)
            diffs.append(abs(true_d - euclid_d) / true_d)
        rel_errors_by_offset[offset_frac] = np.mean(diffs)

    assert rel_errors_by_offset[0.05] < 0.02
    assert rel_errors_by_offset[0.05] < rel_errors_by_offset[0.4] < rel_errors_by_offset[0.95]
    assert rel_errors_by_offset[0.95] > 0.15


def test_mobius_add_associates_with_recenter_at_n2():
    for _ in range(200):
        a, b, x = _rand_in_ball(2), _rand_in_ball(2), _rand_in_ball(2)
        left = mobius_add(a, mobius_add(-a, x, C), C)
        assert np.max(np.abs(left - x)) < 1e-9
