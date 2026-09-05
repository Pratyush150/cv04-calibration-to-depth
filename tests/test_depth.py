"""Disparity to depth: the formula, the error law, Q, and triangulation."""

import cv2
import numpy as np
import pytest

from geo import depth as dp
from geo import pinhole as ph

F, B = 800.0, 0.12


def test_depth_matches_the_hand_computed_values():
    # f*B = 96 metre-pixels, so Z = 96/d.
    assert dp.depth_from_disparity(30.0, F, B) == pytest.approx(3.2)
    assert dp.depth_from_disparity(4.0, F, B) == pytest.approx(24.0)
    assert dp.depth_from_disparity(96.0, F, B) == pytest.approx(1.0)


def test_zero_and_negative_disparity_are_infinity_not_a_number():
    assert np.isinf(dp.depth_from_disparity(0.0, F, B))
    assert np.isinf(dp.depth_from_disparity(-3.0, F, B))


def test_disparity_and_depth_round_trip():
    Z = np.array([1.0, 3.2, 12.0, 24.0])
    d = dp.disparity_from_depth(Z, F, B)
    assert dp.depth_from_disparity(d, F, B) == pytest.approx(Z)


def test_the_depth_error_formula_matches_hand_computation():
    # |dZ| = Z^2/(f*B) * |dd|, with f*B = 96.
    # At Z = 24 m and dd = 1 px: 576/96 = 6.0 m.
    # At Z = 3.2 m and dd = 1 px: 10.24/96 = 0.10667 m.
    assert dp.depth_error(24.0, F, B, 1.0) == pytest.approx(6.0)
    assert dp.depth_error(3.2, F, B, 1.0) == pytest.approx(0.1066666, abs=1e-6)
    # And a quarter pixel is a quarter of the error - it is linear in dd.
    assert dp.depth_error(24.0, F, B, 0.25) == pytest.approx(1.5)


def test_the_error_law_is_quadratic_in_depth():
    a = dp.depth_error(3.2, F, B, 1.0)
    b = dp.depth_error(24.0, F, B, 1.0)
    assert b / a == pytest.approx((24.0 / 3.2) ** 2, rel=1e-9)


def test_the_linear_estimate_matches_a_finite_difference():
    for Z in (2.0, 5.0, 20.0):
        d = dp.disparity_from_depth(Z, F, B)
        h = 1e-4
        numeric = abs(dp.depth_from_disparity(d + h, F, B) -
                      dp.depth_from_disparity(d - h, F, B)) / (2 * h)
        assert numeric == pytest.approx(dp.depth_error(Z, F, B, 1.0), rel=1e-6)


def test_the_exact_interval_is_asymmetric_and_brackets_the_linear_one():
    near, far = dp.depth_error_exact(24.0, F, B, 1.0)
    assert near == pytest.approx(19.2)          # d = 5 px
    assert far == pytest.approx(32.0)           # d = 3 px
    # The linear estimate (6.0 m) sits between the two one-sided intervals.
    assert (24.0 - near) < dp.depth_error(24.0, F, B, 1.0) < (far - 24.0)


def test_max_useful_range_solves_the_relative_error_equation():
    r = dp.max_useful_range(F, B, 0.25, 0.10)
    assert r == pytest.approx(0.10 * F * B / 0.25)
    assert dp.depth_error(r, F, B, 0.25) == pytest.approx(0.10 * r)


def test_doffs_on_a_real_rig():
    # Middlebury Motorcycle: f = 3979.911 px, B = 193.001 mm, doffs = 124.343 px.
    f_mb, b_mb, doffs = 3979.911, 0.193001, 124.343
    assert dp.depth_from_disparity(200.0, f_mb, b_mb, doffs) == pytest.approx(2.3683,
                                                                             abs=1e-3)
    naive = dp.depth_from_disparity(200.0, f_mb, b_mb, 0.0)
    assert naive == pytest.approx(3.8406, abs=1e-3)
    assert naive / dp.depth_from_disparity(200.0, f_mb, b_mb, doffs) > 1.6
    # And the error grows with distance, which is the trap.
    far_ratio = (dp.depth_from_disparity(28.16, f_mb, b_mb, 0.0) /
                 dp.depth_from_disparity(28.16, f_mb, b_mb, doffs))
    assert far_ratio > 5.0


def test_the_Q_matrix_drill():
    Q = dp.Q_matrix(463.7446, 0.12, 320.2591, 240.2001)
    pts = dp.reproject_disparity(np.full((480, 640), 40.0), Q)
    X, Y, Z = pts[300, 400]
    assert X == pytest.approx(0.2392, abs=1e-4)
    assert Y == pytest.approx(0.1794, abs=1e-4)
    assert Z == pytest.approx(1.3912, abs=1e-4)
    # Z from Q must equal Z from the scalar formula with the same f - they are
    # the same equation written twice.
    assert Z == pytest.approx(dp.depth_from_disparity(40.0, 463.7446, 0.12), abs=1e-9)


def test_reproject_matches_opencv():
    Q = dp.Q_matrix(463.7446, 0.12, 320.2591, 240.2001)
    disp = np.random.default_rng(0).uniform(5, 60, (40, 50))
    mine = dp.reproject_disparity(disp, Q)
    theirs = cv2.reprojectImageTo3D(disp.astype(np.float32), Q.astype(np.float32))
    assert np.abs(mine - theirs).max() < 1e-5      # their side is float32


def geometry():
    K = ph.intrinsic_matrix(800.0, 800.0, 320.0, 240.0)
    R, _ = cv2.Rodrigues(np.array([0.02, 0.20, 0.01]))
    t = np.array([-1.0, 0.0, 0.0])
    P1 = ph.projection_matrix(K, np.eye(3), np.zeros(3))
    P2 = ph.projection_matrix(K, R, t)
    X = np.array([[0.30, -0.10, 5.00]])
    return K, R, t, P1, P2, X


def test_triangulation_recovers_a_known_point_from_clean_observations():
    K, R, t, P1, P2, X = geometry()
    x1 = ph.project(K, np.eye(3), np.zeros(3), X)
    x2 = ph.project(K, R, t, X)
    assert np.abs(dp.triangulate_dlt(P1, P2, x1, x2) - X).max() < 1e-9


def test_triangulation_matches_opencv():
    K, R, t, P1, P2, X = geometry()
    rng = np.random.default_rng(1)
    x1 = ph.project(K, np.eye(3), np.zeros(3), X) + rng.normal(0, 0.5, (1, 2))
    x2 = ph.project(K, R, t, X) + rng.normal(0, 0.5, (1, 2))
    mine = dp.triangulate_dlt(P1, P2, x1, x2)
    theirs = cv2.triangulatePoints(P1, P2, x1.T, x2.T)
    theirs = (theirs[:3] / theirs[3]).T
    assert np.abs(mine - theirs).max() < 1e-9


def test_the_rays_meet_only_when_the_data_is_perfect():
    K, R, t, P1, P2, X = geometry()
    x1 = ph.project(K, np.eye(3), np.zeros(3), X)
    x2 = ph.project(K, R, t, X)
    assert dp.ray_gap(P1, P2, x1[0], x2[0]) < 1e-12
    rng = np.random.default_rng(1)
    gap = dp.ray_gap(P1, P2, (x1 + rng.normal(0, 0.5, (1, 2)))[0],
                     (x2 + rng.normal(0, 0.5, (1, 2)))[0])
    assert gap > 1e-4          # millimetres, from half a pixel and nothing else


def test_triangulated_uncertainty_is_elongated_along_the_ray():
    K, R, t, P1, P2, X = geometry()
    x1 = ph.project(K, np.eye(3), np.zeros(3), X)
    x2 = ph.project(K, R, t, X)
    rng = np.random.default_rng(5)
    errs = np.array([(dp.triangulate_dlt(P1, P2,
                                         x1 + rng.normal(0, 0.5, (1, 2)),
                                         x2 + rng.normal(0, 0.5, (1, 2))) - X)[0]
                     for _ in range(200)])
    sx, sy, sz = errs.std(axis=0)
    assert sz > 5 * sx and sz > 5 * sy
