"""Two-view geometry: F from eight points, normalisation, and E's four poses."""

import cv2
import numpy as np
import pytest

from geo import epipolar as ep
from geo import pinhole as ph

K = ph.intrinsic_matrix(900.0, 900.0, 640.0, 360.0)
R_TRUE, _ = cv2.Rodrigues(np.array([0.03, 0.12, -0.02]))
T_TRUE = np.array([-0.40, 0.05, 0.06])


def two_views(n=150, seed=3):
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.uniform(-2, 2, n), rng.uniform(-1.5, 1.5, n),
                         rng.uniform(4, 10, n)])
    p1 = ph.project(K, np.eye(3), np.zeros(3), X)
    p2 = ph.project(K, R_TRUE, T_TRUE, X)
    return X, p1, p2


def test_skew_turns_a_cross_product_into_a_matmul():
    rng = np.random.default_rng(0)
    for _ in range(10):
        a, b = rng.normal(size=3), rng.normal(size=3)
        assert ep.skew(a) @ b == pytest.approx(np.cross(a, b))


def test_E_and_F_are_rank_2():
    E = ep.essential_from_Rt(R_TRUE, T_TRUE)
    F = ep.fundamental_from_KRt(K, K, R_TRUE, T_TRUE)
    assert np.linalg.matrix_rank(E, tol=1e-9) == 2
    assert np.linalg.matrix_rank(F, tol=1e-9) == 2
    assert abs(np.linalg.det(F)) < 1e-20


def test_the_epipolar_constraint_holds_on_true_correspondences():
    _, p1, p2 = two_views()
    F = ep.fundamental_from_KRt(K, K, R_TRUE, T_TRUE)
    assert ep.symmetric_epipolar_distance(F, p1, p2).max() < 1e-9


def test_eight_point_recovers_the_known_fundamental_matrix():
    _, p1, p2 = two_views()
    F_true = ep.fundamental_from_KRt(K, K, R_TRUE, T_TRUE)
    F_est = ep.eight_point(p1[:8], p2[:8])
    assert np.abs(F_est - F_true).max() < 1e-8
    assert np.linalg.matrix_rank(F_est, tol=1e-9) == 2
    # And with more than the minimum, which is the overdetermined case.
    assert np.abs(ep.eight_point(p1, p2) - F_true).max() < 1e-8


def test_eight_point_matches_opencv_on_clean_points():
    _, p1, p2 = two_views()
    mine = ep.eight_point(p1[:40], p2[:40])
    theirs, _ = cv2.findFundamentalMat(p1[:40], p2[:40], cv2.FM_8POINT)
    assert np.abs(mine - ep.normalize_matrix(theirs)).max() < 1e-6


def test_skipping_normalisation_measurably_degrades_the_estimate():
    """Report both numbers, then assert the gap.

    Measured on this fixture (0.5 px of correspondence noise, 60 points, scored
    as mean symmetric epipolar distance against the CLEAN correspondences):

        normalised   0.1967 px
        raw pixels   4.6171 px      23.5x worse

    The gap is not a constant of nature - it grows with how far the points'
    centroid sits from the coordinate origin, which is why the 1280x720 frame
    here is worse than a 480x320 one and an ROI inside a 4K frame is worse
    still.
    """
    _, p1, p2 = two_views(n=60)
    rng = np.random.default_rng(11)
    norm_errs, raw_errs = [], []
    for _ in range(15):
        q1 = p1 + rng.normal(0, 0.5, p1.shape)
        q2 = p2 + rng.normal(0, 0.5, p2.shape)
        Fn = ep.eight_point(q1, q2, normalize=True)
        Fr = ep.eight_point(q1, q2, normalize=False)
        norm_errs.append(ep.symmetric_epipolar_distance(Fn, p1, p2).mean())
        raw_errs.append(ep.symmetric_epipolar_distance(Fr, p1, p2).mean())
    normalised = float(np.mean(norm_errs))
    raw = float(np.mean(raw_errs))
    assert normalised < 0.5
    assert raw > 5 * normalised


def test_normalisation_is_invariant_to_where_pixel_zero_is():
    """The normalised solve does not care about the coordinate origin; the raw
    one falls apart.  This is the same experiment as above with the points
    shifted by 8000 px, which is what an ROI inside a large sensor looks like."""
    _, p1, p2 = two_views(n=60)
    rng = np.random.default_rng(11)
    shift = np.array([8000.0, 8000.0])
    q1 = p1 + rng.normal(0, 0.5, p1.shape)
    q2 = p2 + rng.normal(0, 0.5, p2.shape)
    a = ep.symmetric_epipolar_distance(ep.eight_point(q1, q2), p1, p2).mean()
    b = ep.symmetric_epipolar_distance(
        ep.eight_point(q1 + shift, q2 + shift), p1 + shift, p2 + shift).mean()
    assert b == pytest.approx(a, rel=1e-6)


def test_rank_2_projection_is_not_optional():
    _, p1, p2 = two_views(n=60)
    rng = np.random.default_rng(5)
    q1 = p1 + rng.normal(0, 0.5, p1.shape)
    q2 = p2 + rng.normal(0, 0.5, p2.shape)
    F = ep.eight_point(q1, q2)
    assert np.linalg.matrix_rank(F, tol=1e-8) == 2
    # The epipole only exists because the rank is 2: it is the null vector.
    e = ep.epipole(F, which=1)
    assert np.all(np.isfinite(e))


def test_decompose_essential_gives_the_twisted_pair():
    E = ep.essential_from_Rt(R_TRUE, T_TRUE / np.linalg.norm(T_TRUE))
    R1, R2, t = ep.decompose_essential(E)
    assert ph.is_rotation(R1, tol=1e-8)
    assert ph.is_rotation(R2, tol=1e-8)
    twist = R2 @ R1.T
    angle = np.degrees(np.arccos(np.clip((np.trace(twist) - 1) / 2, -1, 1)))
    assert angle == pytest.approx(180.0, abs=1e-3)
    axis = cv2.Rodrigues(twist)[0].ravel()
    axis /= np.linalg.norm(axis)
    t_hat = T_TRUE / np.linalg.norm(T_TRUE)
    assert abs(abs(axis @ t_hat) - 1.0) < 1e-6          # the axis IS the baseline


def test_cheirality_picks_exactly_one_of_the_four():
    _, p1, p2 = two_views(n=40)
    t_hat = T_TRUE / np.linalg.norm(T_TRUE)
    E = ep.essential_from_Rt(R_TRUE, t_hat)
    R_rec, t_rec, votes = ep.select_pose_by_cheirality(E, K, p1, p2)
    assert sorted(votes)[-1] == 40                       # one candidate takes them all
    assert sorted(votes)[-2] == 0                        # and the runner-up takes none
    ang = np.degrees(np.arccos(np.clip((np.trace(R_rec.T @ R_TRUE) - 1) / 2, -1, 1)))
    assert ang < 1e-4
    assert abs(abs(t_rec @ t_hat) - 1.0) < 1e-9


def test_recovered_translation_has_unit_norm_whatever_the_baseline():
    for scale in (0.05, 1.0, 17.0):
        t = T_TRUE / np.linalg.norm(T_TRUE) * scale
        E = ep.essential_from_Rt(R_TRUE, t)
        _, _, t_dir = ep.decompose_essential(E)
        assert np.linalg.norm(t_dir) == pytest.approx(1.0)
