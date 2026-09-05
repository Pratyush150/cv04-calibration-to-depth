"""Distortion: the forward polynomial, the iterative inverse, and the traps."""

import cv2
import numpy as np
import pytest

from geo import distortion as dm
from geo import pinhole as ph

K = ph.intrinsic_matrix(800.0, 802.0, 325.0, 238.0)
D = dm.coefficients(k1=-0.30, k2=0.10, p1=0.0012, p2=-0.0009)


def test_the_hand_drill():
    # (0.6, 0.8) sits at r = 1 exactly, so the radial factor is
    # 1 + k1 + k2 = 1 - 0.28 + 0.10 = 0.82 and the point moves inward to 0.82r.
    d = dm.coefficients(k1=-0.28, k2=0.10)
    xd, yd = dm.distort_normalized(0.60, 0.80, d)
    assert xd == pytest.approx(0.492, abs=1e-12)
    assert yd == pytest.approx(0.656, abs=1e-12)


def test_negative_k1_pulls_in_and_positive_pushes_out():
    r = np.array([0.2, 0.5, 0.8, 1.0])
    barrel, _ = dm.distort_normalized(r, np.zeros_like(r), dm.coefficients(k1=-0.28))
    pincushion, _ = dm.distort_normalized(r, np.zeros_like(r), dm.coefficients(k1=+0.28))
    assert np.all(barrel < r)
    assert np.all(pincushion > r)
    assert np.all(np.diff(barrel / r) < 0)          # the effect grows with radius


def test_zero_coefficients_change_nothing():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=50), rng.normal(size=50)
    xd, yd = dm.distort_normalized(x, y, np.zeros(5))
    assert np.abs(xd - x).max() == 0.0
    assert np.abs(yd - y).max() == 0.0


def test_the_inverse_converges_and_how_fast():
    rng = np.random.default_rng(0)
    th = rng.uniform(0, 2 * np.pi, 400)
    rad = np.sqrt(rng.uniform(0, 1, 400))
    x, y = rad * np.cos(th), rad * np.sin(th)
    xd, yd = dm.distort_normalized(x, y, D)
    r5 = np.hypot(*(np.array(dm.undistort_normalized(xd, yd, D, iters=5)) -
                    np.array([x, y]))).max()
    r10 = np.hypot(*(np.array(dm.undistort_normalized(xd, yd, D, iters=10)) -
                     np.array([x, y]))).max()
    r20 = np.hypot(*(np.array(dm.undistort_normalized(xd, yd, D, iters=20)) -
                     np.array([x, y]))).max()
    assert r5 < 1e-3
    assert r10 < 1e-5
    assert r20 < 1e-10
    assert r20 < r10 < r5                            # monotone, not accidental


def test_the_inverse_agrees_with_opencv():
    rng = np.random.default_rng(1)
    uv = np.column_stack([rng.uniform(0, 640, 200), rng.uniform(0, 480, 200)])
    mine = dm.undistort_pixels(K, D, uv, iters=20)
    theirs = cv2.undistortPoints(uv.reshape(-1, 1, 2), K, D, P=K).reshape(-1, 2)
    # OpenCV stops after a fixed iteration count, so this is their tolerance and
    # not ours - the check is that we land on the same answer, not that they are
    # bit-identical.
    assert np.abs(mine - theirs).max() < 5e-3


def test_the_forward_model_matches_projectPoints():
    from geo import synthetic as syn
    objp = syn.board_object_points((9, 6), 0.025)
    rvec = np.array([0.31, -0.22, 0.06])
    tvec = np.array([-0.10, -0.07, 0.55])
    R, _ = cv2.Rodrigues(rvec)
    mine = dm.project_distorted(K, D, R, tvec, objp)
    theirs, _ = cv2.projectPoints(objp, rvec, tvec, K, D)
    assert np.abs(mine - theirs.reshape(-1, 2)).max() < 1e-9


def test_undistort_then_distort_is_the_identity():
    rng = np.random.default_rng(2)
    uv = np.column_stack([rng.uniform(20, 620, 100), rng.uniform(20, 460, 100)])
    ideal = dm.undistort_pixels(K, D, uv, iters=25)
    xy = dm.normalize_pixels(K, ideal)
    xd, yd = dm.distort_normalized(xy[:, 0], xy[:, 1], D)
    back = dm.denormalize_points(K, np.column_stack([xd, yd]))
    assert np.abs(back - uv).max() < 1e-8


def test_distortion_coefficients_survive_a_resolution_change():
    # The point of the claim: normalised coordinates are resolution-invariant,
    # so the polynomial that acts on them cannot need different coefficients.
    K_small = ph.scale_intrinsics(K, 0.5)
    uv = np.array([[600.0, 400.0], [40.0, 60.0]])
    xn_full = dm.normalize_pixels(K, uv)
    xn_half = dm.normalize_pixels(K_small, uv * 0.5)
    assert np.abs(xn_full - xn_half).max() < 1e-12


def test_anisotropic_resize_also_preserves_normalised_coordinates():
    K_aniso = ph.scale_intrinsics(K, 1 / 3, 4 / 9)
    uv = np.array([[600.0, 400.0]])
    a = dm.normalize_pixels(K, uv)
    b = dm.normalize_pixels(K_aniso, uv * np.array([1 / 3, 4 / 9]))
    assert np.abs(a - b).max() < 1e-12


def test_the_coefficient_order_is_not_interchangeable():
    # [k1, k2, p1, p2, k3] against the "natural" [k1, k2, k3, p1, p2].  OpenCV
    # accepts either without a word, which is why this is worth a test.
    right = dm.coefficients(k1=-0.30, k2=0.10, k3=0.02)
    wrong = np.array([-0.30, 0.10, 0.02, 0.0, 0.0])
    a = dm.distort_normalized(0.6, 0.8, right)
    b = dm.distort_normalized(0.6, 0.8, wrong)
    assert abs(a[0] - b[0]) > 1e-3


def test_undistort_image_straightens_a_rendered_board():
    from geo import synthetic as syn
    rvec = np.array([0.16, -0.12, 0.04])
    R, _ = cv2.Rodrigues(rvec)
    objp = syn.board_object_points((9, 6), 0.025)
    tvec = np.array([0.0, 0.0, 0.42]) - R @ objp.mean(axis=0)
    img = syn.render_checkerboard(K, D, rvec, tvec, (640, 480), seed=3)
    und = dm.undistort_image(img, K, D)

    def max_row_bow(pts):
        p = pts.reshape(6, 9, 2)
        worst = 0.0
        for row in p:
            a, b = row[0], row[-1]
            d = b - a
            n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
            worst = max(worst, float(np.abs((row - a) @ n).max()))
        return worst

    ok_a, corners = syn.detect_corners(img, (9, 6))
    ok_b, corners_u = syn.detect_corners(und, (9, 6))
    assert ok_a and ok_b
    assert max_row_bow(corners_u) < 0.35 * max_row_bow(corners)
