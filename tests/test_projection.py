"""Projection: the arithmetic, the conventions, and the guards."""

import numpy as np
import pytest

from geo import pinhole as ph

K = ph.intrinsic_matrix(800.0, 800.0, 320.0, 240.0)


def test_known_point_projects_to_known_pixel():
    # Worked by hand: u = 800*(0.1/2.0) + 320 = 360, v = 800*(-0.2/2.0) + 240 = 160.
    uv = ph.project_camera_points(K, [[0.1, -0.2, 2.0]])[0]
    assert uv == pytest.approx([360.0, 160.0], abs=1e-9)


def test_the_homogeneous_triple_is_not_the_pixel():
    # K @ X is (su, sv, s) = (720, 320, 2).  The pixel only exists after the
    # divide, and printing the triple as if it were a pixel is the classic bug.
    h = K @ np.array([0.1, -0.2, 2.0])
    assert h == pytest.approx([720.0, 320.0, 2.0])
    assert h[2] == pytest.approx(2.0)             # the scale IS the depth
    assert (h[:2] / h[2]) == pytest.approx([360.0, 160.0])


def test_doubling_depth_halves_the_offset():
    uv = ph.project_camera_points(K, [[0.1, -0.2, 2.0], [0.1, -0.2, 4.0],
                                      [0.1, -0.2, 8.0]])
    offsets = uv - np.array([320.0, 240.0])
    assert offsets[0] == pytest.approx([40.0, -80.0])
    assert offsets[1] == pytest.approx(offsets[0] / 2)
    assert offsets[2] == pytest.approx(offsets[0] / 4)


def test_optical_axis_lands_on_the_principal_point_at_any_depth():
    uv = ph.project_camera_points(K, [[0.0, 0.0, 5.0], [0.0, 0.0, 500.0]])
    assert uv[0] == pytest.approx([320.0, 240.0])
    assert uv[1] == pytest.approx([320.0, 240.0])


def test_matvec3_matches_the_library_product():
    rng = np.random.default_rng(0)
    for _ in range(20):
        v = rng.normal(size=3)
        assert ph.matvec3(K, v) == pytest.approx(K @ v)


def test_real_and_virtual_image_planes_are_point_reflections():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    X[:, 2] = rng.uniform(0.5, 10.0, 200)
    virtual = ph.project_camera_points(K, X)
    real = ph.project_real_plane(K, X)
    reflected = 2 * np.array([320.0, 240.0]) - virtual
    assert np.abs(real - reflected).max() < 1e-9


def test_points_behind_the_camera_are_rejected_not_projected():
    # Without the cheirality guard this returns (280, 320) - a plausible pixel
    # inside the image for an object behind the lens.
    uv = ph.project_camera_points(K, [[0.1, -0.2, -2.0]])
    assert np.isnan(uv).all()


def test_camera_centre_is_not_t():
    R = np.eye(3)
    C = np.array([0.0, 0.0, -5.0])
    t = ph.extrinsics_from_centre(R, C)
    assert t == pytest.approx([0.0, 0.0, 5.0])         # opposite sign
    assert ph.camera_centre(R, t) == pytest.approx(C)
    assert ph.project(K, R, t, [[0, 0, 0]])[0] == pytest.approx([320.0, 240.0])


def test_camera_centre_round_trips_for_a_general_pose():
    import cv2
    R, _ = cv2.Rodrigues(np.array([0.3, -0.2, 0.1]))
    C = np.array([1.5, -0.4, 2.2])
    t = ph.extrinsics_from_centre(R, C)
    assert ph.camera_centre(R, t) == pytest.approx(C)


def test_rotation_validity_needs_both_conditions():
    import cv2
    R, _ = cv2.Rodrigues(np.array([0.05, 0.20, -0.03]))
    assert ph.is_rotation(R)
    mirrored = R.copy()
    mirrored[:, 0] *= -1                                # orthonormal but det = -1
    assert np.allclose(mirrored.T @ mirrored, np.eye(3))
    assert not ph.is_rotation(mirrored)


def test_orthonormalise_repairs_drift():
    import cv2
    R, _ = cv2.Rodrigues(np.array([0.05, 0.20, -0.03]))
    drifted = R + 1e-3 * np.random.default_rng(0).normal(size=(3, 3))
    assert not ph.is_rotation(drifted, tol=1e-6)
    fixed = ph.orthonormalise(drifted)
    assert ph.is_rotation(fixed)


def test_resize_scales_all_four_entries():
    K_full = ph.intrinsic_matrix(1450.0, 1452.0, 962.0, 541.0)
    K_small = ph.scale_intrinsics(K_full, 1 / 3)
    assert K_small[0, 0] == pytest.approx(1450.0 / 3)
    assert K_small[1, 1] == pytest.approx(1452.0 / 3)
    assert K_small[0, 2] == pytest.approx(962.0 / 3)
    assert K_small[1, 2] == pytest.approx(541.0 / 3)


def test_the_scaled_K_predicts_the_scaled_pixel():
    import cv2
    K_full = ph.intrinsic_matrix(1450.0, 1452.0, 962.0, 541.0)
    R, _ = cv2.Rodrigues(np.array([0.05, 0.20, -0.03]))
    t = np.array([0.10, -0.05, 4.00])
    X = np.array([[0.30, -0.15, 0.00], [0.1, 0.2, 0.05]])
    full = ph.project(K_full, R, t, X)
    third = ph.project(ph.scale_intrinsics(K_full, 1 / 3), R, t, X)
    assert np.abs(third * 3 - full).max() < 1e-9

    # And the bug: scale the focal lengths, forget the principal point.
    K_bad = K_full.copy()
    K_bad[0, 0] /= 3
    K_bad[1, 1] /= 3
    bad = ph.project(K_bad, R, t, X)
    assert (bad - third)[0] == pytest.approx([(1 - 1 / 3) * 962.0,
                                              (1 - 1 / 3) * 541.0], abs=1e-6)


def test_crop_shifts_only_the_principal_point():
    K_full = ph.intrinsic_matrix(1450.0, 1452.0, 962.0, 541.0)
    K_crop = ph.crop_intrinsics(K_full, 320, 180)
    assert K_crop[0, 0] == pytest.approx(1450.0)        # focal lengths untouched
    assert K_crop[1, 1] == pytest.approx(1452.0)
    assert K_crop[0, 2] == pytest.approx(642.0)
    assert K_crop[1, 2] == pytest.approx(361.0)


def test_projection_matrix_agrees_with_the_two_step_form():
    import cv2
    R, _ = cv2.Rodrigues(np.array([0.1, 0.2, -0.05]))
    t = np.array([0.2, -0.1, 3.0])
    X = np.array([[0.3, -0.2, 0.1], [0.0, 0.0, 0.0]])
    P = ph.projection_matrix(K, R, t)
    Xh = ph.to_homogeneous(X)
    via_P = ph.from_homogeneous(Xh @ P.T)
    via_steps = ph.project(K, R, t, X)
    assert np.abs(via_P - via_steps).max() < 1e-9


def test_field_of_view_matches_the_hand_formula():
    # fx = (w/2)/tan(FOV/2): 640/tan(30 deg) = 1108.5 px for a 60 degree lens.
    k = ph.intrinsic_matrix(1108.5, 1108.5, 640.0, 360.0)
    hfov, _ = ph.fov_degrees(k, 1280, 720)
    assert hfov == pytest.approx(60.0, abs=0.05)
