"""Calibration against known ground truth - the check a real rig cannot make."""

import numpy as np
import pytest

from geo import calibrate as cal
from geo import distortion as dm
from geo import pinhole as ph
from geo import synthetic as syn

PATTERN, SQUARE = (9, 6), 0.025
SIZE = (320, 240)
K_TRUE = ph.intrinsic_matrix(400.0, 401.0, 163.0, 119.0)
D_TRUE = dm.coefficients(k1=-0.28, k2=0.09, p1=0.0010, p2=-0.0008)


@pytest.fixture(scope="module")
def rendered_calibration():
    """Render views, detect corners, calibrate.  Module-scoped: the rendering is
    the slow part of this file and three tests want the same result."""
    objp = syn.board_object_points(PATTERN, SQUARE)
    object_points, image_points = [], []
    for i, (rvec, tvec) in enumerate(syn.board_poses(16, tilt_sigma=0.40, spread=True,
                                                     seed=17, pattern=PATTERN,
                                                     square=SQUARE)):
        img = syn.render_checkerboard(K_TRUE, D_TRUE, rvec, tvec, SIZE,
                                      pattern=PATTERN, square=SQUARE, seed=i)
        ok, corners = syn.detect_corners(img, PATTERN)
        if ok:
            object_points.append(objp)
            image_points.append(corners)
    res = cal.calibrate(object_points, image_points, SIZE)
    return res, object_points, image_points


def test_intrinsics_recovered_from_rendered_images_match_the_truth(rendered_calibration):
    """The headline claim of the repository, stated as a tolerance.

    Rendered at 320x240 from K = (fx 400.0, fy 401.0, cx 163.0, cy 119.0) and
    D = (-0.28, 0.09, 0.0010, -0.0008, 0), then recovered from the detected
    corners with no knowledge of any of those numbers.

    Tolerances, and why they are what they are:
      * focal lengths within 1% - the dominant remaining error is the corner
        detector's own sub-pixel bias, measured at ~0.1 px in
        tests/test_synthetic.py.
      * principal point within 4 px - it is the weakest-constrained of the
        four, because a chessboard capture cannot put many corners in the outer
        ring of the frame without the detector losing the board.
      * k1 within 0.05 - k2 and k3 are deliberately NOT checked individually,
        because they are strongly correlated and the solver trades one against
        the other freely.  What has to be right is the curve, which the next
        test checks directly.
    """
    res, obj, img = rendered_calibration
    assert res.n_views >= 5
    assert res.rms < 0.5
    err = cal.intrinsics_error(K_TRUE, D_TRUE, res.K, res.D)
    assert abs(err["fx"]["rel_pct"]) < 1.0
    assert abs(err["fy"]["rel_pct"]) < 1.0
    assert abs(err["cx"]["abs"]) < 4.0
    assert abs(err["cy"]["abs"]) < 4.0
    assert abs(res.D[0] - D_TRUE[0]) < 0.05


def test_the_recovered_lens_model_agrees_with_the_true_one_as_a_CURVE(
        rendered_calibration):
    res, _, _ = rendered_calibration
    grid = np.column_stack([np.repeat(np.linspace(0, SIZE[0] - 1, 30), 30),
                            np.tile(np.linspace(0, SIZE[1] - 1, 30), 30)])
    a = dm.undistort_pixels(K_TRUE, D_TRUE, grid)
    b = dm.undistort_pixels(res.K, res.D, grid)
    gap = np.linalg.norm(a - b, axis=1)
    assert gap.max() < 2.0
    assert gap.mean() < 0.5


def test_per_view_errors_reproduce_the_aggregate_rms(rendered_calibration):
    res, obj, img = rendered_calibration
    per = cal.per_view_errors(res, obj, img)
    assert len(per) == res.n_views
    # The aggregate is the root-mean-square of the per-view root-mean-squares,
    # which is only true because every view has the same number of corners.
    # The tolerance is loose at 1e-4 because OpenCV accumulates its own RMS in a
    # different order, and float addition is not associative.
    assert np.sqrt((per ** 2).mean()) == pytest.approx(res.rms, rel=1e-4)


def test_residuals_are_centred_and_sub_pixel(rendered_calibration):
    res, obj, img = rendered_calibration
    residuals, pixels = cal.per_point_residuals(res, obj, img)
    assert residuals.shape == (res.n_views * PATTERN[0] * PATTERN[1], 2)
    assert np.abs(residuals.mean(axis=0)).max() < 1e-6
    assert np.linalg.norm(residuals, axis=1).max() < 1.0
    assert pixels.shape == residuals.shape


def test_board_tilt_is_measured_from_the_pose_not_assumed():
    # A board with no rotation is 0 degrees of tilt; one rotated 30 degrees
    # about the x axis reads 30.
    import cv2
    rvecs = [np.zeros(3), np.array([np.deg2rad(30.0), 0.0, 0.0]),
             np.array([0.0, np.deg2rad(45.0), 0.0])]
    tilts = cal.board_tilt_degrees(rvecs)
    assert tilts[0] == pytest.approx(0.0, abs=1e-9)
    assert tilts[1] == pytest.approx(30.0, abs=1e-6)
    assert tilts[2] == pytest.approx(45.0, abs=1e-6)
    assert cv2 is not None


def test_the_fronto_parallel_capture_cannot_determine_the_focal_length():
    """The result that matters most in this repository, as an assertion.

    Same ground truth, same corner noise, same solver, same number of views.
    The only difference is whether the board was tilted and swept or held
    fronto-parallel and centred.  Measured over the five seeds this test uses,
    at 640x480:

        tilted and swept : fx spans 799.3 to 800.8   (true 800.0)
        fronto-parallel  : fx spans 769.7 to 797.2
        every RMS in both columns: 0.066 to 0.070 px

    Five seeds understate it.  examples/04 runs ten, and the bad column there
    reaches fx = 2910.1 - 264% high - at an RMS of 0.131 px.

    The bad capture is not biased - it is UNCONSTRAINED, so it lands low on
    some seeds and enormously high on others, and no threshold on reprojection
    error separates the two columns.
    """
    K_true = ph.intrinsic_matrix(800.0, 802.0, 325.0, 238.0)
    D_true = dm.coefficients(k1=-0.30, k2=0.10, p1=0.0012, p2=-0.0009)
    size = (640, 480)
    good_fx, bad_fx, rms_all = [], [], []
    for seed in range(5):
        for tilt, spread, store in ((0.35, True, good_fx), (0.02, False, bad_fx)):
            poses = syn.board_poses(15, tilt_sigma=tilt, spread=spread, seed=seed,
                                    pattern=PATTERN, square=SQUARE)
            obj, img = cal.simulate_capture(K_true, D_true, poses, PATTERN, SQUARE,
                                            size, corner_noise=0.05, seed=1000 + seed)
            res = cal.calibrate(obj, img, size)
            store.append(res.K[0, 0])
            rms_all.append(res.rms)

    good_spread = max(good_fx) - min(good_fx)
    bad_spread = max(bad_fx) - min(bad_fx)
    assert good_spread < 5.0                      # tight around the truth
    assert bad_spread > 20.0                      # and the other one is not
    assert bad_spread > 10 * good_spread
    assert max(np.abs(np.array(good_fx) - 800.0)) < 5.0
    # The point of the whole experiment: the error metric cannot tell them apart.
    assert max(rms_all) < 0.2


def test_corner_coverage_reports_what_it_claims():
    # Corners packed into the middle of the frame leave every border cell empty;
    # corners spread to the edges do not.
    centre = [np.column_stack([np.full(50, 160.0), np.full(50, 120.0)])]
    assert cal.corner_coverage(centre, SIZE) == 1.0
    rng = np.random.default_rng(0)
    everywhere = [np.column_stack([rng.uniform(0, SIZE[0], 4000),
                                   rng.uniform(0, SIZE[1], 4000)])]
    assert cal.corner_coverage(everywhere, SIZE) == 0.0


def test_simulate_capture_uses_a_separate_noise_stream():
    """Two capture geometries must see the SAME noise realisation, or the
    comparison in the test above is not a controlled experiment."""
    K_true = ph.intrinsic_matrix(800.0, 800.0, 320.0, 240.0)
    poses_a = syn.board_poses(3, tilt_sigma=0.35, spread=True, seed=1)
    poses_b = syn.board_poses(3, tilt_sigma=0.02, spread=False, seed=1)
    _, img_a = cal.simulate_capture(K_true, np.zeros(5), poses_a, PATTERN, SQUARE,
                                    SIZE, corner_noise=0.05, seed=42)
    _, img_b = cal.simulate_capture(K_true, np.zeros(5), poses_b, PATTERN, SQUARE,
                                    SIZE, corner_noise=0.05, seed=42)
    clean_a = np.array([dm.project_distorted(K_true, np.zeros(5),
                                             __import__("cv2").Rodrigues(r)[0], t,
                                             syn.board_object_points(PATTERN, SQUARE))
                        for r, t in poses_a])
    clean_b = np.array([dm.project_distorted(K_true, np.zeros(5),
                                             __import__("cv2").Rodrigues(r)[0], t,
                                             syn.board_object_points(PATTERN, SQUARE))
                        for r, t in poses_b])
    noise_a = np.array(img_a) - clean_a
    noise_b = np.array(img_b) - clean_b
    assert np.abs(noise_a - noise_b).max() < 1e-12
