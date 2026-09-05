"""The block matcher, and the rectification that has to happen before it."""

import cv2
import numpy as np
import pytest

from geo import distortion as dm
from geo import pinhole as ph
from geo import stereo as st
from geo import synthetic as syn

SIZE = (320, 224)
NDISP, WINDOW = 48, 9   # ndisp is a multiple of 16 because cv2.StereoBM demands it


@pytest.fixture(scope="module")
def scene():
    return syn.render_stereo_pair(size=SIZE, fx=360.0, baseline=0.12, seed=7)


def textured(scene):
    """Non-occluded pixels on the three surfaces that carry real texture."""
    return (~scene.unmatchable &
            (scene.region_mask("near_slab") | scene.region_mask("ramp") |
             scene.region_mask("wall")))


def test_the_scene_ground_truth_is_self_consistent(scene):
    ok = np.isfinite(scene.depth_left) & np.isfinite(scene.disparity_left)
    expected = 360.0 * 0.12 / scene.depth_left[ok]
    assert np.abs(scene.disparity_left[ok] - expected).max() < 1e-9
    assert 0.005 < scene.occluded.mean() < 0.15       # a plausible occlusion fraction
    # Every pixel closer to the left edge than its own disparity has no match
    # in the right image at all: x - d < 0.  The guaranteed-empty band is
    # therefore as wide as the SMALLEST disparity in the scene, and the widest
    # part of it is wherever the scene is nearest.
    assert scene.outside_right[:, :int(np.nanmin(scene.disparity_left))].all()
    assert not scene.outside_right[:, int(np.nanmax(scene.disparity_left)) + 1:].any()


def test_block_matching_recovers_a_known_disparity(scene):
    disp = st.block_match(scene.left, scene.right, ndisp=NDISP, window=WINDOW)
    sc = st.score_disparity(disp, scene.disparity_left, textured(scene))
    assert sc["density"] > 0.80
    assert sc["mae"] < 0.5
    assert sc["bad_pct"] < 8.0


def test_sad_and_ssd_land_in_the_same_place(scene):
    sad = st.block_match(scene.left, scene.right, ndisp=NDISP, window=WINDOW, metric="SAD")
    ssd = st.block_match(scene.left, scene.right, ndisp=NDISP, window=WINDOW, metric="SSD")
    m = textured(scene) & np.isfinite(sad) & np.isfinite(ssd)
    assert np.median(np.abs(sad[m] - ssd[m])) < 0.15


def test_subpixel_refinement_reduces_the_error(scene):
    integer = st.block_match(scene.left, scene.right, ndisp=NDISP, window=WINDOW,
                             subpixel=False, lr_check=False)
    refined = st.block_match(scene.left, scene.right, ndisp=NDISP, window=WINDOW,
                             subpixel=True, lr_check=False)
    m = textured(scene) & np.isfinite(integer) & np.isfinite(refined)
    e_int = np.abs(integer[m] - scene.disparity_left[m]).mean()
    e_sub = np.abs(refined[m] - scene.disparity_left[m]).mean()
    assert e_sub < e_int
    # The integer map really is quantised and the refined one really is not.
    assert np.all(np.abs(integer[np.isfinite(integer)] % 1.0) < 1e-6)
    assert np.any(np.abs(refined[m] % 1.0) > 1e-3)


def test_the_left_right_check_targets_occlusions(scene):
    vol = st.cost_volume(scene.left, scene.right, NDISP, WINDOW, "SAD")
    best = np.argmin(vol, axis=0)
    agree = st.left_right_consistency(vol, best)
    rejected_occluded = (~agree)[scene.occluded].mean()
    # Compared against the TEXTURED visible pixels: the check also (correctly)
    # throws out the flat patch and the stripes, so measuring against every
    # visible pixel would compare occlusion rejection with texture rejection.
    rejected_visible = (~agree)[textured(scene)].mean()
    assert rejected_occluded > 0.4
    assert rejected_occluded > 3 * rejected_visible


def test_the_invalid_band_is_the_most_expensive_place_to_match(scene):
    vol = st.cost_volume(scene.left, scene.right, NDISP, WINDOW, "SAD")
    # For d = 20 the first 20 columns have no right-image pixel to compare
    # against.  Filling them with zeros would make them the cheapest match in
    # the volume and paint a confident stripe down the left edge.
    assert vol[20, :, :5].min() > vol[20, :, 70:].max()


def test_a_textureless_region_produces_a_flat_cost_curve(scene):
    vol = st.cost_volume(scene.left, scene.right, NDISP, WINDOW, "SAD")
    flat = scene.region_mask("textureless")
    good = scene.region_mask("wall")
    ys, xs = np.nonzero(flat[:, NDISP + WINDOW:])
    fy, fx = int(ys[len(ys) // 2]), int(xs[len(xs) // 2]) + NDISP + WINDOW
    ys, xs = np.nonzero(good[:, NDISP + WINDOW:])
    gy, gx = int(ys[len(ys) // 2]), int(xs[len(xs) // 2]) + NDISP + WINDOW
    spread_flat = vol[:, fy, fx].max() - vol[:, fy, fx].min()
    spread_good = vol[:, gy, gx].max() - vol[:, gy, gx].min()
    assert spread_flat < spread_good


def test_a_repeated_pattern_produces_several_equal_minima(scene):
    vol = st.cost_volume(scene.left, scene.right, NDISP, WINDOW, "SAD")
    stripes = scene.region_mask("repeated")
    ys, xs = np.nonzero(stripes[:, NDISP + WINDOW:])
    y, x = int(ys[len(ys) // 2]), int(xs[len(xs) // 2]) + NDISP + WINDOW
    curve = vol[:, y, x]
    winner = int(np.argmin(curve))
    far = [i for i in np.argsort(curve) if abs(int(i) - winner) > 2]
    # The runner-up that is a genuinely different disparity is nearly as cheap,
    # which is what makes the matcher confidently wrong rather than uncertain.
    assert (curve[far[0]] - curve[winner]) < 0.25 * (curve.max() - curve.min())


def test_opencv_matchers_run_and_are_in_the_same_ballpark(scene):
    bm = st.opencv_bm(scene.left, scene.right, ndisp=NDISP, block=15)
    sgbm = st.opencv_sgbm(scene.left, scene.right, ndisp=NDISP, block=5)
    for d in (bm, sgbm):
        sc = st.score_disparity(d, scene.disparity_left, textured(scene))
        assert sc["density"] > 0.4
        assert sc["mae"] < 1.5


def test_rectification_puts_correspondences_on_the_same_row():
    """The end-to-end check on a deliberately misaligned, distorted rig.

    Measured with ground-truth correspondences rather than feature matches,
    because a feature matcher on a repetitive scene contributes its own errors
    and this test is about the warp.
    """
    size = (320, 224)
    K = ph.intrinsic_matrix(360.0, 360.0, (size[0] - 1) / 2, (size[1] - 1) / 2)
    D = dm.coefficients(k1=-0.22, k2=0.06, p1=0.0008, p2=-0.0006)
    R, _ = cv2.Rodrigues(np.array([0.020, 0.045, 0.030]))
    C2 = np.array([0.12, 0.006, -0.004])
    T = ph.extrinsics_from_centre(R, C2)
    rect = st.rectify_pair(K, D, K, D, size, R, T, alpha=0.0)

    _, depth, _ = syn.render_view(K, np.eye(3), np.zeros(3), size, D=D, seed=21)
    rng = np.random.default_rng(0)
    u = rng.integers(30, size[0] - 30, 300)
    v = rng.integers(30, size[1] - 30, 300)
    xy = dm.normalize_pixels(K, np.column_stack([u, v]))
    xn, yn = dm.undistort_normalized(xy[:, 0], xy[:, 1], D)
    X = np.column_stack([xn, yn, np.ones_like(xn)]) * depth[v, u][:, None]

    h1 = (rect.R1 @ X.T).T @ rect.P1[:, :3].T + rect.P1[:, 3]
    h2 = (rect.R2 @ (X @ R.T + T).T).T @ rect.P2[:, :3].T + rect.P2[:, 3]
    dy = np.abs(h1[:, 1] / h1[:, 2] - h2[:, 1] / h2[:, 2])
    assert np.median(dy) < 1e-6
    assert dy.max() < 1e-6

    # Disparity must be positive for a scene in front of the rig, or the sign
    # convention is inverted somewhere and every depth comes out negative.
    disparity = h1[:, 0] / h1[:, 2] - h2[:, 0] / h2[:, 2]
    assert np.all(disparity > 0)


def test_rectification_changes_the_focal_length():
    """Documented because it costs people days: the f you calibrated is not the
    f you measure disparity in."""
    size = (320, 224)
    K = ph.intrinsic_matrix(360.0, 360.0, (size[0] - 1) / 2, (size[1] - 1) / 2)
    D = dm.coefficients(k1=-0.22, k2=0.06)
    R, _ = cv2.Rodrigues(np.array([0.02, 0.045, 0.03]))
    T = ph.extrinsics_from_centre(R, np.array([0.12, 0.006, -0.004]))
    rect = st.rectify_pair(K, D, K, D, size, R, T, alpha=0.0)
    assert abs(rect.f_rect - K[0, 0]) / K[0, 0] > 0.01
    # The baseline read back out of P2 must still be the physical one.
    assert rect.baseline == pytest.approx(np.linalg.norm([0.12, 0.006, -0.004]),
                                          rel=1e-6)
