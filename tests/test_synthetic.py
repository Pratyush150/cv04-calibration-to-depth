"""The synthetic scenes themselves: if these are wrong, everything else is."""

import cv2
import numpy as np
import pytest

from geo import distortion as dm
from geo import pinhole as ph
from geo import synthetic as syn

K = ph.intrinsic_matrix(800.0, 802.0, 325.0, 238.0)
D = dm.coefficients(k1=-0.30, k2=0.10, p1=0.0012, p2=-0.0009)
PATTERN, SQUARE = (9, 6), 0.025


def test_object_points_are_inner_corners_on_a_plane():
    objp = syn.board_object_points(PATTERN, SQUARE)
    assert objp.shape == (54, 3)
    assert np.all(objp[:, 2] == 0.0)
    assert objp[:, 0].max() == pytest.approx((PATTERN[0] - 1) * SQUARE)
    assert objp[:, 1].max() == pytest.approx((PATTERN[1] - 1) * SQUARE)


def test_the_renderer_puts_corners_where_the_projection_says_they_go():
    """The renderer is a backward ray tracer and the projection is a forward
    matrix product.  They are written independently, so their agreement is
    evidence that the geometry is right rather than merely self-consistent.

    The comparison allows for the chessboard detector's 180-degree ambiguity:
    nothing in a 9x6 pattern distinguishes the board from the same board turned
    upside down, so the detector may return the corners in reverse order.  That
    ambiguity is real, it is why ChArUco boards exist, and a calibration
    absorbs it into the pose.
    """
    objp = syn.board_object_points(PATTERN, SQUARE)
    errors = []
    # Six poses to be sure of at least three detections: the default pose spread
    # deliberately pushes some boards off the edge of the frame, and the
    # detector needs the whole board.
    for i, (rvec, tvec) in enumerate(syn.board_poses(6, tilt_sigma=0.35, spread=True,
                                                     seed=5)):
        img = syn.render_checkerboard(K, D, rvec, tvec, (640, 480), pattern=PATTERN,
                                      square=SQUARE, seed=i)
        ok, corners = syn.detect_corners(img, PATTERN)
        if not ok:
            continue
        R, _ = cv2.Rodrigues(rvec)
        proj = dm.project_distorted(K, D, R, tvec, objp)
        forward = np.abs(corners - proj).max()
        reversed_ = np.abs(corners[::-1] - proj).max()
        errors.append(min(forward, reversed_))
    assert len(errors) >= 3
    assert max(errors) < 0.6            # sub-pixel, on every detected view


def test_rendered_images_look_like_a_board():
    rvec, tvec = syn.board_poses(1, tilt_sigma=0.2, spread=False, seed=1)[0]
    img = syn.render_checkerboard(K, np.zeros(5), rvec, tvec, (640, 480))
    assert img.dtype == np.uint8
    assert img.min() < 60 and img.max() > 200        # ink and paper both present
    assert 60 < img.mean() < 200


def test_stereo_disparity_is_exactly_f_B_over_Z():
    scene = syn.render_stereo_pair(size=(240, 160), fx=300.0, baseline=0.10)
    ok = np.isfinite(scene.depth_left)
    expected = 300.0 * 0.10 / scene.depth_left[ok]
    assert np.abs(scene.disparity_left[ok] - expected).max() < 1e-9


def test_every_scene_pixel_hits_a_surface():
    scene = syn.render_stereo_pair(size=(240, 160))
    assert np.all(scene.surface_left >= 0)
    assert np.all(np.isfinite(scene.depth_left))


def test_occlusions_sit_beside_the_near_object():
    scene = syn.render_stereo_pair(size=(320, 224))
    assert 0.005 < scene.occluded.mean() < 0.15
    # An occluded pixel is one the right camera cannot see because something
    # nearer got in the way, so occlusions cluster at depth discontinuities.
    # The occlusion shadow beside an object is as wide as the DISPARITY STEP at
    # its edge - about 16 px on this scene - so the structuring element has to
    # be at least that wide or the test measures the dilation, not the geometry.
    edges = cv2.Canny((scene.depth_left / scene.depth_left.max() * 255)
                      .astype(np.uint8), 30, 90) > 0
    near_edge = cv2.dilate(edges.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
    assert near_edge[scene.occluded].mean() > 0.9
    # And occluded pixels are on the FAR side of the discontinuity: the near
    # object is what does the hiding.
    assert scene.depth_left[scene.occluded].mean() > scene.depth_left.mean()


def test_named_regions_are_disjoint_and_cover_the_image():
    scene = syn.render_stereo_pair(size=(240, 160))
    masks = [scene.region_mask(n) for n in scene.names]
    total = np.sum([m.astype(int) for m in masks], axis=0)
    assert total.max() == 1                       # no pixel belongs to two surfaces
    assert total.min() == 1                       # and none belongs to none


def test_render_view_with_rotation_and_distortion_runs():
    R, _ = cv2.Rodrigues(np.array([0.03, -0.08, 0.01]))
    img, depth, sid = syn.render_view(K, R, np.array([0.2, 0.0, -0.1]), (240, 160),
                                      D=dm.coefficients(k1=-0.2))
    assert img.shape == (160, 240)
    assert np.isfinite(depth).all()
    assert sid.min() >= 0
