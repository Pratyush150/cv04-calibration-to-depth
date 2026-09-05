"""The homography DLT: against the truth, against the library, and its conditioning."""

import cv2
import numpy as np
import pytest

from geo import homography as hg

H_TRUE = np.array([[1.05, 0.08, -30.0],
                   [0.03, 0.98, 17.0],
                   [1.2e-4, -5e-5, 1.0]])


def clean_correspondences(n=30, seed=0):
    rng = np.random.default_rng(seed)
    src = rng.uniform([50, 40], [590, 440], (n, 2))
    return src, hg.apply_transform(H_TRUE, src)


def test_dlt_recovers_a_known_homography():
    src, dst = clean_correspondences()
    H = hg.homography_dlt(src, dst)
    assert np.abs(H - H_TRUE).max() < 1e-9


def test_dlt_matches_opencv_on_clean_correspondences():
    src, dst = clean_correspondences()
    H = hg.homography_dlt(src, dst)
    H_cv, _ = cv2.findHomography(src, dst, 0)
    H_cv = H_cv / H_cv[2, 2]
    assert np.abs(H - H_cv).max() < 1e-4
    # And ours is the more exact of the two on noiseless data, because OpenCV
    # stops its refinement on a tolerance.  Worth asserting so that the claim in
    # examples/06 is checked rather than asserted.
    assert hg.transfer_error(H, src, dst).max() < hg.transfer_error(H_cv, src, dst).max()


def test_four_points_are_enough_and_three_are_not():
    src, dst = clean_correspondences(n=4)
    H = hg.homography_dlt(src, dst)
    assert hg.transfer_error(H, src, dst).max() < 1e-8
    with pytest.raises(ValueError):
        hg.homography_dlt(src[:3], dst[:3])


def test_mismatched_input_lengths_raise():
    src, dst = clean_correspondences()
    with pytest.raises(ValueError):
        hg.homography_dlt(src, dst[:-1])


def test_normalisation_transform_does_what_it_claims():
    rng = np.random.default_rng(3)
    pts = rng.uniform([100, 100], [500, 400], (200, 2))
    T = hg.normalizing_transform(pts)
    q = hg.apply_transform(T, pts)
    assert np.abs(q.mean(axis=0)).max() < 1e-12                 # zero mean
    assert np.linalg.norm(q, axis=1).mean() == pytest.approx(np.sqrt(2.0))


def test_normalisation_improves_conditioning_by_orders_of_magnitude():
    src, dst = clean_correspondences()
    good = hg.condition_number(src, dst, normalize=True)
    raw = hg.condition_number(src, dst, normalize=False)
    assert good < 20.0
    assert raw / good > 1e3


def test_transfer_error_is_zero_for_the_generating_map():
    src, dst = clean_correspondences()
    assert hg.transfer_error(H_TRUE, src, dst).max() < 1e-9


def test_estimate_degrades_gracefully_with_noise():
    src, dst = clean_correspondences(n=60)
    rng = np.random.default_rng(7)
    errs = []
    for sigma in (0.0, 0.5, 2.0):
        noisy = dst + rng.normal(0, sigma, dst.shape)
        H = hg.homography_dlt(src, noisy)
        errs.append(hg.transfer_error(H, src, dst).mean())
    assert errs[0] < 1e-9
    assert errs[0] < errs[1] < errs[2]
    assert errs[2] < 2.0            # 2 px of point noise must not become 2 px of map error
