"""Rectification, and a block matcher written from scratch.

Rectification is the warp that turns the epipolar constraint from "somewhere on
this diagonal line" into "somewhere on this ROW".  A diagonal line through a
pixel grid has to be walked with interpolation, accumulating rounding error and
resampling a patch per candidate; a row is a contiguous slice of memory.  After
rectification, matching is an integer horizontal scan.

Two words that get used interchangeably and must not be:

  * UNDISTORTION removes lens curvature.  One camera, no reference to any
    other.
  * RECTIFICATION removes lens curvature AND rotates both virtual image planes
    until they are coplanar and row-aligned.  It is a PAIR operation - "aligned
    with what?" has no answer for a single camera.

Rectification includes undistortion.  Applying both in sequence undistorts
twice and warps the image into nonsense.

The other surprise: stereoRectify CHOOSES A NEW FOCAL LENGTH.  The two cameras
must end up sharing one, so the function picks a common value and returns it in
P1[0,0] and Q[2,3].  Use the old K's fx in Z = f*B/d afterwards and every depth
is wrong by a constant percentage - which looks exactly like a mis-measured
baseline, so people re-measure the rig with calipers and find nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Rectification:
    """Everything stereoRectify hands back, kept together so it cannot drift apart."""
    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray
    map1: tuple
    map2: tuple
    roi1: tuple
    roi2: tuple

    @property
    def f_rect(self) -> float:
        """The rectified focal length - the ONLY f that belongs in Z = f*B/d."""
        return float(self.P1[0, 0])

    @property
    def baseline(self) -> float:
        """Baseline in metres, read back out of P2: P2[0,3] = -f_rect * B."""
        return float(-self.P2[0, 3] / self.P1[0, 0])


def rectify_pair(K1, D1, K2, D2, size, R, T, *, alpha: float = 0.0) -> Rectification:
    """Run the full stereoRectify -> initUndistortRectifyMap chain once.

    The maps are built ONCE at startup and applied per frame with cv2.remap,
    because the undistortion inside them has no closed form and would otherwise
    be iterated per pixel per frame - ten million polynomial evaluations for a
    1080p frame, thirty times a second.

    alpha=0 crops to only-valid pixels and changes the effective K; alpha=1
    keeps every source pixel and leaves black wedges at the border.  Whichever
    one produced the image you are measuring in is the one whose P you must use
    downstream, and mixing them is a quiet few-percent error.
    """
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2, size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha)
    map1 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, size, cv2.CV_32FC1)
    map2 = cv2.initUndistortRectifyMap(K2, D2, R2, P2, size, cv2.CV_32FC1)
    return Rectification(R1, R2, P1, P2, Q, map1, map2, roi1, roi2)


def remap_pair(rect: Rectification, left: np.ndarray, right: np.ndarray):
    """Apply the precomputed maps.  This is what runs every frame."""
    return (cv2.remap(left, rect.map1[0], rect.map1[1], cv2.INTER_LINEAR),
            cv2.remap(right, rect.map2[0], rect.map2[1], cv2.INTER_LINEAR))


def row_alignment_error(left: np.ndarray, right: np.ndarray, *, n_features: int = 800):
    """The number that tells you whether rectification actually worked.

    Rectification fails QUIETLY.  A mediocre calibration produces a slightly
    wrong warp, no exception, no visible tear - and the block matcher then
    confidently matches the wrong row against the wrong row and returns a
    disparity map that looks smooth, structured and believable, and is wrong.
    A loud failure costs ten minutes; this one costs two days.

    So: detect features in both RECTIFIED images, match them, and measure
    |y_left - y_right|.  Returns (median, mean, p90, count).

    Gate on the MEDIAN, not the mean.  Stereo feature matching always leaks a
    few gross mismatches, and a handful of matches wrong by thirty pixels drags
    the mean over any threshold while nine-tenths of the matches sit at zero.
    Median under 0.5 px: proceed.  Median over 1 px: fix the calibration before
    looking at a single disparity map.  A mean far above the median is a
    matching problem, not a rectification one.
    """
    orb = cv2.ORB_create(n_features)
    kL, dL = orb.detectAndCompute(left, None)
    kR, dR = orb.detectAndCompute(right, None)
    if dL is None or dR is None or len(kL) < 8 or len(kR) < 8:
        return np.nan, np.nan, np.nan, 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(dL, dR)
    if len(matches) < 8:
        return np.nan, np.nan, np.nan, len(matches)
    dy = np.array([abs(kL[m.queryIdx].pt[1] - kR[m.trainIdx].pt[1]) for m in matches])
    return float(np.median(dy)), float(dy.mean()), float(np.percentile(dy, 90)), len(dy)


# --------------------------------------------------------------------------
# The block matcher
# --------------------------------------------------------------------------

def cost_volume(left: np.ndarray, right: np.ndarray, ndisp: int, window: int,
                metric: str = "SAD") -> np.ndarray:
    """The D x H x W array where entry (d, y, x) is the cost of disparity d at (x, y).

    The obvious implementation - for each pixel, for each disparity, compare a
    patch - is four nested Python loops and tens of millions of iterations.
    Turn it inside out: for each disparity, score EVERY pixel at once.  There
    are only `ndisp` disparities, so that is a few dozen vectorised passes
    instead of tens of millions of interpreted ones.  The difference on the
    scene in examples/08 is roughly two seconds against several minutes.

    SAD sums |L - R|; SSD sums (L - R)^2.  SSD weights a single large
    difference far more heavily, so it is the more decisive of the two on clean
    synthetic data and the more fragile of the two under an outlier - a
    specular highlight visible in one camera only will dominate an SSD window
    and merely annoy a SAD one.  Both are here so the comparison can be
    measured rather than asserted.

    The invalid band matters.  For x < d there is no right pixel to compare
    against, and filling it with zeros would make it the CHEAPEST match
    everywhere - a black stripe of confident nonsense down the left edge of
    every disparity map.  It is filled with the maximum possible cost instead.
    """
    L = left.astype(np.float32)
    R = right.astype(np.float32)
    h, w = L.shape
    worst = 255.0 ** 2 if metric.upper() == "SSD" else 255.0
    vol = np.full((ndisp, h, w), np.inf, dtype=np.float32)
    for d in range(ndisp):
        diff = np.full((h, w), worst, dtype=np.float32)
        if d < w:
            delta = L[:, d:] - R[:, :w - d]
            diff[:, d:] = delta * delta if metric.upper() == "SSD" else np.abs(delta)
        vol[d] = cv2.boxFilter(diff, -1, (window, window), normalize=True,
                               borderType=cv2.BORDER_REPLICATE)
    return vol


def subpixel_parabola(vol: np.ndarray, best: np.ndarray):
    """Fit a parabola through the three costs around the winner and take its vertex.

        d* = d + 0.5 * (C[d-1] - C[d+1]) / (C[d-1] - 2 C[d] + C[d+1])

    The numerator is an asymmetry measure: equal neighbours give zero
    correction and the integer winner stands.  This is not cosmetic.  Integer
    disparity quantises depth brutally at range - a quarter of a pixel is
    45 mm at 5.8 m on a typical rig - and the fit costs three array lookups.

    Two guards, both load-bearing.  A non-positive denominator means the three
    samples are flat or CONCAVE, so there is no minimum to refine; the integer
    winner is kept rather than a division by nearly-zero being clamped into a
    plausible-looking number.  And a vertex more than half a pixel from the
    winner is not a refinement of that winner, so the shift is clipped.
    """
    ndisp = vol.shape[0]
    k = np.clip(best, 1, ndisp - 2)[None]
    c0 = np.take_along_axis(vol, k - 1, 0)[0]
    c1 = np.take_along_axis(vol, k, 0)[0]
    c2 = np.take_along_axis(vol, k + 1, 0)[0]
    den = c0 - 2.0 * c1 + c2
    bad = den <= 1e-6
    shift = 0.5 * (c0 - c2) / np.where(bad, 1.0, den)
    shift[bad] = 0.0
    return k[0].astype(np.float32) + np.clip(shift, -0.5, 0.5).astype(np.float32)


def left_right_consistency(vol: np.ndarray, best: np.ndarray, tol: float = 1.0):
    """Mark pixels whose match does not agree when the roles of the images swap.

    Some pixels visible on the left simply are not present on the right - stand
    behind a lamp post and the strip of wall it hides from one camera has no
    correct match in the other.  That is OCCLUSION, and for those pixels any
    disparity a matcher reports is invention.  The detector: ask the right
    image what IT thinks pixel x - d matches; if it points back to roughly
    where you came from, the match is mutual.

    The right-to-left cost volume is not a second matching pass - it is a SHIFT
    of the one you already have, because cost_R[d, y, x] and cost_L[d, y, x+d]
    describe the same pair of pixels indexed from opposite ends.  Recomputing
    it doubles the runtime for nothing.
    """
    ndisp, h, w = vol.shape
    volR = np.full_like(vol, np.inf)
    for d in range(ndisp):
        volR[d, :, :w - d] = vol[d, :, d:]
    bestR = np.argmin(volR, axis=0)
    xs = np.arange(w)[None, :].repeat(h, 0)
    partner = np.clip(xs - best, 0, w - 1)
    return np.abs(best - np.take_along_axis(bestR, partner, 1)) <= tol


def block_match(left: np.ndarray, right: np.ndarray, *, ndisp: int = 64, window: int = 9,
                metric: str = "SAD", subpixel: bool = True, lr_check: bool = True,
                tol: float = 1.0) -> np.ndarray:
    """Full from-scratch matcher.  Returns float disparity with NaN where invalid.

    NaN rather than 0 for invalid pixels, deliberately.  Zero is a legal
    disparity meaning "infinitely far away", so a map that encodes "no idea" as
    zero has silently put a horizon behind every occlusion, and every mean you
    take of it afterwards is wrong.  An honest matcher says what it does not
    know, and the left `ndisp` columns - where the search window falls off the
    edge of the right image - are the first thing it does not know.
    """
    vol = cost_volume(left, right, ndisp, window, metric)
    best = np.argmin(vol, axis=0)
    disp = subpixel_parabola(vol, best) if subpixel else best.astype(np.float32)
    disp = disp.astype(np.float32)
    if lr_check:
        disp = np.where(left_right_consistency(vol, best, tol), disp, np.nan)
    disp[:, :ndisp] = np.nan
    return disp


def opencv_bm(left: np.ndarray, right: np.ndarray, *, ndisp: int = 64,
              block: int = 15) -> np.ndarray:
    """cv2.StereoBM - the library's own SAD block matcher, for comparison.

    ndisp must be a multiple of 16 and block must be odd; both are asserted by
    OpenCV with a message that does not name which argument is wrong.  Output
    is fixed-point with 4 fractional bits, hence the /16, and invalid pixels
    come back as (minDisparity - 1) * 16.  Testing validity with
    `disp > disp.min()` therefore keeps every invalid pixel, because that value
    IS the minimum - the test has to be against the configured minimum.
    """
    bm = cv2.StereoBM_create(numDisparities=ndisp, blockSize=block)
    d = bm.compute(left, right).astype(np.float32) / 16.0
    d[d <= 0] = np.nan
    return d


def opencv_sgbm(left: np.ndarray, right: np.ndarray, *, ndisp: int = 64,
                block: int = 5) -> np.ndarray:
    """cv2.StereoSGBM - semi-global matching, the practical baseline.

    SGBM differs from the matcher above in three ways, not one, and being able
    to name all three is the difference between having read the docs and having
    read the class:

      1. A different cost - a Birchfield-Tomasi sub-pixel metric, not SAD.
      2. Semi-global aggregation along scanline directions (five by default,
         eight with MODE_HH), penalising disparity changes between neighbours
         so a confident pixel can pull an ambiguous neighbour to the right
         answer.  This is the part the from-scratch matcher genuinely does not
         have: winner-take-all treats every pixel as independent.
      3. Post-filters this configuration switches on - uniquenessRatio,
         speckle filtering, and its own left-right check.
    """
    sg = cv2.StereoSGBM_create(minDisparity=0, numDisparities=ndisp, blockSize=block,
                               P1=8 * block * block, P2=32 * block * block,
                               uniquenessRatio=10, speckleWindowSize=100,
                               speckleRange=2, disp12MaxDiff=1)
    d = sg.compute(left, right).astype(np.float32) / 16.0
    d[d <= 0] = np.nan
    return d


def score_disparity(est: np.ndarray, truth: np.ndarray, mask: np.ndarray | None = None,
                    bad_threshold: float = 1.0) -> dict:
    """Compare an estimate against ground truth on the pixels where both exist.

    Reports the density alongside the error, because the two trade off directly
    and quoting either alone is misleading: a matcher that rejects 90% of the
    image can post a beautiful mean error, and one that answers everywhere
    posts a worse one while being more useful.  `bad_%` is the fraction of
    valid pixels wrong by more than `bad_threshold` px, which is the metric
    stereo benchmarks actually rank on - a mean is dominated by a few gross
    outliers.
    """
    valid = np.isfinite(est) & np.isfinite(truth)
    if mask is not None:
        valid &= mask
    n = int(valid.sum())
    if n == 0:
        return {"n": 0, "density": 0.0, "mae": np.nan, "rmse": np.nan, "bad_pct": np.nan}
    err = np.abs(est[valid] - truth[valid])
    total = int((np.isfinite(truth) & (mask if mask is not None else True)).sum())
    return {"n": n,
            "density": n / max(total, 1),
            "mae": float(err.mean()),
            "rmse": float(np.sqrt((err ** 2).mean())),
            "bad_pct": float(100.0 * (err > bad_threshold).mean())}
