"""Brown-Conrady lens distortion: the forward polynomial and its iterative inverse.

Two facts govern this whole module, and both of them are easy to get wrong in a
way that produces plausible-looking output:

  1. Distortion is defined in NORMALISED camera coordinates x = X/Z, y = Y/Z -
     the values you have after dividing by depth and BEFORE multiplying by K.
     There r is around 0..1, so r^4 and r^6 stay well behaved, the coefficients
     come out dimensionless, and they survive a change of resolution unchanged.
     Feed the polynomial raw pixels instead and r^2 is ~250000, r^6 overflows,
     and you get coordinates in the billions.
  2. The polynomial runs IDEAL -> DISTORTED.  That direction is a closed form.
     The direction you actually want at runtime - measured pixel back to the
     pinhole prediction - has no closed form at all, so it is iterated.  This
     is why undistortion is done as a precomputed remap table instead of per
     pixel per frame: a 1080p frame at five iterations per pixel is ten million
     polynomial evaluations, thirty times a second.

OpenCV packs the coefficients as [k1, k2, p1, p2, k3] - k3 LAST, after the
tangential pair, for backward-compatibility reasons that will not help you at
three in the morning.  Passing them in the "natural" order [k1, k2, k3, p1, p2]
raises no error whatsoever; it just bends the image slightly wrong.
"""

from __future__ import annotations

import cv2
import numpy as np


def coefficients(k1: float = 0.0, k2: float = 0.0, p1: float = 0.0,
                 p2: float = 0.0, k3: float = 0.0) -> np.ndarray:
    """Build the 5-vector in OpenCV's order, with the order spelled out.

    Written as a keyword-only-in-practice constructor so that a reader of the
    calling code sees `coefficients(k1=-0.28, k2=0.10)` instead of an array
    literal whose third slot they have to remember is p1 and not k3.
    """
    return np.array([k1, k2, p1, p2, k3], dtype=np.float64)


def distort_normalized(x, y, D):
    """IDEAL normalised (x, y) -> DISTORTED normalised (xd, yd).

    radial = 1 + k1 r^2 + k2 r^4 + k3 r^6, applied along the radius, plus the
    two tangential terms.  Negative k1 shrinks the radius - points are pulled
    inward and straight lines bow outward, which is barrel distortion and what
    wide lenses do.  Positive k1 is pincushion, typical of telephoto.  The
    tangential pair p1, p2 is not lens shape at all: it is the sensor sitting a
    fraction of a degree out of parallel with the lens, and it is fitted even
    when it is tiny because omitting it pushes that error into the radial terms
    and corrupts those instead.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    k1, k2, p1, p2, k3 = np.asarray(D, dtype=np.float64).ravel()[:5]
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return x * radial + dx, y * radial + dy


def undistort_normalized(xd, yd, D, iters: int = 20):
    """DISTORTED normalised -> IDEAL normalised, by fixed-point iteration.

    There is no closed form: inverting a sixth-order-in-radius polynomial in
    two coupled variables is not something anybody has a formula for.  So:
    assume the current guess is the ideal point, compute the radial factor and
    tangential offsets it implies, undo them on the MEASURED point, repeat.

    The iteration converges because the correction is small near the centre and
    only mildly large at the rim.  With k1 = -0.28, k2 = 0.10 over the unit
    disc it reaches ~1e-4 in five iterations and ~1e-8 in ten (measured by
    tests/test_distortion.py).  OpenCV runs a fixed iteration count internally
    and stops slightly short of exact - its undistortion is an approximation
    with a tolerance, not an exact operation.

    Approximating the inverse by negating the coefficients and running the
    forward polynomial is the popular shortcut.  It is a first-order
    approximation: right in the middle, visibly wrong by several pixels at the
    corners, and it never announces itself.
    """
    xd = np.asarray(xd, dtype=np.float64)
    yd = np.asarray(yd, dtype=np.float64)
    k1, k2, p1, p2, k3 = np.asarray(D, dtype=np.float64).ravel()[:5]
    x, y = xd.copy(), yd.copy()          # start the guess AT the distorted point
    for _ in range(int(iters)):
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        x = (xd - dx) / radial
        y = (yd - dy) / radial
    return x, y


def normalize_pixels(K: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Pixels -> normalised coordinates, x = (u - cx)/fx, y = (v - cy)/fy.

    Separate denominators per axis: that is what absorbs non-square pixels, and
    it is why an anisotropic resize (different sx and sy) still leaves the
    distortion coefficients untouched.
    """
    uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
    K = np.asarray(K, dtype=np.float64)
    return np.column_stack([(uv[:, 0] - K[0, 2]) / K[0, 0],
                            (uv[:, 1] - K[1, 2]) / K[1, 1]])


def denormalize_points(K: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Normalised coordinates -> pixels: u = fx*x + cx, v = fy*y + cy."""
    xy = np.atleast_2d(np.asarray(xy, dtype=np.float64))
    K = np.asarray(K, dtype=np.float64)
    return np.column_stack([K[0, 0] * xy[:, 0] + K[0, 2],
                            K[1, 1] * xy[:, 1] + K[1, 2]])


def project_distorted(K: np.ndarray, D: np.ndarray, R: np.ndarray, t: np.ndarray,
                      X_world: np.ndarray) -> np.ndarray:
    """The full forward camera model, with the lens in the right place.

        world -> [R|t] -> camera -> divide by Z -> DISTORT -> multiply by K

    Anywhere else in that chain and the model is simply a different one.  The
    result matches cv2.projectPoints to ~1e-10 px; tests/test_distortion.py
    asserts it, which is the point of writing it out rather than calling.
    """
    from . import pinhole

    X_cam = pinhole.world_to_camera(R, t, X_world)
    z = X_cam[:, 2]
    xn, yn = X_cam[:, 0] / z, X_cam[:, 1] / z
    xd, yd = distort_normalized(xn, yn, D)
    return denormalize_points(K, np.column_stack([xd, yd]))


def undistort_pixels(K: np.ndarray, D: np.ndarray, uv: np.ndarray,
                     iters: int = 20) -> np.ndarray:
    """Measured pixels -> the pixels a true pinhole would have produced.

    Returns PIXELS, deliberately.  cv2.undistortPoints returns NORMALISED
    coordinates unless you pass P=K, which is the single most common surprise
    in this API: the "undistorted pixel coordinates" all come back between -1
    and 1 and every downstream drawing call plots them in the top-left corner.
    """
    K = np.asarray(K, dtype=np.float64)
    xy_d = normalize_pixels(K, uv)
    x, y = undistort_normalized(xy_d[:, 0], xy_d[:, 1], D, iters=iters)
    return denormalize_points(K, np.column_stack([x, y]))


def undistortion_maps(K: np.ndarray, D: np.ndarray, size: tuple[int, int],
                      new_K: np.ndarray | None = None):
    """Build the remap lookup tables once, per the note at the top of the file.

    `size` is (width, height).  If new_K is None the original K is reused, so
    the undistorted image is measured in the same intrinsics as the raw one -
    which is what you want unless you have a specific reason to crop or expand
    the valid region.  cv2.getOptimalNewCameraMatrix hands back a DIFFERENT K,
    and using the old K to measure an image produced with the new one is a
    quiet few-percent error in every distance you compute afterwards.
    """
    new_K = K if new_K is None else new_K
    return cv2.initUndistortRectifyMap(K, D, None, new_K, size, cv2.CV_32FC1)


def undistort_image(img: np.ndarray, K: np.ndarray, D: np.ndarray,
                    new_K: np.ndarray | None = None) -> np.ndarray:
    """Convenience wrapper: build the maps and remap once."""
    h, w = img.shape[:2]
    map_x, map_y = undistortion_maps(K, D, (w, h), new_K)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def max_model_radius_degrees(fov_limit_deg: float = 120.0) -> float:
    """Half-angle past which the pinhole+polynomial model stops existing.

    The model puts an incoming ray at r = f*tan(theta).  tan diverges at 90
    degrees: between 60 and 89 degrees r grows by a factor of 33, and at 90 it
    is infinite.  No set of k1..k3 approximates that, so past roughly a
    120-degree FULL field of view a fisheye is not a badly fitted Brown-Conrady
    lens, it is a different model (Kannala-Brandt: r = f*theta, four radial
    coefficients, no tangential terms, cv2.fisheye.*).  The symptom of using
    the wrong one is an RMS stuck at several pixels that more views never
    improve.
    """
    return fov_limit_deg / 2.0
