"""Disparity to metres, and how wrong those metres are.

    Z = f * B / d

f in pixels (the RECTIFIED focal length, not the one from the original K), B in
metres, d in pixels.  Derived by subtracting the two projections of the same
point in a rectified pair:

    x_L = f*X/Z + cx        x_R = f*(X - B)/Z + cx
    d   = x_L - x_R = f*B/Z

Read what it says: depth is INVERSELY proportional to disparity.  Everything
uncomfortable about stereo follows from where that d sits in the fraction.
Differentiating and substituting back gives the law this module exists to make
concrete:

    |dZ| = (Z^2 / (f*B)) * |dd|

Depth error grows with the SQUARE of depth.  One pixel of matching error is
millimetres at one metre and metres at twenty, and no amount of code repairs
it - it is the geometry.  The three levers are a longer baseline, a longer
focal length, and better sub-pixel matching.
"""

from __future__ import annotations

import numpy as np


def depth_from_disparity(d, f: float, baseline: float, doffs: float = 0.0):
    """Z = f*B / (d + doffs), in the units of `baseline`.

    `doffs` is the horizontal offset between the two cameras' principal
    points, cx_right - cx_left.  The clean derivation quietly assumes the two
    cameras share a principal point so the cx terms cancel; real rigs do not,
    and what survives the subtraction is this constant.  It is zero after a
    stereoRectify with CALIB_ZERO_DISPARITY - which is one of the real reasons
    that flag exists - and it is 124.343 px on Middlebury's Motorcycle rig,
    where ignoring it makes near objects 62% too far and distant ones 441% too
    far.  The error grows with distance because doffs is a constant added to a
    shrinking denominator, so a scene sanity-checked only on a nearby object
    passes.

    Non-positive denominators return inf: a disparity of zero is a point at
    infinity, and dividing by it silently would put a wall at 1e17 metres in
    the middle of your point cloud.
    """
    d = np.asarray(d, dtype=np.float64)
    den = d + doffs
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = np.where(den > 0, f * baseline / den, np.inf)
    return Z


def disparity_from_depth(Z, f: float, baseline: float, doffs: float = 0.0):
    """The inverse: d = f*B/Z - doffs.  Used to build synthetic ground truth."""
    Z = np.asarray(Z, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(Z > 0, f * baseline / Z - doffs, np.nan)


def depth_error(Z, f: float, baseline: float, disparity_error: float = 1.0):
    """|dZ| = Z^2 / (f*B) * |dd| - the linearised depth uncertainty in metres.

    This is a first-order (derivative) estimate, so it is exact in the limit of
    small dd and slightly optimistic for large ones - `depth_error_exact`
    computes the true asymmetric interval when you need it.  Quote this one for
    the scaling law and that one for a number you will act on.
    """
    Z = np.asarray(Z, dtype=np.float64)
    return Z ** 2 / (f * baseline) * abs(disparity_error)


def depth_error_exact(Z, f: float, baseline: float, disparity_error: float = 1.0,
                      doffs: float = 0.0):
    """The true depth interval for +/- dd of disparity error, as (Z_near, Z_far).

    The interval is ASYMMETRIC, and increasingly so with range, because
    Z = f*B/d is a hyperbola rather than a line.  At the far end the far side
    of the interval runs away much faster than the near side approaches - at
    f=800, B=0.12 and d=4 px, one pixel either way is 19.2 m and 32.0 m around
    a nominal 24 m.  Reporting a symmetric +/- there understates the risk by a
    factor of two on the side that matters.
    """
    d = disparity_from_depth(Z, f, baseline, doffs)
    near = depth_from_disparity(d + abs(disparity_error), f, baseline, doffs)
    far = depth_from_disparity(d - abs(disparity_error), f, baseline, doffs)
    return near, far


def max_useful_range(f: float, baseline: float, disparity_error: float,
                     relative_error: float = 0.10) -> float:
    """The range at which depth error first exceeds a fraction of the depth.

    Set |dZ|/Z = Z*dd/(f*B) = relative_error and solve: Z = relative*f*B/dd.
    This is the number to put in a design document, because "how far can this
    rig see" is meaningless without an accuracy attached, and it is linear in
    both f and B - the two things you can actually buy.
    """
    return relative_error * f * baseline / abs(disparity_error)


def Q_matrix(f: float, baseline: float, cx: float, cy: float,
             cx_right: float | None = None) -> np.ndarray:
    """The 4x4 disparity-to-depth matrix, in OpenCV's layout.

        (u, v, d, 1) -> (X', Y', Z', W'),  then divide by W'

    Q is nothing more than Z = f*B/d generalised to recover X and Y as well:
    the first two rows subtract the principal point, the third row carries f,
    and the bottom row's -1/Tx is 1/B.  The bottom-right entry is where doffs
    lives.  cv2.reprojectImageTo3D pushes an entire disparity map through this
    matrix at once.
    """
    cx_right = cx if cx_right is None else cx_right
    Tx = -abs(baseline)                 # OpenCV's sign for a right camera to the right
    return np.array([[1.0, 0.0, 0.0, -cx],
                     [0.0, 1.0, 0.0, -cy],
                     [0.0, 0.0, 0.0, f],
                     [0.0, 0.0, -1.0 / Tx, (cx - cx_right) / Tx]], dtype=np.float64)


def reproject_disparity(disparity: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """(H, W) disparity -> (H, W, 3) metric points, by hand rather than by cv2.

    Identical to cv2.reprojectImageTo3D to ~1e-9 m (asserted in
    tests/test_depth.py); written out so the Q matrix is a matrix a reader can
    follow rather than an argument to a call.
    """
    h, w = disparity.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    ones = np.ones_like(u)
    vec = np.stack([u, v, disparity.astype(np.float64), ones], axis=-1)
    out = vec @ np.asarray(Q, dtype=np.float64).T
    with np.errstate(divide="ignore", invalid="ignore"):
        pts = out[..., :3] / out[..., 3:4]
    return pts


def triangulate_dlt(P1: np.ndarray, P2: np.ndarray,
                    pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Triangulate N points from two views, by the linear (DLT) method.

    The picture everybody draws - two rays meeting at the point - is wrong, and
    the way it is wrong is the lesson.  Two lines in 3-D have four degrees of
    freedom of relative position and intersecting imposes one equation on them,
    so generic lines do not meet.  Perfect measurements make them meet by
    construction; half a pixel of noise, or merely reporting an integer pixel,
    destroys that.  The rays are SKEW.  There is no intersection to compute, so
    triangulation must minimise something instead.

    The construction: x is parallel to P X as homogeneous vectors, i.e.
    x cross (P X) = 0, which gives two independent rows per view:

        u * P[2] - P[0]         v * P[2] - P[1]

    Four rows from two views, and the X minimising ||A X|| with ||X|| = 1 is
    the last row of Vt.  That is what cv2.triangulatePoints does internally.

    The honest caveat: this minimises ALGEBRAIC error, which has no geometric
    meaning.  The statistically correct estimate minimises reprojection error
    in pixels and is normally reached by running a couple of Gauss-Newton steps
    from this result.  For well-conditioned pairs they agree closely; at
    grazing angles they do not.
    """
    pts1 = np.atleast_2d(np.asarray(pts1, dtype=np.float64))
    pts2 = np.atleast_2d(np.asarray(pts2, dtype=np.float64))
    P1 = np.asarray(P1, dtype=np.float64)
    P2 = np.asarray(P2, dtype=np.float64)

    n = pts1.shape[0]
    X = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        A = np.vstack([pts1[i, 0] * P1[2] - P1[0],
                       pts1[i, 1] * P1[2] - P1[1],
                       pts2[i, 0] * P2[2] - P2[0],
                       pts2[i, 1] * P2[2] - P2[1]])
        _, _, Vt = np.linalg.svd(A)
        Xh = Vt[-1]
        X[i] = Xh[:3] / Xh[3]
    return X


def ray_gap(P1: np.ndarray, P2: np.ndarray, pt1: np.ndarray, pt2: np.ndarray) -> float:
    """Minimum distance in metres between the two back-projected rays.

    Prints the number that kills the intersecting-rays picture.  With 0.5 px of
    noise on each observation of a point five metres away and NOTHING else
    wrong - no calibration error, no distortion - the two rays miss each other
    by millimetres.  They always miss.
    """
    def ray(P, x):
        M = P[:, :3]
        C = -np.linalg.inv(M) @ P[:, 3]              # camera centre
        d = np.linalg.inv(M) @ np.array([x[0], x[1], 1.0])
        return C, d / np.linalg.norm(d)

    C1, d1 = ray(np.asarray(P1, float), np.asarray(pt1, float).ravel())
    C2, d2 = ray(np.asarray(P2, float), np.asarray(pt2, float).ravel())
    n = np.cross(d1, d2)
    nn = np.linalg.norm(n)
    if nn < 1e-12:                                   # parallel rays: no unique gap
        return float(np.linalg.norm(np.cross(C2 - C1, d1)))
    return float(abs((C2 - C1) @ (n / nn)))
