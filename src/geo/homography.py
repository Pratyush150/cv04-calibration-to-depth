"""The linear (DLT) estimate of a homography, with Hartley normalisation.

A homography is the 3x3 matrix that maps one plane to another through a
projective camera - a calibration board to its image, one photo of a wall to
another photo of the same wall, an image to its bird's-eye rectification.  It
is the object that makes camera calibration possible at all: each view of a
planar board gives you one homography, and K is what all of those homographies
have in common.

This module implements the estimate from scratch, because "solve for the null
space of a stacked constraint matrix" is the single most reused pattern in
geometric vision - the eight-point algorithm in epipolar.py and the
triangulation DLT in depth.py are the same three lines with a different A.
"""

from __future__ import annotations

import numpy as np


def normalizing_transform(pts: np.ndarray) -> np.ndarray:
    """Hartley normalisation: translate to zero mean, scale to mean radius sqrt(2).

    This is not cosmetic.  Raw pixel coordinates are in the hundreds, so a
    design matrix built from them has entries spanning u*u ~ 1e6 next to
    entries equal to 1 - six orders of magnitude in one matrix.  Its smallest
    singular value is then dominated by float rounding rather than by the
    geometry, and the null vector you extract is noise.  Centring and scaling
    puts every entry within an order of magnitude of every other.

    sqrt(2) specifically because it makes the average point sit at (1, 1) in
    the normalised frame, so all three homogeneous components are the same
    size.
    """
    pts = np.atleast_2d(np.asarray(pts, dtype=np.float64))
    mean = pts.mean(axis=0)
    centred = pts - mean
    rms = np.sqrt((centred ** 2).sum(axis=1)).mean()
    scale = np.sqrt(2.0) / max(rms, 1e-12)
    return np.array([[scale, 0.0, -scale * mean[0]],
                     [0.0, scale, -scale * mean[1]],
                     [0.0, 0.0, 1.0]])


def apply_transform(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 3x3 projective transform to (N, 2) points, dividing through."""
    pts = np.atleast_2d(np.asarray(pts, dtype=np.float64))
    h = np.hstack([pts, np.ones((pts.shape[0], 1))]) @ np.asarray(T, dtype=np.float64).T
    return h[:, :2] / h[:, 2:3]


def homography_dlt(src: np.ndarray, dst: np.ndarray, *, normalize: bool = True) -> np.ndarray:
    """Estimate H with dst ~ H @ src from >= 4 correspondences, by DLT.

    The constraint per correspondence.  We want (u', v', 1) parallel to
    H @ (u, v, 1) - parallel, not equal, because homogeneous vectors are only
    defined up to scale.  "Parallel" is a zero cross product, and writing out
    x' x (H x) = 0 gives three equations of which two are independent:

        [ -u -v -1   0   0   0   u'u  u'v  u' ]
        [  0   0   0  -u  -v  -1   v'u  v'v  v' ]   @ vec(H) = 0

    Four correspondences give eight rows, and vec(H) has nine unknowns with one
    of them being overall scale, so eight rows is exactly enough.  With more
    than four the system is overdetermined and inconsistent under noise, so
    take the h minimising ||A h|| subject to ||h|| = 1: the right-singular
    vector of A with the smallest singular value, which is the last row of Vt.

    Caveat worth stating out loud: this minimises ALGEBRAIC error ||A h||,
    which has no units and no geometric meaning.  The statistically correct
    thing minimises reprojection error in pixels, usually by feeding this
    result to Levenberg-Marquardt - which is exactly what cv2.findHomography
    does after its own DLT.  On clean correspondences the two agree to ~1e-12
    (tests/test_homography.py); on noisy ones they do not, and the DLT is the
    initialiser rather than the answer.
    """
    src = np.atleast_2d(np.asarray(src, dtype=np.float64))
    dst = np.atleast_2d(np.asarray(dst, dtype=np.float64))
    if src.shape[0] < 4 or src.shape != dst.shape:
        raise ValueError("need at least 4 matched points, same count in both sets")

    if normalize:
        T_src = normalizing_transform(src)
        T_dst = normalizing_transform(dst)
        s = apply_transform(T_src, src)
        d = apply_transform(T_dst, dst)
    else:
        T_src = T_dst = np.eye(3)
        s, d = src, dst

    n = s.shape[0]
    A = np.zeros((2 * n, 9), dtype=np.float64)
    u, v = s[:, 0], s[:, 1]
    up, vp = d[:, 0], d[:, 1]
    A[0::2, 0] = -u
    A[0::2, 1] = -v
    A[0::2, 2] = -1.0
    A[0::2, 6] = up * u
    A[0::2, 7] = up * v
    A[0::2, 8] = up
    A[1::2, 3] = -u
    A[1::2, 4] = -v
    A[1::2, 5] = -1.0
    A[1::2, 6] = vp * u
    A[1::2, 7] = vp * v
    A[1::2, 8] = vp

    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)

    # Undo the normalisation: the H we just solved for maps normalised source
    # points to normalised destination points, so sandwich it back.
    H = np.linalg.inv(T_dst) @ H @ T_src
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]          # fix the free scale so two estimates compare
    return H


def transfer_error(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Per-point distance in pixels between H @ src and dst.

    Reported instead of the algebraic residual because pixels are the unit the
    reader can judge: 0.3 px is good, 3 px is a bad fit, and ||A h|| is neither.
    """
    return np.linalg.norm(apply_transform(H, src) - np.atleast_2d(dst), axis=1)


def condition_number(src: np.ndarray, dst: np.ndarray, *, normalize: bool) -> float:
    """Condition number of the DLT design matrix, with and without normalisation.

    This is the number that makes the case for Hartley normalisation concrete
    rather than folkloric: examples/06 prints both, and the unnormalised one is
    orders of magnitude larger on ordinary 640x480 pixel coordinates.
    """
    src = np.atleast_2d(np.asarray(src, dtype=np.float64))
    dst = np.atleast_2d(np.asarray(dst, dtype=np.float64))
    if normalize:
        src = apply_transform(normalizing_transform(src), src)
        dst = apply_transform(normalizing_transform(dst), dst)
    n = src.shape[0]
    A = np.zeros((2 * n, 9))
    u, v = src[:, 0], src[:, 1]
    up, vp = dst[:, 0], dst[:, 1]
    A[0::2, 0] = -u; A[0::2, 1] = -v; A[0::2, 2] = -1.0
    A[0::2, 6] = up * u; A[0::2, 7] = up * v; A[0::2, 8] = up
    A[1::2, 3] = -u; A[1::2, 4] = -v; A[1::2, 5] = -1.0
    A[1::2, 6] = vp * u; A[1::2, 7] = vp * v; A[1::2, 8] = vp
    sv = np.linalg.svd(A, compute_uv=False)
    # sv[-1] is the null direction we are trying to extract, and on noiseless
    # data it is ~0, so sv[0]/sv[-1] measures nothing but float epsilon.  What
    # governs how well that direction is resolved is the gap to the NEXT
    # smallest, so the meaningful ratio is sv[0]/sv[-2].
    return float(sv[0] / sv[-2])
