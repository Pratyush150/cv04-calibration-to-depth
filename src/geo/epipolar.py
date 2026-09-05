"""Two-view geometry: the epipolar constraint, E, F, and the eight-point algorithm.

The result this module exists to exploit: given a point in one image, its match
in the other image lies on a LINE that can be computed before you look.  A 3-D
point you cannot locate along its ray still projects to somewhere on the image
of that ray in the other view, and the image of a line is a line.  Matching
therefore drops from a 2-D search to a 1-D one, which is what makes real-time
stereo possible at all.

Written down, "x' lies on the line F x" is one dot product:

        x'^T F x = 0

F is the FUNDAMENTAL matrix - raw pixels, no calibration needed, 7 degrees of
freedom.  E is the ESSENTIAL matrix - normalised (calibrated) coordinates, 5
degrees of freedom, and it factors into the relative rotation and translation
between the two cameras.  E = K'^T F K, and F = K'^-T E K^-1.

Both are rank 2.  That is not trivia: rank 2 is exactly what forces every
epipolar line in an image to pass through a single point, the epipole.  A
full-rank F describes a geometry that does not exist, and its lines miss their
points by tens of pixels.
"""

from __future__ import annotations

import numpy as np

from .homography import normalizing_transform, apply_transform


def skew(t: np.ndarray) -> np.ndarray:
    """[t]_x, the matrix that turns a cross product into a matrix multiply.

    [t]_x @ v == np.cross(t, v).  It is singular by construction (t is in its
    null space), which is where E's rank deficiency comes from.
    """
    t = np.asarray(t, dtype=np.float64).ravel()
    return np.array([[0.0, -t[2], t[1]],
                     [t[2], 0.0, -t[0]],
                     [-t[1], t[0], 0.0]])


def essential_from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """E = [t]_x R for the relative pose (R, t) taking camera 1 to camera 2.

    Not a black box: the epipolar constraint says the two rays and the baseline
    are coplanar, the triple product of three coplanar vectors is zero, and
    writing that triple product as a matrix is exactly [t]_x R.
    """
    return skew(t) @ np.asarray(R, dtype=np.float64)


def fundamental_from_KRt(K1: np.ndarray, K2: np.ndarray,
                         R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """F = K2^-T [t]_x R K1^-1, normalised so its largest entry is 1.

    The scale normalisation matters for comparison only: F is homogeneous, so
    F and 17*F describe the same geometry, and any comparison between two
    estimates has to divide the free scale out first.
    """
    E = essential_from_Rt(R, t)
    F = np.linalg.inv(np.asarray(K2, dtype=np.float64)).T @ E @ \
        np.linalg.inv(np.asarray(K1, dtype=np.float64))
    return normalize_matrix(F)


def normalize_matrix(F: np.ndarray) -> np.ndarray:
    """Divide out the arbitrary scale of a homogeneous matrix, sign included.

    Fixing the sign as well as the magnitude (by the sign of the largest-
    magnitude entry) is what lets two estimates be subtracted elementwise; a
    comparison that ignores the sign reports a 2x error on a perfect match.
    """
    F = np.asarray(F, dtype=np.float64)
    idx = np.unravel_index(np.argmax(np.abs(F)), F.shape)
    return F / F[idx]


def enforce_rank2(F: np.ndarray) -> np.ndarray:
    """Project F onto the closest rank-2 matrix by zeroing its smallest singular value.

    The eight-point solve knows nothing about rank: it returns the least-squares
    null vector of a 9-column system, and under noise that vector reshapes into
    a full-rank matrix.  The consequence is visible rather than abstract - the
    epipolar lines it generates do not all meet at a point, so points miss
    their lines by tens of pixels.  Zeroing sigma_3 is the Frobenius-closest
    rank-2 matrix, and it is the second half of the algorithm, not a polish
    step.
    """
    U, S, Vt = np.linalg.svd(np.asarray(F, dtype=np.float64))
    S[2] = 0.0
    return U @ np.diag(S) @ Vt


def eight_point(pts1: np.ndarray, pts2: np.ndarray, *, normalize: bool = True) -> np.ndarray:
    """Estimate F from >= 8 correspondences.  x2^T F x1 = 0.

    Each correspondence gives one linear equation in the nine entries of F,
    because x2^T F x1 = 0 is linear in F:

        [u2*u1, u2*v1, u2, v2*u1, v2*v1, v2, u1, v1, 1] @ vec(F) = 0

    Eight of those pin F down up to scale; more make it overdetermined, and the
    answer is again the smallest right-singular vector.  Then rank-2 projection.

    `normalize=False` exists to be run and lost with.  Without Hartley
    normalisation the design matrix mixes entries of order u2*u1 ~ 1e5 with
    entries equal to 1, so the smallest singular value is dominated by rounding
    rather than by geometry.  Measured in examples/07 at 0.5 px of correspondence
    noise: 4x worse than the normalised solve on a 480x320 pair, and 2869x worse
    once the same points are indexed from an origin 8000 px away, which is what
    an ROI inside a large sensor does.  cv2.findFundamentalMat normalises
    internally and never mentions it.
    """
    pts1 = np.atleast_2d(np.asarray(pts1, dtype=np.float64))
    pts2 = np.atleast_2d(np.asarray(pts2, dtype=np.float64))
    if pts1.shape[0] < 8 or pts1.shape != pts2.shape:
        raise ValueError("need at least 8 matched points, same count in both sets")

    if normalize:
        T1 = normalizing_transform(pts1)
        T2 = normalizing_transform(pts2)
        p1 = apply_transform(T1, pts1)
        p2 = apply_transform(T2, pts2)
    else:
        T1 = T2 = np.eye(3)
        p1, p2 = pts1, pts2

    u1, v1 = p1[:, 0], p1[:, 1]
    u2, v2 = p2[:, 0], p2[:, 1]
    A = np.column_stack([u2 * u1, u2 * v1, u2,
                         v2 * u1, v2 * v1, v2,
                         u1, v1, np.ones_like(u1)])
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)
    F = enforce_rank2(F)

    # Denormalise.  The rank-2 projection has to happen in the NORMALISED
    # frame, before this line: rank is preserved by the sandwich, but the
    # closest rank-2 matrix in the badly-scaled frame is not the closest one in
    # the well-scaled frame, which is the subtlety that makes the ordering of
    # these two lines matter.
    F = T2.T @ F @ T1
    return normalize_matrix(F)


def epipolar_lines(F: np.ndarray, pts: np.ndarray, *, which: int = 2) -> np.ndarray:
    """Lines (a, b, c) with a*u + b*v + c = 0, one per input point.

    which=2 gives the lines in image 2 for points in image 1 (l' = F x), which
    is the usual direction; which=1 gives lines in image 1 for points in image
    2 (l = F^T x').  Getting the transpose backwards produces lines that look
    entirely plausible and are simply wrong, so the direction is an explicit
    argument rather than something the caller has to remember.
    """
    pts = np.atleast_2d(np.asarray(pts, dtype=np.float64))
    h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    M = np.asarray(F, dtype=np.float64) if which == 2 else np.asarray(F, dtype=np.float64).T
    return h @ M.T


def symmetric_epipolar_distance(F: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Per-correspondence distance in PIXELS from each point to its epipolar line.

    The raw residual x2^T F x1 is an algebraic quantity whose size depends on
    how F happens to be scaled, so it cannot be compared between two estimates
    or judged against a threshold.  Dividing by the line's gradient turns it
    into a point-to-line distance in pixels, and taking it in both images
    (hence "symmetric") stops an estimate from hiding its error in one view.
    """
    pts1 = np.atleast_2d(np.asarray(pts1, dtype=np.float64))
    pts2 = np.atleast_2d(np.asarray(pts2, dtype=np.float64))
    x1 = np.hstack([pts1, np.ones((pts1.shape[0], 1))])
    x2 = np.hstack([pts2, np.ones((pts2.shape[0], 1))])
    F = np.asarray(F, dtype=np.float64)
    l2 = x1 @ F.T                       # lines in image 2
    l1 = x2 @ F                         # lines in image 1
    num = np.abs(np.sum(x2 * l2, axis=1))
    d2 = num / np.maximum(np.hypot(l2[:, 0], l2[:, 1]), 1e-12)
    d1 = num / np.maximum(np.hypot(l1[:, 0], l1[:, 1]), 1e-12)
    return 0.5 * (d1 + d2)


def epipole(F: np.ndarray, *, which: int = 1) -> np.ndarray:
    """The epipole: the image of the other camera's centre, in pixels.

    It is the null vector of F (image 1) or of F^T (image 2), which is another
    way of saying every epipolar line passes through it.  Returned
    inhomogeneously; a rig with parallel optical axes has its epipole at
    infinity and the third component goes to zero, which is exactly the
    condition rectification manufactures.
    """
    F = np.asarray(F, dtype=np.float64)
    _, _, Vt = np.linalg.svd(F if which == 1 else F.T)
    e = Vt[-1]
    return e[:2] / e[2] if abs(e[2]) > 1e-12 else e


def decompose_essential(E: np.ndarray):
    """The FOUR candidate poses hiding inside one essential matrix.

    E = U S V^T, and with W = [[0,-1,0],[1,0,0],[0,0,1]] the decomposition
    yields two rotations R1 = U W V^T and R2 = U W^T V^T, and a translation
    direction t = +/- u3.  Two rotations times two signs is four candidates,
    and ALL FOUR satisfy the epipolar constraint exactly - the constraint is a
    statement about lines, and a line does not care which side of the camera
    its point is on.

    Geometrically: negating t flips which camera is in front; R2 is the
    "twisted pair", R1 rotated 180 degrees about the baseline.  Exactly one
    candidate puts the scene in front of both cameras, and finding it is the
    cheirality check (`select_pose_by_cheirality`).  cv2.recoverPose runs that
    check inside one call, which is precisely why so few people know the four
    solutions exist.

    Also note what does NOT come out: |t| = 1, always.  E fixes the DIRECTION
    of the baseline and says nothing about its length, so every reconstruction
    from E is correct only up to one unknown positive scale.  For a stereo rig
    with a measured baseline that costs nothing; for a moving single camera it
    is the defining problem of monocular odometry.
    """
    U, _, Vt = np.linalg.svd(np.asarray(E, dtype=np.float64))
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1.0            # keep both factors rotations, not reflections
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1.0
    W = np.array([[0.0, -1.0, 0.0],
                  [1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0]])
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]
    return R1, R2, t


def select_pose_by_cheirality(E: np.ndarray, K: np.ndarray,
                              pts1: np.ndarray, pts2: np.ndarray):
    """Pick the one physically real pose out of the four, by positive depth.

    Triangulate a handful of correspondences under each candidate and count how
    many land in front of BOTH cameras.  A camera cannot photograph what is
    behind it, so any candidate that puts points behind either camera is not a
    worse fit - it is impossible.  Returns (R, t, votes) with votes as the
    per-candidate inlier counts, so the caller can see the margin rather than
    trust the winner.
    """
    from .depth import triangulate_dlt
    from .pinhole import projection_matrix

    R1, R2, t = decompose_essential(E)
    P1 = projection_matrix(K, np.eye(3), np.zeros(3))
    best, votes = None, []
    for Rc, tc in ((R1, t), (R1, -t), (R2, t), (R2, -t)):
        P2 = projection_matrix(K, Rc, tc)
        X = triangulate_dlt(P1, P2, pts1, pts2)
        z1 = X[:, 2]
        z2 = (X @ Rc.T + tc)[:, 2]
        n = int(np.count_nonzero((z1 > 0) & (z2 > 0)))
        votes.append(n)
        if best is None or n > best[0]:
            best = (n, Rc, tc)
    return best[1], best[2], votes
