"""The pinhole camera: similar triangles, homogeneous coordinates, K and [R|t].

Everything downstream in this package assumes the conventions fixed here, so
they are stated once, in one place:

  * Camera frame: origin at the centre of projection, +Z out along the optical
    axis, +X right, +Y DOWN.  Y down is not an aesthetic choice - it makes the
    image row index and the camera Y axis point the same way, so no sign flip
    is needed anywhere between projection and array indexing.
  * The image plane is the VIRTUAL one at +f, in front of the pinhole, which is
    what gives the unsigned equation u = fx * X/Z + cx.  The physical sensor
    sits at -f and sees the point-reflected image; `project_real_plane` below
    exists purely to make that difference visible and is never used in the
    pipeline.
  * Extrinsics map WORLD to CAMERA: X_cam = R @ X_world + t.  So `t` is the
    world origin expressed in the camera frame, NOT the camera's position.  The
    camera position is C = -R.T @ t (`camera_centre`).
"""

from __future__ import annotations

import numpy as np


def intrinsic_matrix(fx: float, fy: float, cx: float, cy: float, skew: float = 0.0) -> np.ndarray:
    """Assemble K from the four numbers that a calibration actually reports.

    skew defaults to 0 and stays 0 for every camera you are likely to meet: it
    models the sensor's pixel rows being non-perpendicular to its columns, a
    manufacturing defect that photolithography has not produced in decades.
    The parameter exists so the matrix layout is explicit rather than implied,
    and so a reader can see *where* skew would live if it were ever non-zero.
    """
    return np.array([[float(fx), float(skew), float(cx)],
                     [0.0, float(fy), float(cy)],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def matvec3(M: np.ndarray, v: np.ndarray) -> np.ndarray:
    """A 3x3 matrix-vector product written out as three dot products.

    `M @ v` does exactly this and is what the rest of the package calls.  This
    version exists so the operation is written down once in a form that can be
    read rather than trusted, and so a test can assert the two agree.  Row i of
    the output is row i of M dotted with v - there is nothing else in there.
    """
    out = np.zeros(3, dtype=np.float64)
    for i in range(3):
        out[i] = M[i, 0] * v[0] + M[i, 1] * v[1] + M[i, 2] * v[2]
    return out


def to_homogeneous(pts: np.ndarray) -> np.ndarray:
    """(N, k) -> (N, k+1) by appending a column of ones.

    The appended 1 is what turns "rotate then add t" into a single matrix
    multiply, which is what lets a whole camera chain collapse into one matrix.
    """
    pts = np.atleast_2d(np.asarray(pts, dtype=np.float64))
    return np.hstack([pts, np.ones((pts.shape[0], 1))])


def from_homogeneous(pts: np.ndarray) -> np.ndarray:
    """(N, k+1) -> (N, k) by dividing through by the last coordinate.

    This divide is not tidying up after the multiply: it IS perspective.  The
    last coordinate comes out as the depth Z, and dividing by depth is the
    reason distant things look small.  It is also why homogeneous vectors are
    only defined up to scale - (720, 320, 2) and (360, 160, 1) are the same
    point, and printing the un-divided triple as if it were a pixel is the most
    common first-day bug in this subject.
    """
    pts = np.atleast_2d(np.asarray(pts, dtype=np.float64))
    w = pts[:, -1:]
    return pts[:, :-1] / w


def project_camera_points(K: np.ndarray, X_cam: np.ndarray, *, return_depth: bool = False):
    """Project points already in CAMERA coordinates to pixels.

    Points with Z <= 0 are returned as NaN rather than as numbers.  They have
    to be: u = su/s with a negative s still produces a perfectly plausible
    pixel somewhere inside the image, so a point behind the camera silently
    becomes a phantom detection in front of it.  Rejecting on the sign of the
    third homogeneous coordinate is the cheirality test, and it is the same
    test that picks the one real pose out of the four an essential matrix
    offers (see epipolar.py).
    """
    X_cam = np.atleast_2d(np.asarray(X_cam, dtype=np.float64))
    h = X_cam @ K.T                      # (N, 3) homogeneous pixels (su, sv, s)
    s = h[:, 2]
    valid = s > 0
    uv = np.full((X_cam.shape[0], 2), np.nan)
    uv[valid] = h[valid, :2] / s[valid, None]
    if return_depth:
        return uv, X_cam[:, 2]
    return uv


def world_to_camera(R: np.ndarray, t: np.ndarray, X_world: np.ndarray) -> np.ndarray:
    """X_cam = R @ X_world + t, vectorised over N points."""
    X_world = np.atleast_2d(np.asarray(X_world, dtype=np.float64))
    return X_world @ np.asarray(R, dtype=np.float64).T + np.asarray(t, dtype=np.float64).ravel()


def projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """P = K [R|t], the 3x4 matrix that takes a homogeneous world point to a
    homogeneous pixel.  Kept separate from `project` because P is the object
    that triangulation and the DLT both consume."""
    Rt = np.hstack([np.asarray(R, dtype=np.float64),
                    np.asarray(t, dtype=np.float64).reshape(3, 1)])
    return np.asarray(K, dtype=np.float64) @ Rt


def project(K: np.ndarray, R: np.ndarray, t: np.ndarray, X_world: np.ndarray,
            *, return_depth: bool = False):
    """World point -> pixel, the full chain, with no lens distortion.

    Distortion belongs BETWEEN the divide by Z and the multiply by K, so it
    cannot be bolted on here without lying about where it acts; use
    distortion.project_distorted when you want the lens in the model.
    """
    X_cam = world_to_camera(R, t, X_world)
    return project_camera_points(K, X_cam, return_depth=return_depth)


def project_real_plane(K: np.ndarray, X_cam: np.ndarray) -> np.ndarray:
    """Projection onto the PHYSICAL image plane at -f, behind the pinhole.

    Nothing in the pipeline calls this.  It exists so the sign convention can
    be demonstrated instead of asserted: derive projection from a drawing of a
    real pinhole box and similar triangles give u = -fx * X/Z + cx.  The result
    is the point-reflection of the textbook answer through the principal point,
    which is the 180-degree rotation about the optical axis that makes a real
    camera's image upside down.  Every source silently moves the image plane to
    +f to make the minus signs go away.
    """
    X_cam = np.atleast_2d(np.asarray(X_cam, dtype=np.float64))
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z = X_cam[:, 2]
    return np.column_stack([-fx * X_cam[:, 0] / z + cx,
                            -fy * X_cam[:, 1] / z + cy])


def camera_centre(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """C = -R.T @ t - the camera's position in WORLD coordinates.

    Plotting `t` as a trajectory is the classic silent bug: `t` is the world
    origin seen from the camera, so a plot of it comes out inside-out, and
    nothing in the maths complains.
    """
    R = np.asarray(R, dtype=np.float64)
    return -R.T @ np.asarray(t, dtype=np.float64).ravel()


def extrinsics_from_centre(R: np.ndarray, C: np.ndarray) -> np.ndarray:
    """The inverse of `camera_centre`: given where the camera IS, get t.

    t = -R @ C.  Provided because scene setup naturally thinks in camera
    positions while every projection equation wants t, and doing that flip in
    your head each time is how the sign gets lost.
    """
    R = np.asarray(R, dtype=np.float64)
    return -R @ np.asarray(C, dtype=np.float64).ravel()


def is_rotation(R: np.ndarray, tol: float = 1e-8) -> bool:
    """R.T @ R == I and det(R) == +1.

    Both halves matter.  Drop the determinant check and a reflection
    (det = -1) passes as a rotation, which mirror-images the reconstructed
    scene while every error metric stays happy.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        return False
    orthonormal = np.allclose(R.T @ R, np.eye(3), atol=tol)
    return bool(orthonormal and abs(np.linalg.det(R) - 1.0) < tol)


def orthonormalise(R: np.ndarray) -> np.ndarray:
    """Snap a drifted almost-rotation back onto the rotation manifold.

    Any hand-edited or optimised R accumulates float error until R.T @ R is
    only approximately I, at which point it quietly SHEARS the points it acts
    on - a slow reprojection-error creep with no single bad frame to blame.
    The nearest true rotation in the Frobenius sense is U @ Vt from the SVD.
    """
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64))
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:            # SVD can hand back a reflection; flip
        U[:, -1] *= -1.0                 # the least-significant axis to fix it
        Rn = U @ Vt
    return Rn


def scale_intrinsics(K: np.ndarray, sx: float, sy: float | None = None) -> np.ndarray:
    """K for a RESIZED image.  All four of fx, fy, cx, cy scale; nothing else.

    cx and cy are pixel coordinates like any other, so they scale with the
    pixels.  Scaling the focal lengths and forgetting the principal point is a
    real and expensive bug: the resulting offset is (1-s)*cx pixels, constant
    across the frame, which reads as a calibration bias rather than as the
    four-line code error it is.  The distortion coefficients are NOT touched -
    they act on normalised coordinates (u-cx)/fx, where the scale cancels top
    and bottom.
    """
    sy = sx if sy is None else sy
    S = np.array([[sx, 0.0, 0.0],
                  [0.0, sy, 0.0],
                  [0.0, 0.0, 1.0]])
    return S @ np.asarray(K, dtype=np.float64)


def crop_intrinsics(K: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """K for a CROPPED image: only cx, cy move, by the discarded top-left.

    No pixel changed size, so fx and fy are untouched - the pair with
    `scale_intrinsics` is the whole of the resolution rule.
    """
    T = np.array([[1.0, 0.0, -float(dx)],
                  [0.0, 1.0, -float(dy)],
                  [0.0, 0.0, 1.0]])
    return T @ np.asarray(K, dtype=np.float64)


def fov_degrees(K: np.ndarray, width: int, height: int) -> tuple[float, float]:
    """Horizontal and vertical field of view implied by K, in degrees.

    This is the independent plausibility check on a calibration.  A ray at the
    edge of frame has X/Z = tan(FOV/2) and lands width/2 pixels from the
    principal point, so FOV = 2*atan((width/2)/fx).  If the number it returns
    disagrees with the lens on the box, the calibration is wrong no matter how
    small its reprojection error is.  Note the direction: SMALLER fx means a
    WIDER lens.
    """
    K = np.asarray(K, dtype=np.float64)
    hfov = 2.0 * np.degrees(np.arctan((width / 2.0) / K[0, 0]))
    vfov = 2.0 * np.degrees(np.arctan((height / 2.0) / K[1, 1]))
    return float(hfov), float(vfov)
