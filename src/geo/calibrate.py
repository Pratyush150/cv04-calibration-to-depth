"""Calibration: many views of a known board, one K and one D that explain them all.

The idea is a least-squares fit and nothing more exotic.  You know where the
board's corners are in the board's own frame.  You measure where they landed in
each image.  You ask for the single set of intrinsics and distortion
coefficients, plus one pose per view, that best explains every landing at once,
and "best" means smallest sum of squared pixel residuals - the REPROJECTION
ERROR.

The part every tutorial omits, and the reason this module carries diagnostics
rather than just a wrapper:

    A low RMS reprojection error does not mean a correct calibration.

RMS says how well the model fits the data you HAPPENED to collect.  It says
nothing about whether that data contained enough information to pin the
parameters down.  The specific failure is a confound between focal length and
distance: u = fx * X/Z, so a board twice as far seen with twice the focal
length produces the same image.  If every view is fronto-parallel, nothing in
the data separates those hypotheses and the solver slides freely along that
direction while fitting your corners to a hundredth of a pixel.

examples/04 measures exactly that on synthetic data where the truth is known.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CalibrationResult:
    rms: float
    K: np.ndarray
    D: np.ndarray
    rvecs: list
    tvecs: list
    image_size: tuple
    n_views: int


def calibrate(object_points, image_points, image_size, *, flags: int = 0) -> CalibrationResult:
    """Thin wrapper over cv2.calibrateCamera that keeps the outputs together.

    `image_size` is (width, height) - the opposite order to the shape of the
    array the images came out of, which is the one-line mistake that produces a
    calibration whose principal point is nowhere near the middle of the frame.
    """
    obj = [np.asarray(o, dtype=np.float32).reshape(-1, 1, 3) for o in object_points]
    img = [np.asarray(p, dtype=np.float32).reshape(-1, 1, 2) for p in image_points]
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(obj, img, image_size, None, None,
                                                  flags=flags)
    return CalibrationResult(float(rms), K, D.ravel(), list(rvecs), list(tvecs),
                             tuple(image_size), len(obj))


def per_view_errors(res: CalibrationResult, object_points, image_points) -> np.ndarray:
    """RMS reprojection error per view, in pixels.

    The aggregate RMS hides a single bad view inside twenty good ones.  A view
    at three times the median is usually a bad detection or a blurred frame,
    and dropping it and refitting is a legitimate move; dropping views until
    the number looks nice is not, and the difference is whether you can say why
    the view was bad.
    """
    out = []
    for i in range(res.n_views):
        proj, _ = cv2.projectPoints(np.asarray(object_points[i], np.float32),
                                    res.rvecs[i], res.tvecs[i], res.K, res.D)
        d = proj.reshape(-1, 2) - np.asarray(image_points[i], np.float64).reshape(-1, 2)
        out.append(float(np.sqrt((d ** 2).sum(axis=1).mean())))
    return np.array(out)


def per_point_residuals(res: CalibrationResult, object_points, image_points):
    """Every residual vector, in pixels, as (N, 2), plus the pixel it belongs to.

    Plotted as a scatter this is worth more than the RMS it summarises.  A
    healthy calibration gives an isotropic blob centred on zero.  Structure -
    a ring, a bias, residuals growing toward the frame corners - means the
    model is missing something, and WHICH structure tells you what: radial
    growth is an unfitted distortion term, a constant offset is a principal
    point that never moved off its initial guess.
    """
    residuals, pixels = [], []
    for i in range(res.n_views):
        proj, _ = cv2.projectPoints(np.asarray(object_points[i], np.float32),
                                    res.rvecs[i], res.tvecs[i], res.K, res.D)
        obs = np.asarray(image_points[i], np.float64).reshape(-1, 2)
        residuals.append(proj.reshape(-1, 2) - obs)
        pixels.append(obs)
    return np.vstack(residuals), np.vstack(pixels)


def board_tilt_degrees(rvecs) -> np.ndarray:
    """The angle between each board's normal and the optical axis, in degrees.

    The board's own normal is (0, 0, 1) in the board frame, so in the camera
    frame it is the third column of R, and the tilt is the angle between that
    and the camera's own z axis - i.e. arccos|R[2,2]|.

    This is the diagnostic that makes the fronto-parallel confound detectable
    from your own photographs instead of only in a synthetic experiment.  A
    median tilt below ~15 degrees means f and Z are confounded and no number of
    extra frames will help: the fix is to pick the board up and angle it.
    """
    tilts = []
    for rv in rvecs:
        R, _ = cv2.Rodrigues(np.asarray(rv, dtype=np.float64).reshape(3, 1))
        tilts.append(np.degrees(np.arccos(min(1.0, abs(R[2, 2])))))
    return np.array(tilts)


def corner_coverage(image_points, image_size, grid: int = 10) -> float:
    """Fraction of the outer-20% border cells that contain NO detected corner.

    The radial coefficients are estimated almost entirely from corners near the
    frame edge, because that is where r is large and where the polynomial does
    anything at all.  A calibration set that is a fat blob in the middle of the
    frame leaves k1 and k2 unconstrained no matter how many views it contains,
    and adding more centred views does not fix it.
    """
    w, h = image_size
    occupied = np.zeros((grid, grid), dtype=bool)
    for pts in image_points:
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        gx = np.clip((p[:, 0] / w * grid).astype(int), 0, grid - 1)
        gy = np.clip((p[:, 1] / h * grid).astype(int), 0, grid - 1)
        occupied[gy, gx] = True
    border = np.ones((grid, grid), dtype=bool)
    border[grid // 5:grid - grid // 5, grid // 5:grid - grid // 5] = False
    return float(np.count_nonzero(border & ~occupied) / np.count_nonzero(border))


def intrinsics_error(K_true: np.ndarray, D_true: np.ndarray,
                     K_est: np.ndarray, D_est: np.ndarray) -> dict:
    """Recovered-versus-truth, the comparison a real calibration cannot make.

    This is the whole argument for synthetic data in a teaching repo: on real
    footage the only available number is the reprojection error, and the
    section above explains why that number can be small while fx is 30% wrong.
    Here the truth is a variable in the script, so the claim "the calibration
    worked" becomes a measurement.
    """
    K_true = np.asarray(K_true, dtype=np.float64)
    K_est = np.asarray(K_est, dtype=np.float64)
    D_true = np.asarray(D_true, dtype=np.float64).ravel()
    D_est = np.asarray(D_est, dtype=np.float64).ravel()
    n = min(len(D_true), len(D_est))
    keys = ["fx", "fy", "cx", "cy"]
    idx = [(0, 0), (1, 1), (0, 2), (1, 2)]
    out = {}
    for k, (i, j) in zip(keys, idx):
        out[k] = {"true": float(K_true[i, j]), "est": float(K_est[i, j]),
                  "abs": float(K_est[i, j] - K_true[i, j]),
                  "rel_pct": float(100.0 * (K_est[i, j] - K_true[i, j]) / K_true[i, j])}
    out["D"] = {"true": D_true[:n].tolist(), "est": D_est[:n].tolist(),
                "abs": (D_est[:n] - D_true[:n]).tolist()}
    out["max_focal_rel_pct"] = max(abs(out["fx"]["rel_pct"]), abs(out["fy"]["rel_pct"]))
    return out


def simulate_capture(K_true, D_true, poses, pattern, square, image_size,
                     *, corner_noise: float = 0.05, seed: int = 0):
    """Project the board corners for a set of poses, with detector-like noise.

    Used for the multi-seed sweep in examples/04, where rendering and detecting
    300 images would take minutes and teaches nothing extra: the confound being
    measured lives in the geometry of the poses, not in the pixels.

    The noise comes from its own generator, deliberately.  Draw it from the
    same stream as the pose sampling and the two capture protocols diverge on
    the first view - they consume different numbers of random draws - so the
    two runs no longer see the same noise and the experiment is no longer
    controlled.  Same truth, same noise sample, only the capture geometry
    differs: that is what makes the comparison mean something.
    """
    from .synthetic import board_object_points
    from .distortion import project_distorted

    objp = board_object_points(pattern, square)
    noise = np.random.default_rng(seed)
    object_points, image_points = [], []
    for rvec, tvec in poses:
        R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
        px = project_distorted(K_true, D_true, R, tvec, objp)
        px = px + noise.normal(0.0, corner_noise, px.shape)
        object_points.append(objp)
        image_points.append(px)
    return object_points, image_points
