"""06 - The homography, estimated by hand: DLT, normalisation, and the library check.

Run:  python3 examples/06_homography_dlt.py

What it shows
  * the plane-to-image map that a calibration board gives you for free
  * a DLT written from scratch, agreeing with cv2.findHomography on clean data
  * what Hartley normalisation is actually worth, in condition number and in
    pixels of transfer error under noise
  * the map used for something: warping a slanted board back to fronto-parallel
"""

import cv2
import numpy as np

from _common import rule, save, plt          # noqa: E402
from geo import distortion as dm             # noqa: E402
from geo import homography as hg             # noqa: E402
from geo import pinhole as ph                # noqa: E402
from geo import synthetic as syn             # noqa: E402

SIZE = (640, 480)
PATTERN = (9, 6)
SQUARE = 0.025
K = ph.intrinsic_matrix(800.0, 802.0, 325.0, 238.0)
D = dm.coefficients(k1=-0.30, k2=0.10, p1=0.0012, p2=-0.0009)

rule("Where the correspondences come from")
print("A calibration board is a PLANE, and a plane imaged by a pinhole camera maps to")
print("the image by a single 3x3 homography.  That is why calibration is possible at")
print("all: each view gives you one H, and K is what every H in the set has in common.")

rvec = np.array([0.42, -0.28, 0.09])
R, _ = cv2.Rodrigues(rvec)
objp = syn.board_object_points(PATTERN, SQUARE)
tvec = np.array([0.0, 0.0, 0.46]) - R @ objp.mean(axis=0)
img = syn.render_checkerboard(K, D, rvec, tvec, SIZE, pattern=PATTERN, square=SQUARE,
                              seed=5)
ok, corners = syn.detect_corners(img, PATTERN)
assert ok
# Work on UNDISTORTED corners: a homography is a pinhole relation, and lens
# distortion is not projective.  Feeding raw distorted corners to a homography
# fit produces a small, structured, and entirely avoidable residual.
corners_ideal = dm.undistort_pixels(K, D, corners)
board_xy = objp[:, :2] * 1000.0                        # board coordinates in mm
# The chessboard detector labels corners only up to a 180-degree rotation of the
# board (nothing in the pattern distinguishes the two), so fix the ordering by
# testing both against the known projection rather than assuming one.
proj = dm.project_distorted(K, D, R, tvec, objp)
if np.abs(corners - proj[::-1]).max() < np.abs(corners - proj).max():
    board_xy = board_xy[::-1]
print(f"detected {len(corners)} inner corners; using board millimetres as the source")
print("plane and undistorted pixels as the destination.")

rule("The DLT, against the library, on clean correspondences")
H_mine = hg.homography_dlt(board_xy, corners_ideal)
H_cv, _ = cv2.findHomography(board_xy, corners_ideal, 0)
H_cv = H_cv / H_cv[2, 2]
print("H (mine):\n", np.array2string(H_mine, precision=6, suppress_small=True))
print(f"max |H_mine - H_cv| (both scaled so H[2,2] = 1) = {np.abs(H_mine - H_cv).max():.2e}")
print(f"transfer error, mine : mean {hg.transfer_error(H_mine, board_xy, corners_ideal).mean():.4f} px, "
      f"max {hg.transfer_error(H_mine, board_xy, corners_ideal).max():.4f} px")
print(f"transfer error, cv2  : mean {hg.transfer_error(H_cv, board_xy, corners_ideal).mean():.4f} px, "
      f"max {hg.transfer_error(H_cv, board_xy, corners_ideal).max():.4f} px")
print("Both fit the same points to the same fraction of a pixel; the residual that is")
print("left is the corner detector, not the estimator.")
print("\nThe 1e-3 gap between the two matrices is not one of them being wrong - it is")
print("that they minimise different things.  The DLT minimises algebraic error ||A h||;")
print("cv2.findHomography runs Levenberg-Marquardt on REPROJECTION error afterwards.")
print("On correspondences with no noise at all there is almost nothing left to")
print("disagree about, and what remains is OpenCV's iteration stopping tolerance:")
exact = ph.project(K, R, tvec, objp)                   # pinhole projection, no lens, no noise
H_exact = hg.homography_dlt(board_xy, exact)
H_cv_exact, _ = cv2.findHomography(board_xy, exact, 0)
print(f"  exact correspondences: max |H_dlt - H_cv2| = "
      f"{np.abs(H_exact - H_cv_exact / H_cv_exact[2, 2]).max():.2e}")
print(f"  exact correspondences: DLT transfer error  = "
      f"{hg.transfer_error(H_exact, board_xy, exact).max():.2e} px  "
      f"(cv2: {hg.transfer_error(H_cv_exact / H_cv_exact[2, 2], board_xy, exact).max():.2e} px)")

rule("What normalisation buys, in one number")
cond_norm = hg.condition_number(board_xy, corners_ideal, normalize=True)
cond_raw = hg.condition_number(board_xy, corners_ideal, normalize=False)
print(f"condition number of the DLT design matrix, normalised   : {cond_norm:10.2f}")
print(f"condition number of the DLT design matrix, raw units    : {cond_raw:10.2e}")
print(f"ratio: {cond_raw / cond_norm:.3e}")
print("The raw matrix mixes entries of order u*u (about 1e5 here) with entries equal")
print("to 1.  Its smallest singular value is then decided by float rounding rather")
print("than by geometry, and that vector is the answer you were about to use.")

rule("The same estimate under noise, with and without normalisation")
rng = np.random.default_rng(0)
print(f"{'noise (px)':>11s} {'normalised':>12s} {'raw':>12s} {'cv2':>10s}")
noise_rows = []
for sigma in (0.0, 0.25, 0.5, 1.0, 2.0):
    errs = []
    for trial in range(30):
        noisy = corners_ideal + rng.normal(0.0, sigma, corners_ideal.shape)
        Hn = hg.homography_dlt(board_xy, noisy, normalize=True)
        Hr = hg.homography_dlt(board_xy, noisy, normalize=False)
        Hc, _ = cv2.findHomography(board_xy, noisy, 0)
        errs.append([hg.transfer_error(Hn, board_xy, corners_ideal).mean(),
                     hg.transfer_error(Hr, board_xy, corners_ideal).mean(),
                     hg.transfer_error(Hc / Hc[2, 2], board_xy, corners_ideal).mean()])
    m = np.mean(errs, axis=0)
    noise_rows.append((sigma, *m))
    print(f"{sigma:11.2f} {m[0]:12.4f} {m[1]:12.4f} {m[2]:10.4f}")
print("Error is measured against the CLEAN corners, so it is the estimate's error and")
print("not the noise being echoed back.  On 640x480 pixel coordinates the raw solve")
print("survives - the conditioning is bad but not fatal at this image size - and it is")
print("the eight-point algorithm in example 07, whose design matrix carries products")
print("of TWO pixel coordinates, where the same neglect costs a factor of 4 on a small")
print("frame and nearly 3000 once the points sit far from the coordinate origin.")

rule("Using the map: warp the slanted board back to fronto-parallel")
H_board = hg.homography_dlt(corners_ideal, board_xy)    # image -> board millimetres
scale = 2.2                                             # px per mm in the output
S = np.array([[scale, 0, 60.0], [0, scale, 40.0], [0, 0, 1.0]])
und = dm.undistort_image(img, K, D)
warped = cv2.warpPerspective(und, S @ H_board, (560, 400), borderValue=105)
warped_corners = hg.apply_transform(S @ H_board, corners_ideal)
grid = warped_corners.reshape(PATTERN[1], PATTERN[0], 2)
spacing = np.linalg.norm(np.diff(grid, axis=1), axis=2)
print(f"square spacing in the warped image: mean {spacing.mean():.2f} px, "
      f"std {spacing.std():.3f} px")
print(f"expected {SQUARE*1000*scale:.2f} px from the {SQUARE*1000:.0f} mm squares and the "
      f"{scale} px/mm output scale")
print("A perspective view has square spacing that shrinks with distance; after the")
print("warp it is constant to a few hundredths of a pixel.  That constancy IS the")
print("check that the homography is right.")

# ---------------------------------------------------------------- the figure
fig, axes = plt.subplots(1, 4, figsize=(17, 4.3))

ax = axes[0]
ax.imshow(img, cmap="gray", vmin=0, vmax=255)
ax.plot(corners[:, 0], corners[:, 1], ".", color="#d62728", ms=3)
quad = corners.reshape(PATTERN[1], PATTERN[0], 2)[[0, 0, -1, -1], [0, -1, -1, 0]]
ax.plot(np.append(quad[:, 0], quad[0, 0]), np.append(quad[:, 1], quad[0, 1]),
        "-", color="#1f77b4", lw=1.5)
ax.set_title("(a) a plane, seen at an angle")
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

ax = axes[1]
ax.imshow(warped, cmap="gray", vmin=0, vmax=255)
ax.plot(warped_corners[:, 0], warped_corners[:, 1], ".", color="#d62728", ms=3)
ax.set_title(f"(b) warped by the hand-rolled H\nspacing {spacing.mean():.2f} "
             f"+/- {spacing.std():.3f} px")
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

ax = axes[2]
rows = np.array(noise_rows)
ax.plot(rows[:, 0], rows[:, 1], "o-", color="#1f77b4", label="DLT, normalised")
ax.plot(rows[:, 0], rows[:, 2], "s-", color="#d62728", label="DLT, raw pixels")
ax.plot(rows[:, 0], rows[:, 3], "^--", color="#2ca02c", label="cv2.findHomography")
ax.set_xlabel("corner noise sigma (px)")
ax.set_ylabel("mean transfer error vs clean corners (px)")
ax.set_title("(c) estimator error under noise")
ax.legend(fontsize=8)

ax = axes[3]
ax.bar(["normalised", "raw pixels"], [cond_norm, cond_raw],
       color=["#1f77b4", "#d62728"])
ax.set_yscale("log")
ax.set_ylabel("condition number of the design matrix")
ax.set_title(f"(d) conditioning: {cond_raw/cond_norm:.0e}x apart")
for i, v in enumerate((cond_norm, cond_raw)):
    ax.annotate(f"{v:.3g}", (i, v * 1.3), ha="center", fontsize=9)

fig.suptitle("The homography by DLT: 4 correspondences, 8 rows, one null vector", y=1.02)
save(fig, "07_homography_dlt.png")
