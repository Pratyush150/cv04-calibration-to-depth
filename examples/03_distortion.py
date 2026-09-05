"""03 - Lens distortion: the forward polynomial, the iterative inverse, and undistortion.

Run:  python3 examples/03_distortion.py

What it shows
  * the radial and tangential terms, and what each does to a straight line
  * that the model runs IDEAL -> DISTORTED and has no closed-form inverse
  * a checkerboard rendered from known coefficients, then undistorted, with the
    residual bow measured in pixels before and after
"""

import cv2
import numpy as np

from _common import rule, save, plt          # noqa: E402
from geo import distortion as dm             # noqa: E402
from geo import pinhole as ph                # noqa: E402
from geo import synthetic as syn             # noqa: E402

K = ph.intrinsic_matrix(800.0, 802.0, 325.0, 238.0)
D = dm.coefficients(k1=-0.30, k2=0.10, p1=0.0012, p2=-0.0009)
SIZE = (640, 480)

rule("The coefficient vector, in OpenCV's order")
print(f"D = [k1, k2, p1, p2, k3] = {D}")
print("k3 comes LAST, after the tangential pair.  Passing [k1,k2,k3,p1,p2] raises no")
print("error at all - it just bends the image slightly wrong, for an afternoon.")

rule("The hand drill: (0.60, 0.80) with k1=-0.28, k2=0.10")
D_drill = dm.coefficients(k1=-0.28, k2=0.10)
xd, yd = dm.distort_normalized(0.60, 0.80, D_drill)
print("r^2 = 0.36 + 0.64 = 1.00 exactly, so radial = 1 - 0.28 + 0.10 = 0.82")
print(f"(0.60, 0.80) -> ({xd:.4f}, {yd:.4f})   = 0.82 x the original radius")
print("Pulled INWARD: negative k1 is barrel distortion, which is what wide lenses do.")

rule("The inverse has no closed form, so it is iterated")
rng = np.random.default_rng(0)
th = rng.uniform(0, 2 * np.pi, 500)
rad = np.sqrt(rng.uniform(0, 1, 500))         # uniform over the unit disc
x, y = rad * np.cos(th), rad * np.sin(th)
xdd, ydd = dm.distort_normalized(x, y, D)
for iters in (1, 2, 3, 5, 10, 20):
    xr, yr = dm.undistort_normalized(xdd, ydd, D, iters=iters)
    print(f"  {iters:2d} iteration(s): max residual over the unit disc = "
          f"{np.hypot(xr - x, yr - y).max():.2e}")
print("Repeated correction, nothing clever.  Five iterations is already 1e-4.")

px = np.array([[[0.492 * 500 + 320, 0.656 * 500 + 240]]], np.float32)
K_drill = ph.intrinsic_matrix(500.0, 500.0, 320.0, 240.0)
cv_norm = cv2.undistortPoints(px, K_drill, D_drill).ravel()
mine = dm.undistort_normalized(np.array(0.492), np.array(0.656), D_drill, iters=20)
print(f"\nOpenCV undistortPoints: ({cv_norm[0]:.6f}, {cv_norm[1]:.6f})")
print(f"this module, 20 iters : ({float(mine[0]):.6f}, {float(mine[1]):.6f})")
print("truth                 : (0.600000, 0.800000)")
print("OpenCV is the one slightly off - it stops after a fixed iteration count.  Its")
print("undistortion is an approximation with a tolerance, not an exact operation.")
cv_px = cv2.undistortPoints(px, K_drill, D_drill, P=K_drill).ravel()
print(f"\nundistortPoints(..., P=K) : {cv_px}   <- PIXELS")
print(f"undistortPoints(...)      : {cv_norm}   <- NORMALISED")
print("Same function, same input, one keyword apart, three orders of magnitude between")
print("the answers.  Decide which one you want rather than discovering it from a plot.")

rule("The forward model matches cv2.projectPoints")
objp = syn.board_object_points((9, 6), 0.025)
rvec = np.array([0.16, -0.12, 0.04])
R, _ = cv2.Rodrigues(rvec)
# Aim the board CENTRE at the optical axis and bring it close enough to fill the
# frame: the radial terms only do anything where r is large, so a board sitting
# in the middle third of the image demonstrates almost nothing.
tvec = np.array([0.0, 0.0, 0.42]) - R @ objp.mean(axis=0)
mine_px = dm.project_distorted(K, D, R, tvec, objp)
cv_px2, _ = cv2.projectPoints(objp, rvec, tvec, K, D)
print(f"max |mine - cv2.projectPoints| = {np.abs(mine_px - cv_px2.reshape(-1, 2)).max():.2e} px")
print("Same model, written out rather than called - which is what makes the agreement")
print("evidence rather than a coincidence.")

rule("Render a distorted board, then undistort it and measure the bow")
img = syn.render_checkerboard(K, D, rvec, tvec, SIZE, seed=3)
ok, corners = syn.detect_corners(img, (9, 6))
assert ok, "corner detection failed on the rendered board"
und_img = dm.undistort_image(img, K, D)
ok2, corners_u = syn.detect_corners(und_img, (9, 6))
assert ok2, "corner detection failed on the undistorted board"


def max_row_bow(pts, cols=9, rows=6):
    """Largest deviation of any board row from the straight line through its ends.

    A straight line of corners in the world must be a straight line of pixels
    in a pinhole image.  How far they bow IS the distortion, measured in the
    only unit that matters here.
    """
    p = pts.reshape(rows, cols, 2)
    worst = 0.0
    for r in range(rows):
        a, b = p[r, 0], p[r, -1]
        d = b - a
        n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        worst = max(worst, float(np.abs((p[r] - a) @ n).max()))
    return worst


bow_before = max_row_bow(corners)
bow_after = max_row_bow(corners_u)

# The same measurement on a line that spans the WHOLE frame, which is where the
# polynomial actually does something: the board, even filling the middle of the
# image, never reaches the corners.
line_ideal = np.column_stack([np.linspace(2, SIZE[0] - 2, 60), np.full(60, 30.0)])
xy = dm.normalize_pixels(K, line_ideal)
xdl, ydl = dm.distort_normalized(xy[:, 0], xy[:, 1], D)
line_seen = dm.denormalize_points(K, np.column_stack([xdl, ydl]))
chord = line_seen[-1] - line_seen[0]
nrm = np.array([-chord[1], chord[0]]) / np.linalg.norm(chord)
frame_bow = float(np.abs((line_seen - line_seen[0]) @ nrm).max())
print(f"a straight line across the top of the frame bows by {frame_bow:6.2f} px")
print(f"max bow of a board row, distorted   : {bow_before:6.2f} px")
print(f"max bow of a board row, undistorted : {bow_after:6.2f} px")
print(f"reduction: {bow_before / max(bow_after, 1e-9):.0f}x")

# ---------------------------------------------------------------- the figure
fig = plt.figure(figsize=(13.5, 8.0))
gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0], hspace=0.28, wspace=0.22)

ax = fig.add_subplot(gs[0, 0])
ax.imshow(img, cmap="gray", vmin=0, vmax=255)
p = corners.reshape(6, 9, 2)
for r in range(6):
    ax.plot(p[r, :, 0], p[r, :, 1], ".", color="#d62728", ms=3)
    ax.plot([p[r, 0, 0], p[r, -1, 0]], [p[r, 0, 1], p[r, -1, 1]], "-",
            color="#1f77b4", lw=0.9)
ax.set_title(f"(a) rendered with k1={D[0]}, k2={D[1]}\nrows bow off the chord by "
             f"{bow_before:.2f} px")
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

ax = fig.add_subplot(gs[0, 1])
ax.imshow(und_img, cmap="gray", vmin=0, vmax=255)
pu = corners_u.reshape(6, 9, 2)
for r in range(6):
    ax.plot(pu[r, :, 0], pu[r, :, 1], ".", color="#d62728", ms=3)
    ax.plot([pu[r, 0, 0], pu[r, -1, 0]], [pu[r, 0, 1], pu[r, -1, 1]], "-",
            color="#1f77b4", lw=0.9)
ax.set_title(f"(b) undistorted with the same coefficients\nbow now {bow_after:.2f} px")
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

ax = fig.add_subplot(gs[0, 2])
step = 40
gu, gv = np.meshgrid(np.arange(20, SIZE[0], step), np.arange(20, SIZE[1], step))
grid_px = np.column_stack([gu.ravel(), gv.ravel()])
ideal = dm.undistort_pixels(K, D, grid_px)
disp = grid_px - ideal
mag = np.hypot(disp[:, 0], disp[:, 1])
q = ax.quiver(ideal[:, 0], ideal[:, 1], disp[:, 0], disp[:, 1], mag,
              angles="xy", scale_units="xy", scale=1, cmap="viridis", width=0.004)
fig.colorbar(q, ax=ax, label="displacement (px)")
ax.plot([K[0, 2]], [K[1, 2]], "r+", ms=10)
ax.set_xlim(0, SIZE[0]); ax.set_ylim(SIZE[1], 0)
ax.set_title(f"(c) where the lens moves each pixel\nmax {mag.max():.1f} px at the corners")
ax.set_xlabel("u (px)"); ax.set_ylabel("v (px)")

ax = fig.add_subplot(gs[1, 0])
rr = np.linspace(0.02, 1.0, 200)   # from 0.02: the ratio at r = 0 is 0/0
for k1, colour, label in ((-0.28, "#1f77b4", "k1 = -0.28 (barrel, wide lens)"),
                          (0.0, "#666666", "k1 = 0 (no radial term)"),
                          (+0.28, "#d62728", "k1 = +0.28 (pincushion, telephoto)")):
    dd = dm.coefficients(k1=k1, k2=0.10)
    xr, _ = dm.distort_normalized(rr, np.zeros_like(rr), dd)
    ax.plot(rr, xr / rr, color=colour, lw=2, label=label)
ax.axhline(1.0, color="#999999", lw=0.8, ls="--")
ax.set_xlabel("ideal radius r (normalised)"); ax.set_ylabel("radial factor  r_distorted / r")
ax.set_title("(d) the sign of k1 is the whole story"); ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, 1])
iters = np.arange(1, 13)
res = [np.hypot(*(np.array(dm.undistort_normalized(xdd, ydd, D, iters=int(i))) -
                  np.array([x, y]))).max() for i in iters]
ax.semilogy(iters, res, "o-", color="#1f77b4")
ax.set_xlabel("fixed-point iterations"); ax.set_ylabel("max residual (normalised units)")
ax.set_title("(e) the inverse converges by repeated correction")

ax = fig.add_subplot(gs[1, 2])
theta = np.linspace(0, 89.5, 300)
ax.plot(theta, np.tan(np.deg2rad(theta)), color="#d62728", lw=2)
ax.axvline(60, color="#1f77b4", ls="--", lw=1.2)
ax.annotate("60 deg half-angle =\n120 deg full FOV:\nthe model's limit", (32, 12), fontsize=8,
            color="#1f77b4")
ax.set_ylim(0, 40)
ax.set_xlabel("incoming ray angle theta (deg)"); ax.set_ylabel("r / f  =  tan(theta)")
ax.set_title("(f) why fisheye needs a different model")

fig.suptitle("Lens distortion: rendered from known coefficients, then recovered", y=0.97)
save(fig, "03_distortion.png")
