"""07 - Two-view geometry: the epipolar constraint, F by hand, and E's four poses.

Run:  python3 examples/07_eight_point_epipolar.py

What it shows
  * what an epipolar line physically is, drawn on a rendered stereo pair
  * the eight-point algorithm implemented from scratch, checked against a
    fundamental matrix built from the known R and t
  * what Hartley normalisation is worth here: a factor of ~40, not a rounding
  * the four candidate poses inside E, and the cheirality test that picks one
"""

import cv2
import numpy as np

from _common import rule, save, plt          # noqa: E402
from geo import epipolar as ep               # noqa: E402
from geo import pinhole as ph                # noqa: E402
from geo import synthetic as syn             # noqa: E402

SIZE = (480, 320)
K = ph.intrinsic_matrix(520.0, 520.0, (SIZE[0] - 1) / 2, (SIZE[1] - 1) / 2)

# Camera 1 at the origin looking down +Z; camera 2 displaced and genuinely
# ROTATED, so the epipolar lines are not horizontal and there is something for
# rectification (example 09) to undo.
R1, C1 = np.eye(3), np.zeros(3)
R2, _ = cv2.Rodrigues(np.array([0.035, -0.10, 0.02]))
C2 = np.array([0.55, 0.03, -0.10])
t2 = ph.extrinsics_from_centre(R2, C2)

rule("The rig")
print(f"camera 2 sits at C = {C2} m in camera 1's frame")
print(f"and is rotated by {np.degrees(np.linalg.norm(cv2.Rodrigues(R2)[0])):.2f} degrees")
print(f"extrinsic t = -R C = {np.round(t2, 4)}   |t| = {np.linalg.norm(t2):.4f} m")

img1, depth1, sid1 = syn.render_view(K, R1, C1, SIZE, seed=11)
img2, depth2, sid2 = syn.render_view(K, R2, C2, SIZE, seed=12)

rule("Correspondences straight from the scene's ground truth")
rng = np.random.default_rng(4)
us = rng.integers(20, SIZE[0] - 20, 900)
vs = rng.integers(20, SIZE[1] - 20, 900)
Z = depth1[vs, us]
rays = np.column_stack([(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones(len(us))])
X = rays * Z[:, None]                                   # 3-D points in camera 1's frame
p1 = np.column_stack([us, vs]).astype(np.float64)
p2 = ph.project(K, R2, t2, X)
keep = (np.isfinite(p2).all(axis=1) & (p2[:, 0] > 2) & (p2[:, 0] < SIZE[0] - 3) &
        (p2[:, 1] > 2) & (p2[:, 1] < SIZE[1] - 3) & np.isfinite(Z))
# Drop points that camera 2 cannot actually see: the ones the near slab hides.
# Without this filter the "correspondences" include pairs that do not exist,
# which is what a real matcher's outliers are and what RANSAC exists to remove.
uu = np.rint(p2[:, 0]).astype(int).clip(0, SIZE[0] - 1)
vv = np.rint(p2[:, 1]).astype(int).clip(0, SIZE[1] - 1)
keep &= sid2[vv, uu] == sid1[vs, us]
p1, p2, X = p1[keep], p2[keep], X[keep]
print(f"{len(p1)} correspondences, taken from the renderer's depth map so they are")
print("exact by construction - the geometry is being tested here, not a matcher.")

rule("The fundamental matrix, built from what we know")
F_true = ep.fundamental_from_KRt(K, K, R2, t2)
print(np.array2string(F_true, precision=6))
print(f"rank = {np.linalg.matrix_rank(F_true, tol=1e-9)}   "
      f"det = {np.linalg.det(F_true):.2e}")
print("Rank 2 is not trivia: it is what forces every epipolar line in an image to")
print("pass through one point, the epipole.  A full-rank F describes a geometry that")
print("does not exist.")
e1 = ep.epipole(F_true, which=1)
e2 = ep.epipole(F_true, which=2)
print(f"epipole in image 1: ({e1[0]:9.1f}, {e1[1]:8.1f}) px")
print(f"epipole in image 2: ({e2[0]:9.1f}, {e2[1]:8.1f}) px")
print("Both are far outside a 480x320 frame, because the cameras are nearly")
print("side-by-side: the image of one camera's centre in the other is way off to the")
print("side, and the epipolar lines are therefore nearly parallel inside the frame.")
print(f"residual x2^T F x1 over the exact correspondences: "
      f"max {np.abs(np.sum(np.hstack([p2, np.ones((len(p2),1))]) * (np.hstack([p1, np.ones((len(p1),1))]) @ F_true.T), axis=1)).max():.2e}")

rule("The eight-point algorithm, with and without normalisation")
print(f"{'noise':>7s} | {'normalised':>11s} {'raw pixels':>11s} {'cv2 8POINT':>11s} "
      f"{'raw/norm':>9s}")
rows = []
for sigma in (0.0, 0.1, 0.25, 0.5, 1.0):
    en, er, ec = [], [], []
    trial_rng = np.random.default_rng(7)
    for _ in range(20):
        q1 = p1 + trial_rng.normal(0, sigma, p1.shape)
        q2 = p2 + trial_rng.normal(0, sigma, p2.shape)
        idx = trial_rng.choice(len(q1), 60, replace=False)
        Fn = ep.eight_point(q1[idx], q2[idx], normalize=True)
        Fr = ep.eight_point(q1[idx], q2[idx], normalize=False)
        Fc, _ = cv2.findFundamentalMat(q1[idx], q2[idx], cv2.FM_8POINT)
        # Scored on the CLEAN correspondences: this is the estimate's error, not
        # an echo of the noise that was just added.
        en.append(ep.symmetric_epipolar_distance(Fn, p1, p2).mean())
        er.append(ep.symmetric_epipolar_distance(Fr, p1, p2).mean())
        ec.append(ep.symmetric_epipolar_distance(ep.normalize_matrix(Fc), p1, p2).mean())
    rows.append((sigma, np.mean(en), np.mean(er), np.mean(ec)))
    print(f"{sigma:7.2f} | {np.mean(en):11.4f} {np.mean(er):11.4f} {np.mean(ec):11.4f} "
          f"{np.mean(er)/max(np.mean(en),1e-12):9.1f}x")
print("Units are pixels: mean distance from a point to the epipolar line it should")
print("lie on, measured in both images.  With no noise both solves are exact, which")
print("is the trap - normalisation looks unnecessary until real data arrives.  The")
print("design matrix here holds products of TWO pixel coordinates (u2*u1 ~ 1e5) beside")
print("entries equal to 1, so under noise the raw solve's smallest singular vector is")
print("decided by rounding rather than by geometry.")
best = rows[3]
print(f"\nAt {best[0]:.2f} px of noise: normalised {best[1]:.4f} px, "
      f"raw {best[2]:.4f} px, cv2 {best[3]:.4f} px")
print("cv2.findFundamentalMat normalises internally and never mentions it, which is")
print("why our normalised result and theirs land on top of each other.")
print("\nWhat actually decides how bad the raw solve is: how far the points\' centroid")
print("sits from the coordinate origin, relative to their spread.  Add a constant")
print("offset to every coordinate - exactly what happens when your points live in one")
print("corner of a 4K frame, or when an ROI is indexed from the full sensor - and:")
print(f"{'offset (px)':>12s} {'normalised':>11s} {'raw pixels':>12s} {'raw/norm':>10s}")
for off in (0, 500, 2000, 8000):
    en, er = [], []
    srng = np.random.default_rng(7)
    shift = np.array([off, off], dtype=float)
    for _ in range(20):
        a1 = p1 + srng.normal(0, 0.5, p1.shape) + shift
        a2 = p2 + srng.normal(0, 0.5, p2.shape) + shift
        sub = srng.choice(len(a1), 60, replace=False)
        Fn = ep.eight_point(a1[sub], a2[sub], normalize=True)
        Fr = ep.eight_point(a1[sub], a2[sub], normalize=False)
        en.append(ep.symmetric_epipolar_distance(Fn, p1 + shift, p2 + shift).mean())
        er.append(ep.symmetric_epipolar_distance(Fr, p1 + shift, p2 + shift).mean())
    print(f"{off:12d} {np.mean(en):11.4f} {np.mean(er):12.4f} "
          f"{np.mean(er)/max(np.mean(en), 1e-12):9.0f}x")
print("The normalised column does not move by a digit - it does not care what units or")
print("origin you hand it, which is the entire point.  The raw column goes from 4x")
print("worse to nearly 3000x worse for a change that is not geometry at all, only")
print("bookkeeping about where pixel (0, 0) is.")

rule("From F to E, and the four poses hiding inside it")
E = ep.essential_from_Rt(R2, t2 / np.linalg.norm(t2))
R_a, R_b, t_dir = ep.decompose_essential(E)
tw = R_b @ R_a.T
ang = np.degrees(np.arccos(np.clip((np.trace(tw) - 1) / 2, -1, 1)))
axis = cv2.Rodrigues(tw)[0].ravel()
axis = axis / np.linalg.norm(axis)
print(f"the two rotations differ by a {ang:.3f} degree rotation about "
      f"{np.round(axis, 3)}")
print(f"the baseline direction is {np.round(t2/np.linalg.norm(t2), 3)}; "
      f"|axis . t_hat| = {abs(axis @ (t2/np.linalg.norm(t2))):.6f}")
print("That is the twisted pair: R2 is R1 rotated 180 degrees about the baseline.")
print("Four candidates = {R1, twisted} x {+t, -t}, and all four satisfy the epipolar")
print("constraint exactly, because the constraint is about LINES and a line does not")
print("care which side of the camera its point is on.\n")
R_rec, t_rec, votes = ep.select_pose_by_cheirality(E, K, p1[:40], p2[:40])
print(f"cheirality votes (points in front of BOTH cameras) per candidate: {votes} of 40")
print(f"rotation error of the winner : "
      f"{np.degrees(np.arccos(np.clip((np.trace(R_rec.T @ R2) - 1) / 2, -1, 1))):.6f} deg")
print(f"translation direction error  : "
      f"{np.degrees(np.arccos(np.clip(abs(t_rec @ (t2/np.linalg.norm(t2))), -1, 1))):.6f} deg")
print(f"|t| recovered = {np.linalg.norm(t_rec):.4f}   |t| true = {np.linalg.norm(t2):.4f}")
print(f"missing scale factor = {np.linalg.norm(t2):.4f}")
print("E fixes the DIRECTION of the baseline and nothing about its length.  For a")
print("stereo rig with a measured baseline that costs nothing; for a moving single")
print("camera every frame pair gets its own arbitrary unit, and that is the problem")
print("monocular odometry exists to solve.")

# ---------------------------------------------------------------- the figure
sel = np.linspace(0, len(p1) - 1, 12).astype(int)
noise_rng = np.random.default_rng(3)
q1 = p1 + noise_rng.normal(0, 0.5, p1.shape)
q2 = p2 + noise_rng.normal(0, 0.5, p2.shape)
idx = noise_rng.choice(len(q1), 60, replace=False)
F_norm = ep.eight_point(q1[idx], q2[idx], normalize=True)
# For the picture, show the raw solve in the situation that actually breaks it:
# the same correspondences indexed from an origin 2000 px away, as an ROI inside
# a larger sensor would be.  The estimate is mapped back into image coordinates
# with x_off = T x, so F = T^T F_off T, and the lines are then directly
# comparable with the normalised ones.
T_off = np.array([[1.0, 0.0, 2000.0], [0.0, 1.0, 2000.0], [0.0, 0.0, 1.0]])
F_raw_off = ep.eight_point(q1[idx] + 2000.0, q2[idx] + 2000.0, normalize=False)
F_raw = ep.normalize_matrix(T_off.T @ F_raw_off @ T_off)

fig, axes = plt.subplots(1, 4, figsize=(17.5, 3.9))
colours = plt.cm.tab20(np.linspace(0, 1, len(sel)))


def draw_lines(ax, image, lines, pts, title):
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    w = image.shape[1]
    for (a, b, c), pt, col in zip(lines, pts, colours):
        if abs(b) < 1e-9:
            continue
        ax.plot([0, w], [-c / b, -(c + a * w) / b], "-", color=col, lw=1.0)
        ax.plot([pt[0]], [pt[1]], "o", color=col, ms=5, mec="k", mew=0.5)
    ax.set_xlim(0, image.shape[1]); ax.set_ylim(image.shape[0], 0)
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)


ax = axes[0]
ax.imshow(img1, cmap="gray", vmin=0, vmax=255)
for pt, col in zip(p1[sel], colours):
    ax.plot([pt[0]], [pt[1]], "o", color=col, ms=5, mec="k", mew=0.5)
ax.set_title("(a) image 1: 12 points")
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

d_norm = ep.symmetric_epipolar_distance(F_norm, p1, p2).mean()
d_raw = ep.symmetric_epipolar_distance(F_raw, p1, p2).mean()
draw_lines(axes[1], img2, ep.epipolar_lines(F_norm, p1[sel]), p2[sel],
           f"(b) image 2: lines from the normalised\n8-point F ({d_norm:.3f} px mean error)")
draw_lines(axes[2], img2, ep.epipolar_lines(F_raw, p1[sel]), p2[sel],
           f"(c) same points, raw solve with the ROI\nindexed from +2000 px "
           f"({d_raw:.1f} px mean error)")

ax = axes[3]
r = np.array(rows)[1:]      # drop the noiseless row: both solves are exact there
ax.plot(r[:, 0], r[:, 1], "o-", color="#1f77b4", label="normalised (this repo)")
ax.plot(r[:, 0], r[:, 2], "s-", color="#d62728", label="raw pixels")
ax.plot(r[:, 0], r[:, 3], "^--", color="#2ca02c", label="cv2.findFundamentalMat")
ax.set_xlabel("correspondence noise sigma (px)")
ax.set_ylabel("mean symmetric epipolar distance (px)")
ax.set_title("(d) the cost of skipping normalisation")
ax.legend(fontsize=8)

fig.suptitle("Epipolar geometry: every match lies on a line you can compute in advance",
             y=1.03)
save(fig, "08_epipolar_lines.png")
