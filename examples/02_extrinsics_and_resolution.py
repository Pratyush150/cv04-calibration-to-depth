"""02 - Extrinsics: world to camera, the sign everybody gets wrong, and K under a resize.

Run:  python3 examples/02_extrinsics_and_resolution.py

What it shows
  * t is the world origin in camera coordinates, NOT the camera's position
  * C = -R^T t recovers the position, demonstrated on a real pose
  * what a resize does to K (all four entries) and what a crop does (two), and
    the size of the bug when you scale the focal lengths and forget the
    principal point
"""

import cv2
import numpy as np

from _common import rule, save, plt          # noqa: E402
from geo import pinhole as ph                # noqa: E402

K = ph.intrinsic_matrix(800.0, 800.0, 320.0, 240.0)

rule("t is not where the camera is")
R = np.eye(3)
C = np.array([0.0, 0.0, -5.0])               # camera five metres back along -Z
t = ph.extrinsics_from_centre(R, C)
print(f"camera position C   = {C}")
print(f"extrinsic t = -R C  = {t}     <- opposite sign, and this is the one K needs")
print(f"world origin lands at pixel {ph.project(K, R, t, [[0, 0, 0]])[0]}  (the principal point)")
print(f"(1,0,0) lands at pixel      {ph.project(K, R, t, [[1, 0, 0]])[0]}  (right of centre, correct)")
print(f"recovered C = -R^T t = {ph.camera_centre(R, t)}")
print("Plot t as a trajectory instead of C and the path comes out inside-out, with")
print("nothing in the maths complaining.")

rule("A rotation matrix has two conditions, and both matter")
R2, _ = cv2.Rodrigues(np.array([0.05, 0.20, -0.03]))
print(f"R^T R = I ?  {np.allclose(R2.T @ R2, np.eye(3))}      det(R) = {np.linalg.det(R2):+.6f}")
mirror = R2.copy()
mirror[:, 0] *= -1                            # a reflection, not a rotation
print(f"after mirroring one column: det = {np.linalg.det(mirror):+.6f}, "
      f"is_rotation = {ph.is_rotation(mirror)}")
print("det = -1 is a reflection.  It reconstructs a mirror-imaged scene and every")
print("error metric stays perfectly happy about it.")
drifted = R2 + 1e-3 * np.random.default_rng(0).normal(size=(3, 3))
print(f"a drifted R: max|R^T R - I| = {np.abs(drifted.T @ drifted - np.eye(3)).max():.2e} "
      f"-> after SVD re-orthonormalisation: "
      f"{np.abs(ph.orthonormalise(drifted).T @ ph.orthonormalise(drifted) - np.eye(3)).max():.2e}")

rule("Changing resolution: RESIZE scales all four entries, CROP shifts two")
K_full = ph.intrinsic_matrix(1450.0, 1452.0, 962.0, 541.0)
s = 1.0 / 3.0
K_small = ph.scale_intrinsics(K_full, s)
K_crop = ph.crop_intrinsics(K_full, (1920 - 1280) / 2, (1080 - 720) / 2)
print(f"calibrated at 1920x1080:  fx={K_full[0,0]:7.1f} fy={K_full[1,1]:7.1f} "
      f"cx={K_full[0,2]:6.1f} cy={K_full[1,2]:6.1f}")
print(f"resized  to  640x360   :  fx={K_small[0,0]:7.1f} fy={K_small[1,1]:7.1f} "
      f"cx={K_small[0,2]:6.1f} cy={K_small[1,2]:6.1f}")
print(f"cropped  to 1280x720   :  fx={K_crop[0,0]:7.1f} fy={K_crop[1,1]:7.1f} "
      f"cx={K_crop[0,2]:6.1f} cy={K_crop[1,2]:6.1f}")
print("The distortion coefficients change in NEITHER case, because they act on")
print("normalised coordinates (u-cx)/fx where the scale cancels top and bottom.")

X_w = np.array([[0.30, -0.15, 0.00], [0.10, 0.22, 0.05], [-0.25, 0.05, -0.10]])
t2 = np.array([0.10, -0.05, 4.00])
uv_full = ph.project(K_full, R2, t2, X_w)
uv_small = ph.project(K_small, R2, t2, X_w)
print(f"\nmax |3 * (scaled projection) - (full projection)| = "
      f"{np.abs(3 * uv_small - uv_full).max():.2e} px")

K_bad = K_full.copy()
K_bad[0, 0] *= s
K_bad[1, 1] *= s                              # focal lengths scaled, cx/cy forgotten
uv_bad = ph.project(K_bad, R2, t2, X_w)
offset = uv_bad - uv_small
print(f"scaling fx,fy but not cx,cy offsets every point by "
      f"({offset[0,0]:.1f}, {offset[0,1]:.1f}) px")
print(f"predicted (1-s)*cx = {(1 - s) * K_full[0, 2]:.1f}, "
      f"(1-s)*cy = {(1 - s) * K_full[1, 2]:.1f}")
print("The horizontal error is larger than the entire 640-pixel-wide image, and it is")
print("CONSTANT - which is what makes it look like a calibration bias instead of a")
print("four-line code bug.")

rule("The field-of-view plausibility check")
for fx in (1108.0, 780.0, 1450.0):
    kk = ph.intrinsic_matrix(fx, fx, 640.0, 360.0)
    hfov, _ = ph.fov_degrees(kk, 1280, 720)
    print(f"fx = {fx:7.1f} px on a 1280-wide image implies a {hfov:5.1f} deg horizontal FOV")
print("A spec sheet saying 60 degrees means fx should be about "
      f"{(1280/2)/np.tan(np.deg2rad(60)/2):.0f} px.  Smaller fx = WIDER lens.")

# ---------------------------------------------------------------- the figure
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

# (a) A top-down view of the scene, the camera, and the two vectors.
ax = axes[0]
pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.6, 2.0], [-0.8, 1.4]])
ax.plot(pts[:, 0], pts[:, 1], "o", color="#1f77b4", ms=7, label="world points (X, Z)")
ax.plot([C[0]], [C[2]], "^", color="#d62728", ms=13, label="camera centre C")
ax.annotate("", xy=(C[0], C[2] + 1.6), xytext=(C[0], C[2]),
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=2))
ax.annotate("optical axis", (C[0] + 0.12, C[2] + 1.0), fontsize=8, color="#d62728")
ax.annotate("", xy=(0, 0), xytext=(C[0], C[2]),
            arrowprops=dict(arrowstyle="->", color="#666666", ls="--", lw=1.2))
ax.annotate("C = -R$^T$t = (0, 0, -5)", (C[0] + 0.1, C[2] - 0.35), fontsize=9, color="#d62728")
ax.annotate("t = (0, 0, +5): the world origin\nas the camera sees it",
            (0.15, 0.15), fontsize=9, color="#444444")
ax.set_xlim(-2.2, 2.2); ax.set_ylim(-6, 3)
ax.set_xlabel("world X (m)"); ax.set_ylabel("world Z (m)")
ax.set_title("(a) t and C point opposite ways"); ax.legend(fontsize=8, loc="upper left")

# (b) The same scene from two poses - intrinsics fixed, extrinsics changed.
ax = axes[1]
grid = np.array([[x, y, z] for x in (-0.4, 0, 0.4) for y in (-0.3, 0, 0.3)
                 for z in (0.0, 0.4)])
for rvec, colour, label in (([0.0, 0.0, 0.0], "#1f77b4", "pose A: no rotation"),
                            ([0.15, 0.45, 0.05], "#d62728", "pose B: rotated and moved")):
    Rp, _ = cv2.Rodrigues(np.array(rvec))
    tp = ph.extrinsics_from_centre(Rp, np.array([0.0, 0.0, -2.5]) if rvec[1] == 0
                                   else np.array([0.7, -0.2, -2.2]))
    uv = ph.project(K, Rp, tp, grid)
    ax.plot(uv[:, 0], uv[:, 1], "o", ms=5, color=colour, label=label)
ax.set_xlim(0, 640); ax.set_ylim(480, 0)
ax.set_xlabel("u (px)"); ax.set_ylabel("v (px)")
ax.set_title("(b) same K, different [R|t]"); ax.legend(fontsize=8, loc="upper left")

# (c) The resolution bug, drawn to scale.
ax = axes[2]
ax.add_patch(plt.Rectangle((0, 0), 640, 360, fill=False, ec="#444444", lw=1.5))
ax.plot(uv_small[:, 0], uv_small[:, 1], "o", color="#1f77b4", ms=8,
        label="correct K' (all four scaled)")
ax.plot(uv_bad[:, 0], uv_bad[:, 1], "X", color="#d62728", ms=9,
        label="fx,fy scaled, cx,cy forgotten")
for a, b in zip(uv_small, uv_bad):
    ax.annotate("", xy=(b[0], b[1]), xytext=(a[0], a[1]),
                arrowprops=dict(arrowstyle="->", color="#999999", lw=1))
ax.set_xlim(-100, 1400); ax.set_ylim(800, -100)
ax.set_xlabel("u (px)"); ax.set_ylabel("v (px)")
ax.set_title(f"(c) the offset is ({offset[0,0]:.0f}, {offset[0,1]:.0f}) px - "
             "off the frame entirely")
ax.legend(fontsize=8, loc="lower right")

fig.suptitle("Extrinsics and the resolution rule", y=1.02)
save(fig, "02_extrinsics_and_resolution.png")
