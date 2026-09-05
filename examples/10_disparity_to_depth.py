"""10 - Disparity to metres, and why far points are numerically poisonous.

Run:  python3 examples/10_disparity_to_depth.py

What it shows
  * Z = f*B/d derived, and the doffs term that the clean derivation drops
  * the law that governs every stereo rig ever built: depth error grows with
    the SQUARE of depth, quantified for a stated f and B and then MEASURED off
    the disparity map from example 08
  * the Q matrix, multiplied out by hand and against cv2
  * triangulation: the two rays do not meet, and the uncertainty is an
    elongated ellipsoid pointing down the viewing ray, never a sphere
"""

import cv2
import numpy as np

from _common import rule, save, plt          # noqa: E402
from geo import depth as dp                  # noqa: E402
from geo import pinhole as ph                # noqa: E402
from geo import stereo as st                 # noqa: E402
from geo import synthetic as syn             # noqa: E402

F_PX = 800.0
BASELINE = 0.12
SUBPIX_ERR = 0.25          # what a decent sub-pixel matcher achieves in practice

rule(f"Z = f*B/d, for f = {F_PX:.0f} px and B = {BASELINE*100:.0f} cm")
print(f"f*B = {F_PX*BASELINE:.1f} metre-pixels, so Z = {F_PX*BASELINE:.1f} / d")
print(f"{'d (px)':>8s} {'Z (m)':>9s} {'Z at d-1':>10s} {'Z at d+1':>10s} {'spread':>9s}")
for d in (60, 30, 12, 6, 4, 2):
    z = dp.depth_from_disparity(d, F_PX, BASELINE)
    near, far = dp.depth_error_exact(z, F_PX, BASELINE, 1.0)
    print(f"{d:8d} {z:9.3f} {far:10.3f} {near:10.3f} {far-near:9.3f}")
print("One pixel of matching error, the same one pixel at every row.  At d = 30 it is")
print("21 cm; at d = 4 it is 12.8 m.  Same rig, same error, sixty times the")
print("consequence - and the interval is ASYMMETRIC, because Z = f*B/d is a hyperbola")
print("and not a line.  Quoting a symmetric +/- at range understates the far side.")

rule("Where that comes from: differentiate")
print("dZ/dd = -f*B/d^2, and substituting d = f*B/Z gives")
print("    |dZ| = (Z^2 / (f*B)) * |dd|")
print("Depth error grows with the SQUARE of depth.  Check it against the table above:")
for z_a, z_b in ((3.2, 24.0),):
    ratio_measured = ((dp.depth_error_exact(z_b, F_PX, BASELINE, 1.0)[1] -
                       dp.depth_error_exact(z_b, F_PX, BASELINE, 1.0)[0]) /
                      (dp.depth_error_exact(z_a, F_PX, BASELINE, 1.0)[1] -
                       dp.depth_error_exact(z_a, F_PX, BASELINE, 1.0)[0]))
    print(f"  spread at {z_b:.1f} m / spread at {z_a:.1f} m = {ratio_measured:.1f}")
    print(f"  predicted by the square law: ({z_b:.1f}/{z_a:.1f})^2 = {(z_b/z_a)**2:.1f}")
print("The small gap is because a one-pixel step is not infinitesimal, so the")
print("derivative is only an approximation of it.  The three levers are all in the")
print("formula: a longer baseline B, a longer focal length f, and better sub-pixel")
print("matching dd.  There is nothing else, and no amount of code is a fourth lever.")

rule(f"Design numbers: what range is this rig good for, at {SUBPIX_ERR} px of matching error?")
print(f"{'B (cm)':>7s} {'10% error at':>14s} {'5% error at':>13s} "
      f"{'error at 10 m':>15s} {'error at 30 m':>15s}")
for B in (0.06, 0.12, 0.25, 0.50):
    print(f"{B*100:7.0f} {dp.max_useful_range(F_PX, B, SUBPIX_ERR, 0.10):13.1f} m "
          f"{dp.max_useful_range(F_PX, B, SUBPIX_ERR, 0.05):12.1f} m "
          f"{dp.depth_error(10.0, F_PX, B, SUBPIX_ERR):14.2f} m "
          f"{dp.depth_error(30.0, F_PX, B, SUBPIX_ERR):14.2f} m")
print("Baseline buys precision linearly and costs you the near field: a wide baseline")
print("sees the scene from two very different angles, so more of it is occluded in one")
print("view, and close objects fall outside the disparity search entirely.  There is no")
print("setting that wins both, which is why 'how far can it see' is not a question you")
print("can answer without an accuracy attached.")

rule("The doffs term the clean derivation drops")
print("Z = f*B/d assumes both cameras share a principal point, so the two cx terms")
print("cancel in x_L - x_R.  Real rigs do not.  What survives is a constant:")
print("    doffs = cx_right - cx_left,  and  Z = f*B/(d + doffs)")
f_mb, b_mb, doffs = 3979.911, 0.193001, 124.343       # Middlebury Motorcycle rig
print(f"\nOn a real published rig (f = {f_mb:.1f} px, B = {b_mb*1000:.1f} mm, "
      f"doffs = {doffs:.3f} px):")
print(f"{'d (px)':>8s} {'correct Z':>11s} {'ignoring doffs':>15s} {'error':>10s}")
for d in (200.0, 60.0, 28.16):
    good = dp.depth_from_disparity(d, f_mb, b_mb, doffs)
    bad = dp.depth_from_disparity(d, f_mb, b_mb, 0.0)
    print(f"{d:8.2f} {good:10.3f} m {bad:14.3f} m {100*(bad-good)/good:9.0f}%")
print("62%, then 207%, then 441%.  The error GROWS with distance, because doffs is a")
print("constant added to a shrinking denominator - so a scene sanity-checked on one")
print("nearby object passes, and the back of the room reads as 27 m when it is 5 m.")
print("After a stereoRectify with CALIB_ZERO_DISPARITY it is zero by construction,")
print("which is one of the real reasons that flag exists.")

rule("The Q matrix, multiplied out")
Q = dp.Q_matrix(463.7446, 0.12, 320.2591, 240.2001)
print(Q)
u, v, d = 400, 300, 40.0
grid_disp = np.full((480, 640), d, dtype=np.float64)
pts = dp.reproject_disparity(grid_disp, Q)
X, Y, Z = pts[v, u]
print(f"\n(u, v, d, 1) = ({u}, {v}, {d:.0f}, 1) ->")
print(f"  X' = {u} - {-Q[0,3]:.4f} = {u + Q[0,3]:.4f}")
print(f"  Y' = {v} - {-Q[1,3]:.4f} = {v + Q[1,3]:.4f}")
print(f"  Z' = {Q[2,3]:.4f}")
print(f"  W' = {Q[3,2]:.4f} * {d:.0f} = {Q[3,2]*d:.4f}")
print(f"  divide: X = {X:.4f} m, Y = {Y:.4f} m, Z = {Z:.4f} m")
print(f"cross-check straight from the depth formula: "
      f"Z = {Q[2,3]:.4f} * 0.12 / {d:.0f} = "
      f"{dp.depth_from_disparity(d, Q[2,3], 0.12):.4f} m")
cv_pts = cv2.reprojectImageTo3D(grid_disp.astype(np.float32), Q.astype(np.float32))
print(f"max |this module - cv2.reprojectImageTo3D| = "
      f"{np.abs(pts - cv_pts).max():.2e} m  (float32 on their side)")
print("Q is Z = f*B/d generalised to recover X and Y too: the principal point folded")
print("into the first two rows, f in the third, and -1/Tx in the bottom.")

rule("Triangulation: the rays do not meet")
K = ph.intrinsic_matrix(800.0, 800.0, 320.0, 240.0)
R_t, _ = cv2.Rodrigues(np.array([0.02, 0.20, 0.01]))
t_t = np.array([-1.0, 0.0, 0.0])
P1 = ph.projection_matrix(K, np.eye(3), np.zeros(3))
P2 = ph.projection_matrix(K, R_t, t_t)
X_true = np.array([[0.30, -0.10, 5.00]])
x1 = ph.project(K, np.eye(3), np.zeros(3), X_true)
x2 = ph.project(K, R_t, t_t, X_true)
print(f"clean observations: gap between the two rays = "
      f"{dp.ray_gap(P1, P2, x1[0], x2[0]):.2e} m")
rng = np.random.default_rng(1)
n1 = x1 + rng.normal(0, 0.5, x1.shape)
n2 = x2 + rng.normal(0, 0.5, x2.shape)
print(f"with 0.5 px of noise on each: gap = {dp.ray_gap(P1, P2, n1[0], n2[0])*1000:.2f} mm")
print("Two lines in 3-D have four degrees of freedom of relative position, and")
print("intersecting is one equation on those four - a measure-zero condition.  Perfect")
print("measurements make them meet by construction; ANY noise, including nothing worse")
print("than reporting an integer pixel, destroys it.  So triangulation cannot be an")
print("intersection.  It has to minimise something, and the DLT minimises ||A X||.")
X_dlt = dp.triangulate_dlt(P1, P2, n1, n2)
X_cv = cv2.triangulatePoints(P1, P2, n1.T, n2.T)
X_cv = (X_cv[:3] / X_cv[3]).T
print(f"\ntriangulated (this module): {np.round(X_dlt[0], 5)}")
print(f"triangulated (cv2)        : {np.round(X_cv[0], 5)}")
print(f"agreement                 : {np.abs(X_dlt - X_cv).max():.2e} m")
err_mm = (X_dlt - X_true)[0] * 1000
print(f"error vs truth (mm)       : X {err_mm[0]:+.2f}, Y {err_mm[1]:+.2f}, "
      f"Z {err_mm[2]:+.2f}")

trials = 400
noise_rng = np.random.default_rng(5)
errs = []
for _ in range(trials):
    a = x1 + noise_rng.normal(0, 0.5, x1.shape)
    b = x2 + noise_rng.normal(0, 0.5, x2.shape)
    errs.append((dp.triangulate_dlt(P1, P2, a, b) - X_true)[0])
errs = np.array(errs)
print(f"\nover {trials} noise draws, standard deviation of the triangulated point:")
print(f"  sigma_X = {errs[:,0].std()*1000:6.2f} mm")
print(f"  sigma_Y = {errs[:,1].std()*1000:6.2f} mm")
print(f"  sigma_Z = {errs[:,2].std()*1000:6.2f} mm   "
      f"({errs[:,2].std()/errs[:,0].std():.0f}x the lateral spread)")
print("The uncertainty of a stereo point is an elongated ellipsoid pointing down the")
print("viewing ray, never a sphere.  Anyone fusing stereo points with another sensor")
print("under an isotropic covariance is discarding the most important thing they know")
print("about the measurement.")

rule("The law, measured off a real disparity map")
SIZE = (480, 320)
scene = syn.render_stereo_pair(size=SIZE, fx=520.0, baseline=0.12)
disp = st.block_match(scene.left, scene.right, ndisp=48, window=9)
ok = (np.isfinite(disp) & ~scene.unmatchable &
      (scene.region_mask("near_slab") | scene.region_mask("ramp") |
       scene.region_mask("wall")))
Z_est = dp.depth_from_disparity(disp, 520.0, 0.12)
Z_true = scene.depth_left
d_err = np.abs(disp[ok] - scene.disparity_left[ok])
z_err = np.abs(Z_est[ok] - Z_true[ok])
zt = Z_true[ok]
print(f"{'depth band':>16s} {'n':>7s} {'median |dd|':>12s} {'median |dZ|':>12s} "
      f"{'predicted':>11s}")
bands = [(2.5, 3.0), (3.0, 3.5), (3.5, 4.0), (7.4, 7.6)]
for lo, hi in bands:
    m = (zt >= lo) & (zt < hi)
    if m.sum() < 50:
        continue
    dd = np.median(d_err[m])
    pred = dp.depth_error(np.median(zt[m]), 520.0, 0.12, dd)
    print(f"{f'{lo:.1f}-{hi:.1f} m':>16s} {int(m.sum()):7d} {dd:12.4f} "
          f"{np.median(z_err[m]):12.4f} {pred:11.4f}")
print("\nThe error grows faster than Z^2 alone across those bands, and that is not a")
print("contradiction: the disparity error |dd| grows too, from 0.02 px on the near")
print("slab to 0.09 px on the low-contrast back wall.  Multiply the two and the")
print("arithmetic closes.  The square law is what sits ON TOP of whatever your matcher")
print("does, not instead of it.")
print("\nMeasured depth error against the error the square law predicts from the")
print("measured disparity error at that range.  They track, which is the point: the")
print("depth error is not a property of the matcher, it is the matcher's error passed")
print("through a 1/d that steepens as d shrinks.")

# ---------------------------------------------------------------- the figure
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

ax = axes[0]
Zs = np.linspace(1.0, 40.0, 400)
for B, colour in ((0.06, "#d62728"), (0.12, "#1f77b4"), (0.25, "#2ca02c"),
                  (0.50, "#9467bd")):
    ax.plot(Zs, dp.depth_error(Zs, F_PX, B, SUBPIX_ERR), color=colour, lw=2,
            label=f"B = {B*100:.0f} cm")
ax.plot(Zs, 0.10 * Zs, "--", color="#333333", lw=1.2, label="10% of range")
ax.set_xlabel("true depth Z (m)")
ax.set_ylabel(f"depth error for {SUBPIX_ERR} px of disparity error (m)")
ax.set_ylim(0, 6)
ax.set_title(f"(a) error grows as Z$^2$  (f = {F_PX:.0f} px)")
ax.legend(fontsize=8)

ax = axes[1]
ref = np.linspace(2.0, 9.0, 50)
ax.loglog(ref, 0.0025 * (ref / 2.7) ** 2, "--", color="#999999", lw=1.4,
          label="a pure Z$^2$ slope, for reference")
for lo, hi in bands:
    m = (zt >= lo) & (zt < hi)
    if m.sum() < 50:
        continue
    z_mid = np.median(zt[m])
    dd = np.median(d_err[m])
    ax.loglog([z_mid], [np.median(z_err[m])], "o", color="#d62728", ms=9,
              label="measured off the disparity map" if lo == bands[0][0] else None)
    ax.loglog([z_mid], [dp.depth_error(z_mid, 520.0, 0.12, dd)], "x", color="#1f77b4",
              ms=11, mew=2,
              label="Z$^2$ law, fed the measured |dd|" if lo == bands[0][0] else None)
ax.set_xlabel("depth Z (m)"); ax.set_ylabel("median depth error (m)")
ax.set_title("(b) the law, and the measurement (f = 520 px, B = 12 cm)")
ax.legend(fontsize=8, loc="upper left")

ax = axes[2]
Zvis = np.where(ok, Z_est, np.nan)
im = ax.imshow(Zvis, cmap="magma_r", vmin=2.4, vmax=8.0)
ax.set_title("(c) the disparity map, in metres")
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
cb = fig.colorbar(im, ax=ax, fraction=0.046)
cb.set_label("depth Z (m)")

fig.suptitle("Disparity to depth: one pixel of matching error is millimetres near "
             "and metres far", y=1.02)
save(fig, "12_depth_error_vs_distance.png")
