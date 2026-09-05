"""04 - Calibration: recover K and D from rendered views, and check against the truth.

Run:  python3 examples/04_calibrate_from_synthetic_views.py   (~30 s)

What it shows
  * a full calibration on IMAGES - render, detect corners, solve - where the
    ground truth is known, so "it worked" is a measurement and not a claim
  * the diagnostics that catch a bad capture: board tilt, corner coverage, and
    the field-of-view plausibility check
  * the experiment that matters: two capture protocols with identical
    reprojection error, one of which cannot determine the focal length at all
"""

import time

import cv2
import numpy as np

from _common import rule, save, plt          # noqa: E402
from geo import calibrate as cal             # noqa: E402
from geo import distortion as dm             # noqa: E402
from geo import pinhole as ph                # noqa: E402
from geo import synthetic as syn             # noqa: E402

SIZE = (640, 480)
PATTERN = (9, 6)
SQUARE = 0.025
K_TRUE = ph.intrinsic_matrix(800.0, 802.0, 325.0, 238.0)
D_TRUE = dm.coefficients(k1=-0.30, k2=0.10, p1=0.0012, p2=-0.0009)

rule("Ground truth used to render the views")
print(f"fx = {K_TRUE[0,0]:.1f}   fy = {K_TRUE[1,1]:.1f}   "
      f"cx = {K_TRUE[0,2]:.1f}   cy = {K_TRUE[1,2]:.1f}")
print(f"D  = {D_TRUE}")
print(f"board: {PATTERN[0]}x{PATTERN[1]} inner corners, {SQUARE*1000:.0f} mm squares")

rule("Render 20 views, detect corners, keep what detects")
objp = syn.board_object_points(PATTERN, SQUARE)
poses = syn.board_poses(20, tilt_sigma=0.40, spread=True, seed=17,
                        pattern=PATTERN, square=SQUARE)
object_points, image_points, kept_images, kept_idx = [], [], [], []
t0 = time.time()
for i, (rvec, tvec) in enumerate(poses):
    img = syn.render_checkerboard(K_TRUE, D_TRUE, rvec, tvec, SIZE,
                                  pattern=PATTERN, square=SQUARE, seed=i)
    ok, corners = syn.detect_corners(img, PATTERN)
    if ok:
        object_points.append(objp)
        image_points.append(corners)
        kept_images.append(img)
        kept_idx.append(i)
print(f"{len(object_points)} of {len(poses)} views detected in {time.time()-t0:.1f} s")
print("The rejected ones are the poses that pushed the board off the edge of the")
print("frame.  findChessboardCornersSB needs the WHOLE board, which pulls directly")
print("against the advice to put corners near the frame edge - and is exactly the")
print("tension a ChArUco board removes, because every corner there carries an ID and")
print("a clipped view is still usable.")

rule("Calibrate, then compare against the truth")
res = cal.calibrate(object_points, image_points, SIZE)
err = cal.intrinsics_error(K_TRUE, D_TRUE, res.K, res.D)
print(f"RMS reprojection error: {res.rms:.4f} px over {res.n_views} views")
print(f"{'':6s} {'true':>10s} {'recovered':>12s} {'error':>10s} {'rel':>9s}")
for k in ("fx", "fy", "cx", "cy"):
    e = err[k]
    print(f"{k:6s} {e['true']:10.3f} {e['est']:12.3f} {e['abs']:+10.3f} "
          f"{e['rel_pct']:+8.3f}%")
print(f"\nD true      = {np.array2string(D_TRUE, precision=5)}")
print(f"D recovered = {np.array2string(res.D, precision=5)}")
print("Compare the k2 and k3 columns and do not panic.  Those two are strongly")
print("correlated - r^4 and r^6 look alike over the range of r an image spans - so")
print("the solver trades one against the other freely.  The thing that has to match")
print("is the CURVE they describe, not the individual coefficients:")

grid = np.column_stack([np.repeat(np.linspace(0, SIZE[0] - 1, 40), 40),
                        np.tile(np.linspace(0, SIZE[1] - 1, 40), 40)])
u_true = dm.undistort_pixels(K_TRUE, D_TRUE, grid)
u_est = dm.undistort_pixels(res.K, res.D, grid)
model_gap = np.linalg.norm(u_true - u_est, axis=1)
print(f"  max disagreement between the true and recovered undistortion, over the")
print(f"  whole frame: {model_gap.max():.3f} px  (mean {model_gap.mean():.3f} px)")

rule("The three diagnostics that catch a bad capture")
tilts = cal.board_tilt_degrees(res.rvecs)
coverage = cal.corner_coverage(image_points, SIZE)
hfov, vfov = ph.fov_degrees(res.K, *SIZE)
print(f"board tilt      : min {tilts.min():.1f}, median {np.median(tilts):.1f}, "
      f"max {tilts.max():.1f} deg  -> "
      f"{'ok' if np.median(tilts) > 15 else 'RECAPTURE: f and Z are confounded'}")
print(f"border coverage : {coverage*100:.0f}% of outer-20% cells hold no corner  -> "
      f"{'ok' if coverage < 0.30 else 'thin at the edges: k1/k2 weakly constrained'}")
print(f"implied FOV     : {hfov:.1f} deg horizontal, {vfov:.1f} deg vertical")
print(f"                  (truth: {ph.fov_degrees(K_TRUE, *SIZE)[0]:.1f} deg)")
print("Coverage is the honest weak spot of a full-board chessboard capture, and it is")
print("why the recovered k2/k3 wander while k1 does not: the outer ring of the frame")
print("is where those terms live and the board never quite gets there.")

per_view = cal.per_view_errors(res, object_points, image_points)
print(f"\nper-view RMS: min {per_view.min():.3f}, median {np.median(per_view):.3f}, "
      f"max {per_view.max():.3f} px")

rule("Two capture protocols, identical RMS, and only one of them works")
print("Rendering 300 images for this sweep would take minutes and teach nothing extra,")
print("so it runs on projected corners with 0.05 px of noise - the confound lives in")
print("the geometry of the poses, not in the pixels.\n")
print(f"{'seed':>4s} | {'GOOD fx':>9s} {'GOOD RMS':>9s} | {'BAD fx':>9s} {'BAD RMS':>9s}")
good_fx, bad_fx, good_rms, bad_rms = [], [], [], []
for seed in range(10):
    row = []
    for tilt, spread, store_fx, store_rms in ((0.35, True, good_fx, good_rms),
                                              (0.02, False, bad_fx, bad_rms)):
        poses_s = syn.board_poses(15, tilt_sigma=tilt, spread=spread, seed=seed,
                                  pattern=PATTERN, square=SQUARE)
        op, ip = cal.simulate_capture(K_TRUE, D_TRUE, poses_s, PATTERN, SQUARE, SIZE,
                                      corner_noise=0.05, seed=1000 + seed)
        r = cal.calibrate(op, ip, SIZE)
        store_fx.append(r.K[0, 0])
        store_rms.append(r.rms)
        row += [r.K[0, 0], r.rms]
    print(f"{seed:4d} | {row[0]:9.1f} {row[1]:9.3f} | {row[2]:9.1f} {row[3]:9.3f}")
print(f"\nGOOD fx spans {min(good_fx):.1f} to {max(good_fx):.1f}   "
      f"(true {K_TRUE[0,0]:.1f}, a {max(good_fx)-min(good_fx):.1f} px window)")
print(f"BAD  fx spans {min(bad_fx):.1f} to {max(bad_fx):.1f}   "
      f"(a {max(bad_fx)-min(bad_fx):.1f} px window)")
print(f"every RMS in the table is between {min(good_rms+bad_rms):.3f} and "
      f"{max(good_rms+bad_rms):.3f} px")
print("\nNo threshold on RMS separates those two columns.  The bad capture is not")
print("biased, it is UNCONSTRAINED - it lands under the truth on some seeds and well")
print("over it on others, because fronto-parallel views cannot tell a longer lens")
print("from a further board.  Propagate the worst one into a stereo depth with a")
print("12 cm baseline at 30 px of disparity:")
for label, fx in (("true", K_TRUE[0, 0]), ("worst BAD seed", max(bad_fx, key=abs)),
                  ("worst GOOD seed", max(good_fx, key=lambda v: abs(v - 800)))):
    print(f"  {label:16s} fx = {fx:7.1f} -> Z = {fx*0.12/30:.3f} m")

# ------------------------------------------------------- figure: the views
n_show = min(8, len(kept_images))
fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.4))
for ax, idx in zip(axes.ravel(), range(n_show)):
    ax.imshow(kept_images[idx], cmap="gray", vmin=0, vmax=255)
    c = image_points[idx]
    ax.plot(c[:, 0], c[:, 1], ".", color="#d62728", ms=2.5)
    ax.set_title(f"view {kept_idx[idx]}: tilt {tilts[idx]:.0f} deg, "
                 f"RMS {per_view[idx]:.3f} px", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
for ax in axes.ravel()[n_show:]:
    ax.axis("off")
fig.suptitle(f"Eight of the {len(kept_images)} rendered calibration views, with the "
             f"corners the detector found", y=0.99)
save(fig, "04_calibration_views.png")

# --------------------------------------------- figure: the confound, measured
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

ax = axes[0]
seeds = np.arange(10)
ax.axhline(K_TRUE[0, 0], color="#333333", lw=1.2, ls="--", label="true fx = 800.0")
ax.plot(seeds, good_fx, "o-", color="#1f77b4", label="tilted and swept (good)")
ax.plot(seeds, bad_fx, "s-", color="#d62728", label="fronto-parallel, centred (bad)")
# Clip the axis to the interesting range: one bad seed is so far off that
# plotting it to scale flattens every other row into a single line.  The
# off-scale points are labelled rather than hidden.
lo, hi = 730.0, 860.0
ax.set_ylim(lo, hi)
for sd, val in zip(seeds, bad_fx):
    if val > hi or val < lo:
        ax.annotate(f"seed {sd}: {val:.0f}\n(off this axis)",
                    xy=(sd, hi - 3), xytext=(sd - 3.2, hi - 26), fontsize=8,
                    color="#d62728",
                    arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2))
ax.set_xlabel("random seed"); ax.set_ylabel("recovered fx (px)")
ax.set_title("(a) the same solver, the same noise,\nonly the capture geometry differs")
ax.legend(fontsize=8)

ax = axes[1]
ax.plot(seeds, good_rms, "o-", color="#1f77b4", label="good capture")
ax.plot(seeds, bad_rms, "s-", color="#d62728", label="bad capture")
ax.set_ylim(0, max(good_rms + bad_rms) * 1.6)
ax.set_xlabel("random seed"); ax.set_ylabel("RMS reprojection error (px)")
ax.set_title("(b) and the error metric cannot tell\nthem apart")
ax.legend(fontsize=8)

ax = axes[2]
allpts = np.vstack(image_points)
ax.plot(allpts[:, 0], allpts[:, 1], ".", ms=1.4, color="#1f77b4")
ax.add_patch(plt.Rectangle((SIZE[0] * 0.2, SIZE[1] * 0.2), SIZE[0] * 0.6, SIZE[1] * 0.6,
                           fill=False, ec="#d62728", lw=1.4, ls="--"))
ax.annotate("outside this box is where\nk1 and k2 are observable", (12, 462),
            fontsize=8, color="#d62728")
ax.set_xlim(0, SIZE[0]); ax.set_ylim(SIZE[1], 0)
ax.set_xlabel("u (px)"); ax.set_ylabel("v (px)")
ax.set_title(f"(c) corner coverage: {coverage*100:.0f}% of border\ncells are empty")

fig.suptitle("Low reprojection error is not a correct calibration", y=1.02)
save(fig, "05_calibration_confound.png")
