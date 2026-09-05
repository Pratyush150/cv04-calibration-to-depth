"""05 - Reprojection error: what the number means, and what its structure tells you.

Run:  python3 examples/05_reprojection_error.py   (~15 s)

What it shows
  * reprojection error defined by construction: project the known corner with
    the estimated model, subtract the detected pixel, that vector is the residual
  * what a healthy residual cloud looks like (isotropic, centred, sub-pixel)
  * what an under-parameterised model looks like (the same RMS story, but the
    residuals acquire radial STRUCTURE - the model is missing a term and the
    residuals are where it went)
"""

import numpy as np
import cv2

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

objp = syn.board_object_points(PATTERN, SQUARE)
poses = syn.board_poses(14, tilt_sigma=0.40, spread=True, seed=23,
                        pattern=PATTERN, square=SQUARE)
object_points, image_points = [], []
for i, (rvec, tvec) in enumerate(poses):
    img = syn.render_checkerboard(K_TRUE, D_TRUE, rvec, tvec, SIZE,
                                  pattern=PATTERN, square=SQUARE, seed=100 + i)
    ok, corners = syn.detect_corners(img, PATTERN)
    if ok:
        object_points.append(objp)
        image_points.append(corners)

rule("What the residual actually is")
full = cal.calibrate(object_points, image_points, SIZE)
res, pix = cal.per_point_residuals(full, object_points, image_points)
print(f"{len(object_points)} views, {len(res)} corners, {len(res)*2} residual components")
print(f"aggregate RMS from cv2.calibrateCamera : {full.rms:.5f} px")
print(f"the same number recomputed by hand     : "
      f"{np.sqrt((res ** 2).sum(axis=1).mean()):.5f} px")
print("Recomputed as sqrt(mean over corners of (du^2 + dv^2)).  That is the entire")
print("definition, and it is worth writing out once so the number stops being an")
print("output and becomes a quantity.")

rule("What a good value looks like, and what it does not tell you")
mag = np.linalg.norm(res, axis=1)
print(f"residual magnitude: median {np.median(mag):.4f} px, "
      f"90th pct {np.percentile(mag, 90):.4f} px, max {mag.max():.4f} px")
print(f"mean residual vector: ({res[:,0].mean():+.4f}, {res[:,1].mean():+.4f}) px "
      "- centred, as it must be")
print("Rules of thumb for a real camera, in pixels of RMS:")
print("  < 0.3   a good calibration on a decent detector")
print("  0.3-1.0 usable, but look at the per-view spread before trusting it")
print("  > 1.0   something is wrong: pattern size, blur, a bad view, wrong board")
print("Those thresholds are about the FIT.  They say nothing about whether fx is")
print("right - example 04 measures a 0.131 px RMS sitting on a focal length that is")
print("264% too large - so read them as a necessary condition, never a sufficient one.")

rule("An under-parameterised model puts its error into the residuals")
flags = (cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
poor = cal.calibrate(object_points, image_points, SIZE, flags=flags)
res_p, pix_p = cal.per_point_residuals(poor, object_points, image_points)
print("Refit with k2, k3 and both tangential terms forced to zero - k1 only:")
print(f"  full model  RMS = {full.rms:.4f} px    fx = {full.K[0,0]:.2f}  "
      f"D = {np.array2string(full.D, precision=4)}")
print(f"  k1-only     RMS = {poor.rms:.4f} px    fx = {poor.K[0,0]:.2f}  "
      f"D = {np.array2string(poor.D, precision=4)}")
err_full = cal.intrinsics_error(K_TRUE, D_TRUE, full.K, full.D)
err_poor = cal.intrinsics_error(K_TRUE, D_TRUE, poor.K, poor.D)
print(f"  fx error: full {err_full['fx']['rel_pct']:+.3f}%   "
      f"k1-only {err_poor['fx']['rel_pct']:+.3f}%")

r_full = np.linalg.norm(pix - full.K[:2, 2], axis=1)
r_poor = np.linalg.norm(pix_p - poor.K[:2, 2], axis=1)
mag_p = np.linalg.norm(res_p, axis=1)


def radial_trend(r, m, edges):
    """Mean residual magnitude in radius bins - the shape, not the summary."""
    idx = np.digitize(r, edges) - 1
    return np.array([m[idx == b].mean() if np.any(idx == b) else np.nan
                     for b in range(len(edges) - 1)])


edges = np.linspace(0, max(r_full.max(), r_poor.max()), 9)
centres = 0.5 * (edges[:-1] + edges[1:])
trend_full = radial_trend(r_full, mag, edges)
trend_poor = radial_trend(r_poor, mag_p, edges)
print("\nmean |residual| against distance from the principal point:")
print(f"{'r (px)':>8s} {'full model':>12s} {'k1 only':>10s}")
for c, a, b in zip(centres, trend_full, trend_poor):
    print(f"{c:8.0f} {a:12.4f} {b:10.4f}")
print("The full model's row is flat.  The k1-only row climbs with radius, because")
print("the terms that were removed are the ones that act at the edge of the frame.")
print("A residual scatter that has STRUCTURE is a model that is missing something,")
print("and where the structure lives says which something.")

per_view = cal.per_view_errors(full, object_points, image_points)
worst = int(np.argmax(per_view))
print(f"\nper-view RMS: median {np.median(per_view):.4f} px, "
      f"worst view #{worst} at {per_view[worst]:.4f} px "
      f"({per_view[worst]/np.median(per_view):.1f}x the median)")
print("A view at 3x the median is worth investigating - a blurred frame or a bad")
print("detection.  Dropping views until the number looks good is not calibration.")

# ---------------------------------------------------------------- the figure
fig, axes = plt.subplots(1, 4, figsize=(18, 4.4), layout="constrained")

lim = max(np.abs(res).max(), np.abs(res_p).max()) * 1.05
for ax, r, p, title in ((axes[0], res, pix, f"(a) full model, RMS {full.rms:.3f} px"),
                        (axes[1], res_p, pix_p, f"(b) k1 only, RMS {poor.rms:.3f} px")):
    rad = np.linalg.norm(p - full.K[:2, 2], axis=1)
    sc = ax.scatter(r[:, 0], r[:, 1], c=rad, s=6, cmap="viridis", alpha=0.85)
    ax.axhline(0, color="#999999", lw=0.7); ax.axvline(0, color="#999999", lw=0.7)
    circ = plt.Circle((0, 0), np.percentile(np.linalg.norm(r, axis=1), 90),
                      fill=False, ec="#d62728", lw=1.2, ls="--")
    ax.add_patch(circ)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xlabel("du (px)"); ax.set_ylabel("dv (px)"); ax.set_title(title)
    fig.colorbar(sc, ax=ax, label="distance from principal point (px)")

ax = axes[2]
ax.plot(centres, trend_full, "o-", color="#1f77b4", label="full model")
ax.plot(centres, trend_poor, "s-", color="#d62728", label="k1 only")
ax.set_xlabel("distance from principal point (px)")
ax.set_ylabel("mean |residual| (px)")
ax.set_title("(c) structure lives at the frame edge")
ax.legend(fontsize=8)

ax = axes[3]
ax.bar(np.arange(len(per_view)), per_view, color="#1f77b4")
ax.axhline(np.median(per_view), color="#d62728", ls="--", lw=1.2,
           label=f"median {np.median(per_view):.3f} px")
ax.axhline(2 * np.median(per_view), color="#ff7f0e", ls=":", lw=1.2,
           label="2x median: investigate above this")
ax.set_xlabel("view index"); ax.set_ylabel("per-view RMS (px)")
ax.set_title("(d) the aggregate hides the outlier")
ax.legend(fontsize=8)

fig.suptitle("Reprojection residuals: the number, and the shape of the number")
save(fig, "06_reprojection_residuals.png")
