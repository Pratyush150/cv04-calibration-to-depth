"""08 - Rectification, then a block matcher built from scratch.

Run:  python3 examples/08_rectify_and_block_match.py   (~20 s)

What it shows
  * why rectification exists: it turns "search along a diagonal line" into
    "search along a row", measured as row-alignment error before and after
  * that stereoRectify picks a NEW focal length, and what using the old one
    costs in depth
  * a cost volume, winner-take-all, parabolic sub-pixel and a left-right check,
    written out, scored against ground truth and against StereoBM/StereoSGBM
"""

import time

import cv2
import numpy as np

from _common import rule, save, plt          # noqa: E402
from geo import depth as dp                  # noqa: E402
from geo import distortion as dm             # noqa: E402
from geo import pinhole as ph                # noqa: E402
from geo import stereo as st                 # noqa: E402
from geo import synthetic as syn             # noqa: E402

SIZE = (480, 320)
BASELINE = 0.12
NDISP, WINDOW = 48, 9
K = ph.intrinsic_matrix(520.0, 520.0, (SIZE[0] - 1) / 2, (SIZE[1] - 1) / 2)
D_LENS = dm.coefficients(k1=-0.22, k2=0.06, p1=0.0008, p2=-0.0006)

rule("A deliberately misaligned rig, because a perfect one teaches nothing")
R_rel, _ = cv2.Rodrigues(np.array([0.020, 0.045, 0.030]))   # camera 2's rotation
C2 = np.array([BASELINE, 0.006, -0.004])                    # and its position
T_rel = ph.extrinsics_from_centre(R_rel, C2)
print(f"baseline           : {np.linalg.norm(C2)*1000:.1f} mm")
print(f"relative rotation  : {np.degrees(np.linalg.norm(cv2.Rodrigues(R_rel)[0])):.2f} deg")
print(f"lens distortion    : {D_LENS}")
print("Both cameras carry the same lens model here.  Nothing about this rig is")
print("catastrophic - three degrees of relative rotation and a few millimetres of")
print("vertical offset is an ordinary bracket - and it is entirely enough to stop a")
print("row-wise matcher from working at all.")

left_raw, _, _ = syn.render_view(K, np.eye(3), np.zeros(3), SIZE, D=D_LENS, seed=21)
right_raw, _, _ = syn.render_view(K, R_rel, C2, SIZE, D=D_LENS, seed=22)

rule("The metric that tells you whether rectification worked")
before = st.row_alignment_error(left_raw, right_raw)
rect = st.rectify_pair(K, D_LENS, K, D_LENS, SIZE, R_rel, T_rel, alpha=0.0)
left_rect, right_rect = st.remap_pair(rect, left_raw, right_raw)
after = st.row_alignment_error(left_rect, right_rect)
print(f"{'':22s} {'median':>8s} {'mean':>8s} {'p90':>8s} {'matches':>8s}")
print(f"{'before rectification':22s} {before[0]:8.2f} {before[1]:8.2f} "
      f"{before[2]:8.2f} {before[3]:8d}")
print(f"{'after rectification':22s} {after[0]:8.2f} {after[1]:8.2f} "
      f"{after[2]:8.2f} {after[3]:8d}")
print("ORB features matched between the two images, then |y_left - y_right|.")
print("Gate on the MEDIAN: stereo matching always leaks a few gross mismatches, and a")
print("handful of matches wrong by thirty pixels drags the mean over any threshold")
print("while nine-tenths of them sit at zero.  Median under 0.5 px, proceed; over")
print("1 px, fix the calibration before looking at a single disparity map.")
print("\nThis is the number nobody computes, and rectification is the step that fails")
print("QUIETLY - a slightly wrong warp produces a smooth, structured, believable and")
print("completely wrong disparity map.")
print("\nRead the row above the way the rule says to.  The median is 0.00 px and the")
print("mean is not, which by that rule is a MATCHING failure and not a rectification")
print("one.  Here we can prove it, because the scene has ground truth: take known")
print("3-D points, push them through the rectified projection matrices P1 and P2, and")
print("measure the row difference directly.")
_, depth_l_raw, _ = syn.render_view(K, np.eye(3), np.zeros(3), SIZE, D=D_LENS, seed=21)
grng = np.random.default_rng(0)
gu = grng.integers(30, SIZE[0] - 30, 400)
gv = grng.integers(30, SIZE[1] - 30, 400)
gxy = dm.normalize_pixels(K, np.column_stack([gu, gv]))
gxn, gyn = dm.undistort_normalized(gxy[:, 0], gxy[:, 1], D_LENS)
Xg = np.column_stack([gxn, gyn, np.ones_like(gxn)]) * depth_l_raw[gv, gu][:, None]
h1 = (rect.R1 @ Xg.T).T @ rect.P1[:, :3].T + rect.P1[:, 3]
h2 = ((rect.R2 @ (Xg @ R_rel.T + T_rel).T).T @ rect.P2[:, :3].T + rect.P2[:, 3])
row_gap = np.abs(h1[:, 1] / h1[:, 2] - h2[:, 1] / h2[:, 2])
print(f"on 400 ground-truth correspondences: median |dy| = {np.median(row_gap):.2e} px, "
      f"max = {row_gap.max():.2e} px")
print("The rectification is exact to machine precision.  Every pixel of that ORB mean")
print("is a wrong match, and most of them come from the striped band at the bottom of")
print("the scene, where every stripe looks like every other stripe.")

rule("stereoRectify changes the focal length, and nobody warns you")
print(f"original fx from K   : {K[0,0]:.4f} px")
print(f"rectified f (P1[0,0]): {rect.f_rect:.4f} px")
print(f"difference           : {100*(rect.f_rect - K[0,0])/K[0,0]:+.2f}%")
print(f"baseline read back from P2[0,3]/f : {rect.baseline*1000:.2f} mm "
      f"(true {np.linalg.norm(C2)*1000:.2f} mm)")
d_demo = 30.0
z_right = dp.depth_from_disparity(d_demo, rect.f_rect, np.linalg.norm(C2))
z_wrong = dp.depth_from_disparity(d_demo, K[0, 0], np.linalg.norm(C2))
print(f"\nat d = {d_demo:.0f} px: correct Z = {z_right:.4f} m, "
      f"using the pre-rectification fx = {z_wrong:.4f} m "
      f"({100*(z_wrong-z_right)/z_right:+.2f}%)")
print("A constant percentage error in every depth looks exactly like a mis-measured")
print("baseline, so people go and re-measure the rig with calipers and find nothing.")

rule("Now the matcher, on a pair that is rectified by construction")
scene = syn.render_stereo_pair(size=SIZE, fx=K[0, 0], baseline=BASELINE)
print(f"scene: {SIZE[0]}x{SIZE[1]}, fx = {K[0,0]:.0f} px, baseline = {BASELINE*100:.0f} cm")
print(f"true disparity range: {np.nanmin(scene.disparity_left):.2f} to "
      f"{np.nanmax(scene.disparity_left):.2f} px "
      f"(depths {np.nanmin(scene.depth_left):.2f} to {np.nanmax(scene.depth_left):.2f} m)")
print(f"occluded pixels: {100*scene.occluded.mean():.1f}% of the left image, plus "
      f"{100*scene.outside_right.mean():.1f}% whose match would fall off the left edge")
print("The scoring below uses matchable pixels only.  An occluded pixel has no correct")
print("answer to score against - it is not in the right image at all - and neither does")
print("one whose match would sit off the left edge of that image.  Scoring a matcher on")
print("pixels where no answer exists measures the scoring, not the matcher.")

valid = ~scene.unmatchable
results = {}
timings = {}
for tag, kwargs in (("SAD, WTA only", dict(metric="SAD", subpixel=False, lr_check=False)),
                    ("SAD + sub-pixel", dict(metric="SAD", subpixel=True, lr_check=False)),
                    ("SAD + sub-pixel + LR", dict(metric="SAD", subpixel=True, lr_check=True)),
                    ("SSD + sub-pixel + LR", dict(metric="SSD", subpixel=True, lr_check=True))):
    t0 = time.time()
    d = st.block_match(scene.left, scene.right, ndisp=NDISP, window=WINDOW, **kwargs)
    timings[tag] = time.time() - t0
    results[tag] = d

t0 = time.time()
results["cv2.StereoBM"] = st.opencv_bm(scene.left, scene.right, ndisp=NDISP, block=15)
timings["cv2.StereoBM"] = time.time() - t0
t0 = time.time()
results["cv2.StereoSGBM"] = st.opencv_sgbm(scene.left, scene.right, ndisp=NDISP, block=5)
timings["cv2.StereoSGBM"] = time.time() - t0

print(f"\n{'matcher':<22s} {'density':>8s} {'MAE px':>8s} {'RMSE px':>8s} "
      f"{'bad>1px':>8s} {'time s':>7s}")
for tag, d in results.items():
    sc = st.score_disparity(d, scene.disparity_left, valid)
    print(f"{tag:<22s} {sc['density']*100:7.1f}% {sc['mae']:8.3f} {sc['rmse']:8.3f} "
          f"{sc['bad_pct']:7.1f}% {timings[tag]:7.2f}")
print("\nRead the first three rows as one experiment.  Sub-pixel refinement is three")
print("array lookups and a parabola; the left-right check throws pixels away and the")
print("error drops because of it - the matcher got better by admitting what it does")
print("not know.  Density and error trade off directly, which is why quoting either")
print("one alone is misleading.")
print("\nStereoSGBM is the one to beat and this matcher does not beat it.  It differs")
print("in three ways, not one: a Birchfield-Tomasi cost rather than SAD, semi-global")
print("aggregation along scanlines (the part genuinely not implemented here - our")
print("winner-take-all treats every pixel as independent), and its own post-filters.")

textured = valid & (scene.region_mask("near_slab") | scene.region_mask("ramp") |
                    scene.region_mask("wall"))
print("\nThose whole-image numbers are dominated by two regions that were put in the")
print("scene specifically to break a matcher.  Score the same maps on the textured")
print("surfaces only - the three where the data actually contains an answer:")
print(f"\n{'matcher':<22s} {'density':>8s} {'MAE px':>8s} {'bad>1px':>8s}")
for tag, d in results.items():
    sc = st.score_disparity(d, scene.disparity_left, textured)
    print(f"{tag:<22s} {sc['density']*100:7.1f}% {sc['mae']:8.3f} {sc['bad_pct']:7.1f}%")
print("\nThat is the honest comparison, and it is a different story from the one above:")
print("where the images contain the information, this matcher is within a few")
print("hundredths of a pixel of SGBM.  The whole-image gap is almost entirely about")
print("what each of them does where the information is NOT there, which is example 09.")

wta = results["SAD, WTA only"]
sub = results["SAD + sub-pixel"]
# On the textured surfaces, where a minimum exists to refine.  Scored on the
# same pixels for both, so this is the refinement's effect and nothing else -
# on a flat cost curve the parabola is correctly declining to do anything.
m = textured & np.isfinite(wta) & np.isfinite(sub)
print(f"\nsub-pixel refinement alone, textured surfaces, identical pixels:")
print(f"  integer winner-take-all : MAE {np.abs(wta[m]-scene.disparity_left[m]).mean():.4f} px")
print(f"  parabolic sub-pixel     : MAE {np.abs(sub[m]-scene.disparity_left[m]).mean():.4f} px")
d_int, d_sub = 8.0, 8.3
print(f"in metres at the far end of this scene: d = {d_int:.1f} px is "
      f"{dp.depth_from_disparity(d_int, K[0,0], BASELINE):.3f} m, "
      f"d = {d_sub:.1f} px is {dp.depth_from_disparity(d_sub, K[0,0], BASELINE):.3f} m")
print(f"- three tenths of a pixel is "
      f"{1000*abs(dp.depth_from_disparity(d_int, K[0,0], BASELINE) - dp.depth_from_disparity(d_sub, K[0,0], BASELINE)):.0f} mm there.")

# ------------------------------------------------- figure: rectification
fig, axes = plt.subplots(2, 1, figsize=(11, 7.4))
for ax, (l, r), title in ((axes[0], (left_raw, right_raw),
                           f"before rectification: median |dy| = {before[0]:.2f} px"),
                          (axes[1], (left_rect, right_rect),
                           f"after rectification: median |dy| = {after[0]:.2f} px")):
    pair = np.hstack([l, r])
    ax.imshow(pair, cmap="gray", vmin=0, vmax=255)
    for y in range(20, SIZE[1], 30):
        ax.axhline(y, color="#d62728", lw=0.6, alpha=0.8)
    ax.axvline(SIZE[0] - 0.5, color="#1f77b4", lw=1.5)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
fig.suptitle("Rectification: the same scene point must land on the same ROW in both "
             "images", y=0.98)
save(fig, "09_rectification.png")

# ------------------------------------------------- figure: disparity maps
show = ["SAD, WTA only", "SAD + sub-pixel + LR", "SSD + sub-pixel + LR",
        "cv2.StereoBM", "cv2.StereoSGBM"]
fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.2))
vmin, vmax = np.nanmin(scene.disparity_left), np.nanmax(scene.disparity_left)
ax = axes[0, 0]
im = ax.imshow(scene.disparity_left, cmap="viridis", vmin=vmin, vmax=vmax)
ax.set_title("ground truth disparity (px)")
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
fig.colorbar(im, ax=ax, fraction=0.046)
for ax, tag in zip(axes.ravel()[1:], show):
    d = results[tag]
    sc = st.score_disparity(d, scene.disparity_left, valid)
    im = ax.imshow(np.where(np.isfinite(d), d, np.nan), cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(f"{tag}\nMAE {sc['mae']:.2f} px, density {sc['density']*100:.0f}%",
                 fontsize=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle(f"Disparity: {NDISP} candidates, {WINDOW}x{WINDOW} window, "
             f"white pixels are 'no answer'", y=0.98)
save(fig, "10_disparity_maps.png")
