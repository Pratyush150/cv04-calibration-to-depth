"""09 - The three ways stereo matching fails, taken apart one at a time.

Run:  python3 examples/09_failure_modes.py

What it shows
  * textureless: the cost curve is FLAT.  There is an answer; the data does not
    contain it.
  * repeated pattern: the cost curve has SEVERAL equal minima.  The matcher
    picks one confidently and is usually wrong.
  * occlusion: there is NO answer.  The pixel is not in the other image at all,
    and the left-right check is how you find that out.

Three different diseases with three different symptoms.  Being able to separate
them is what turns "the depth map is noisy" into a diagnosis.
"""

import cv2
import numpy as np

from _common import rule, save, plt          # noqa: E402
from geo import stereo as st                 # noqa: E402
from geo import synthetic as syn             # noqa: E402

SIZE = (480, 320)
NDISP, WINDOW = 48, 9
scene = syn.render_stereo_pair(size=SIZE, fx=520.0, baseline=0.12)
vol = st.cost_volume(scene.left, scene.right, NDISP, WINDOW, "SAD")
best = np.argmin(vol, axis=0)
disp = st.block_match(scene.left, scene.right, ndisp=NDISP, window=WINDOW)
agree = st.left_right_consistency(vol, best)

rule("Where the matcher is right, and where it is not")
print(f"{'region':<14s} {'true disp':>10s} {'density':>9s} {'MAE px':>9s} "
      f"{'bad>1px':>9s}")
for name in scene.names:
    mask = scene.region_mask(name) & ~scene.unmatchable
    if mask.sum() == 0:
        continue
    sc = st.score_disparity(disp, scene.disparity_left, mask)
    truth = scene.disparity_left[mask]
    print(f"{name:<14s} {truth.mean():10.2f} {sc['density']*100:8.1f}% "
          f"{sc['mae']:9.3f} {sc['bad_pct']:8.1f}%")
print("\nThe wall, the ramp and the near slab are all textured, at three different")
print("depths, and the matcher handles all three.  The two failures are not about")
print("depth or distance - they are about what is painted on the surface.")

rule("Occlusion: no correct answer exists")
occ = scene.occluded
print(f"occluded pixels                       : {100*occ.mean():5.2f}% of the image")
print(f"of those, the LR check rejected       : {100*(~agree[occ]).mean():5.1f}%")
print(f"of the non-occluded, it rejected      : {100*(~agree[~occ]).mean():5.1f}%")
kept_occ = np.isfinite(disp) & occ
if kept_occ.any():
    print(f"error on occluded pixels it KEPT      : "
          f"{np.abs(disp[kept_occ] - scene.disparity_left[kept_occ]).mean():.2f} px")
print("A left-right check is not a smoothing filter and it is not tuning.  It asks the")
print("right image what IT thinks it matches, and throws away the pixels where the two")
print("answers disagree.  Those pixels are the ones the geometry hid.")

rule("Three probe pixels, and their cost curves")
def deepest_pixel(mask):
    """The pixel furthest from the edge of a mask, by distance transform.

    Probing a pixel near a region boundary measures the boundary, not the
    region: a 9x9 window there straddles two surfaces and its cost curve is a
    mixture of both.  This picks the most interior pixel available, which is
    the only place a single cost curve says something about one surface.
    """
    usable = (mask & (np.arange(SIZE[0])[None, :] > NDISP + WINDOW)).astype(np.uint8)
    # Also keep a window's margin off the image border: distanceTransform has no
    # opinion about the frame edge, but cv2.boxFilter replicates there, so a
    # probe on row 0 would be reading invented pixels.
    usable[:WINDOW, :] = 0
    usable[-WINDOW:, :] = 0
    usable[:, -WINDOW:] = 0
    dist = cv2.distanceTransform(usable, cv2.DIST_L2, 3)
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return int(y), int(x)


probes = {}
for name, label in (("wall", "well textured"), ("textureless", "flat grey patch"),
                    ("repeated", "stripe pattern")):
    y, x = deepest_pixel(scene.region_mask(name) & ~scene.unmatchable)
    probes[name] = (y, x, label)
y, x = deepest_pixel(occ)
probes["occluded"] = (y, x, "hidden from the right camera")

print(f"{'probe':<14s} {'pixel':>12s} {'true d':>8s} {'argmin':>8s} "
      f"{'cost spread':>12s} {'2nd min gap':>12s}")
curves = {}
for name, (y, x, label) in probes.items():
    c = vol[:, y, x]
    curves[name] = c
    order = np.argsort(c)
    # "How much better is the winner than the best genuinely different
    # candidate": the margin a matcher would need to be confident.
    far = [i for i in order if abs(int(i) - int(order[0])) > 2]
    gap = c[far[0]] - c[order[0]] if far else np.nan
    print(f"{name:<14s} {f'({x},{y})':>12s} {scene.disparity_left[y, x]:8.2f} "
          f"{order[0]:8d} {c.max()-c.min():12.2f} {gap:12.2f}")
print("\nRead the last two columns together.  The well-textured pixel has a large")
print("spread and a large gap to its runner-up: one clear answer.  The flat patch has")
print("almost no spread at all - every disparity costs the same, so the argmin is")
print("decided by sensor noise.  The stripe pixel has a big spread AND a tiny gap:")
print("several answers look equally good, which is the worst case, because the")
print("matcher reports one of them with no hint that it was a coin toss.")

# ---------------------------------------------------------------- the figure
fig = plt.figure(figsize=(15, 8.4))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)

ax = fig.add_subplot(gs[0, 0])
ax.imshow(scene.left, cmap="gray", vmin=0, vmax=255)
for name, (y, x, label) in probes.items():
    ax.plot([x], [y], "o", ms=9, mfc="none", mew=2,
            color={"wall": "#1f77b4", "textureless": "#d62728",
                   "repeated": "#ff7f0e", "occluded": "#2ca02c"}[name])
    ax.annotate(name, (x + 8, y - 6), fontsize=8, color="white",
                bbox=dict(fc="black", alpha=0.5, pad=1))
ax.set_title("(a) the left image, with four probes")
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

ax = fig.add_subplot(gs[0, 1])
for name, colour in (("wall", "#1f77b4"), ("textureless", "#d62728"),
                     ("repeated", "#ff7f0e"), ("occluded", "#2ca02c")):
    y, x, _ = probes[name]
    ax.plot(curves[name], color=colour, lw=1.6, label=name)
    ax.axvline(scene.disparity_left[y, x], color=colour, ls=":", lw=1.0)
ax.set_xlabel("candidate disparity d (px)")
ax.set_ylabel("windowed SAD cost")
ax.set_title("(b) one column of the cost volume, four times\n"
             "(dotted lines mark the true disparity)")
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[0, 2])
err = np.abs(disp - scene.disparity_left)
im = ax.imshow(np.where(np.isfinite(err), err, np.nan), cmap="magma", vmin=0, vmax=8)
ax.contour(occ.astype(float), levels=[0.5], colors="#00ffff", linewidths=0.8)
ax.set_title("(c) |error| in px, occlusions outlined in cyan")
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
fig.colorbar(im, ax=ax, fraction=0.046)

ax = fig.add_subplot(gs[1, 0])
names, mae, dens = [], [], []
for name in scene.names:
    mask = scene.region_mask(name) & ~scene.unmatchable
    if mask.sum() == 0:
        continue
    sc = st.score_disparity(disp, scene.disparity_left, mask)
    names.append(name)
    mae.append(sc["mae"])
    dens.append(sc["density"] * 100)
xs = np.arange(len(names))
ax.bar(xs, mae, color=["#1f77b4" if n in ("near_slab", "ramp", "wall") else "#d62728"
                       for n in names])
ax.set_xticks(xs); ax.set_xticklabels(names, rotation=20, fontsize=8)
ax.set_ylabel("mean absolute disparity error (px)")
ax.set_title("(d) error by region")

ax = fig.add_subplot(gs[1, 1])
ax.bar(xs, dens, color=["#1f77b4" if n in ("near_slab", "ramp", "wall") else "#d62728"
                        for n in names])
ax.set_xticks(xs); ax.set_xticklabels(names, rotation=20, fontsize=8)
ax.set_ylabel("% of pixels the matcher answered")
ax.set_title("(e) density by region: the flat patch is\nthe only one it admits to")

ax = fig.add_subplot(gs[1, 2])
ax.imshow(scene.left, cmap="gray", vmin=0, vmax=255, alpha=0.45)
show = np.zeros(scene.left.shape + (4,))
show[occ] = (1.0, 0.0, 0.0, 0.85)
show[~agree & ~occ] = (0.0, 0.4, 1.0, 0.6)
ax.imshow(show)
ax.set_title(f"(f) red: truly occluded ({100*occ.mean():.1f}%)\n"
             f"blue: other pixels the LR check rejected")
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

fig.suptitle("Three failure modes: no information, too many answers, no answer at all",
             y=0.97)
save(fig, "11_failure_modes.png")
