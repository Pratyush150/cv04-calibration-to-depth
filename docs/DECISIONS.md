# Decisions

One record per real architectural choice: what was decided, what else was on the
table, why this one, and what it costs. A decision with no cost listed has not
been thought about hard enough.

---

## D1 — All data is synthetic and rendered by this repository

**Decision.** Every image is produced by `src/geo/synthetic.py` from intrinsics,
distortion coefficients, poses and depths that are variables in a script. No
camera, no download, no dataset.

**Alternatives.**
- *Photograph a printed board.* The normal way to teach calibration.
- *Use a public dataset* — Middlebury for stereo, KITTI for scale.
- *Ship pre-rendered images as binary assets.*

**Why this one.** The central lesson of the repository is that **reprojection
error and correctness are different things**, and you cannot demonstrate that
without knowing the correct answer in advance. On real photographs the only
available number is the reprojection error, so "the calibration worked" is an
assertion. Here it is a measurement: `fx` recovered to −0.076% of a value that
was typed into the script.

The same argument applies to every other stage. Ground-truth disparity means the
block matcher can be scored per region rather than admired. Ground-truth
correspondences mean the eight-point comparison measures the *estimator* and not
the feature detector in front of it. Exact occlusion masks mean the left-right
check can be scored on the pixels it is actually for.

A dataset also brings a download at run time, which the specification for these
repositories forbids, and pre-rendered assets would hide the renderer — which is
itself a third of the teaching material.

**What it costs.** No sensor noise model beyond additive Gaussian, no motion
blur, no rolling shutter, no chromatic aberration, no printing error in the
board, and none of the practical friction of a real capture session. A
calibration that works here is a **necessary** condition for one that works on a
rig, not a sufficient one, and the README says so. Published real-rig constants
do appear where they are the right tool — Middlebury's `f`, `B` and `doffs` in
`examples/10`, because the `doffs` lesson needs a rig that genuinely has one.

---

## D2 — The checkerboard renderer is a backward ray tracer

**Decision.** For each output pixel: normalise, undistort iteratively, form a
ray, intersect the board plane, look up the checker colour. Supersample 2× and
box-average down.

**Alternatives.**
- *Forward projection plus polygon fill*: project the board's squares into the
  image and rasterise them.
- *Warp a fronto-parallel board image with the homography*, then apply a
  distortion remap.

**Why this one.** Forward rasterisation gives a pinhole image; distortion then
has to be applied as a second remap pass, which resamples an already-resampled
image and softens exactly the corners the calibration is going to measure. The
backward tracer applies the lens model *once*, analytically, at ray-generation
time — so the distortion baked into the rendered image is the exact inverse of
the undistortion the calibration later applies. That makes recovering `D` a real
round trip rather than a tautology.

**What it costs.** About 1 s per 640×480 view, most of it in the fixed-point
undistortion running on 1.2 million supersampled rays. `examples/04` spends 21 s
rendering. The first implementation solved a 3×3 linear system per ray and was
100× slower still; the current one exploits the fact that the board's normal is
the third column of `R`, which turns the intersection into one dot product.

---

## D3 — Two stereo scenes, not one

**Decision.** `render_stereo_pair` produces a pair that is rectified *by
construction* — same intrinsics, same orientation, pure baseline translation.
`render_view` separately produces an arbitrarily posed, distorted view, and
`examples/08` uses it to build a genuinely misaligned pair for a real
`cv2.stereoRectify` round trip.

**Alternatives.**
- *One misaligned scene, rectified before matching.* More realistic.
- *One perfect scene, with rectification described but not run.*

**Why this one.** They teach different things and each would contaminate the
other. Scoring a matcher on a rectified-from-misaligned pair means every
disparity error is a mixture of matching error and residual rectification error,
and the failure-mode analysis in `examples/09` would no longer be attributable.
Conversely, a repository that never runs `stereoRectify` cannot show that it
changes the focal length, or measure the row-alignment metric that catches a
quiet failure.

**What it costs.** Ground-truth disparity is only available in the by-construction
scene, so the rectification demonstration is validated by row alignment (and by
pushing known 3-D points through `P1` and `P2`) rather than by a disparity score.
A reader has to hold two scenes in their head.

---

## D4 — Implement by hand, then check against OpenCV, then say which to use

**Decision.** Six algorithms are implemented from scratch and asserted equal to
their library equivalents in the test suite: the distortion forward model, its
iterative inverse, the homography DLT, the eight-point algorithm, DLT
triangulation, and disparity reprojection through `Q`.

**Alternatives.**
- *Call the library and explain what it does.* Shorter, and what production code
  should do.
- *Implement everything, including `calibrateCamera`.*

**Why this one.** The comparison **is** the lesson, and asserting the agreement in
a test is what makes it credible rather than decorative. It also produces
findings that reading the docs does not: that `cv2.undistortPoints` is an
approximation with a tolerance and is the *less* exact of the two on a clean
point; that `cv2.findHomography` differs from a DLT because it refines on
reprojection error afterwards; that our normalised eight-point solve lands
exactly on `findFundamentalMat` because that function normalises internally and
never mentions it.

The line is drawn at `calibrateCamera`. Implementing Levenberg-Marquardt over
`6N + 9` parameters would be a numerical-optimisation lesson wearing a camera
costume, and it would bury the actual point of that section, which is capture
geometry.

**What it costs.** Roughly 400 lines of code that a production system would
delete, and a small ongoing risk that the hand-rolled version drifts from the
library across OpenCV releases — which is why the version is pinned and quoted
next to every number.

---

## D5 — Hartley normalisation is a parameter, not a hidden step

**Decision.** `homography_dlt` and `eight_point` both take `normalize=True` as a
default and accept `normalize=False`, and both examples run the comparison.

**Alternative.** Always normalise, and explain in a comment why.

**Why this one.** "Normalise your coordinates" is folklore until it is a number.
Being able to run both makes it measurable: at 0.5 px of correspondence noise the
raw eight-point solve is 4× worse, and 2869× worse once the same points are
indexed from an origin 8000 px away — which is a bookkeeping choice, not geometry.
That last experiment is the one that actually explains *why* normalisation works,
and it is impossible to run without the switch.

**What it costs.** An API with a foot-gun in it. Mitigated by the default being
correct, by the docstring stating the measured penalty, and by a test that fails
if the gap ever closes.

---

## D6 — The cost volume is materialised in full

**Decision.** `cost_volume` returns the whole `D × H × W` float32 array.
Winner-take-all, the sub-pixel fit and the left-right check all consume it.

**Alternatives.**
- *Stream it*: accumulate the running minimum per disparity slice and keep only
  the best cost and its index.
- *Call `cv2.StereoBM` and skip the volume entirely.*

**Why this one.** The cost volume is the most transferable object in this
subject — the same `D × H × W` array sits at the centre of every learned stereo
network, with SAD replaced by a feature correlation and the argmin replaced by a
3-D convolution. A reader should be able to slice it, plot one column of it, and
see a flat curve, a multi-modal curve and a sharp one side by side, which is
exactly what `examples/09` does. Streaming would make that impossible.

Materialising it also makes the left-right check nearly free, because the
right-to-left volume is a *shift* of the left-to-right one rather than a second
matching pass.

**What it costs.** Memory: 29 MB at 480×320 with 48 disparities, 118 MB at
741×500 with 80. That is fine on a laptop and wrong for an embedded target. The
scene sizes in the examples were chosen so that everything fits comfortably and
runs in well under a second.

---

## D7 — Invalid disparities are NaN, and density is always reported

**Decision.** `block_match` returns float32 with `NaN` where it has no answer.
`score_disparity` returns `density` next to `mae`, `rmse` and `bad_pct`, and
every table in the README prints all four.

**Alternatives.**
- *Zero for invalid*, as `StereoBM` effectively does.
- *A separate boolean validity mask.*

**Why this one.** Zero is a **legal disparity** meaning "infinitely far away", so
a map that encodes ignorance as zero has silently placed a horizon behind every
occlusion, and every statistic computed from it afterwards is wrong. This is not
hypothetical: OpenCV writes invalid pixels as `(minDisparity − 1) × 16`, which is
why the common `disparity > disparity.min()` validity test keeps every invalid
pixel — the test compares against exactly that value.

Reporting density alongside error is the same principle at the level of results.
Density and error trade off directly: a matcher that answers 40% of the image can
post a beautiful mean, and one that answers everywhere posts a worse number while
being more useful. Either alone is misleading, which is why the README's tables
never quote one without the other.

**What it costs.** `NaN` propagates, so every consumer needs `np.isfinite`
guards and `nanmin`/`nanmax` in the plotting code. A separate mask would be
faster; it would also be droppable, and a validity convention that can be
forgotten will be.

---

## D8 — Scoring excludes pixels where no answer exists, and says so

**Decision.** `StereoScene` distinguishes `occluded` (hidden behind something
nearer) from `outside_right` (its match would fall off the left edge), exposes
`unmatchable` as the union, and every score in the examples is computed on
matchable pixels only.

**Alternative.** Score every pixel, and let occlusions count against the matcher.

**Why this one.** Scoring a matcher where no correct answer exists measures the
scoring, not the matcher. Keeping the two causes apart matters too, even though
they have the same consequence: occlusion is a property of the *scene* and is
what a left-right check detects, while falling off the edge is a property of the
*rig* — it is the left band, as wide as the largest disparity, that every stereo
system has and that no algorithm can fill in.

**What it costs.** The headline numbers are not directly comparable with
published benchmarks, which have their own masking conventions. Mitigated by
stating the convention wherever a number appears, and by `examples/09` reporting
what the matcher does on the excluded pixels anyway (it keeps 31% of occluded
pixels, and is wrong by 13 px on them).

---

## D9 — Two traps built into the stereo scene on purpose

**Decision.** The back wall carries a flat grey band and a stripe band, both at
the same depth as their surroundings and both clear of anything that could
occlude them.

**Alternative.** A realistic scene, with failures wherever they happen to fall.

**Why this one.** Textureless, repeated and occluded are three genuinely different
diseases — no information, too many answers, no answer — and a reader who has
seen them separated will recognise them later. Putting the traps at the *same
depth* as their surroundings is what makes the attribution clean: any error
measured there is caused by the texture and not by the geometry. Keeping them
clear of the foreground objects means they can be masked and scored
independently.

The result is worth the artificiality: the flat patch answers for 36% of its
pixels and is wrong on 82% of those; the stripes answer for 87% and are wrong on
95%. Confident nonsense and an admitted gap are different failures with different
fixes, and no naturally occurring scene would have separated them so cleanly.

**What it costs.** The whole-image error numbers are dominated by two regions
that were designed to be hard, which makes them look worse than the matcher
deserves. Handled by always printing the textured-surface table beside the
whole-image one.

---

## D10 — Ten examples that each stand alone, sharing only a path shim

**Decision.** Each example re-renders whatever it needs and can be run on its
own. `examples/_common.py` holds only the `sys.path` insertion, the matplotlib
style and a `save()` helper.

**Alternatives.**
- *A pipeline*: example 04 writes `calib.npz`, example 08 reads it.
- *A notebook.*

**Why this one.** A reader lands on one example, usually from a link in the
README, and it has to work. A pipeline means an unhelpful `FileNotFoundError`
for anyone who starts in the middle, and an implicit ordering that has to be
documented and obeyed. Notebooks do not diff, do not run in CI without extra
machinery, and hide execution order.

**What it costs.** Repeated work: examples 04 and 05 each render their own
calibration set, and examples 08, 09 and 10 each render the stereo scene. That
is about 20 s of duplicated rendering across the suite. Cheap, and the tests use
a module-scoped fixture where the same cost would have been paid repeatedly
inside one file.

---

## D11 — Figures are matplotlib on white, committed to the repository

**Decision.** Twelve figures in `docs/figures/`, written by the examples, on a
white background, committed as PNGs.

**Alternatives.**
- *Generate figures in CI and do not commit them.*
- *Match the dark theme of the portfolio site.*

**Why this one.** A README that renders on GitHub needs its images present in the
repository. These are teaching materials — printed, pasted into slides, read on a
phone — and a light background survives all three; a dark one does not survive
printing at all. Committing them also means a reader can see the result before
deciding whether to install anything.

**What it costs.** About 3 MB of PNGs in the history, and the discipline of
re-running the examples whenever a number changes, since a stale figure that
contradicts the README is worse than no figure.

---

## D12 — OpenCV is pinned at 4.14.0.94 and quoted next to every number

**Decision.** `requirements.txt` pins `opencv-contrib-python==4.14.0.94`, and the
README states the version alongside the measured results.

**Alternative.** Depend on `opencv-python>=4.8`.

**Why this one.** OpenCV 5 moved things that this repository touches — the ArUco
module, several free functions — and `StereoSGBM`'s output changes between
releases, so a number measured on one build is not a universal constant. Anyone
reproducing a table needs to know which build produced it.

**What it costs.** The pin will age, and someone will eventually have to port
this to OpenCV 5. The comments name the specific API differences where they are
known, which is the part of that work that is hard to reconstruct later.
