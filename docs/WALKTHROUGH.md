# Walkthrough

The README is the map. This is the terrain: every stage in order, with the code
that implements it, the reasoning behind it, and the output it produces. Read it
next to a terminal — every block of output here is copied from a run of the file
named above it, so you can reproduce any of it in a few seconds.

**Contents**

1. [A 3-D point becomes a pixel](#1-a-3-d-point-becomes-a-pixel)
2. [Homogeneous coordinates, and the divide that is perspective](#2-homogeneous-coordinates-and-the-divide-that-is-perspective)
3. [Extrinsics, and the sign that catches everyone](#3-extrinsics-and-the-sign-that-catches-everyone)
4. [What a resize does to K](#4-what-a-resize-does-to-k)
5. [The lens](#5-the-lens)
6. [Making the data: why the renderer runs backwards](#6-making-the-data-why-the-renderer-runs-backwards)
7. [Calibration](#7-calibration)
8. [Reprojection error, and the shape of it](#8-reprojection-error-and-the-shape-of-it)
9. [The homography by DLT](#9-the-homography-by-dlt)
10. [Two views: F, E, and four poses](#10-two-views-f-e-and-four-poses)
11. [Rectification](#11-rectification)
12. [The block matcher](#12-the-block-matcher)
13. [The three failure modes](#13-the-three-failure-modes)
14. [Disparity to metres](#14-disparity-to-metres)

---

## 1. A 3-D point becomes a pixel

Start with the physics, which is simpler than the algebra. A sealed box with one
tiny hole in the front. Light from a point in the world reaches the back wall
along exactly one straight ray — the one through the hole — so every world point
paints exactly one spot. That is an image.

Put the origin at the hole, `Z` out along the direction the camera looks, `X`
right, `Y` **down**. `Y` down is not aesthetic: it makes the camera's Y axis and
the image row index point the same way, so no sign flip is needed anywhere
between the projection and the array indexing.

Similar triangles on the ray from `(X, Y, Z)` through the origin give
`x = f·X/Z`. Two more steps make that a pixel: divide by the physical size of a
pixel (fold it into `f`, and now `f` is measured **in pixels**), and shift the
origin from the optical axis to the top-left corner of the image (add the
principal point). Four numbers:

```python
def intrinsic_matrix(fx, fy, cx, cy, skew=0.0):
    return np.array([[float(fx), float(skew), float(cx)],
                     [0.0, float(fy), float(cy)],
                     [0.0, 0.0, 1.0]], dtype=np.float64)
```

`fx` is the focal length in pixel *widths*, `fy` in pixel *heights*; they differ
only if the pixels are not square. A lens stamped 6 mm on a sensor with 3 μm
pixels has `fx = 6 / 0.003 = 2000 px`. If your `fx` ever prints as a number
between 3 and 50 you have put millimetres into `K`, and every depth you compute
downstream is wrong by a factor of a few hundred.

`skew` models sensor rows that are not perpendicular to sensor columns. It is
zero on every camera you will meet; the parameter exists so that the layout of
the matrix is explicit rather than implied.

Run `examples/01_pinhole_projection.py`:

```
Z =  2.0 m -> pixel ( 360.0,  160.0)   offset from principal point ( +40.0,  -80.0)
Z =  4.0 m -> pixel ( 340.0,  200.0)   offset from principal point ( +20.0,  -40.0)
Z =  8.0 m -> pixel ( 330.0,  220.0)   offset from principal point ( +10.0,  -20.0)
```

Double the depth, halve the offset. That is the whole of perspective, and it is
also why a single image cannot tell you scale: a point twice as far and twice as
big lands on exactly the same pixel.

**The sanity check.** A point on the optical axis lands on the principal point at
any depth, under either image-plane convention:

```
points on the optical axis project to [320. 240.] and [320. 240.]
```

If your projection code ever fails that, nothing downstream can work, and no
amount of staring at the rest of the pipeline will help.

### The minus sign nobody mentions

The physical image plane is at `Z = -f`, *behind* the hole. Similar triangles
there give `x = -f·X/Z`, an inverted image. Every textbook writes `x = +f·X/Z`
because they quietly move the image plane to a *virtual* one at `+f`, in front of
the hole. Same rays, easier arithmetic, and nobody says out loud that a choice
was made.

`pinhole.project_real_plane` exists purely so the difference can be measured
rather than asserted:

```
max |real - reflection of virtual through (cx, cy)| = 4.55e-13 px over 200 random points
```

The two conventions differ by a **180-degree rotation about the optical axis** —
a point reflection through the principal point, not a left-right mirror, which is
what people usually guess. That distinction matters exactly once: when you derive
a projection by hand and your reconstruction comes out both upside down and
flipped, and you spend an evening looking for two bugs when there is one.

---

## 2. Homogeneous coordinates, and the divide that is perspective

Matrix multiplication can rotate, scale, shear and mirror. There is one everyday
transform it cannot do: **translation**. No 3×3 matrix times a vector ever adds a
constant — feed it zero and you get zero back.

So we append a 1 and grow the matrix. Now translation is a multiply, and the
real payoff is **chaining**: rotate into the camera frame, translate, project —
three operations of the same kind, so they multiply into one matrix.

```python
def from_homogeneous(pts):
    w = pts[:, -1:]
    return pts[:, :-1] / w
```

Three lines, and the middle one is the subject of this section. When you multiply
a point by `K` you get **three** numbers, `(su, sv, s)`, and the pixel is only the
first two divided by the third:

```
K @ X (row-dot-col)  = [720. 320.   2.]      <- NOT pixels yet
divide by the third  = [360. 160.]  <- pixels
```

That `s` came out as `2.0`, which is `Z`. **The divide is by depth, and dividing
by depth is what perspective is.** A railway track converges in a photograph
because both rails were divided by a bigger and bigger number the further down
the track you look. Perspective is not an effect bolted onto the projection; it
is the divide.

Two consequences fall straight out of it.

**Homogeneous vectors are only defined up to scale.** `(720, 320, 2)` and
`(360, 160, 1)` and `(1440, 640, 4)` are the same point. This is why the
projection equation is written with an equals sign that means "equal up to an
unknown positive scale", and why anyone reporting "my pixel coordinates are in
the tens of thousands and they get *bigger* with distance" has skipped the
divide.

**The sign of `s` matters.** A point behind the camera has `Z < 0`, so `s < 0`,
and `u = su/s` still produces a perfectly plausible pixel inside the image:

```python
    s = h[:, 2]
    valid = s > 0
    uv = np.full((X_cam.shape[0], 2), np.nan)
    uv[valid] = h[valid, :2] / s[valid, None]
```

Without that guard, `(0.1, -0.2, -2.0)` projects to `(280, 320)` — a phantom
detection, in frame, from an object behind the lens. This check has a name, the
**cheirality test**, and it comes back in section 10 as the thing that picks one
pose out of four.

---

## 3. Extrinsics, and the sign that catches everyone

`K` only works on points that are already in camera coordinates. Getting them
there is the extrinsics' job:

```
X_cam = R @ X_world + t
```

and the trap is in what `t` is. Feed the world origin through that equation and
you get `t` back, so **`t` is the world origin expressed in the camera's frame**.
It is not the camera's position. Those are different vectors pointing in roughly
opposite directions.

```python
def camera_centre(R, t):
    return -R.T @ np.asarray(t, dtype=np.float64).ravel()
```

From `examples/02_extrinsics_and_resolution.py`:

```
camera position C   = [ 0.  0. -5.]
extrinsic t = -R C  = [0. 0. 5.]     <- opposite sign, and this is the one K needs
world origin lands at pixel [320. 240.]  (the principal point)
recovered C = -R^T t = [ 0.  0. -5.]
```

Plot `t` as a trajectory instead of `C` and the path comes out inside-out, with
nothing in the maths complaining. `C = -Rᵀt` is worth memorising in that form.

`R` also has to actually be a rotation, and that is two conditions, not one:

```
R^T R = I ?  True      det(R) = +1.000000
after mirroring one column: det = -1.000000, is_rotation = False
```

Drop the determinant check and a **reflection** passes as a rotation. Your
reconstructed scene comes out mirror-imaged and every error metric stays happy
about it. And after any optimisation or hand-edit, float drift makes `R` only
approximately orthonormal, at which point it silently *shears* the points it acts
on — a slow reprojection-error creep with no single bad frame to blame. The fix
is one SVD:

```
a drifted R: max|R^T R - I| = 2.05e-03 -> after SVD re-orthonormalisation: 3.09e-16
```

---

## 4. What a resize does to K

This is the section that costs teams real money, and it is four lines of code.

`K` is expressed in pixels, so it is tied to the resolution you calibrated at.
There are two different operations people call "changing the resolution" and they
do different things:

```python
def scale_intrinsics(K, sx, sy=None):
    sy = sx if sy is None else sy
    S = np.array([[sx, 0.0, 0.0],
                  [0.0, sy, 0.0],
                  [0.0, 0.0, 1.0]])
    return S @ np.asarray(K, dtype=np.float64)
```

Write `S @ K` out on paper and you can see why all four entries scale: `S`
multiplies the whole of the top two rows, and `fx`, `fy`, `cx` and `cy` all live
there. `cx` is a pixel coordinate like any other, so it scales with the pixels.

Cropping is the other one: no pixel changed size, so the focal lengths are
untouched and only the principal point moves, by the amount discarded off the
top-left.

```
calibrated at 1920x1080:  fx= 1450.0 fy= 1452.0 cx= 962.0 cy= 541.0
resized  to  640x360   :  fx=  483.3 fy=  484.0 cx= 320.7 cy= 180.3
cropped  to 1280x720   :  fx= 1450.0 fy= 1452.0 cx= 642.0 cy= 361.0
```

And now the bug, which is scaling the focal lengths and forgetting the principal
point:

```
scaling fx,fy but not cx,cy offsets every point by (641.3, 360.7) px
predicted (1-s)*cx = 641.3, (1-s)*cy = 360.7
```

The horizontal error is **larger than the entire 640-pixel-wide image**. Your
projected point is not slightly off, it is off the frame. And because the offset
is a constant, it reads as a calibration bias rather than as the four-line code
bug it is.

**The distortion coefficients do not change in either case.** The reason is worth
being able to give on demand: distortion is defined on normalised coordinates
`x_n = (u - cx)/fx`, which divides a pixel distance by a pixel focal length, so
the scale cancels top and bottom. `tests/test_distortion.py` asserts it, and it
holds even for an anisotropic resize, because `x_n` and `y_n` have separate
denominators.

---

## 5. The lens

Everything so far assumed a pinhole. A real camera has a stack of curved glass,
and the deviation from the pinhole prediction is **distortion**: the amount by
which a straight line in the world fails to be a straight line in your image.

Two physically different effects. **Radial** distortion depends only on distance
from the optical centre and comes from the shape of the glass — negative `k1`
pulls points inward and bows lines outward (barrel, wide lenses), positive `k1`
pushes them out (pincushion, telephoto). **Tangential** distortion, `p1` and
`p2`, is the sensor sitting a fraction of a degree out of parallel with the lens;
it is small on modern cameras, and you fit it anyway, because leaving it out
pushes its error into the radial terms and corrupts those instead.

```python
def distort_normalized(x, y, D):
    k1, k2, p1, p2, k3 = np.asarray(D, dtype=np.float64).ravel()[:5]
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return x * radial + dx, y * radial + dy
```

Three things about that function.

**It works in normalised coordinates**, not pixels. There `r` runs about 0 to 1,
so `r⁴` and `r⁶` stay well behaved and the coefficients come out dimensionless.
Feed it raw pixels and `r²` is about 250000, `r⁶` overflows, and you get
coordinates in the billions.

**The order is `[k1, k2, p1, p2, k3]`** — `k3` last, after the tangential pair,
for backward-compatibility reasons that will not help you at three in the
morning. OpenCV accepts a wrongly ordered array without a word of complaint,
because all it sees is five floats.

**It runs ideal → distorted.** That direction is a closed-form polynomial. The
direction you actually want at runtime — a measured pixel, back to where a
pinhole would have put it — has no closed form at all, because you are asking to
invert a sixth-order-in-radius polynomial in two coupled variables. So it is
iterated: assume the guess is right, compute the factor it implies, undo it on
the *measured* point, repeat.

```python
    x, y = xd.copy(), yd.copy()          # start the guess AT the distorted point
    for _ in range(int(iters)):
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        ...
        x = (xd - dx) / radial
        y = (yd - dy) / radial
```

From `examples/03_distortion.py`:

```
   1 iteration(s): max residual over the unit disc = 6.07e-02
   5 iteration(s): max residual over the unit disc = 3.34e-04
  10 iteration(s): max residual over the unit disc = 5.55e-07
  20 iteration(s): max residual over the unit disc = 1.71e-12
```

Nothing clever, just repeated correction, converging fast because the correction
is small near the centre and only mildly large at the rim. Cross-checked against
the library on one point:

```
OpenCV undistortPoints: (0.599959, 0.799945)
this module, 20 iters : (0.600000, 0.800000)
truth                 : (0.600000, 0.800000)
```

**OpenCV is the one that is slightly off**, because it stops after a fixed
iteration count. Sit with that: the library's undistortion is an approximation
with a tolerance, not an exact operation.

That iteration is also why undistortion at runtime is a **lookup table**. A
1080p frame is 2.07 million pixels; five iterations each is 10.4 million
polynomial evaluations per frame, over 300 million a second at 30 fps. So you
compute, once, a map saying "output pixel `(u,v)` reads from input pixel
`(u',v')`", and every frame after that is a `cv2.remap` — memory-bound lookup,
not arithmetic.

Measured on a rendered board:

```
a straight line across the top of the frame bows by   9.05 px
max bow of a board row, distorted   :   2.10 px
max bow of a board row, undistorted :   0.14 px
```

### Where the model stops existing

The pinhole geometry underneath all of this puts an incoming ray at
`r = f·tan(θ)`. `tan` diverges at 90 degrees: between 60° and 89° the radius
grows by a factor of 33, and at 90° it is infinite. No polynomial in `r²`
approximates that. Past roughly a **120-degree full field of view** a fisheye is
not a badly fitted Brown-Conrady lens — it is a different model
(Kannala-Brandt: `r = f·θ`, four radial coefficients, no tangential terms,
`cv2.fisheye.*`). The symptom of using the wrong one is an RMS stuck at several
pixels that more views never improve, and the fix is the other API, not more
coefficients.

---

## 6. Making the data: why the renderer runs backwards

Every image in this repository is rendered by `src/geo/synthetic.py`. The reason
is in section 7, but the *mechanism* is worth a page of its own, because a
renderer that is subtly wrong would make everything downstream a tautology.

Projecting the board forward gives you the corners but not the pixels: to fill a
pixel you need to know which point of the board it sees, which is the inverse
question. So the checkerboard renderer works backwards, one ray per pixel:

```
pixel -> normalised distorted -> UNDISTORT (iteratively) -> ray
      -> intersect the board plane -> board (x, y) -> black or white
```

```python
    n = R[:, 2]
    denom = dirs @ n
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (n @ t) / denom
    P = s[:, None] * dirs                       # the board point each ray hits
    rel = P - t
    bx = rel @ R[:, 0]                          # board-frame coordinates, metres
    by = rel @ R[:, 1]
```

The board's own normal is its `+Z` axis carried over by `R`, which is the third
column, and the plane passes through `t`. So the ray meets it at
`s = (n·t)/(n·dir)` — one dot product per pixel. The first version of this code
solved the 3×3 system `[r0 r1 -dir](bx, by, s) = -t` per pixel instead; same
answer, 100× slower on the 1.2 million rays a 640×480 view needs at the
default 2× supersampling.

Two details that are not decoration:

**Supersampling.** The render happens at 2× linear resolution and is
box-averaged down. Without it the checker edges alias into staircases, sub-pixel
corner refinement latches onto the staircase, and the corner error floor stops
being about geometry and starts being about rasterisation.

**Half-pixel bookkeeping.** Sub-pixel `(i + 0.5)/ss - 0.5` is the pixel-centre
convention. Get it wrong and the whole render shifts by half a pixel, which is
exactly the size of the effect the calibration is trying to measure.

Does it work? `tests/test_synthetic.py` compares the detected corners against the
forward projection — two independently written pieces of geometry — and they
agree to **under 0.6 px on every detected view**, with a mean signed error of
about 0.01 px, so there is no systematic bias hiding in the rasteriser.

One honest wrinkle, asserted rather than hidden: a 9×6 chessboard looks identical
upside down, so the detector may return the corners in reverse order. The test
accepts either ordering. A calibration absorbs the ambiguity into the board pose;
the real fix is identified corners, which is what a ChArUco board is for.

The stereo scene is the same idea with bounded planes and procedural textures.
Each surface is `n·P = c` with a rectangle of validity, tested nearest-first
along each ray. Because it is built in 3-D rather than warped from a depth map,
**occlusion comes out for free and correctly**: a left pixel is occluded when the
right camera's ray at the corresponding column lands on a different surface.

```python
    xr = np.rint(xs - np.nan_to_num(disp, nan=1e6)).astype(np.int64)
    in_frame = (xr >= 0) & (xr < w)
    occluded[in_frame] = sid_r[ys[in_frame], xr[in_frame]] != sid_l[in_frame]
```

Note that occlusion and "the match would fall off the left edge" are kept apart.
They have the same consequence — no correct answer — but different causes: the
first is the scene, the second is the rig, and the second is exactly the left
band, as wide as the largest disparity, that every stereo system has.

---

## 7. Calibration

Calibration is a least-squares fit and nothing more exotic. You know where the
board's corners are in the board's frame. You measure where they landed in each
image. You ask for the single `K` and `D`, plus one pose per view, that best
explain every landing at once, where "best" means smallest sum of squared pixel
residuals.

The call is four lines. The reason this module exists is everything around it.

### Every tutorial ends here, and it is not enough

> If your RMS is under half a pixel, your calibration is good.

That is wrong, and it is wrong in a way that silently corrupts every metric
number downstream. RMS says how well the model fits **the data you happened to
collect**. It says nothing about whether that data contained enough information
to pin the parameters down.

The specific failure is a confound between focal length and distance. Look at the
projection equation: `u = fx·X/Z + cx`. A board at distance `Z` seen with focal
length `f` produces the same image as a board at `2Z` seen with `2f`. If every
view is fronto-parallel, **nothing in your data separates those hypotheses**, and
the solver slides freely along that direction while fitting your corners to a
hundredth of a pixel. Tilt the board and the confound breaks, because a tilted
plane's foreshortening depends on `f` and `Z` differently.

`examples/04` runs both protocols against known truth, ten seeds each, with the
noise drawn from its own generator so that both protocols see the *same* noise
realisation and the only difference is the capture geometry:

```
seed |   GOOD fx  GOOD RMS |    BAD fx   BAD RMS
   0 |     800.0     0.067 |     775.5     0.067
   ...
   7 |     800.0     0.067 |    2910.1     0.131
   ...
GOOD fx spans 798.9 to 800.8   (true 800.0, a 1.9 px window)
BAD  fx spans 760.1 to 2910.1   (a 2150.1 px window)
every RMS in the table is between 0.066 and 0.131 px
```

Read that twice. The bad capture is **not biased** — it lands under the truth on
some seeds and 264% over it on another. It is *unconstrained*, and an
unconstrained parameter goes wherever the noise pushes it. Every reprojection
error in the table is under a seventh of a pixel. No threshold you could set
separates the two columns.

Propagate the worst one into a stereo depth:

```
  true             fx =   800.0 -> Z = 3.200 m
  worst BAD seed   fx =  2910.1 -> Z = 11.640 m
```

The focal-length error passes straight into every depth, undiluted. It does not
average out over frames, it does not shrink with more measurements, and it does
not look like noise — it looks like a clean scale factor, which is exactly why
you will blame your baseline measurement instead of your calibration.

### The diagnostics that catch it

Since RMS cannot tell you, three things that can:

```python
def board_tilt_degrees(rvecs):
    for rv in rvecs:
        R, _ = cv2.Rodrigues(...)
        tilts.append(np.degrees(np.arccos(min(1.0, abs(R[2, 2])))))
```

The board's normal is `(0,0,1)` in its own frame, so in the camera frame it is
the third column of `R`, and the tilt is the angle between that and the optical
axis. A median tilt below about 15 degrees means `f` and `Z` are confounded, and
no number of extra frames will help — the fix is to pick the board up and angle
it.

`corner_coverage` counts how many cells in the outer 20% ring of the frame hold
no corner at all. The radial coefficients are estimated almost entirely from
corners near the frame edge, because that is where `r` is large and where the
polynomial does anything.

`fov_degrees` is the independent plausibility check, and the only one here that
uses information from outside the calibration. A ray at the edge of frame has
`X/Z = tan(FOV/2)` and lands `width/2` px from the principal point, so
`FOV = 2·atan((width/2)/fx)`. If that disagrees with the number printed on the
box the camera came in, the calibration is wrong regardless of its RMS. Note the
direction, because it is easy to get backwards: **smaller `fx` means a wider
lens**.

### The result

Rendered from a known `K` and `D`, recovered from the detected corners:

```
RMS reprojection error: 0.0720 px over 13 views
             true    recovered      error       rel
fx        800.000      799.393     -0.607   -0.076%
fy        802.000      801.385     -0.615   -0.077%
cx        325.000      323.678     -1.322   -0.407%
cy        238.000      238.033     +0.033   +0.014%

D true      = [-0.3     0.1     0.0012 -0.0009  0.    ]
D recovered = [-0.29885  0.09427  0.00133 -0.00077  0.01347]
```

`k2` and `k3` moved, and that is expected rather than alarming: `r⁴` and `r⁶` are
nearly collinear over the radius an image spans, so the solver trades one against
the other freely. Do not compare the coefficients — compare the **curve**:

```
  max disagreement between the true and recovered undistortion, over the
  whole frame: 0.332 px  (mean 0.071 px)
```

The diagnostics also report honestly that this capture is thin at the edges — 41%
of border cells hold no corner — which is exactly why `k2` and `k3` are the
parameters that moved. A full-board chessboard detector refuses any view where
the board clips the frame, which pulls directly against the advice to push
corners to the edge. That tension is real, it is why ChArUco boards exist, and it
is not papered over here.

---

## 8. Reprojection error, and the shape of it

The definition, written out once so the number stops being an output and becomes
a quantity:

```python
        proj, _ = cv2.projectPoints(objp, rvec, tvec, K, D)
        d = proj.reshape(-1, 2) - imgpoints
        out.append(float(np.sqrt((d ** 2).sum(axis=1).mean())))
```

Take a known 3-D corner, push it through the estimated pose and the estimated
model, subtract the pixel you actually detected. That vector is the residual.
Square, average, root:

```
aggregate RMS from cv2.calibrateCamera : 0.07352 px
the same number recomputed by hand     : 0.07352 px
```

Rules of thumb for a real camera, in pixels of RMS: under 0.3 is a good
calibration on a decent detector; 0.3 to 1.0 is usable but look at the per-view
spread first; over 1.0 means something is wrong — pattern size, blur, a bad view,
the wrong board. All three are statements about the **fit**, and section 7 is why
none of them is a statement about correctness.

The more useful thing is the residual's *structure*. `examples/05` fits the same
data twice, once with the full model and once with `k1` only:

```
  full model  RMS = 0.0735 px    k1-only RMS = 0.0856 px

mean |residual| against distance from the principal point:
  r (px)   full model    k1 only
      20       0.0411     0.0477
     216       0.0551     0.0810
     294       0.0505     0.1189
```

As a summary the two are almost the same. As a *shape* they are not: the full
model's residuals are flat with radius and the under-parameterised one's climb,
because the terms that were removed are the ones that act at the frame edge. A
healthy residual cloud is an isotropic blob centred on zero. **Structure means
the model is missing something, and where the structure lives says which
something.**

The per-view breakdown matters for the same reason — an aggregate hides a single
bad view inside twenty good ones:

```
per-view RMS: median 0.0418 px, worst view #3 at 0.1401 px (3.4x the median)
```

A view at three times the median is worth investigating: a blurred frame, a bad
detection. Dropping views until the number looks nice is not calibration; the
difference is whether you can say *why* the view was bad.

---

## 9. The homography by DLT

A plane imaged by a pinhole camera maps to the image by a single 3×3 matrix.
That is why calibration is possible at all: each view of a board gives you one
homography, and `K` is what every homography in the set has in common.

The constraint per correspondence: we want `(u', v', 1)` **parallel** to
`H·(u, v, 1)` — parallel, not equal, because homogeneous vectors are only defined
up to scale. Parallel means a zero cross product, and writing that out gives two
independent rows per correspondence:

```python
    A[0::2, 0] = -u
    A[0::2, 1] = -v
    A[0::2, 2] = -1.0
    A[0::2, 6] = up * u
    A[0::2, 7] = up * v
    A[0::2, 8] = up
```

Four correspondences give eight rows, and `vec(H)` has nine unknowns of which one
is overall scale, so four is exactly enough. With more, the system is
overdetermined and inconsistent under noise, so take the `h` minimising `‖Ah‖`
subject to `‖h‖ = 1` — the right-singular vector with the smallest singular
value, which is the last row of `Vt`:

```python
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
```

That pattern — stack the constraints, take the smallest singular vector — is the
single most reused idea in geometric vision. The eight-point algorithm in the
next section is the same three lines with a different `A`, and so is
triangulation in section 14.

### Normalisation is not a detail

```python
def normalizing_transform(pts):
    mean = pts.mean(axis=0)
    centred = pts - mean
    rms = np.sqrt((centred ** 2).sum(axis=1)).mean()
    scale = np.sqrt(2.0) / max(rms, 1e-12)
```

Translate to zero mean, scale so the mean distance from the origin is `√2` (which
puts the average point at `(1,1)`, so all three homogeneous components are the
same size). Without it, a design matrix built from raw pixels mixes entries of
order `u·u ≈ 1e5` with entries equal to 1, and its smallest singular value is
decided by float rounding rather than by geometry:

```
condition number of the DLT design matrix, normalised   :       3.56
condition number of the DLT design matrix, raw units    :   1.59e+05
```

### Against the library

```
  exact correspondences: max |H_dlt - H_cv2| = 4.54e-06
  exact correspondences: DLT transfer error  = 1.91e-13 px  (cv2: 7.72e-06 px)
```

On exact data the from-scratch DLT is seven orders of magnitude more exact than
`cv2.findHomography`. That is not a defect in OpenCV: the DLT minimises
*algebraic* error `‖Ah‖`, which has no units, while `findHomography` runs
Levenberg-Marquardt on *reprojection* error afterwards and stops on a tolerance.
On noisy correspondences their difference is about 1e-3 and OpenCV's is the
better estimate. Knowing which is which — and that the DLT is the initialiser
rather than the answer — is the actual lesson.

Finally, the map used for something. Warping a slanted board back to
fronto-parallel with the hand-rolled `H`:

```
square spacing in the warped image: mean 55.00 px, std 0.070 px
expected 55.00 px from the 25 mm squares and the 2.2 px/mm output scale
```

A perspective view has square spacing that shrinks with distance; after the warp
it is constant to seven hundredths of a pixel. That constancy is the check.

---

## 10. Two views: F, E, and four poses

Pick a pixel in the left image. Where is its match in the right? The naive
answer is "search the whole right image" — a 2-D search, millions of comparisons
per pixel. Epipolar geometry says you never have to.

The argument is worth holding rather than memorising. You do not know how far
away the 3-D point is, so it could sit anywhere along the ray through that pixel.
Look at that entire ray from the *other* camera's position. A straight line in
3-D projects to a straight line in an image. So every possible position of the
point lands somewhere on **one line** in the other image. The match is on that
line and cannot be anywhere else.

Written down, "x' lies on the line Fx" is a single dot product: `x'ᵀ F x = 0`.

- **F**, the fundamental matrix: raw pixels, no calibration needed, 7 degrees of
  freedom.
- **E**, the essential matrix: normalised coordinates, needs `K`, 5 degrees of
  freedom, and it factors into the relative rotation and translation.
- `E = K'ᵀ F K`, and both are **rank 2**.

Rank 2 is not trivia. It is exactly what forces every epipolar line in an image
to pass through one point — the **epipole**, which is the image of the other
camera's centre. A full-rank `F` describes a geometry that does not exist, and
its lines miss their points by tens of pixels.

```python
def essential_from_Rt(R, t):
    return skew(t) @ np.asarray(R, dtype=np.float64)
```

Not a black box: the epipolar constraint says the two rays and the baseline are
coplanar, the triple product of three coplanar vectors is zero, and writing that
triple product as a matrix is exactly `[t]ₓ R`.

### The eight-point algorithm

`x'ᵀ F x = 0` is linear in the nine entries of `F`, so each correspondence gives
one row:

```python
    A = np.column_stack([u2 * u1, u2 * v1, u2,
                         v2 * u1, v2 * v1, v2,
                         u1, v1, np.ones_like(u1)])
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)
    F = enforce_rank2(F)
```

The second half of the algorithm is that `enforce_rank2`. The solve knows nothing
about rank, so under noise its null vector reshapes into a full-rank matrix.
Zeroing the smallest singular value gives the Frobenius-closest rank-2 matrix.
It is not a polish step; without it the epipolar lines do not concur.

There is an ordering subtlety in the code worth pointing at:

```python
    F = enforce_rank2(F)

    # Denormalise.  The rank-2 projection has to happen in the NORMALISED
    # frame, before this line: rank is preserved by the sandwich, but the
    # closest rank-2 matrix in the badly-scaled frame is not the closest one in
    # the well-scaled frame, which is the subtlety that makes the ordering of
    # these two lines matter.
    F = T2.T @ F @ T1
```

### What normalisation is worth here

From `examples/07`, mean symmetric epipolar distance in pixels, scored against
clean correspondences:

```
  noise |  normalised  raw pixels  cv2 8POINT  raw/norm
   0.00 |      0.0000      0.0000      0.0000       8.5x
   0.25 |      0.1074      0.3445      0.1074       3.2x
   1.00 |      0.4321      2.7387      0.4321       6.3x
```

With no noise both solves are exact, which is the trap — normalisation looks
unnecessary until real data arrives. And our normalised solve lands exactly on
`cv2.findFundamentalMat`, because that function normalises internally and never
mentions it.

The size of the penalty is not a constant of nature. What decides it is how far
the points' centroid sits from the coordinate origin relative to their spread —
which is bookkeeping, not geometry:

```
 offset (px)  normalised   raw pixels   raw/norm
           0      0.2153       0.8491         4x
        2000      0.2153      85.5598       397x
        8000      0.2153     617.6451      2869x
```

The normalised column does not move by a digit. That is what "the estimate should
not depend on where you put pixel (0,0)" means as a number.

### E does not give you one pose

The part that separates people who have read the geometry from people who have
read the API docs. The SVD of `E` gives **four** candidates, not one: two
rotations `R₁ = U W Vᵀ` and `R₂ = U Wᵀ Vᵀ`, each with `±t`.

All four satisfy the epipolar constraint **exactly**, because the constraint is a
statement about lines and a line does not care which side of the camera its point
is on. Geometrically: negating `t` flips which camera is in front; `R₂` is the
**twisted pair**, `R₁` rotated 180° about the baseline. Demonstrated rather than
asserted:

```
the two rotations differ by a 180.000 degree rotation about [ 0.994  0.078 -0.077]
the baseline direction is [-0.994 -0.078  0.077]; |axis . t_hat| = 1.000000
```

Exactly one candidate is physically real, and the test that finds it is the
**cheirality check** from section 2: triangulate a point under each candidate and
keep the one with positive depth in **both** cameras.

```
cheirality votes (points in front of BOTH cameras) per candidate: [0, 0, 40, 0] of 40
rotation error of the winner : 0.000001 deg
translation direction error  : 0.000001 deg
```

One candidate takes all forty points and the other three take none. Note *which*
candidate won: the third, which is `R₂` — the twisted pair. Which of `R₁` and
`R₂` is the true rotation is **not fixed**; cheirality is what tells you, and the
ordering is not.

`cv2.recoverPose` runs this loop inside one call, which is precisely why so few
people know the four solutions exist and precisely why interviewers ask.

### And the scale it will not give you

```
|t| recovered = 1.0000   |t| true = 0.5598
missing scale factor = 0.5598
```

`E` fixes the **direction** of the baseline and nothing about its length. For a
stereo rig with a measured baseline that costs nothing — multiply through by `B`.
For a moving single camera it is fatal, because every frame pair gets its own
arbitrary unit and the trajectory has no consistent scale *between segments*.
That is a worse problem than drift, and it is the problem monocular odometry
exists to solve.

---

## 11. Rectification

The epipolar constraint already reduced matching from 2-D to 1-D. But a diagonal
line through a pixel grid is awkward: you interpolate to walk it, you accumulate
rounding error, and every candidate costs a resampled patch. What you want is for
that line to be **a row**.

Rectification is the warp that manufactures that condition. Two words that get
used interchangeably and must not be:

- **Undistortion** removes lens curvature. One camera, no reference to another.
- **Rectification** removes lens curvature **and** rotates both virtual image
  planes until they are coplanar and row-aligned. It is a *pair* operation —
  "aligned with what?" has no answer for a single camera.

Rectification *includes* undistortion. Doing both in sequence undistorts twice
and warps the image into nonsense.

```python
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2, size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha)
    map1 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, size, cv2.CV_32FC1)
    map2 = cv2.initUndistortRectifyMap(K2, D2, R2, P2, size, cv2.CV_32FC1)
```

The maps are built once at startup; `cv2.remap` runs per frame. That split is the
practical consequence of section 5's "the inverse has no closed form".

### The focal length changes

You calibrated. You got an `fx`. You feel you own that number. `stereoRectify`
re-chooses it, because the two cameras have to end up sharing one focal length or
a row in one image would not correspond to a row in the other:

```
original fx from K   : 520.0000 px
rectified f (P1[0,0]): 555.2688 px
difference           : +6.78%

at d = 30 px: correct Z = 2.2251 m, using the pre-rectification fx = 2.0838 m (-6.35%)
```

A constant percentage error in every depth looks **exactly** like a mis-measured
baseline, so you will go and re-measure your rig with calipers for an hour and
find nothing wrong with it. The number that belongs in `Z = fB/d` is `P1[0,0]`,
which is also `Q[2,3]`.

### Rectification fails quietly, so you must instrument it

If your calibration is mediocre, rectification does not throw. It does not
produce a black image or a visible tear. It produces a slightly wrong warp, and
your block matcher then confidently matches the wrong row against the wrong row
and returns a disparity map that looks smooth, structured and believable, and is
wrong. A loud failure costs ten minutes; this one costs two days.

So build a number: match features between the two **rectified** images and
measure `|y_left − y_right|`.

```
                         median     mean      p90  matches
before rectification      12.00    16.24    17.28      342
after rectification        0.00     7.90     9.60      363
```

**Gate on the median.** Stereo feature matching always leaks a few gross
mismatches, and a handful wrong by thirty pixels drags the mean over any
threshold while nine-tenths of the matches sit at zero. Median under 0.5 px:
proceed. Median over 1 px: fix the calibration before looking at a single
disparity map. A mean far above the median is a *matching* failure, not a
rectification one.

That last rule is testable here, because the scene has ground truth. Push known
3-D points through `P1` and `P2` directly:

```
on 400 ground-truth correspondences: median |dy| = 2.48e-13 px, max = 4.55e-13 px
```

The warp is exact to machine precision. Every pixel of that ORB mean is a wrong
match, most of them on the striped band where every stripe looks like every other
stripe. The rule diagnosed it correctly.

---

## 12. The block matcher

Two rectified images. For a left pixel at column `x`, the match is at `x - d` on
the same row for some non-negative integer `d`. The whole problem is: for every
pixel, choose a `d`.

### The cost volume

The obvious implementation is four nested loops — for each pixel, for each
disparity, compare a patch. On this 480×320 scene with 48 disparities that is 7.4
million patch comparisons in interpreted Python, and you will wait minutes per
run. Iteration speed is the difference between finishing this section and
abandoning it.

Turn it inside out: **for each disparity, score every pixel at once.**

```python
    for d in range(ndisp):
        diff = np.full((h, w), worst, dtype=np.float32)
        if d < w:
            delta = L[:, d:] - R[:, :w - d]
            diff[:, d:] = delta * delta if metric.upper() == "SSD" else np.abs(delta)
        vol[d] = cv2.boxFilter(diff, -1, (window, window), normalize=True,
                               borderType=cv2.BORDER_REPLICATE)
```

Forty-eight vectorised passes instead of 7.4 million interpreted ones, and it
runs in 0.07 s. What you build in the process is the **cost volume**: a
`D × H × W` array where entry `(d, y, x)` is the cost of assigning disparity `d`
to pixel `(x, y)`. That definition is the single most transferable idea in this
subject — it is the object at the centre of every learned stereo network, and the
same `D × H × W` shape reappears as segmentation logits with `D` = classes.

Note the initialisation. For `x < d` there is no right-image pixel to compare
against, and filling that band with zeros would make it the **cheapest** match
everywhere — a stripe of confident nonsense down the left edge of every disparity
map. It is filled with the maximum possible cost instead, and
`tests/test_stereo.py` asserts that the invalid band is the most expensive place
in the volume.

### Winner-take-all, and what it omits

```python
best = np.argmin(vol, axis=0)
```

One line. Note the word *independently*: WTA makes no use of the fact that
neighbouring pixels in the real world are usually at similar depths. That
omission is exactly what semi-global matching fixes, and holding onto it now is
what lets you name the gap later instead of hand-waving about "SGBM being
better".

### Sub-pixel

The true minimum of the cost curve almost never falls on an integer. Fit a
parabola through the three costs around the winner and take its vertex:

```python
    den = c0 - 2.0 * c1 + c2
    bad = den <= 1e-6
    shift = 0.5 * (c0 - c2) / np.where(bad, 1.0, den)
    shift[bad] = 0.0
    return k[0].astype(np.float32) + np.clip(shift, -0.5, 0.5).astype(np.float32)
```

Two guards, both load-bearing. A non-positive denominator means the three samples
are flat or *concave* — there is no minimum to refine — so the integer winner is
kept rather than a division by nearly-zero being clamped into a plausible-looking
number. And a vertex more than half a pixel from the winner is not a refinement
of that winner, so the shift is clipped.

Measured on the textured surfaces, same pixels for both:

```
  integer winner-take-all : MAE 0.7481 px
  parabolic sub-pixel     : MAE 0.6059 px
```

Three array lookups for a 19% error reduction. And it is not cosmetic in metres:
at the far end of this scene, three tenths of a pixel is 282 mm.

### The left-right check

Some pixels visible on the left are simply not present on the right. Stand behind
a lamp post: the strip of wall it hides from the right camera has no correct
match at all. For those pixels, any disparity a matcher reports is a fabrication.

```python
    volR = np.full_like(vol, np.inf)
    for d in range(ndisp):
        volR[d, :, :w - d] = vol[d, :, d:]
```

The right-to-left volume is **a shift of the one you already have**, not a second
matching pass: `cost_R[d,y,x]` and `cost_L[d,y,x+d]` describe the same pair of
pixels indexed from opposite ends. Recomputing it doubles the runtime for
nothing, and in a learned-stereo context the equivalent mistake doubles the
memory.

Then keep a pixel only if the two maps point at each other:

```
of the occluded pixels, the LR check rejected :  68.8%
of the non-occluded, it rejected              :  18.0%
```

### The result

Whole image, matchable pixels only:

```
matcher                 density   MAE px  RMSE px  bad>1px  time s
SAD, WTA only             91.7%    6.585   11.276    42.7%    0.16
SAD + sub-pixel + LR      77.2%    4.848    9.023    33.2%    0.14
cv2.StereoSGBM            75.3%    3.566    6.954    28.1%    0.03
```

And on the textured surfaces — the three where the images actually contain an
answer:

```
SAD + sub-pixel + LR      90.1%    0.354     2.0%
cv2.StereoBM              88.4%    0.160     0.7%
cv2.StereoSGBM            91.2%    0.298     1.3%
```

Two different stories, and both are true. Where the data supports matching, this
matcher is within 0.056 px of SGBM. Where it does not, SGBM's smoothness term
lets a confident neighbour pull an ambiguous pixel to the right answer, and a
per-pixel argmin has nothing to pull with.

Be precise about what SGBM does differently, because there are three things and
not one: a Birchfield-Tomasi sub-pixel matching cost rather than SAD, semi-global
aggregation along scanline directions (five by default, eight with `MODE_HH`),
and the post-filters this configuration switches on — `uniquenessRatio`,
speckle filtering, and its own left-right check.

Finally, a small decision with large consequences: invalid disparities are
`NaN`, never 0. Zero is a legal disparity meaning "infinitely far away", so a map
that encodes ignorance as zero has silently put a horizon behind every occlusion,
and every mean taken of it afterwards is wrong.

---

## 13. The three failure modes

Fix these in your head, because they are genuinely different diseases and
separating them is the difference between "the depth map is noisy" and a
diagnosis.

- **Textureless** — the cost curve is *flat*. Every disparity looks equally good.
  There is an answer; the data does not contain it.
- **Repeated pattern** — the curve has *several equal minima*. There are too many
  answers, and the matcher picks one confidently.
- **Occlusion** — there is *no answer*. The pixel is not in the other image.

`examples/09` probes one pixel deep inside each region — chosen by distance
transform, because a probe near a boundary measures the boundary, not the region
— and prints its cost column:

```
probe                 pixel   true d   argmin  cost spread  2nd min gap
wall               (229,99)     8.32        8        16.10         7.14
textureless         (88,39)     8.32       45         0.38         0.04
repeated           (87,281)     8.32       22       146.67         2.43
occluded          (264,101)     8.32       40        41.46         0.96
```

Read the last two columns together. The well-textured pixel has a large spread
and a large gap to its runner-up: one clear answer. The flat patch has a spread
of **0.38 total** — every disparity costs the same, so the argmin is decided by
sensor noise. The stripe pixel has a huge spread *and* a tiny gap: several
answers look equally good, which is the worst case of the three, because the
matcher reports one with no hint that it was a coin toss.

By region:

```
region          true disp   density    MAE px   bad>1px
near_slab           24.00     99.6%     0.045      0.2%
ramp                20.99    100.0%     0.077      0.2%
textureless          8.32     35.9%    13.379     82.0%
repeated             8.32     86.6%    13.154     95.4%
wall                 8.32     76.8%     0.874      5.2%
```

The two failures have nearly the same error and completely different densities.
The textureless patch is wrong *and mostly refuses to answer* — 36% density. The
stripes answer for 87% of their pixels and are wrong about 95% of them.
**Confident nonsense is worse than an admitted gap**, and the two need different
fixes: texture projection for the first (which is what structured light *is*), a
larger window or a global smoothness prior for the second.

---

## 14. Disparity to metres

Put the left camera at the origin and the right at `(B, 0, 0)`, both looking down
`+Z` with parallel axes — which is exactly what rectification manufactures. A
point at `(X, Y, Z)` projects to `x_L = f·X/Z + cx` and `x_R = f·(X−B)/Z + cx`.
Subtract:

```
d = x_L - x_R = f*B/Z        ->        Z = f*B/d
```

Read what that says: depth is **inversely** proportional to disparity. Everything
uncomfortable about stereo follows from where `d` sits in that fraction.

```python
def depth_from_disparity(d, f, baseline, doffs=0.0):
    den = d + doffs
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = np.where(den > 0, f * baseline / den, np.inf)
    return Z
```

Non-positive denominators return `inf` deliberately: a disparity of zero is a
point at infinity, and dividing by it silently would put a wall at 1e17 metres in
the middle of your point cloud.

### The term the clean derivation drops

That derivation assumed both cameras share a principal point, so the two `cx`
terms cancel. Real rigs do not, and what survives is a constant,
`doffs = cx_right − cx_left`. On a real published rig:

```
  d (px)   correct Z  ignoring doffs      error
  200.00      2.368 m          3.841 m        62%
   60.00      4.167 m         12.802 m       207%
   28.16      5.037 m         27.277 m       442%
```

62%, then 207%, then 442% — the error **grows with distance**, because `doffs` is
a constant added to a shrinking denominator. A whole scene 2.4 m deep reads as
27 m at the back, and a sanity check on one nearby object passes. After a
`stereoRectify` with `CALIB_ZERO_DISPARITY` it is zero by construction, which is
one of the real reasons that flag exists.

### The law

Differentiate `Z = fB/d` and substitute back:

```
|dZ| = (Z^2 / (f*B)) * |dd|
```

**Depth error grows with the square of depth.** For `f = 800 px`, `B = 12 cm`:

```
 d (px)     Z (m)   Z at d-1   Z at d+1    spread
     30     3.200      3.310      3.097     0.214
      4    24.000     32.000     19.200    12.800
```

One pixel of matching error — the same one pixel — is 21 cm at 3.2 m and 12.8 m
at 24 m. `12.800 / 0.214 = 60`, against the `(24/3.2)² = 56` the square law
predicts; the small gap is because one pixel is not an infinitesimal. And notice
the interval is **asymmetric**: `Z = fB/d` is a hyperbola, so the far side runs
away faster than the near side approaches, and quoting a symmetric `±` at range
understates the side that matters.

The three levers are all in the formula and there is no fourth: a longer baseline
`B`, a longer focal length `f`, and better sub-pixel matching `dd`. That is why
section 12 spent effort on a parabola fit.

Measured off a real disparity map, at three depth bands, against what the law
predicts from the *measured* disparity error at that range:

```
      depth band       n  median |dd|  median |dZ|   predicted
       2.5-3.0 m   34730       0.0232       0.0026      0.0025
       3.0-3.5 m   12623       0.0578       0.0095      0.0095
       7.4-7.6 m   26425       0.0892       0.0812      0.0804
```

They track. The depth error is not a property of the matcher; it is the matcher's
error passed through a `1/d` that steepens as `d` shrinks. Note that the measured
error grows *faster* than `Z²` alone across those bands — because `|dd|` grows
too, from 0.02 px on the near slab to 0.09 px on the low-contrast back wall. The
square law is what sits on top of whatever your matcher does, not instead of it.

### Q, and what it generalises

Depth alone gives you a distance image. For a 3-D point per pixel you also need
`X = (u − cx)·Z/f` and `Y = (v − cy)·Z/f`. OpenCV packs all three into one 4×4
matrix, and multiplying it out by hand once is worth more than reading its
docstring:

```
(u, v, d, 1) = (400, 300, 40, 1) ->
  X' = 400 - 320.2591 = 79.7409
  Y' = 300 - 240.2001 = 59.7999
  Z' = 463.7446
  W' = 8.3333 * 40 = 333.3333
  divide: X = 0.2392 m, Y = 0.1794 m, Z = 1.3912 m
cross-check straight from the depth formula: Z = 463.7446 * 0.12 / 40 = 1.3912 m
```

Identical, because it is the same equation. `Q` is `Z = fB/d` generalised to
recover `X` and `Y` too, with the principal point folded into the first two rows
and `−1/Tx` in the bottom.

### Triangulation, and the picture that is wrong

Sometimes you do not have a dense disparity map — you have two projection
matrices and one point seen in both images. The picture everyone draws is two
rays meeting at the point.

**That picture is wrong, and how it is wrong is the lesson.** Two lines in 3-D
have four degrees of freedom of relative position, and requiring them to intersect
imposes one equation on those four: a measure-zero condition. Perfect
measurements make them meet by construction. Any error at all — half a pixel of
noise, or merely reporting an integer pixel — destroys it:

```
clean observations: gap between the two rays = 2.92e-16 m
with 0.5 px of noise on each: gap = 6.51 mm
```

There is no intersection to compute. Triangulation has to **minimise** something,
and the DLT is the same construction as section 9 with a different `A`:

```python
        A = np.vstack([pts1[i, 0] * P1[2] - P1[0],
                       pts1[i, 1] * P1[2] - P1[1],
                       pts2[i, 0] * P2[2] - P2[0],
                       pts2[i, 1] * P2[2] - P2[1]])
        _, _, Vt = np.linalg.svd(A)
        Xh = Vt[-1]
```

That is what `cv2.triangulatePoints` does internally, and the two agree to
`1.7e-16 m`. The honest caveat is the same as before: this minimises algebraic
error, which has no geometric meaning, while the statistically correct estimate
minimises reprojection error in both images — usually via a couple of
Gauss-Newton steps from this result.

One last measurement, and it is the one to carry into any fusion work:

```
over 400 noise draws, standard deviation of the triangulated point:
  sigma_X =   2.31 mm
  sigma_Y =   2.11 mm
  sigma_Z =  21.44 mm   (9x the lateral spread)
```

Half a pixel of noise costs you two millimetres laterally and twenty-one in
depth. The uncertainty of a triangulated stereo point is an **elongated ellipsoid
pointing down the viewing ray**, never a sphere. Anyone fusing stereo points with
another sensor under an isotropic covariance is throwing away the single most
important thing they know about the measurement.

---

## Where to go from here

Things worth trying in this repository, roughly in order of how much they teach:

1. **Break the calibration on purpose.** Set `tilt_sigma=0.02, spread=False` in
   `examples/04` and watch a 0.07 px RMS sit on top of a focal length that is
   hundreds of pixels wrong.
2. **Take the rank-2 projection out of `eight_point`** and draw the epipolar
   lines again. They stop concurring.
3. **Change the window size in the block matcher.** Run `examples/08` with
   `WINDOW` at 5, 9 and 15 and compare the printed errors. A larger window is
   more distinctive (fewer false matches on the low-contrast wall) and blunter
   at depth discontinuities (fatter halos around the near slab); the region
   table in `examples/09` separates the two effects.
4. **Widen the baseline in `render_stereo_pair`** and watch the near slab leave
   the disparity range while the far wall gets more precise. That trade is the
   whole of stereo rig design.
5. **Add a smoothness term.** Aggregate the cost volume along scanlines with a
   penalty for disparity changes and see how much of the gap to `StereoSGBM`
   closes on the textureless patch.
