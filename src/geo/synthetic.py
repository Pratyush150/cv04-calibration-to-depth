"""Synthetic scenes with exact ground truth: checkerboards and a stereo pair.

Why this file exists at all is the central methodological choice of the repo.
A calibration demo built on photographs of a real board can show you a
reprojection error and nothing else - you have no independent access to the
camera's true focal length, so "the calibration worked" is an assertion.  Here
the intrinsics, the distortion coefficients, the board poses and the per-pixel
depth are all CHOSEN, then rendered, then recovered by the same code path a
real image would take.  Every claim downstream is checkable against a number
that was known before the algorithm ran.

The cost is honest and worth stating: synthetic images have no sensor noise
model, no motion blur, no rolling shutter, no printing error in the board, and
no chromatic aberration.  A calibration that works here is a NECESSARY
condition for one that works on a real rig, not a sufficient one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import distortion as dist


# --------------------------------------------------------------------------
# Checkerboard: object points, poses, and rendering through a real lens model
# --------------------------------------------------------------------------

def board_object_points(pattern: tuple[int, int], square: float) -> np.ndarray:
    """The board's INNER corners in the board's own frame, Z = 0, in metres.

    `pattern` is the count of inner corners, (cols, rows) - NOT the count of
    squares.  A board printed as 10x7 squares has 9x6 inner corners, because an
    inner corner is where four squares meet and the outer edge has none.  Every
    OpenCV chessboard call takes the inner-corner count, and getting it wrong
    makes findChessboardCorners return False on perfectly good images.
    """
    cols, rows = pattern
    objp = np.zeros((cols * rows, 3), dtype=np.float64)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * float(square)
    return objp


def board_poses(n_views: int, *, tilt_sigma: float, spread: bool, seed: int = 17,
                pattern: tuple[int, int] = (9, 6), square: float = 0.025,
                z_range: tuple[float, float] = (0.42, 0.62),
                centre_span: tuple[float, float] = (0.26, 0.20)):
    """Generate `n_views` board poses as (rvec, tvec) pairs.

    The two capture protocols this function can produce are the point of the
    whole calibration example, and they differ in exactly one structural way:

      spread=True, tilt_sigma ~0.35  - the board tilted 20-40 degrees in varied
        directions and swept across the frame.  Tilting breaks the confound
        between focal length and distance, because a tilted plane's
        foreshortening depends on f and Z differently.  Sweeping puts corners
        near the frame edge, which is the only place the radial coefficients
        are observable.

      spread=False, tilt_sigma ~0.02 - the board fronto-parallel and centred.
        Every view now satisfies u = fx*X/Z + cx with f and Z appearing only as
        a ratio, so a board twice as far with twice the focal length produces
        the same image and the solver cannot tell them apart.  It still fits
        the corners beautifully; the parameter is simply not observable.

    The board's CENTRE is aimed at the requested position, not its origin
    corner: tvec = target - R @ centre.  Aiming the origin corner instead makes
    the sweep lopsided, because the corner is 100 mm from the centre and the
    board then leaves the frame on one side long before it reaches the other.
    `centre_span` is in normalised coordinates (x/z), so at fx = 800 the default
    span of 0.26 moves the board centre up to 208 px either side of the
    principal point.  That is deliberately aggressive: it pushes corners out to
    where the radial terms are observable, at the cost of some views in which
    the board clips the frame and the detector - which needs the whole board -
    refuses.  Losing a few views to get corners near the edge is the right
    trade, and it is the trade a ChArUco board would let you avoid.
    """
    rng = np.random.default_rng(seed)
    ctr = board_object_points(pattern, square).mean(axis=0)
    poses = []
    for _ in range(n_views):
        rvec = rng.normal(0.0, tilt_sigma, 3)
        R, _ = cv2.Rodrigues(rvec)
        z = rng.uniform(*z_range)
        if spread:
            nx = rng.uniform(-centre_span[0], centre_span[0])
            ny = rng.uniform(-centre_span[1], centre_span[1])
            target = np.array([nx * z, ny * z, z])
        else:
            target = np.array([0.0, 0.0, z])
        poses.append((rvec, target - R @ ctr))
    return poses


def render_checkerboard(K: np.ndarray, D: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
                        size: tuple[int, int], *, pattern: tuple[int, int] = (9, 6),
                        square: float = 0.025, supersample: int = 2,
                        margin_squares: float = 0.7, noise_sigma: float = 1.2,
                        undistort_iters: int = 12, seed: int = 0) -> np.ndarray:
    """Render one view of a checkerboard through the FULL camera model.

    This is an inverse (backward) renderer, and it has to be.  Projecting the
    board forward onto the image gives you the corners but not the pixels: to
    fill a pixel you need to know which point of the board it sees, which is
    the inverse question.  So for every output pixel:

        pixel -> normalised distorted -> UNDISTORT (iteratively) -> ray
              -> intersect the board plane -> board (x, y) -> black or white

    Doing it this way means the distortion in the rendered image is the exact
    inverse of the undistortion the calibration will later apply, so a
    successful recovery of D is a real round trip and not a tautology.

    `supersample` renders at N times the linear resolution and box-averages
    down.  Without it the checker edges alias into staircases, sub-pixel corner
    refinement latches onto the staircase, and the corner error floor stops
    being about geometry and starts being about rasterisation.

    `pattern` is the INNER-corner count, matching board_object_points, and the
    squares are laid out so that inner corner (0, 0) sits exactly on the board
    frame's origin.  That alignment is what lets a test assert the detector
    recovers the projected object points to a fraction of a pixel; get it wrong
    by one square and the whole board is offset by 25 mm, which is ~28 px at
    0.7 m and looks like a calibration failure rather than a bookkeeping one.
    """
    w, h = size
    ss = int(supersample)
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3))
    t = np.asarray(tvec, dtype=np.float64).reshape(3)

    # Pixel centres of the supersampled grid, mapped back to the ORIGINAL pixel
    # coordinate system: sub-pixel (i + 0.5)/ss - 0.5 is the centre convention,
    # and getting it wrong shifts the whole render by half a pixel, which is
    # exactly the size of the effect the calibration is trying to measure.
    us = (np.arange(w * ss) + 0.5) / ss - 0.5
    vs = (np.arange(h * ss) + 0.5) / ss - 0.5
    uu, vv = np.meshgrid(us, vs)
    uv = np.column_stack([uu.ravel(), vv.ravel()])

    xy_d = dist.normalize_pixels(K, uv)
    xn, yn = dist.undistort_normalized(xy_d[:, 0], xy_d[:, 1], D, iters=undistort_iters)
    dirs = np.column_stack([xn, yn, np.ones_like(xn)])          # rays in camera frame

    # Board plane in the camera frame: its normal is the board's own +Z axis
    # carried over by R (the third column), and it passes through t.  So a ray
    # P = s*dir meets it at s = (n . t) / (n . dir) - one dot product per
    # pixel.  The alternative, solving the 3x3 system [r0 r1 -dir](bx,by,s) =
    # -t per pixel, is the same answer and was 100x slower on the 1.2 million
    # rays a 640x480 view needs at the default 2x supersampling.
    n = R[:, 2]
    denom = dirs @ n
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (n @ t) / denom
    P = s[:, None] * dirs                       # the board point each ray hits
    rel = P - t
    bx = rel @ R[:, 0]                          # board-frame coordinates, metres
    by = rel @ R[:, 1]

    # Inner corner (i, j) sits at (i*square, j*square), so the squares run from
    # -square to cols*square: one extra square on each side of the corner grid.
    cols, rows = pattern
    lo_x, hi_x = -square, cols * square
    lo_y, hi_y = -square, rows * square
    m = margin_squares * square

    inside_paper = ((bx > lo_x - m) & (bx < hi_x + m) &
                    (by > lo_y - m) & (by < hi_y + m) & (s > 0))
    inside_grid = ((bx >= lo_x) & (bx < hi_x) & (by >= lo_y) & (by < hi_y) & (s > 0))

    ix = np.floor(bx / square).astype(np.int64)
    iy = np.floor(by / square).astype(np.int64)
    black = inside_grid & (((ix + iy) % 2) == 0)

    img = np.full(bx.shape, 105.0)       # the desk the board is lying on
    img[inside_paper] = 235.0            # paper white, not 255: real paper is not blown out
    img[black] = 25.0                    # ink black, not 0, for the same reason
    img = img.reshape(h * ss, w * ss)

    # Box-average the supersampled render down to the requested size.
    img = img.reshape(h, ss, w, ss).mean(axis=(1, 3))

    if noise_sigma > 0:
        rng = np.random.default_rng(seed)
        img = img + rng.normal(0.0, noise_sigma, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def detect_corners(img: np.ndarray, pattern: tuple[int, int]):
    """Sub-pixel inner-corner detection, returning (ok, corners as (N, 2)).

    findChessboardCornersSB is the newer detector and it already refines to
    sub-pixel internally, so no cornerSubPix call follows it.  With the older
    findChessboardCorners you MUST add that refinement or your corners are
    integer-quantised and your calibration inherits a ~0.3 px noise floor that
    no amount of extra views removes.
    """
    ok, corners = cv2.findChessboardCornersSB(img, pattern, cv2.CALIB_CB_EXHAUSTIVE)
    if not ok:
        return False, None
    return True, corners.reshape(-1, 2).astype(np.float64)


# --------------------------------------------------------------------------
# A textured stereo pair with a known depth map
# --------------------------------------------------------------------------

@dataclass
class Surface:
    """One textured plane in the scene.

    The plane is n . P = c in the LEFT camera's frame, bounded by a rectangle
    in world X and Y.  A general plane rather than a fronto-parallel one
    because a slanted surface has a disparity GRADIENT, and a matcher that only
    ever sees constant-disparity regions is not being asked a real question.
    """
    normal: np.ndarray
    offset: float
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    texture: np.ndarray
    tex_scale: float = 0.35          # metres per texture tile
    name: str = ""


def _band_limited_texture(shape=(256, 256), seed=0, blur=2.5, low=40, high=225):
    """Random texture with a controlled spatial frequency content.

    White noise is the wrong texture for a stereo demo: it is so distinctive
    that every window matches uniquely and the matcher looks better than it is.
    Blurring the noise removes the highest frequencies and leaves something
    closer to a real surface, where nearby windows genuinely resemble each
    other and the cost curve has a finite width.
    """
    rng = np.random.default_rng(seed)
    t = rng.normal(0.0, 1.0, shape).astype(np.float32)
    t = cv2.GaussianBlur(t, (0, 0), blur)
    t -= t.min()
    t /= max(t.max(), 1e-9)
    return (low + t * (high - low)).astype(np.float32)


def _striped_texture(shape=(256, 256), period_px=16, low=60, high=210):
    """A vertical stripe pattern - the repeated-texture failure mode, on purpose.

    Repetition is a different disease from lack of texture.  Here the cost
    curve has SEVERAL equally deep minima, one per period, and winner-take-all
    picks whichever noise favours - confidently, and often wrongly.
    """
    x = np.arange(shape[1])
    stripe = ((x // (period_px // 2)) % 2).astype(np.float32)
    row = low + stripe * (high - low)
    return np.tile(row, (shape[0], 1)).astype(np.float32)


def _sample_texture(tex: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinear sample of a tiled texture at continuous coordinates.

    Tiled (wrap-around) rather than clamped so a surface can be any size
    without the texture stretching, and bilinear rather than nearest because
    nearest re-introduces exactly the aliasing that ruins sub-pixel disparity.
    """
    h, w = tex.shape
    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    fu, fv = u - u0, v - v0
    u0m, v0m = u0 % w, v0 % h
    u1m, v1m = (u0 + 1) % w, (v0 + 1) % h
    a = tex[v0m, u0m]
    b = tex[v0m, u1m]
    c = tex[v1m, u0m]
    d = tex[v1m, u1m]
    return (a * (1 - fu) * (1 - fv) + b * fu * (1 - fv) +
            c * (1 - fu) * fv + d * fu * fv)


@dataclass
class StereoScene:
    """A rendered stereo pair plus every ground-truth array it implies."""
    left: np.ndarray
    right: np.ndarray
    depth_left: np.ndarray           # metres, per left pixel
    disparity_left: np.ndarray       # pixels, = f*B/Z, per left pixel
    surface_left: np.ndarray         # integer surface id, for failure-mode masks
    occluded: np.ndarray             # bool: in the right camera's field of view,
                                     # but hidden behind something nearer
    outside_right: np.ndarray        # bool: its match would fall off the left
                                     # edge of the right image entirely
    K: np.ndarray
    baseline: float
    names: list = field(default_factory=list)

    def region_mask(self, name: str) -> np.ndarray:
        """Boolean mask of one named surface in the LEFT image."""
        return self.surface_left == self.names.index(name)

    @property
    def unmatchable(self) -> np.ndarray:
        """Every left pixel with no correct match, for either reason.

        The two reasons are worth keeping apart even though they have the same
        consequence.  OCCLUSION is a property of the scene - something nearer
        got in the way - and it is what a left-right check detects.  Falling
        outside the right image is a property of the RIG: those pixels are the
        left band, as wide as the largest disparity, that every stereo system
        has and that no algorithm can fill in.
        """
        return self.occluded | self.outside_right


def default_scene_surfaces() -> list[Surface]:
    """The scene: a back wall, a slanted ramp, a near slab, and two traps.

    The two traps are deliberate and are the reason the scene is not just three
    random planes.  `textureless` is a flat grey patch on the back wall where
    the cost curve is flat and the matcher has no information to work with;
    `repeated` is a stripe pattern whose period is a few pixels of disparity,
    where the matcher has too much.  Both are on the back wall, at the same
    depth as their surroundings, so any error there is attributable to the
    texture and not to the geometry.
    """
    # Each surface gets its own brightness band as well as its own texture, so
    # the regions are distinguishable by eye in the figures.  It also stops the
    # matcher from being handed an unrealistically easy problem: identical
    # statistics everywhere would let a window match anywhere at the same cost.
    wall_tex = _band_limited_texture(seed=1, blur=2.0, low=30, high=150)
    ramp_tex = _band_limited_texture(seed=2, blur=1.6, low=70, high=205)
    slab_tex = _band_limited_texture(seed=3, blur=1.4, low=120, high=248)
    flat_tex = np.full((256, 256), 150.0, dtype=np.float32)
    stripes = _striped_texture(period_px=14, low=60, high=210)

    return [
        # Near slab, fronto-parallel at 2.6 m - the large-disparity end, and the
        # thing that casts the occlusion shadow the left-right check has to find.
        Surface(np.array([0.0, 0.0, 1.0]), 2.60, (-0.90, -0.20), (-0.30, 0.40),
                slab_tex, 0.22, "near_slab"),
        # A plane slanted about the Y axis: X + Z = 3.677, so depth runs from
        # ~3.4 m to ~2.7 m across it and disparity has a GRADIENT rather than a
        # plateau.  A matcher only ever asked for constant disparity is not
        # being asked a real question, and sub-pixel refinement has nothing to
        # do on a plateau.
        Surface(np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0), 2.60, (0.20, 1.05),
                (-0.42, 0.48), ramp_tex, 0.30, "ramp"),
        # The two traps, as bands across the top and bottom of the back wall at
        # 7.5 m, clear of anything that could occlude them so that any error
        # measured there is attributable to the TEXTURE and not the geometry.
        Surface(np.array([0.0, 0.0, 1.0]), 7.50, (-4.0, 4.0), (-3.0, -1.30),
                flat_tex, 0.5, "textureless"),
        Surface(np.array([0.0, 0.0, 1.0]), 7.50, (-4.0, 4.0), (1.33, 3.0),
                stripes, 0.45, "repeated"),
        Surface(np.array([0.0, 0.0, 1.0]), 7.50, (-6.0, 6.0), (-4.0, 4.0),
                wall_tex, 0.55, "wall"),
    ]


def _render_camera(surfaces, K, D, R_cw, centre, size, noise_sigma, rng):
    """Render one view, returning (image, depth, surface id).

    `R_cw` maps world to camera and `centre` is the camera's position in the
    world - the pair (R, C) rather than (R, t) because a scene description
    reads better in camera positions, and pinhole.extrinsics_from_centre does
    the flip when the projection equations need it.

    Rays are built the same way as in the checkerboard renderer: pixel ->
    normalised -> UNDISTORT -> direction.  Passing a non-zero D therefore
    produces a genuinely distorted image whose undistortion is the exact
    inverse, which is what lets examples/09 rectify a pair and measure that the
    warp put the rows back where they belong.

    Surfaces are tested nearest-first along each ray and the first hit wins - a
    painter's algorithm run backwards.  Correct here because every surface is a
    bounded plane and none of them intersect; interpenetrating geometry would
    need a real z-buffer.
    """
    w, h = size
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    uv = np.column_stack([uu.ravel(), vv.ravel()])
    xy_d = dist.normalize_pixels(K, uv)
    xn, yn = dist.undistort_normalized(xy_d[:, 0], xy_d[:, 1], D, iters=12)
    dirs_cam = np.column_stack([xn, yn, np.ones_like(xn)]).reshape(h, w, 3)
    dirs = dirs_cam @ R_cw                      # world dirs: (R_cw.T @ d).T == d @ R_cw

    depth = np.full((h, w), np.inf)             # camera-frame Z, which is the ray parameter
    img = np.zeros((h, w), dtype=np.float64)
    sid = np.full((h, w), -1, dtype=np.int32)

    for i, surf in enumerate(surfaces):
        n = surf.normal / np.linalg.norm(surf.normal)
        denom = dirs @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (surf.offset - n @ centre) / denom               # ray parameter
        P = centre + s[..., None] * dirs                         # world point per pixel
        hit = ((s > 0) &
               (P[..., 0] >= surf.xlim[0]) & (P[..., 0] <= surf.xlim[1]) &
               (P[..., 1] >= surf.ylim[0]) & (P[..., 1] <= surf.ylim[1]) &
               (s < depth))                                      # nearer than what we have
        if not hit.any():
            continue
        tu = P[..., 0][hit] / surf.tex_scale * 64.0
        tv = P[..., 1][hit] / surf.tex_scale * 64.0
        img[hit] = _sample_texture(surf.texture, tu, tv)
        depth[hit] = s[hit]
        sid[hit] = i

    if noise_sigma > 0:
        img = img + rng.normal(0.0, noise_sigma, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), depth, sid


def render_view(K, R, C, size, *, D=None, surfaces=None, noise_sigma=1.0, seed=11):
    """Public single-view renderer: any pose, any distortion, same scene.

    Used by the two-view examples, where the second camera is genuinely rotated
    rather than merely displaced, so the epipolar lines are not all horizontal
    and rectification has something real to undo.
    """
    surfaces = default_scene_surfaces() if surfaces is None else surfaces
    D = np.zeros(5) if D is None else np.asarray(D, dtype=np.float64)
    rng = np.random.default_rng(seed)
    img, depth, sid = _render_camera(surfaces, np.asarray(K, dtype=np.float64), D,
                                     np.asarray(R, dtype=np.float64),
                                     np.asarray(C, dtype=np.float64).ravel(),
                                     size, noise_sigma, rng)
    return img, depth, sid


def render_stereo_pair(*, size=(480, 320), fx=520.0, baseline=0.12,
                       surfaces=None, noise_sigma=1.0, seed=7) -> StereoScene:
    """Render a rectified stereo pair with exact per-pixel ground truth.

    The rig is rectified BY CONSTRUCTION: identical intrinsics, identical
    orientation, and the right camera displaced along +X by the baseline.  That
    is the geometry cv2.stereoRectify manufactures out of a real pair, and
    building it directly here keeps the block-matching lesson free of
    rectification error - when the matcher is wrong in this scene, it is the
    matcher.  examples/08 does run a real stereoRectify round trip separately,
    so the step is demonstrated rather than assumed.

    Both cameras share a principal point, so `doffs` is zero and Z = f*B/d is
    exact.  depth.py carries the doffs term anyway, because real rigs (and
    every Middlebury scene) do not share one.
    """
    surfaces = default_scene_surfaces() if surfaces is None else surfaces
    w, h = size
    K = np.array([[fx, 0.0, (w - 1) / 2.0],
                  [0.0, fx, (h - 1) / 2.0],
                  [0.0, 0.0, 1.0]])
    rng = np.random.default_rng(seed)

    no_lens = np.zeros(5)                # this pair is rectified by construction
    left, depth_l, sid_l = _render_camera(surfaces, K, no_lens, np.eye(3),
                                          np.zeros(3), size, noise_sigma, rng)
    right, _, sid_r = _render_camera(surfaces, K, no_lens, np.eye(3),
                                     np.array([baseline, 0.0, 0.0]), size, noise_sigma, rng)

    with np.errstate(divide="ignore", invalid="ignore"):
        disp = fx * baseline / depth_l
    disp[~np.isfinite(depth_l)] = np.nan

    # Occlusion: follow each left pixel to where its match should be and ask
    # the right image what surface is actually there.  A mismatch means the
    # left pixel's surface is hidden on the right - there is no correct
    # disparity to find, and anything a matcher reports there is invention.
    xs = np.arange(w)[None, :].repeat(h, 0)
    xr = np.rint(xs - np.nan_to_num(disp, nan=1e6)).astype(np.int64)
    in_frame = (xr >= 0) & (xr < w)
    ys = np.arange(h)[:, None].repeat(w, 1)
    occluded = np.zeros((h, w), dtype=bool)
    occluded[in_frame] = sid_r[ys[in_frame], xr[in_frame]] != sid_l[in_frame]

    return StereoScene(left=left, right=right, depth_left=depth_l, disparity_left=disp,
                       surface_left=sid_l, occluded=occluded,
                       outside_right=~in_frame, K=K, baseline=baseline,
                       names=[s.name for s in surfaces])
