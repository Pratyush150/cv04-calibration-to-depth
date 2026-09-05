"""01 - The pinhole camera, derived: similar triangles, K, and the divide by Z.

Run:  python3 examples/01_pinhole_projection.py

What it shows
  * the projection of one point, worked through by hand and by matrix, agreeing
  * why the divide by the third homogeneous coordinate IS perspective
  * the two image-plane conventions, and that they differ by a point reflection
    through the principal point rather than by a left-right mirror
"""

import numpy as np

from _common import rule, save, plt          # noqa: E402  (path shim first)
from geo import pinhole as ph                # noqa: E402

fx = fy = 800.0
cx, cy = 320.0, 240.0
K = ph.intrinsic_matrix(fx, fy, cx, cy)

rule("K, and what each entry is in units you can picture")
print(K)
print(f"fx = {fx:.0f} px : the focal length measured in PIXEL WIDTHS.  A lens stamped")
print("             6 mm on a sensor with 3 um pixels gives 6/0.003 = 2000 px.")
print(f"fy = {fy:.0f} px : the same in pixel HEIGHTS.  It differs from fx only if the")
print("             pixels are not square, which is what absorbs sensor anisotropy.")
print(f"cx, cy = ({cx:.0f}, {cy:.0f}) px : the principal point, where the optical axis")
print("             pierces the image.  Near the middle, never exactly at it.")
print("skew   = 0    : non-perpendicular pixel rows.  Zero on every sensor you will")
print("             meet; the slot exists so the matrix layout is explicit.")

rule("One point, by hand and by matrix")
X = np.array([0.1, -0.2, 2.0])           # 10 cm right, 20 cm UP (y is down), 2 m away
h = ph.matvec3(K, X)
print(f"X_cam                = {X}")
print(f"K @ X (row-dot-col)  = {h}      <- NOT pixels yet")
print(f"same via numpy       = {K @ X}")
print(f"divide by the third  = {h[:2] / h[2]}  <- pixels")
print("The third component came out as 2.0, which is Z.  The divide is by DEPTH,")
print("which is exactly why distant things look small.")

rule("Double the depth, halve the offset from the principal point")
for Z in (2.0, 4.0, 8.0):
    uv = ph.project_camera_points(K, [[0.1, -0.2, Z]])[0]
    print(f"Z = {Z:4.1f} m -> pixel ({uv[0]:6.1f}, {uv[1]:6.1f})   "
          f"offset from principal point ({uv[0]-cx:+6.1f}, {uv[1]-cy:+6.1f})")
print("Exactly halved each time.  A point twice as far and twice as big lands on")
print("the same pixel, which is why one image cannot tell you scale.")

rule("The sanity check that any projection code must pass")
axis = ph.project_camera_points(K, [[0.0, 0.0, 5.0], [0.0, 0.0, 500.0]])
print(f"points on the optical axis project to {axis[0]} and {axis[1]}")
print("A point on the optical axis lands on the principal point at ANY depth, under")
print("either image-plane convention.  If this fails, nothing downstream can work.")

rule("Behind the camera: the cheirality guard")
behind = ph.project_camera_points(K, [[0.1, -0.2, -2.0]])
print(f"a point at Z = -2 m projects to {behind[0]}")
print("Without the Z > 0 guard this returns (280.0, 320.0) - a perfectly plausible")
print("pixel inside the image, for an object that is behind the lens.")

rule("Real image plane (-f) versus virtual image plane (+f)")
rng = np.random.default_rng(0)
Xr = rng.normal(size=(200, 3))
Xr[:, 2] = rng.uniform(0.5, 10.0, 200)
virtual = ph.project_camera_points(K, Xr)
real = ph.project_real_plane(K, Xr)
reflected = 2 * np.array([cx, cy]) - virtual
print(f"max |real - reflection of virtual through (cx, cy)| = "
      f"{np.abs(real - reflected).max():.2e} px over 200 random points")
print("So the physical sensor's image is the virtual one rotated 180 degrees about")
print("the optical axis - not mirrored left-right, which is what people guess.")

# ---------------------------------------------------------------- the figure
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

# (a) Perspective: two parallel rails receding from the camera.
ax = axes[0]
Zs = np.linspace(2.0, 40.0, 60)
for X0, colour, label in ((-0.7, "#1f77b4", "rail at X = -0.7 m"),
                          (0.7, "#d62728", "rail at X = +0.7 m")):
    pts = np.column_stack([np.full_like(Zs, X0), np.full_like(Zs, 0.6), Zs])
    uv = ph.project_camera_points(K, pts)
    ax.plot(uv[:, 0], uv[:, 1], ".-", color=colour, ms=3, lw=1, label=label)
for Z in (2, 4, 8, 16, 32):
    sleeper = np.array([[-0.7, 0.6, Z], [0.7, 0.6, Z]])
    uv = ph.project_camera_points(K, sleeper)
    ax.plot(uv[:, 0], uv[:, 1], "-", color="#888888", lw=0.8)
    ax.annotate(f"{Z} m", (uv[1, 0] + 4, uv[1, 1]), fontsize=7, color="#555555")
ax.plot([cx], [cy], "k+", ms=10)
ax.annotate("vanishing point =\nprincipal point", (cx + 8, cy - 26), fontsize=8)
ax.set_xlim(0, 640); ax.set_ylim(480, 0)
ax.set_title("(a) perspective is the divide by Z")
ax.set_xlabel("u (px)"); ax.set_ylabel("v (px)"); ax.legend(loc="lower left", fontsize=8)

# (b) The 1/Z law, as a curve.
ax = axes[1]
Zs = np.linspace(1.0, 20.0, 200)
off = fx * 0.1 / Zs
ax.plot(Zs, off, color="#1f77b4", lw=2)
for Z in (2.0, 4.0, 8.0):
    ax.plot([Z], [fx * 0.1 / Z], "o", color="#d62728")
    ax.annotate(f"Z={Z:.0f} m\n{fx*0.1/Z:.0f} px", (Z + 0.4, fx * 0.1 / Z + 2), fontsize=8)
ax.set_title("(b) pixel offset of a point 10 cm off-axis")
ax.set_xlabel("depth Z (m)"); ax.set_ylabel("offset from principal point (px)")

# (c) The two conventions.
ax = axes[2]
sub = Xr[:40]
v = ph.project_camera_points(K, sub)
r = ph.project_real_plane(K, sub)
ax.plot(v[:, 0], v[:, 1], "o", ms=4, color="#1f77b4", label="virtual plane at +f")
ax.plot(r[:, 0], r[:, 1], "s", ms=4, mfc="none", color="#d62728", label="real plane at -f")
for a, b in zip(v[:12], r[:12]):
    ax.plot([a[0], b[0]], [a[1], b[1]], "-", color="#cccccc", lw=0.6, zorder=0)
ax.plot([cx], [cy], "k+", ms=12)
ax.set_xlim(-100, 740); ax.set_ylim(580, -100)
ax.set_title("(c) the same 40 points, both conventions")
ax.set_xlabel("u (px)"); ax.set_ylabel("v (px)"); ax.legend(fontsize=8, loc="upper left")

fig.suptitle("Pinhole projection: fx=fy=800 px, principal point (320, 240)", y=1.02)
save(fig, "01_pinhole_projection.png")
