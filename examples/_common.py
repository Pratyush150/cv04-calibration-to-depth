"""Shared setup for the examples: import path, figure style, output location.

Every example is runnable on its own from the repository root
(`python3 examples/01_pinhole_projection.py`), so each one imports this first.
The alternative - installing the package - would work too, but it puts a step
between a reader and running the code, and the point of these files is that
they run.
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")                      # no display on CI, and none needed
import matplotlib.pyplot as plt            # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
FIGDIR = ROOT / "docs" / "figures"

# Teaching figures, not the dark portfolio site: white ground, dark ink, and
# enough DPI that a reader can zoom into a residual scatter without it turning
# to mush.
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.grid": True,
    "grid.color": "#d9d9d9",
    "grid.linewidth": 0.6,
    "axes.edgecolor": "#444444",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "font.size": 9,
    "figure.dpi": 110,
    "savefig.dpi": 130,
    "savefig.bbox": "tight",
    "image.cmap": "viridis",
})


def save(fig, name: str) -> pathlib.Path:
    """Write a figure into docs/figures and report where it went."""
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")
    return path


def rule(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))
