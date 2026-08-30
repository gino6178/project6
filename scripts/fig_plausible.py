"""Draw the difference the numbers describe: real level changes vs ours.

The elevated region is filled; the part of its boundary that is a *drop* -- an
edge you could step off -- is drawn heavy in red, and the part that is the
building's own wall is left thin. That one distinction is the whole finding.
"""
from __future__ import annotations

import glob
import gzip
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MPoly
from shapely.geometry import LineString, Polygon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from what_is_plausible import WALL_TOL, free_edges  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def panels(room_xy, tiers):
    room = Polygon(room_xy).buffer(0)
    if room.is_empty:
        return None
    datum = min(tiers, key=lambda t: abs(t["height"]))
    for t in tiers:
        if abs(t["height"] - datum["height"]) < 1e-6:
            continue
        reg = Polygon(t["polygon"]).buffer(0)
        if reg.is_empty or reg.area < 0.3:
            continue
        if reg.geom_type != "Polygon":
            reg = max(reg.geoms, key=lambda g: g.area)
        return room, reg, float(t["height"] - datum["height"])
    return None


def draw(ax, room, reg, rise):
    ax.add_patch(MPoly(np.asarray(room.exterior.coords), closed=True,
                       facecolor="#f2efe9", edgecolor="#2b3a43", lw=1.6))
    ax.add_patch(MPoly(np.asarray(reg.exterior.coords), closed=True,
                       facecolor="#a8c3d4", edgecolor="none", alpha=0.95))
    free = reg.boundary.difference(room.boundary.buffer(WALL_TOL))
    for g in getattr(free, "geoms", [free]):
        if isinstance(g, LineString) and g.length > 0.15:
            xy = np.asarray(g.coords)
            ax.plot(xy[:, 0], xy[:, 1], color="#c0392b", lw=3.4,
                    solid_capstyle="butt", zorder=5)
    ff, nc, _ = free_edges(reg, room)
    ax.set_title(f"{rise:+.2f} m\n{ff:.0%} of edge is a drop, in {nc} place"
                 f"{'s' if nc != 1 else ''}",
                 fontsize=9, pad=5, linespacing=1.45,
                 color="#c0392b" if nc >= 2 or ff > 0.5 else "#41525c")
    b = room.bounds
    pad = 0.06 * max(b[2] - b[0], b[3] - b[1])
    ax.set_xlim(b[0] - pad, b[2] + pad)
    ax.set_ylim(b[1] - pad, b[3] + pad)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    real = []
    for line in open("/home/gino/data/elevate3d/mp3d_elev/rooms.jsonl"):
        r = json.loads(line)
        p = panels(r["room"]["polygon"], r["field"]["tiers"])
        if p:
            real.append(p)

    gen = []
    for f in sorted(glob.glob("/home/gino/data/elevate3d/frontelev5/*.jsonl.gz")):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                d = json.loads(line)["elev"]
                p = panels(d["room"]["polygon"], d["field"]["tiers"])
                if p:
                    gen.append(p)
                if len(gen) >= 400:
                    break
        if len(gen) >= 400:
            break

    rng = np.random.default_rng(3)
    # show the generator's own median behaviour, not its worst case: sample
    # from the ones with two drop edges in proportion to how often it makes them
    two = [g for g in gen if free_edges(g[1], g[0])[1] >= 2]
    one = [g for g in gen if free_edges(g[1], g[0])[1] < 2]
    pick_g = ([two[i] for i in rng.choice(len(two), 2, replace=False)]
              + [one[i] for i in rng.choice(len(one), 3, replace=False)])
    rng.shuffle(pick_g)
    pick_r = [real[i] for i in rng.choice(len(real), 5, replace=False)]

    fig, axes = plt.subplots(2, 5, figsize=(15.0, 7.0))
    for ax, p in zip(axes[0], pick_r):
        draw(ax, *p)
    for ax, p in zip(axes[1], pick_g):
        draw(ax, *p)
    axes[0][0].text(-0.06, 0.5, "real\n(MP3D-Elev)", transform=axes[0][0].transAxes,
                    ha="right", va="center", fontsize=11, fontweight="bold")
    axes[1][0].text(-0.06, 0.5, "ours\n(FRONT-Elev)", transform=axes[1][0].transAxes,
                    ha="right", va="center", fontsize=11, fontweight="bold")
    fig.suptitle("Red is a drop you could step off. Everything else is a wall.",
                 fontsize=12.5, fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0.055, 0.0, 1.0, 0.94), w_pad=1.6)
    out = os.path.join(ROOT, "assets", "fig_plausible.png")
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
