"""Eyeball the corpus: plan and section for a handful of lifted rooms.

A layout corpus that is statistically clean can still be visually absurd, and
the cheapest way to find out is to look at it.  Each row is one room: the flat
original, the lifted version with its tiers and steps, and the section through
the middle of the elevated region.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import elevate3d  # noqa: F401

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPoly

from elevate3d.core.scene import ElevScene, parse_support, TIER
from elevate3d.eval.navigability import ccn_profile
from elevate3d.eval.violations import footprint
from elevate3d.geom.elevation import ElevationField

TIER_COLOURS = ["#dfe7ec", "#b9d2e0", "#e8d3ae", "#cbbdd8", "#c4ddc4"]


class _O:
    def __init__(self, d):
        self.oid = d["oid"]
        self.category = d["category"]
        self.jid = d.get("jid")
        self.yaw = float(d["yaw"])
        self.size = np.asarray(d["size"], float)
        self.position = np.array([d["xy"][0], d["xy"][1], d["z"]], float)

    @property
    def xy(self):
        return self.position[:2].copy()


class _R:
    def __init__(self, d):
        self.polygon = np.asarray(d["polygon"], float)
        self.height = d["height"]
        self.room_type = d.get("room_type", "")
        self.openings = []


def load(d) -> ElevScene:
    es = ElevScene(d["scene_id"], _R(d["room"]),
                   ElevationField.from_dict(d["field"]),
                   [_O(o) for o in d["objects"]],
                   source=d.get("source", ""), meta=d.get("meta", {}))
    for o in d["objects"]:
        es.supports[o["oid"]] = parse_support(o["support"])
        es.dz[o["oid"]] = o["dz"]
    return es


def draw_plan(ax, es: ElevScene, title: str):
    room = np.asarray(es.room.polygon, float)
    for i, t in enumerate(sorted(es.field.tiers, key=lambda t: t.height)):
        ax.add_patch(MPoly(t.polygon, closed=True,
                           facecolor=TIER_COLOURS[i % len(TIER_COLOURS)],
                           edgecolor="#7d8f9a", lw=1.0, zorder=1))
        c = t.shp.representative_point()
        ax.text(c.x, c.y, f"{t.height:+.2f}", ha="center", va="center",
                fontsize=7, color="#41525c", zorder=6)
    ax.add_patch(MPoly(room, closed=True, fill=False,
                       edgecolor="#2b3a43", lw=1.8, zorder=5))

    for o in es.objects:
        s = es.support_of(o.oid)
        if s.kind != TIER or es.dz.get(o.oid, 0.0) > 0.05:
            continue
        fp = np.asarray(footprint(o).exterior.coords)[:-1]
        ax.add_patch(MPoly(fp, closed=True, facecolor="#8a6f4e", alpha=0.55,
                           edgecolor="#5d4a33", lw=0.6, zorder=3))

    for tr in es.field.transitions:
        ax.plot([tr.p0[0], tr.p1[0]], [tr.p0[1], tr.p1[1]],
                color="#b4472f", lw=2.6, solid_capstyle="butt", zorder=4)
        m = (tr.p0 + tr.p1) / 2
        ax.text(m[0], m[1], f"{tr.kind[0].upper()}{tr.n_tread}", fontsize=6,
                color="#b4472f", ha="center", va="bottom", zorder=6)

    b = np.asarray(room, float)
    pad = 0.05 * max(np.ptp(b[:, 0]), np.ptp(b[:, 1]), 1.0)
    ax.set_xlim(b[:, 0].min() - pad, b[:, 0].max() + pad)
    ax.set_ylim(b[:, 1].min() - pad, b[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=8, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def draw_section(ax, es: ElevScene, title: str):
    """Cut through the centroid of the highest-relief tier, along +x."""
    room = es.field
    tgt = max(room.tiers, key=lambda t: abs(t.height))
    y = float(tgt.shp.representative_point().y)
    poly = np.asarray(es.room.polygon, float)
    x0, x1 = poly[:, 0].min(), poly[:, 0].max()
    xs = np.linspace(x0, x1, 400)
    h = np.array([room.height_at([x, y]) for x in xs])
    ax.fill_between(xs, h - 0.35, h, color="#b9d2e0", zorder=1)
    ax.plot(xs, h, color="#2b6b8c", lw=2.0, zorder=2)

    ceil = float(getattr(es.room, "height", 2.8))
    ax.plot([x0, x1], [ceil, ceil], color="#2b3a43", lw=1.4)

    for o in es.objects:
        s = es.support_of(o.oid)
        if s.kind != TIER or es.dz.get(o.oid, 0.0) > 0.05:
            continue                       # pendant lamps are not floor content
        fp = footprint(o)
        if not fp.intersects(__import__("shapely").geometry.LineString(
                [(x0, y), (x1, y)])):
            continue
        b = fp.bounds
        z = float(o.position[2])
        ax.add_patch(plt.Rectangle((b[0], z), b[2] - b[0], float(o.size[2]),
                                   facecolor="#8a6f4e", alpha=0.6,
                                   edgecolor="#5d4a33", lw=0.6, zorder=3))
    ax.axhline(0.0, color="#9aa8b0", lw=0.7, ls=(0, (4, 4)), zorder=0)
    ax.set_ylim(min(-0.8, h.min() - 0.4), ceil + 0.25)
    ax.set_title(title, fontsize=7, pad=6)
    ax.set_xticks([])
    ax.tick_params(labelsize=6)
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/home/gino/data/elevate3d/frontelev")
    ap.add_argument("--out", default="outputs/frontelev_examples.png")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--program", default="")
    args = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(args.corpus, "*.jsonl.gz")))
    if not shards:
        raise SystemExit(f"no shards in {args.corpus}")
    pairs = []
    for sh in shards:
        for line in gzip.open(sh, "rt"):
            d = json.loads(line)
            if args.program and d["elev"]["meta"]["program"] != args.program:
                continue
            pairs.append(d)
            if len(pairs) >= args.n * 6:
                break
        if len(pairs) >= args.n * 6:
            break

    # spread the sample over the programs actually present
    by_prog = {}
    for p in pairs:
        by_prog.setdefault(p["elev"]["meta"]["program"], []).append(p)
    picked, i = [], 0
    while len(picked) < args.n and any(by_prog.values()):
        for k in list(by_prog):
            if by_prog[k] and len(picked) < args.n:
                picked.append(by_prog[k].pop(0))
        i += 1
        if i > 50:
            break

    fig, axes = plt.subplots(len(picked), 3, figsize=(11, 2.9 * len(picked)))
    if len(picked) == 1:
        axes = axes[None, :]
    for r, d in enumerate(picked):
        flat, elev = load(d["flat"]), load(d["elev"])
        m = d["elev"]["meta"]
        prof = ccn_profile(elev)
        draw_plan(axes[r, 0], flat, f"flat · {flat.room.room_type}")
        draw_plan(axes[r, 1],  elev,
                  f"{m['program']} · rise {m['rise']:+.2f} m · "
                  f"{m['region_frac']:.0%} of floor")
        short = {"sweeping_robot": "sweep", "wheelchair": "chair",
                 "wheeled_robot": "wheel", "quadruped": "quad", "adult": "adult"}
        draw_section(axes[r, 2], elev, "section · CCN  " + "  ".join(
            f"{short.get(k, k)}={v:.2f}" for k, v in prof.items()))
    fig.tight_layout(h_pad=1.6, w_pad=1.2)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
