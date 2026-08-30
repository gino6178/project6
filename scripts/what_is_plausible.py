"""What does a real change of floor level in a room actually look like?

This is the question that should have been answered before the corpus generator
was written. The generator picks a program label, cuts a wall-anchored band or a
diagonal corner out of the room, and assigns it a rise -- and nothing in that
procedure asks whether a building would have a reason to do it.

Real floors step for reasons: a structural slab edge, a stair landing, headroom
for something below, or a functional zone that the architecture already
separates. Those reasons leave geometric fingerprints, and this script measures
them on the 72 annotated MP3D-Elev fields, then measures the same quantities on
the generated corpus so the two can be compared directly.

Everything is computed in the *room's* principal frame, because "follows the
structural grid" is only meaningful relative to the walls.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys

import numpy as np
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.ops import unary_union

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import elevate3d  # noqa: F401,E402

WALL_TOL = 0.10          # a boundary this close to a wall is *on* the wall
ANG_TOL = 12.0           # degrees off the grid still counts as aligned


def principal_frame(room: Polygon):
    """The room's dominant wall direction, from its longest edges."""
    c = np.asarray(room.exterior.coords)
    seg = c[1:] - c[:-1]
    ln = np.hypot(seg[:, 0], seg[:, 1])
    ang = np.degrees(np.arctan2(seg[:, 1], seg[:, 0])) % 90.0
    # length-weighted circular-ish mean on a 90 deg period
    w = ln / ln.sum()
    a = np.degrees(0.5 * np.arctan2((w * np.sin(np.radians(2 * ang * 2))).sum(),
                                    (w * np.cos(np.radians(2 * ang * 2))).sum())) / 2.0
    return a % 90.0


def edge_alignment(poly: Polygon, base: float) -> float:
    """Length-weighted share of the region's edges that follow the wall grid."""
    c = np.asarray(poly.exterior.coords)
    seg = c[1:] - c[:-1]
    ln = np.hypot(seg[:, 0], seg[:, 1])
    keep = ln > 0.05
    if not keep.any():
        return float("nan")
    seg, ln = seg[keep], ln[keep]
    ang = np.degrees(np.arctan2(seg[:, 1], seg[:, 0])) % 90.0
    d = np.abs(((ang - base + 45.0) % 90.0) - 45.0)
    return float((ln[d <= ANG_TOL].sum()) / ln.sum())


def free_edges(region: Polygon, room: Polygon):
    """The part of the region's perimeter that is a drop, not a wall.

    Returns (free_fraction, n_components, longest_free_run).
    """
    b = region.boundary
    wall = room.boundary.buffer(WALL_TOL)
    free = b.difference(wall)
    if free.is_empty:
        return 0.0, 0, 0.0
    parts = [g for g in getattr(free, "geoms", [free])
             if isinstance(g, LineString) and g.length > 0.15]
    if not parts:
        return 0.0, 0, 0.0
    total = b.length
    merged = unary_union(parts)
    comps = [g for g in getattr(merged, "geoms", [merged])]
    return (sum(p.length for p in parts) / total, len(comps),
            max(p.length for p in parts))


def spans_room(region: Polygon, room: Polygon) -> bool:
    """Does the region run wall to wall in at least one direction?

    A real split level or sunken lounge is bounded by the building on two
    opposite sides and drops on the third; a patch that floats with floor on
    every side has no structural story.
    """
    wall = room.boundary.buffer(WALL_TOL)
    touch = region.boundary.intersection(wall)
    if touch.is_empty:
        return False
    parts = [g for g in getattr(touch, "geoms", [touch]) if g.length > 0.3]
    if len(parts) < 2:
        return False
    # two contact runs that are far apart, relative to the region's own size
    cs = np.array([list(p.centroid.coords)[0] for p in parts])
    d = np.hypot(*(cs[:, None, :] - cs[None, :, :]).T).max()
    ext = np.hypot(*(np.asarray(region.bounds[2:]) - np.asarray(region.bounds[:2])))
    return bool(d > 0.55 * ext)


def rectangularity(poly: Polygon) -> float:
    r = poly.minimum_rotated_rectangle
    return float(poly.area / r.area) if r.area > 0 else 0.0


def measure(room_xy, tiers, transitions):
    room = Polygon(room_xy).buffer(0)
    if room.is_empty or room.area <= 0:
        return []
    base = principal_frame(room)
    datum = min(tiers, key=lambda t: abs(t["height"]))
    out = []
    for t in tiers:
        if abs(t["height"] - datum["height"]) < 1e-6:
            continue
        reg = Polygon(t["polygon"]).buffer(0)
        if reg.is_empty or reg.area < 0.3:
            continue
        if reg.geom_type != "Polygon":
            reg = max(reg.geoms, key=lambda g: g.area)
        ff, nc, longest = free_edges(reg, room)
        rec = {
            "rise": float(t["height"] - datum["height"]),
            "area_frac": float(reg.area / room.area),
            "free_frac": ff,
            "n_free_runs": nc,
            "spans": spans_room(reg, room),
            "align": edge_alignment(reg, base),
            "rect": rectangularity(reg),
        }
        # is the annotated stair on the free edge, where a step has to be?
        if transitions:
            tl = [LineString([tr["p0"], tr["p1"]]) for tr in transitions]
            b = reg.boundary
            near = [l for l in tl if l.distance(b) < 0.6]
            rec["stair_on_edge"] = bool(near)
        out.append(rec)
    return out


def report(rows, name):
    if not rows:
        print(f"{name}: nothing measured")
        return
    def col(k):
        return np.array([r[k] for r in rows if isinstance(r.get(k), (int, float))],
                        dtype=float)
    print(f"\n=== {name}  ({len(rows)} elevated regions) ===")
    for k, lab in (("rise", "rise (m)"), ("area_frac", "area / room"),
                   ("free_frac", "perimeter that is a drop"),
                   ("align", "edges on the wall grid"),
                   ("rect", "rectangularity")):
        v = col(k)
        print(f"  {lab:28s} median {np.median(v):6.2f}   "
              f"[p10 {np.percentile(v, 10):6.2f}, p90 {np.percentile(v, 90):6.2f}]")
    nf = col("n_free_runs")
    print(f"  {'separate drop edges':28s} median {np.median(nf):6.1f}   "
          + "  ".join(f"{int(k)}:{(nf == k).mean():.0%}"
                      for k in sorted(set(nf.astype(int)))[:5]))
    sp = np.array([bool(r["spans"]) for r in rows])
    print(f"  {'runs wall to wall':28s} {sp.mean():.0%}")
    so = [r for r in rows if "stair_on_edge" in r]
    if so:
        print(f"  {'stair on the region edge':28s} "
              f"{np.mean([r['stair_on_edge'] for r in so]):.0%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mp3d", default="/home/gino/data/elevate3d/mp3d_elev/rooms.jsonl")
    ap.add_argument("--corpus", default="/home/gino/data/elevate3d/frontelev5")
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--out", default="outputs/plausible.json")
    args = ap.parse_args()

    real = []
    for line in open(args.mp3d):
        r = json.loads(line)
        real += measure(r["room"]["polygon"], r["field"]["tiers"],
                        r["field"].get("transitions"))

    gen = []
    for f in sorted(glob.glob(os.path.join(args.corpus, "*.jsonl.gz"))):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                d = json.loads(line)["elev"]
                gen += measure(d["room"]["polygon"], d["field"]["tiers"],
                               d["field"].get("transitions"))
                if len(gen) >= args.n:
                    break
        if len(gen) >= args.n:
            break

    report(real, "MP3D-Elev  (real buildings)")
    report(gen, "FRONT-Elev  (our generator)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"real": real, "generated": gen[:args.n]}, open(args.out, "w"))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
