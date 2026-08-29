"""Do real homes have intra-storey floor height changes?  Measure it.

3D-FRONT answers "no" by construction — its floors are exactly planar.  That is
a fact about a synthetic dataset, not about houses.  HouseLayout3D (NeurIPS
2025) vectorises the structure of 16 real Matterport3D buildings into per-entity
slabs, which makes the question answerable on real scans: find the horizontal
slabs, work out which are floors, group them into storeys, and look for floors
that sit at different heights within one storey and touch each other.

Entity classes are not labelled — the per-entity colours are random — so the
classification is geometric.  A horizontal slab with a matching slab about a
room height above it is a floor; the one above is its ceiling.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

STOREY_GAP = 1.40          # m; a bigger jump between slabs is a new storey
ROOM_HEIGHT = (1.90, 3.60)  # plausible floor-to-ceiling range


def read_ply(path: str):
    """Vertices of an Open3D binary PLY with double xyz + uchar rgb."""
    with open(path, "rb") as fh:
        hdr = b""
        while b"end_header" not in hdr:
            line = fh.readline()
            if not line:
                return None
            hdr += line
        lines = hdr.decode(errors="ignore").splitlines()
        try:
            nv = int([l for l in lines if l.startswith("element vertex")][0].split()[-1])
        except IndexError:
            return None
        has_rgb = any("property uchar red" in l for l in lines)
        dt = ([("x", "<f8"), ("y", "<f8"), ("z", "<f8")]
              + ([("r", "u1"), ("g", "u1"), ("b", "u1")] if has_rgb else []))
        dt = np.dtype(dt)
        buf = fh.read(nv * dt.itemsize)
        if len(buf) < nv * dt.itemsize:
            return None
        return np.frombuffer(buf, dtype=dt, count=nv)


def slabs_of(building_dir: str, min_area: float, flat_tol: float):
    """Horizontal slabs: (height, footprint polygon, plan area)."""
    out = []
    for p in sorted(glob.glob(os.path.join(building_dir, "*.ply"))):
        v = read_ply(p)
        if v is None or len(v) < 4:
            continue
        z = np.asarray(v["z"], float)
        if np.ptp(z) > flat_tol:
            continue                                   # not horizontal
        xy = np.stack([np.asarray(v["x"], float), np.asarray(v["y"], float)], 1)
        try:
            poly = Polygon(xy).convex_hull
        except Exception:
            continue
        if poly.is_empty or poly.area < min_area:
            continue
        out.append((float(z.mean()), poly, float(poly.area)))
    return out


def storeys(slabs, gap: float = STOREY_GAP):
    """Cluster slab heights into storeys."""
    if not slabs:
        return []
    order = sorted(range(len(slabs)), key=lambda i: slabs[i][0])
    groups, cur = [], [order[0]]
    for a, b in zip(order[:-1], order[1:]):
        if slabs[b][0] - slabs[a][0] > gap:
            groups.append(cur)
            cur = []
        cur.append(b)
    groups.append(cur)
    return groups


def floors_in(slabs, idxs, overlap: float = 0.5):
    """A slab is a floor when a slab a room-height above shares its footprint."""
    out = []
    for i in idxs:
        hi, pi, ai = slabs[i]
        for j, (hj, pj, aj) in enumerate(slabs):
            if j == i:
                continue
            d = hj - hi
            if not (ROOM_HEIGHT[0] <= d <= ROOM_HEIGHT[1]):
                continue
            inter = pi.intersection(pj).area
            if inter >= overlap * min(ai, aj):
                out.append(i)
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",
                    default="/home/gino/data/elevate3d/houselayout3d/"
                            "structures/layouts_split_by_entity")
    ap.add_argument("--out", default="outputs/g1c_houselayout_elevation.json")
    ap.add_argument("--min-area", type=float, default=1.5)
    ap.add_argument("--flat-tol", type=float, default=0.08)
    ap.add_argument("--min-rise", type=float, default=0.05)
    ap.add_argument("--max-rise", type=float, default=0.90)
    ap.add_argument("--touch", type=float, default=0.30,
                    help="floors this close in plan count as adjacent (m)")
    args = ap.parse_args()

    buildings = sorted(d for d in glob.glob(os.path.join(args.root, "*"))
                       if os.path.isdir(d))
    print(f"scanning {len(buildings)} buildings")

    rows, per_building = [], []
    rises = []
    for bd in buildings:
        name = os.path.basename(bd)
        slabs = slabs_of(bd, args.min_area, args.flat_tol)
        groups = storeys(slabs)
        n_pairs = 0
        n_floor = 0
        for gi, idxs in enumerate(groups):
            fl = floors_in(slabs, idxs)
            n_floor += len(fl)
            for a in range(len(fl)):
                for b in range(a + 1, len(fl)):
                    i, j = fl[a], fl[b]
                    hi, pi, ai = slabs[i]
                    hj, pj, aj = slabs[j]
                    d = abs(hi - hj)
                    if not (args.min_rise <= d <= args.max_rise):
                        continue
                    if pi.distance(pj) > args.touch:
                        continue               # different rooms, not a split
                    n_pairs += 1
                    rises.append(d)
                    rows.append({
                        "building": name, "storey": gi,
                        "rise": round(d, 4),
                        "lo_h": round(min(hi, hj), 3),
                        "hi_h": round(max(hi, hj), 3),
                        "lo_area": round(min(ai, aj), 2),
                        "hi_area": round(max(ai, aj), 2),
                        "gap": round(float(pi.distance(pj)), 3),
                    })
        per_building.append({"building": name, "slabs": len(slabs),
                             "storeys": len(groups), "floors": n_floor,
                             "elevation_pairs": n_pairs})
        print(f"  {name:16s} slabs={len(slabs):4d} storeys={len(groups)} "
              f"floors={n_floor:3d} elevation_pairs={n_pairs}", flush=True)

    r = np.asarray(rises) if rises else np.zeros(0)
    summary = {
        "buildings": len(buildings),
        "buildings_with_elevation_change": sum(
            1 for b in per_building if b["elevation_pairs"] > 0),
        "total_floor_slabs": sum(b["floors"] for b in per_building),
        "elevation_pairs": int(len(r)),
        "rise_percentiles": {p: round(float(np.percentile(r, p)), 3)
                             for p in (10, 25, 50, 75, 90)} if len(r) else {},
        "rise_histogram": dict(Counter(
            f"{np.floor(x / 0.1) * 0.1:.1f}-{np.floor(x / 0.1) * 0.1 + 0.1:.1f}"
            for x in r).most_common()) if len(r) else {},
        "params": vars(args),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"summary": summary, "per_building": per_building,
                   "pairs": rows}, fh, indent=1)
    print("\n" + json.dumps(summary, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
