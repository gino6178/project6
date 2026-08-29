"""Real elevation fields, recovered from HouseLayout3D's Matterport3D scans.

FRONT-Elev's fields are procedural, so a model trained and tested on them is
partly learning the generator.  These are not: the tier polygons, their heights
and the stairs between them come from annotations of 16 real buildings, and
nothing in ``frontelev.py`` had any hand in them.

There are no furniture annotations, and that is fine — the elevation metrics
(overhang, straddling, blocked steps, headroom, capability-conditioned
reachability) are intrinsic to a generated layout and need no reference layout
to compare against.  What this module produces is a set of real *rooms to
furnish*, which is exactly what an out-of-distribution test needs.

Entity classes are unlabelled in the release, so floors are identified
geometrically: a horizontal slab with a matching slab a room height above it.
"""
from __future__ import annotations

import glob
import os

import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from ..geom.elevation import ElevationField, Tier, make_transition

__all__ = ["read_ply", "building_slabs", "storeys", "floors_in", "stairs_of",
           "elevation_rooms", "STOREY_GAP", "ROOM_HEIGHT"]

STOREY_GAP = 1.40
ROOM_HEIGHT = (1.90, 3.60)


def read_ply(path: str):
    """Vertices of an Open3D binary PLY with double xyz and optional uchar rgb."""
    with open(path, "rb") as fh:
        hdr = b""
        while b"end_header" not in hdr:
            line = fh.readline()
            if not line:
                return None
            hdr += line
        lines = hdr.decode(errors="ignore").splitlines()
        try:
            nv = int([l for l in lines
                      if l.startswith("element vertex")][0].split()[-1])
        except IndexError:
            return None
        fields = [("x", "<f8"), ("y", "<f8"), ("z", "<f8")]
        if any("property uchar red" in l for l in lines):
            fields += [("r", "u1"), ("g", "u1"), ("b", "u1")]
        dt = np.dtype(fields)
        buf = fh.read(nv * dt.itemsize)
        if len(buf) < nv * dt.itemsize:
            return None
        return np.frombuffer(buf, dtype=dt, count=nv)


def _hull(v) -> Polygon | None:
    xy = np.stack([np.asarray(v["x"], float), np.asarray(v["y"], float)], 1)
    try:
        p = Polygon(xy).convex_hull
    except Exception:
        return None
    return p if (p.geom_type == "Polygon" and not p.is_empty) else None


def building_slabs(building_dir: str, min_area: float = 1.5,
                   flat_tol: float = 0.08):
    """Horizontal slabs as ``(height, polygon, area)``."""
    out = []
    for p in sorted(glob.glob(os.path.join(building_dir, "*.ply"))):
        v = read_ply(p)
        if v is None or len(v) < 4:
            continue
        z = np.asarray(v["z"], float)
        if np.ptp(z) > flat_tol:
            continue
        poly = _hull(v)
        if poly is None or poly.area < min_area:
            continue
        out.append((float(z.mean()), poly, float(poly.area)))
    return out


def storeys(slabs, gap: float = STOREY_GAP):
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
    """Slabs with a ceiling a room height above them, and that ceiling's height."""
    out = {}
    for i in idxs:
        hi, pi, ai = slabs[i]
        for j, (hj, pj, aj) in enumerate(slabs):
            d = hj - hi
            if j == i or not (ROOM_HEIGHT[0] <= d <= ROOM_HEIGHT[1]):
                continue
            if pi.intersection(pj).area >= overlap * min(ai, aj):
                out[i] = float(d)
                break
    return out


def stairs_of(stairs_root: str, building: str):
    """Real staircases as ``(footprint, rise, n_tread)``."""
    out = []
    for p in sorted(glob.glob(os.path.join(stairs_root, building, "*.ply"))):
        v = read_ply(p)
        if v is None or len(v) < 4:
            continue
        poly = _hull(v)
        if poly is None:
            continue
        z = np.asarray(v["z"], float)
        rise = float(np.ptp(z))
        out.append((poly, rise, max(1, int(round(rise / 0.18)))))
    return out


def _clean(g):
    if g is None or g.is_empty:
        return None
    if not g.is_valid:
        g = g.buffer(0)
    if isinstance(g, MultiPolygon):
        g = max(g.geoms, key=lambda x: x.area)
    return g if getattr(g, "geom_type", "") == "Polygon" else None


def elevation_rooms(root: str, stairs_root: str = "",
                    min_rise: float = 0.10, max_rise: float = 0.90,
                    touch: float = 0.30, min_area: float = 8.0,
                    max_area: float = 70.0, min_tier: float = 1.5,
                    min_share: float = 0.60):
    """Two-tier rooms built from real adjacent floors at different heights.

    A pair qualifies when the floors touch in plan, differ by a plausible step
    or short flight, and share enough boundary to walk across.  The room is the
    union of the two footprints; its ceiling comes from the annotated ceiling
    above the lower floor.
    """
    rooms = []
    for bd in sorted(d for d in glob.glob(os.path.join(root, "*"))
                     if os.path.isdir(d)):
        name = os.path.basename(bd)
        slabs = building_slabs(bd)
        stairs = stairs_of(stairs_root, name) if stairs_root else []
        for gi, idxs in enumerate(storeys(slabs)):
            fl = floors_in(slabs, idxs)
            keys = sorted(fl)
            for a in range(len(keys)):
                for b in range(a + 1, len(keys)):
                    i, j = keys[a], keys[b]
                    hi, pi, ai = slabs[i]
                    hj, pj, aj = slabs[j]
                    if hi > hj:
                        i, j = j, i
                        hi, pi, ai = slabs[i]
                        hj, pj, aj = slabs[j]
                    rise = hj - hi
                    if not (min_rise <= rise <= max_rise):
                        continue
                    if pi.distance(pj) > touch:
                        continue
                    if min(ai, aj) < min_tier:
                        continue

                    lo = _clean(pi)
                    up = _clean(pj.difference(pi))   # keep the tiers disjoint
                    if lo is None or up is None or up.area < min_tier:
                        continue
                    room = _clean(unary_union([lo, up]))
                    if room is None or not (min_area <= room.area <= max_area):
                        continue

                    # the shared edge: the upper tier's boundary inside the room
                    shared = up.boundary.intersection(lo.buffer(touch + 0.05))
                    width = getattr(shared, "length", 0.0)
                    if width < min_share:
                        continue
                    pts = None
                    for g in getattr(shared, "geoms", [shared]):
                        if getattr(g, "geom_type", "") == "LineString":
                            cc = np.asarray(g.coords)
                            if pts is None or (np.linalg.norm(cc[-1] - cc[0]) >
                                               np.linalg.norm(pts[-1] - pts[0])):
                                pts = cc
                    if pts is None or len(pts) < 2:
                        continue

                    t_lo = Tier(0, np.asarray(lo.exterior.coords)[:-1], 0.0,
                                holes=[np.asarray(r.coords)[:-1]
                                       for r in lo.interiors])
                    t_hi = Tier(1, np.asarray(up.exterior.coords)[:-1],
                                round(rise, 3),
                                holes=[np.asarray(r.coords)[:-1]
                                       for r in up.interiors])

                    # use a real staircase when one sits on this edge
                    kind, tread = None, 0
                    mid = Polygon([pts[0], pts[-1],
                                   pts[-1] + 1e-3, pts[0] + 1e-3]).centroid
                    for sp, srise, sn in stairs:
                        if sp.distance(mid) < 1.0 and abs(srise - rise) < 0.35:
                            kind, tread = "stair", sn
                            break
                    tr = make_transition(t_lo, t_hi, pts[0], pts[-1], kind=kind)
                    if tread:
                        tr.n_tread = tread

                    field = ElevationField(
                        np.asarray(room.exterior.coords)[:-1],
                        [t_lo, t_hi], [tr])
                    if field.validate():
                        continue
                    rooms.append({
                        "scene_id": f"{name}__s{gi}__{i}_{j}",
                        "building": name, "storey": gi,
                        "room": {"polygon": np.round(
                            np.asarray(room.exterior.coords)[:-1], 4).tolist(),
                            "height": round(float(fl[i]), 3),
                            "room_type": "unknown"},
                        "field": field.to_dict(),
                        "meta": {"rise": round(rise, 4),
                                 "source": "HouseLayout3D",
                                 "lo_area": round(float(lo.area), 2),
                                 "hi_area": round(float(up.area), 2),
                                 "transition": tr.kind,
                                 "n_tread": tr.n_tread},
                    })
    return rooms
