"""G1b — the elevation 3D-FRONT *does* carry, that every parser throws away.

``scan_front_elevation.py`` shows the ``Floor`` mesh of a 3D-FRONT room is
exactly planar.  That is not the same as the room being flat: the designers
model raised areas as a separate mesh type, ``CustomizedPlatform``, which no
layout parser reads because they all filter on ``type == "Floor"``.

This pass recovers them.  For every room it measures the platform footprint, its
rise above the floor plane, and — the part that makes it supervision rather than
trivia — which furniture instances are standing *on* the platform rather than on
the floor.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union

HEIGHT_AXIS = 1
PLAN = (0, 2)

# mesh types that could carry a floor-level elevation change
PLATFORM_TYPES = ("CustomizedPlatform",)


def mesh_polygon(mesh: dict, height_lo: float | None = None,
                 height_hi: float | None = None) -> Polygon | None:
    """Footprint of a mesh, optionally only the triangles inside a height band."""
    v = np.asarray(mesh.get("xyz", []), dtype=float).reshape(-1, 3)
    f = np.asarray(mesh.get("faces", []), dtype=int).reshape(-1, 3)
    if len(v) == 0 or len(f) == 0 or f.max() >= len(v):
        return None
    tri = v[f]
    if height_lo is not None:
        h = tri[:, :, HEIGHT_AXIS].mean(axis=1)
        tri = tri[(h >= height_lo) & (h <= height_hi)]
    polys = []
    for t in tri:
        p = Polygon(t[:, PLAN])
        if p.area <= 1e-7:
            continue
        if not p.is_valid:
            p = p.buffer(0)
        if p.is_valid and p.area > 1e-7:
            polys.append(p)
    if not polys:
        return None
    u = unary_union(polys).buffer(1e-4).buffer(-1e-4)
    if u.is_empty:
        return None
    if isinstance(u, MultiPolygon):
        u = max(u.geoms, key=lambda g: g.area)
    return u if u.geom_type == "Polygon" else None


def floor_plane(meshes: list[dict]):
    """(height, footprint polygon) of a room's Floor meshes."""
    hs, areas, polys = [], [], []
    for m in meshes:
        v = np.asarray(m.get("xyz", []), dtype=float).reshape(-1, 3)
        if len(v) == 0:
            continue
        p = mesh_polygon(m)
        if p is None:
            continue
        hs.append(float(np.median(v[:, HEIGHT_AXIS])))
        areas.append(p.area)
        polys.append(p)
    if not polys:
        return None, None
    w = np.asarray(areas)
    h = float((np.asarray(hs) * w).sum() / w.sum())
    u = unary_union(polys).buffer(1e-4).buffer(-1e-4)
    if isinstance(u, MultiPolygon):
        u = max(u.geoms, key=lambda g: g.area)
    return h, (u if u.geom_type == "Polygon" else None)


def _yaw_from_quat(q) -> float:
    x, y, z, w = [float(v) for v in q]
    return math.atan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + x * x))


def furniture_boxes(room, furn, bboxes):
    """[(cat, cx, cz, bottom, top, sx, sz)] in 3D-FRONT world coords."""
    out = []
    for child in room.get("children", []):
        f = furn.get(child.get("ref"))
        if f is None or not f.get("jid"):
            continue
        bb = bboxes.get(f["jid"])
        if bb is None:
            continue
        scale = np.asarray(child.get("scale", [1, 1, 1]), dtype=float)
        lo, hi = bb[:3] * scale, bb[3:] * scale
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        size = hi - lo
        c = (lo + hi) / 2.0
        yaw = _yaw_from_quat(child.get("rot", [0, 0, 0, 1]))
        cs, sn = math.cos(yaw), math.sin(yaw)
        cx = cs * c[0] + sn * c[2]
        cz = -sn * c[0] + cs * c[2]
        pos = np.asarray(child.get("pos", [0, 0, 0]), dtype=float)
        world = pos + np.array([cx, c[1], cz])
        out.append({
            "title": (f.get("title") or "")[:60],
            "jid": f["jid"],
            "x": float(world[0]), "z": float(world[2]),
            "bottom": float(world[1] - size[1] / 2.0),
            "top": float(world[1] + size[1] / 2.0),
            "sx": float(size[0]), "sz": float(size[2]),
        })
    return out


def scan_house(path, bboxes, min_rise, tol):
    try:
        with open(path) as fh:
            d = json.load(fh)
    except Exception:
        return []

    meshes = {m["uid"]: m for m in d.get("mesh", [])}
    furn = {f["uid"]: f for f in d.get("furniture", [])}
    house = os.path.splitext(os.path.basename(path))[0]
    rows = []

    for room in d.get("scene", {}).get("room", []):
        floors, plats = [], []
        for child in room.get("children", []):
            m = meshes.get(child.get("ref"))
            if m is None:
                continue
            t = m.get("type", "")
            if t == "Floor":
                floors.append(m)
            elif t in PLATFORM_TYPES:
                plats.append(m)
        if not plats or not floors:
            continue

        h0, fpoly = floor_plane(floors)
        if h0 is None or fpoly is None or fpoly.area < 4.0:
            continue

        objs = furniture_boxes(room, furn, bboxes)
        tiers = []
        for m in plats:
            v = np.asarray(m.get("xyz", []), dtype=float).reshape(-1, 3)
            if len(v) == 0:
                continue
            top = float(v[:, HEIGHT_AXIS].max())
            bot = float(v[:, HEIGHT_AXIS].min())
            rise = top - h0
            # top face only: triangles within 2 cm of the highest point
            poly = mesh_polygon(m, top - 0.02, top + 1e-3) or mesh_polygon(m)
            if poly is None or poly.area < 0.5:
                continue
            on = [o for o in objs
                  if abs(o["bottom"] - top) <= tol
                  and poly.contains(Point(o["x"], o["z"]))]
            tiers.append({
                "rise": round(rise, 4),
                "thickness": round(top - bot, 4),
                "area": round(float(poly.area), 3),
                "area_frac": round(float(poly.area / fpoly.area), 4),
                "n_vertices": len(poly.exterior.coords) - 1,
                "n_objects_on": len(on),
                "objects_on": [o["title"] for o in on][:12],
            })
        if not tiers:
            continue
        big = [t for t in tiers if abs(t["rise"]) >= min_rise]
        rows.append({
            "house": house,
            "room": room.get("instanceid", ""),
            "type": room.get("type", ""),
            "floor_h": round(h0, 4),
            "floor_area": round(float(fpoly.area), 3),
            "n_platforms": len(tiers),
            "n_platforms_big": len(big),
            "max_rise": round(max((t["rise"] for t in tiers), default=0.0), 4),
            "n_objects": len(objs),
            "n_objects_on_platform": sum(t["n_objects_on"] for t in tiers),
            "platforms": tiers[:8],
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/gino/data/reroom/3D-FRONT_raw/3D-FRONT")
    ap.add_argument("--bboxes", default="/home/gino/data/reroom/future_bboxes.json")
    ap.add_argument("--out", default="outputs/g1b_front_platforms.json")
    ap.add_argument("--min-rise", type=float, default=0.08)
    ap.add_argument("--tol", type=float, default=0.06,
                    help="an object counts as standing on the platform when its "
                         "bottom is within this of the platform top (m)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    with open(args.bboxes) as fh:
        bboxes = {k: np.asarray(v, dtype=float) for k, v in json.load(fh).items()}

    files = sorted(f for f in os.listdir(args.root) if f.endswith(".json"))
    if args.limit:
        files = files[:args.limit]
    paths = [os.path.join(args.root, f) for f in files]
    print(f"scanning {len(paths)} houses for platforms, {args.workers} workers")

    rows = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(scan_house, p, bboxes, args.min_rise, args.tol)
                for p in paths]
        for fut in as_completed(futs):
            rows.extend(fut.result())
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(paths)}  platform-rooms={len(rows)}", flush=True)

    big = [r for r in rows if r["n_platforms_big"] > 0]
    with_obj = [r for r in big if r["n_objects_on_platform"] > 0]
    rises = np.array([t["rise"] for r in rows for t in r["platforms"]])
    fracs = np.array([t["area_frac"] for r in rows for t in r["platforms"]])

    summary = {
        "houses": len(paths),
        "rooms_with_platform": len(rows),
        "rooms_with_platform_rise_ge_min": len(big),
        "rooms_with_furniture_on_platform": len(with_obj),
        "total_platforms": int(len(rises)),
        "rise_percentiles": {p: round(float(np.percentile(rises, p)), 4)
                             for p in (10, 25, 50, 75, 90, 99)} if len(rises) else {},
        "area_frac_percentiles": {p: round(float(np.percentile(fracs, p)), 4)
                                  for p in (10, 50, 90)} if len(fracs) else {},
        "by_room_type": dict(Counter(r["type"] for r in big).most_common(25)),
        "objects_on_platforms": dict(Counter(
            o for r in rows for t in r["platforms"] for o in t["objects_on"]
        ).most_common(40)),
        "params": {"min_rise": args.min_rise, "tol": args.tol},
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"summary": summary, "rooms": rows}, fh, indent=1)
    print(json.dumps(summary, indent=1))
    print(f"\nwrote {args.out}  ({len(rows)} rooms)")


if __name__ == "__main__":
    main()
