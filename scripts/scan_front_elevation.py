"""G1 — does 3D-FRONT contain rooms whose floor is not a single flat plane?

The premise of Elevate3D is that every layout dataset in use projects the floor
onto one plane.  Before building anything we measure how true that is for the
dataset the whole field trains on.

Method.  A 3D-FRONT room owns a set of ``Floor`` meshes.  Each mesh triangle has
a height (3D-FRONT is y-up, so height is component 1) and a footprint area (the
triangle projected onto the xz plane).  Accumulating area against height gives a
per-room elevation histogram; merging bins that sit within ``--merge`` of each
other gives the room's *tiers*.  A room counts as multi-elevation when at least
two tiers are separated by ``--gap`` and each holds ``--min-frac`` of the floor.

The same pass records the mesh ``type`` vocabulary, because a dataset that had
raised platforms would likely name them.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HEIGHT_AXIS = 1          # 3D-FRONT is y-up
PLAN_AXES = (0, 2)       # footprint lives in xz


def tri_areas_heights(mesh: dict):
    """Per-triangle (footprint area, mean height) for one mesh."""
    v = np.asarray(mesh.get("xyz", []), dtype=float).reshape(-1, 3)
    f = np.asarray(mesh.get("faces", []), dtype=int).reshape(-1, 3)
    if len(v) == 0 or len(f) == 0 or f.max() >= len(v):
        return np.zeros(0), np.zeros(0)
    tri = v[f]                                    # (T, 3, 3)
    p = tri[:, :, PLAN_AXES]                      # (T, 3, 2) footprint
    area = 0.5 * np.abs(
        (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
        - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1])
    )
    h = tri[:, :, HEIGHT_AXIS].mean(axis=1)
    return area, h


def tiers_from_hist(area: np.ndarray, h: np.ndarray, merge: float):
    """Greedy area-weighted clustering of heights.  Returns [(height, area)]."""
    if len(area) == 0:
        return []
    order = np.argsort(h)
    h, area = h[order], area[order]
    tiers = []
    start = 0
    for i in range(1, len(h) + 1):
        if i == len(h) or h[i] - h[start] > merge:
            a = float(area[start:i].sum())
            if a > 0:
                w = area[start:i]
                tiers.append((float((h[start:i] * w).sum() / w.sum()), a))
            start = i
    return tiers


def scan_house(path: str, merge: float, gap: float, min_frac: float,
               min_area: float):
    try:
        with open(path) as fh:
            d = json.load(fh)
    except Exception:
        return [], Counter()

    meshes = {m["uid"]: m for m in d.get("mesh", [])}
    types = Counter(m.get("type", "?") for m in d.get("mesh", []))
    house = os.path.splitext(os.path.basename(path))[0]
    rows = []

    for room in d.get("scene", {}).get("room", []):
        areas, heights = [], []
        for child in room.get("children", []):
            m = meshes.get(child.get("ref"))
            if m is None or m.get("type") != "Floor":
                continue
            a, hh = tri_areas_heights(m)
            if len(a):
                areas.append(a)
                heights.append(hh)
        if not areas:
            continue
        area = np.concatenate(areas)
        h = np.concatenate(heights)
        keep = area > 1e-9
        area, h = area[keep], h[keep]
        total = float(area.sum())
        if total < min_area:
            continue

        tiers = tiers_from_hist(area, h, merge)
        tiers.sort(key=lambda t: -t[1])
        big = [t for t in tiers if t[1] / total >= min_frac]
        spread = float(h.max() - h.min()) if len(h) else 0.0

        multi = False
        if len(big) >= 2:
            hs = sorted(t[0] for t in big)
            multi = any(hs[i + 1] - hs[i] > gap for i in range(len(hs) - 1))

        rows.append({
            "house": house,
            "room": room.get("instanceid", ""),
            "type": room.get("type", ""),
            "floor_area": round(total, 3),
            "spread": round(spread, 4),
            "n_tiers": len(tiers),
            "n_tiers_big": len(big),
            "tiers": [[round(t[0], 4), round(t[1], 3)] for t in tiers[:8]],
            "multi": bool(multi),
        })
    return rows, types


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/gino/data/reroom/3D-FRONT_raw/3D-FRONT")
    ap.add_argument("--out", default="outputs/g1_front_elevation.json")
    ap.add_argument("--merge", type=float, default=0.03,
                    help="heights within this are one tier (m)")
    ap.add_argument("--gap", type=float, default=0.10,
                    help="two tiers must differ by at least this (m)")
    ap.add_argument("--min-frac", type=float, default=0.15,
                    help="each tier must hold this fraction of the floor")
    ap.add_argument("--min-area", type=float, default=4.0)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.root) if f.endswith(".json"))
    if args.limit:
        files = files[:args.limit]
    paths = [os.path.join(args.root, f) for f in files]
    print(f"scanning {len(paths)} houses with {args.workers} workers")

    rows, types = [], Counter()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(scan_house, p, args.merge, args.gap,
                          args.min_frac, args.min_area): p for p in paths}
        for fut in as_completed(futs):
            r, t = fut.result()
            rows.extend(r)
            types.update(t)
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(paths)}  rooms={len(rows)}", flush=True)

    n = len(rows)
    multi = [r for r in rows if r["multi"]]
    spreads = np.array([r["spread"] for r in rows]) if n else np.zeros(0)

    summary = {
        "houses": len(paths),
        "rooms": n,
        "multi_elevation_rooms": len(multi),
        "multi_pct": round(100.0 * len(multi) / n, 4) if n else 0.0,
        "spread_percentiles": {
            p: round(float(np.percentile(spreads, p)), 4) for p in (50, 90, 99, 99.9)
        } if n else {},
        "spread_max": round(float(spreads.max()), 4) if n else 0.0,
        "rooms_spread_gt_10cm": int((spreads > 0.10).sum()) if n else 0,
        "rooms_spread_gt_5cm": int((spreads > 0.05).sum()) if n else 0,
        "params": {"merge": args.merge, "gap": args.gap,
                   "min_frac": args.min_frac, "min_area": args.min_area},
        "mesh_types": dict(types.most_common(60)),
        "multi_by_room_type": dict(Counter(r["type"] for r in multi).most_common(20)),
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"summary": summary,
                   "multi_rooms": multi[:500],
                   "top_spread": sorted(rows, key=lambda r: -r["spread"])[:200]},
                  fh, indent=1)

    print(json.dumps(summary, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
