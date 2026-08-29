"""Measure the elevation-field statistics the generator was guessing.

Rise was calibrated against real homes and transferred well; the three that were
not calibrated all disagree with reality (see notes/RESULTS.md). This measures
two of them properly and is explicit about the third.

* **area fraction** — the smaller floor's share of the two-floor union. Directly
  measurable from the annotations.
* **shared edge length** — how much boundary the two floors have in common, i.e.
  how wide the way between them is. Also directly measurable.
* **transitions per room** — *not* measurable here. MP3D-Elev builds exactly one
  transition per room because its builder takes the longest shared segment, so
  the "1.0 per room" in the results table is an artefact of that choice, not an
  observation. Calibrating against it would be fitting to my own construction.
  What the shared-edge distribution does support is a rule: prefer one wide way
  across to several narrow ones.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import elevate3d  # noqa: F401

from elevate3d.data.houselayout import (ROOM_HEIGHT, building_slabs, floors_in,
                                        storeys)

HL = "/home/gino/data/elevate3d/houselayout3d"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(
        HL, "structures/layouts_split_by_entity"))
    ap.add_argument("--out", default="elevate3d/data/geometry_prior.json")
    ap.add_argument("--min-rise", type=float, default=0.10)
    ap.add_argument("--max-rise", type=float, default=0.90)
    ap.add_argument("--touch", type=float, default=0.30)
    args = ap.parse_args()

    import glob
    from shapely.ops import unary_union

    fracs, shares, wall_backed, n = [], [], [], 0
    for bd in sorted(d for d in glob.glob(os.path.join(args.root, "*"))
                     if os.path.isdir(d)):
        slabs = building_slabs(bd)
        for idxs in storeys(slabs):
            fl = sorted(floors_in(slabs, idxs))
            for a in range(len(fl)):
                for b in range(a + 1, len(fl)):
                    i, j = fl[a], fl[b]
                    hi, pi, ai = slabs[i]
                    hj, pj, aj = slabs[j]
                    d = abs(hi - hj)
                    if not (args.min_rise <= d <= args.max_rise):
                        continue
                    if pi.distance(pj) > args.touch:
                        continue
                    n += 1
                    union = unary_union([pi, pj])
                    small = min(ai, aj)
                    fracs.append(small / union.area)
                    # how wide the way between them is: the length of the lower
                    # floor's boundary that runs alongside the upper one
                    lo, up = (pi, pj) if ai >= aj else (pj, pi)
                    shared = up.boundary.intersection(
                        lo.buffer(args.touch + 0.05))
                    shares.append(float(getattr(shared, "length", 0.0)))
                    # does the smaller floor reach the outer edge of the union?
                    wall_backed.append(
                        float(up.boundary.difference(
                            union.boundary.buffer(0.10)).length)
                        < 0.5 * up.boundary.length)

    f = np.asarray(fracs)
    s = np.asarray(shares)
    pct = (10, 25, 50, 75, 90)
    out = {
        "source": "HouseLayout3D structural annotations, 16 Matterport3D buildings",
        "n_pairs": int(n),
        "area_frac": {"percentiles": {p: round(float(np.percentile(f, p)), 4)
                                      for p in pct},
                      "values": [round(float(x), 4) for x in np.sort(f)]},
        "shared_edge_m": {"percentiles": {p: round(float(np.percentile(s, p)), 3)
                                          for p in pct},
                          "values": [round(float(x), 3) for x in np.sort(s)]},
        "wall_backed_frac": round(float(np.mean(wall_backed)), 3),
        "note": ("transitions per room is deliberately absent: MP3D-Elev "
                 "constructs exactly one per room, so it cannot be measured "
                 "from this data without fitting to that construction"),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)

    print(f"{n} real elevation pairs\n")
    print(f"{'percentile':>12s} {'area frac':>11s} {'shared edge':>13s}")
    for p in pct:
        print(f"{p:11d}% {out['area_frac']['percentiles'][p]:11.3f} "
              f"{out['shared_edge_m']['percentiles'][p]:12.2f} m")
    print(f"\nsmaller floor reaches the outer wall: "
          f"{out['wall_backed_frac']:.0%} of pairs")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
