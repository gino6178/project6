"""MP3D-Elev — real elevation fields to furnish, as the out-of-distribution test."""
from __future__ import annotations
import argparse, json, os, sys
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import elevate3d  # noqa
from elevate3d.data.houselayout import elevation_rooms

HL = "/home/gino/data/elevate3d/houselayout3d"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(HL, "structures/layouts_split_by_entity"))
    ap.add_argument("--stairs", default=os.path.join(HL, "stairs"))
    ap.add_argument("--out", default="/home/gino/data/elevate3d/mp3d_elev/rooms.jsonl")
    a = ap.parse_args()
    rooms = elevation_rooms(a.root, a.stairs)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        for r in rooms:
            fh.write(json.dumps(r) + "\n")
    rises = np.array([r["meta"]["rise"] for r in rooms]) if rooms else np.zeros(0)
    areas = np.array([__import__("shapely").geometry.Polygon(r["room"]["polygon"]).area
                      for r in rooms]) if rooms else np.zeros(0)
    print(json.dumps({
        "rooms": len(rooms),
        "buildings": len(set(r["building"] for r in rooms)),
        "transition_kinds": dict(Counter(r["meta"]["transition"] for r in rooms)),
        "rise": {p: round(float(np.percentile(rises, p)), 3) for p in (10, 50, 90)} if len(rises) else {},
        "area": {p: round(float(np.percentile(areas, p)), 1) for p in (10, 50, 90)} if len(areas) else {},
    }, indent=1))
    print(f"wrote {a.out}")

if __name__ == "__main__":
    main()
