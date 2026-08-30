"""What the elevation metrics never looked at.

Every table in this paper counts an elevation-specific failure: F1a-F5, tier
use, CCN. None of them asks whether two pieces of furniture occupy the same
space, or whether an object is inside the room at all. Rendering the layouts
made that gap visible immediately -- the generated rooms read as incoherent in
a way no reported number accounts for -- so it is measured here with PhyScene's
own evaluator rather than a metric of our own invention.

Col_obj is a 3D IoU, so two objects on different tiers that overlap in plan are
correctly *not* a collision. That matters here and is why the flat evaluator can
be reused unchanged.

Sampling uses the same calibrated operating point as the reported tables, read
from the eval JSON rather than restated, so these numbers describe the same
model behaviour the rest of the paper describes.
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("REROOM_ROOT", "/home/gino/project/project5")
import elevate3d  # noqa: F401,E402

import torch  # noqa: E402

from reroom.core.scene import ObjectInstance, Room, Scene  # noqa: E402
from reroom.eval.physcene import physcene_metrics  # noqa: E402

from plot_scenes import load as load_scene  # noqa: E402
from sample_elevate import (load_model_generic, per_tier_sample,  # noqa: E402
                            sample_scene)

KEYS = ("ps_Col_obj", "ps_Col_scene", "ps_R_out", "ps_R_walkable",
        "ps_n_objects")


def to_reroom(es) -> Scene:
    objs = [ObjectInstance(oid=o.oid, category=o.category,
                           position=np.asarray(o.position, float),
                           yaw=float(o.yaw), size=np.asarray(o.size, float))
            for o in es.objects]
    room = Room(polygon=np.asarray(es.room.polygon, float),
                height=float(getattr(es.room, "height", 2.8)), openings=[],
                room_type=getattr(es.room, "room_type", "living_room"))
    return Scene(scene_id=es.scene_id, room=room, objects=objs, source="e3d")


def agg(rows):
    out = {}
    for k in KEYS:
        v = np.array([r[k] for r in rows if not np.isnan(r.get(k, np.nan))])
        out[k] = float(v.mean()) if len(v) else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/home/gino/data/elevate3d/frontelev5")
    ap.add_argument("--runs", default="/home/gino/data/elevate3d/runs6")
    ap.add_argument("--eval", default="outputs/evalG_frontelev.json")
    ap.add_argument("--out", default="outputs/collision_audit.json")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    ep = json.load(open(args.eval))
    stop = ep.get("stop_thresholds") or {}
    temp = ep.get("tier_temps") or {}
    print("operating point from " + os.path.basename(args.eval) + ": "
          + ", ".join(f"{m}(stop={stop.get(m)},t={temp.get(m)})"
                      for m in ("ours", "flatten", "per_tier")), flush=True)

    recs = []
    for f in sorted(glob.glob(os.path.join(args.corpus, "*.jsonl.gz"))):
        with gzip.open(f, "rt") as fh:
            recs += [json.loads(l) for l in fh]
    rng = np.random.default_rng(0)
    keys = np.array([r["flat"]["scene_id"] for r in recs])
    uniq = np.unique(keys)
    rng.shuffle(uniq)
    val = set(uniq[:max(1, int(len(uniq) * 0.1))].tolist())
    val_recs = [r for r, k in zip(recs, keys) if k in val][:args.n]
    print(f"{len(val_recs)} held-out rooms", flush=True)

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ours = load_model_generic(args.runs, "m6_ours", dev)
    flat = load_model_generic(args.runs, "m6_no_tiers", dev)

    res = {}
    res["gt"] = agg([physcene_metrics(to_reroom(load_scene(r["elev"])))
                     for r in val_recs])
    print("gt done", flush=True)

    for name, fn in (
            ("ours", lambda r, rg: sample_scene(
                ours, r["elev"], dev, rng=rg, stop_p=stop.get("ours", 0.8),
                tier_t=temp.get("ours", 1.8))),
            ("flatten", lambda r, rg: sample_scene(
                flat, r["elev"], dev, rng=rg, stop_p=stop.get("flatten", 0.8),
                tier_t=temp.get("flatten", 0.0), flatten=True)),
            ("per_tier", lambda r, rg: per_tier_sample(
                flat, r["elev"], dev, rg, stop.get("per_tier", 0.05),
                temp.get("per_tier", 0.8)))):
        rg = np.random.default_rng(1)
        res[name] = agg([physcene_metrics(to_reroom(fn(r, rg)))
                         for r in val_recs])
        print(f"{name} done", flush=True)

    print(f"\n{'method':12s}" + "".join(f"{k.replace('ps_', ''):>13s}"
                                       for k in KEYS))
    for m in ("gt", "ours", "flatten", "per_tier"):
        print(f"{m:12s}" + "".join(f"{res[m][k]:13.3f}" for k in KEYS))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"n": len(val_recs), "eval_ref": os.path.basename(args.eval),
               "stop_thresholds": stop, "tier_temps": temp, "results": res},
              open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
