"""K = 1: does Elevate3D degrade to a competitive flat-floor generator?

The paper's formalism claims a strict generalisation — one tier at height zero is
the setting every current method already works in. That claim was never tested,
and a reviewer is right to ask: if the degenerate case is worse than a published
flat-floor method, "generalisation" is the wrong word.

project4 already ran **real PhyScene** on 3D-FRONT living/dining rooms and scored
it with PhyScene's own evaluator (`reroom.eval.physcene`), alongside the 3D-FRONT
designs themselves. Their room ids come from PhyScene's cached preprocessing and
cannot be matched to this repo's parser one-for-one, so this is not the same room
sample.

What makes it comparable anyway is the **shared reference row**: the 3D-FRONT
designs are scored on both samples, so if the reference numbers agree the two
room samples are exchangeable for this purpose. The script reports that
agreement first and refuses to draw a conclusion if it fails.
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

import torch

from reroom.core.scene import ObjectInstance, Room, Scene
from reroom.eval.physcene import physcene_metrics

from sample_elevate import load_model_generic, sample_scene  # noqa: E402

KEYS = ("ps_Col_obj", "ps_Col_scene", "ps_R_out", "ps_R_walkable",
        "ps_R_reach", "ps_n_objects")


def to_reroom(es) -> Scene:
    """An ElevScene at K = 1 is a plain flat scene; hand it to PhyScene's own
    evaluator unchanged rather than re-implementing their metrics."""
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
    ap.add_argument("--flat-arm-run", default="m6_flat_arm")
    ap.add_argument("--ref", default="/home/gino/project/project4/compare_physcene.json")
    ap.add_argument("--out", default="outputs/compare_k1.json")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--room-type", default="living")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    prior = json.load(open(args.ref))
    by_method = {}
    for r in prior:
        by_method.setdefault(r["method"], []).append(r)
    prior_agg = {m: agg(rows) for m, rows in by_method.items()}

    recs = []
    for f in sorted(glob.glob(os.path.join(args.corpus, "*.jsonl.gz"))):
        with gzip.open(f, "rt") as fh:
            recs += [json.loads(l) for l in fh]
    rng = np.random.default_rng(0)
    keys = np.array([r["flat"]["scene_id"] for r in recs])
    uniq = np.unique(keys)
    rng.shuffle(uniq)
    val = set(uniq[:max(1, int(len(uniq) * 0.1))].tolist())
    seen, val_recs = set(), []
    for r, k in zip(recs, keys):
        if k in val and k not in seen and args.room_type in \
                r["flat"]["room"].get("room_type", ""):
            seen.add(k)
            val_recs.append(r)
    val_recs = val_recs[:args.n]
    print(f"{len(val_recs)} held-out {args.room_type} rooms", flush=True)

    # the shared anchor: 3D-FRONT designs scored on *this* room sample
    from plot_scenes import load as load_scene_dict
    ref_rows = [physcene_metrics(to_reroom(load_scene_dict(r["flat"])))
                for r in val_recs]
    mine_ref = agg(ref_rows)
    theirs_ref = prior_agg.get("3D-FRONT reference", {})

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model_generic(args.runs, args.flat_arm_run, dev)
    rg = np.random.default_rng(1)
    gen_rows = [physcene_metrics(to_reroom(
        sample_scene(model, r["flat"], dev, rng=rg, stop_p=0.5)))
        for r in val_recs]
    mine_gen = agg(gen_rows)

    # exchangeability check on the anchor before any conclusion is drawn
    dev_keys = ("ps_Col_obj", "ps_Col_scene", "ps_R_out")
    delta = {k: abs(mine_ref[k] - theirs_ref.get(k, float("nan")))
             for k in dev_keys}
    comparable = all(v < 0.10 for v in delta.values() if not np.isnan(v))

    print(f"\n{'method':30s}" + "".join(f"{k.replace('ps_',''):>12s}" for k in KEYS))
    for name, a in (("3D-FRONT ref (project4)", theirs_ref),
                    ("3D-FRONT ref (this sample)", mine_ref),
                    ("PhyScene (project4)", prior_agg.get("PhyScene", {})),
                    ("ReRoom (project4)", prior_agg.get("ReRoom", {})),
                    ("Elevate3D K=1 (this)", mine_gen)):
        print(f"{name:30s}" + "".join(
            f"{a.get(k, float('nan')):12.3f}" for k in KEYS))
    print(f"\nanchor agreement |ref_mine - ref_theirs|: "
          + ", ".join(f"{k.replace('ps_','')}={v:.3f}" for k, v in delta.items()))
    print("comparable:" , comparable,
          "" if comparable else " -- room samples disagree on the shared "
                                "reference, so no conclusion is drawn")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"n": len(val_recs), "room_type": args.room_type,
               "prior": prior_agg, "ref_this_sample": mine_ref,
               "elevate3d_k1": mine_gen, "anchor_delta": delta,
               "comparable": bool(comparable)},
              open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
