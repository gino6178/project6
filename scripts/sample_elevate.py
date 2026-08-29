"""Sample layouts and score them with the elevation-specific metrics.

Training loss says little here on its own. The ablation that is told nothing
about the tiers reaches a support accuracy of 1.000 during training — because
with one floor to point at, pointing at it is trivial. The cost of not knowing
the floor only appears when the layout is resolved against the *real* elevation
field and measured: objects land on the wrong tier, hang off edges and block
steps. That is what this script measures.

Baselines share the sampler, and differ only in what they are allowed to know:

  ours       tier tokens + tier-relative bias
  no-bias    tier tokens, no bias                       (ablation)
  flatten    no tiers; everything is put on the datum   (what current methods do)
  per-tier   no tiers, but run once per tier with that tier as the room
  gt         the corpus itself, as an upper bound
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import elevate3d  # noqa: F401

from elevate3d.core.scene import ElevScene, Support, OBJ, TIER, CEILING
from elevate3d.eval.navigability import AGENTS, ccn_profile
from elevate3d.eval.violations import violation_rates, violations
from elevate3d.geom.elevation import ElevationField
from elevate3d.gen.dataset import (CATEGORIES, MAX_OBJECTS, MAX_TIERS,
                                   N_CATEGORIES, SUPPORT_CEILING, UNKNOWN_CAT,
                                   scene_to_arrays)
from elevate3d.gen.model import Elevate3D


class _O:
    def __init__(self, oid, cat, xy, yaw, size):
        self.oid = oid
        self.category = cat
        self.jid = None
        self.yaw = float(yaw)
        self.size = np.asarray(size, float)
        self.position = np.array([xy[0], xy[1], 0.0], float)

    @property
    def xy(self):
        return self.position[:2].copy()


class _R:
    def __init__(self, d):
        self.polygon = np.asarray(d["polygon"], float)
        self.height = float(d.get("height", 2.8))
        self.room_type = d.get("room_type", "")
        self.openings = []


def load_scene(d) -> ElevScene:
    from elevate3d.core.scene import parse_support
    es = ElevScene(d["scene_id"], _R(d["room"]),
                   ElevationField.from_dict(d["field"]),
                   [_O(o["oid"], o["category"], o["xy"], o["yaw"], o["size"])
                    for o in d["objects"]],
                   source=d.get("source", ""), meta=d.get("meta", {}))
    for o in d["objects"]:
        es.supports[o["oid"]] = parse_support(o["support"])
        es.dz[o["oid"]] = o["dz"]
    return es.resolve()


def _fp(xy, yaw, size):
    from shapely.affinity import rotate, translate
    from shapely.geometry import box
    p = box(-size[0] / 2, -size[1] / 2, size[0] / 2, size[1] / 2)
    p = rotate(p, float(yaw), origin=(0, 0), use_radians=True)
    return translate(p, float(xy[0]), float(xy[1]))


@torch.no_grad()
def sample_scene(model, rec, device, temperature: float = 1.0,
                 max_obj: int = MAX_OBJECTS, flatten: bool = False,
                 rng=None, n_tries: int = 8) -> ElevScene:
    """Autoregressive rollout conditioned on the room and its elevation field."""
    rng = rng or np.random.default_rng(0)
    a = scene_to_arrays(rec)
    room = np.asarray(rec["room"]["polygon"], float)
    c = room.mean(0)
    s = float(a["room"][0])

    field = ElevationField.from_dict(rec["field"])
    slot_tid = {i: t["tid"] for i, t in enumerate(rec["field"]["tiers"][:MAX_TIERS])}
    datum_slot = int(np.argmax([t.area for t in field.tiers]))

    b = {k: torch.as_tensor(v).unsqueeze(0).to(device) for k, v in a.items()}
    for f in ("cat", "box", "dz", "sup"):
        b[f] = torch.zeros_like(b[f])
    b["obj_mask"] = torch.zeros_like(b["obj_mask"])

    objs, sups, dzs = [], [], []
    for i in range(max_obj):
        out, x, m, iob, q = model(b, i)
        if torch.sigmoid(out["stop"])[0].item() > 0.5:
            break
        logits = out["cat"][0] / max(temperature, 1e-3)
        probs = F.softmax(logits, -1).cpu().numpy()
        probs[UNKNOWN_CAT] = 0.0
        if probs.sum() <= 0:
            break
        probs = probs / probs.sum()
        cid = int(rng.choice(len(probs), p=probs))
        onehot = F.one_hot(torch.tensor([cid], device=device),
                           N_CATEGORIES).float()
        g = model.predict_geometry(q, x, m, iob, onehot, i)
        h = g["h"]

        sz = g["size"].sample(h, 0.5)[0].float().cpu().numpy()
        size = np.clip(np.array([abs(sz[0]) * s, abs(sz[1]) * s, abs(sz[2])]),
                       0.05, 4.0)
        yv = F.normalize(g["yaw"].sample(h, 0.5)[0].float(), dim=-1).cpu().numpy()
        yaw = float(np.arctan2(yv[1], yv[0]))

        sl = int(g["support"][0].argmax().item())
        if flatten and sl < MAX_TIERS:
            # what a flat-floor method does: there is one floor, so everything
            # that stands on the floor stands on the datum
            sl = datum_slot
        dz = float(g["dz"].sample(h, 0.5)[0, 0].item())

        # Draw the plan position from the mixture, and keep the least bad of a
        # few draws.  Every method goes through this same loop, so it changes
        # what "a sample" means for all of them equally rather than favouring
        # one — without it a single unlucky draw dominates a whole scene.
        tier_shape = None
        if sl < MAX_TIERS:
            try:
                tier_shape = field.tier(slot_tid.get(sl, field.datum.tid)).shp
            except KeyError:
                tier_shape = None
        best_xy, best_cost = None, float("inf")
        for _ in range(n_tries):
            xy = g["xy"].sample(h, 1.0)[0].float().cpu().numpy() * s + c
            fp = _fp(xy, yaw, size)
            cost = 0.0
            if tier_shape is not None and fp.area > 1e-9:
                cost += 2.0 * (1.0 - tier_shape.intersection(fp).area / fp.area)
            for pobj in objs:
                inter = fp.intersection(_fp(pobj.xy, pobj.yaw, pobj.size)).area
                cost += inter / max(fp.area, 1e-9)
            if cost < best_cost:
                best_cost, best_xy = cost, xy
            if cost < 0.02:
                break
        xy = best_xy

        oid = f"g{i}"
        objs.append(_O(oid, CATEGORIES[cid], xy, yaw, size))
        if sl < MAX_TIERS:
            sups.append(Support(TIER, slot_tid.get(sl, field.datum.tid)))
            dzs.append(max(dz, 0.0))
        elif sl == SUPPORT_CEILING:
            sups.append(Support(CEILING))
            dzs.append(max(dz, 0.0))
        else:
            j = sl - MAX_TIERS
            if j < len(objs) - 1:
                sups.append(Support(OBJ, objs[j].oid))
                dzs.append(max(dz, 0.0))
            else:
                sups.append(Support(TIER, field.datum.tid))
                dzs.append(0.0)

        # feed it back in
        b["cat"][0, i] = cid
        b["box"][0, i] = torch.as_tensor(
            np.concatenate([(xy - c) / s,
                            [np.cos(yaw), np.sin(yaw)],
                            [size[0] / s, size[1] / s, size[2]]]),
            dtype=torch.float32, device=device)
        b["sup"][0, i] = sl
        b["dz"][0, i] = dz
        b["obj_mask"][0, i] = True

    es = ElevScene(rec["scene_id"] + "__sampled", _R(rec["room"]), field, objs,
                   source="Elevate3D", meta=dict(rec.get("meta", {})))
    for o, sp, d in zip(objs, sups, dzs):
        es.supports[o.oid] = sp
        es.dz[o.oid] = d
    return es.resolve()


def per_tier_sample(model, rec, device, rng) -> ElevScene:
    """Baseline: treat every tier as its own flat room, then merge.

    This is the reviewer's "you do not need a new method" answer, and it is
    given the ground-truth elevation field for free.  What it cannot do is
    reason across tiers — about clearance at a step, or where the circulation
    goes — because each call only ever sees one tier.
    """
    field = ElevationField.from_dict(rec["field"])
    merged, sups, dzs = [], [], []
    for t in field.tiers:
        sub = json.loads(json.dumps(rec))
        sub["room"] = dict(rec["room"])
        sub["room"]["polygon"] = np.asarray(t.polygon).tolist()
        sub["field"] = {"boundary": np.asarray(t.polygon).tolist(),
                        "tiers": [{"tid": 0,
                                   "polygon": np.asarray(t.polygon).tolist(),
                                   "height": 0.0}],
                        "transitions": []}
        part = sample_scene(model, sub, device, rng=rng,
                            max_obj=max(4, MAX_OBJECTS // max(field.K, 1)))
        for o in part.objects:
            o.oid = f"t{t.tid}_{o.oid}"
            merged.append(o)
            sups.append(Support(TIER, t.tid))
            dzs.append(part.dz.get(o.oid.split("_", 1)[1], 0.0))
    es = ElevScene(rec["scene_id"] + "__pertier", _R(rec["room"]), field,
                   merged, source="per-tier")
    for o, sp, d in zip(merged, sups, dzs):
        es.supports[o.oid] = sp
        es.dz[o.oid] = d
    return es.resolve()


def evaluate(scenes, ccn_n: int = 150) -> dict:
    out = violation_rates(scenes)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(scenes), min(ccn_n, len(scenes)), replace=False)
    prof = [ccn_profile(scenes[i]) for i in idx]
    for a in AGENTS:
        out[f"ccn_{a.name}"] = float(np.mean([p[a.name] for p in prof]))
    out["ccn_spread"] = float(np.mean(
        [max(p.values()) - min(p.values()) for p in prof]))
    out["objects_per_scene"] = float(np.mean([len(s.objects) for s in scenes]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/home/gino/data/elevate3d/frontelev")
    ap.add_argument("--runs", default="/home/gino/data/elevate3d/runs")
    ap.add_argument("--out", default="outputs/eval_main.json")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--methods", default="gt,ours,no_bias,flatten,per_tier")
    ap.add_argument("--mp3d", default="",
                    help="evaluate on real HouseLayout3D fields instead of the "
                         "held-out FRONT-Elev split (no ground-truth layout, so "
                         "'gt' is skipped)")
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.mp3d:
        rooms = [json.loads(l) for l in open(args.mp3d)][:args.n]
        val_recs = [{"elev": r} for r in rooms]
        print(f"evaluating on {len(val_recs)} real MP3D-Elev rooms "
              f"(out of distribution: these fields were not generated)",
              flush=True)
        want = [w for w in args.methods.split(",") if w.strip() and w != "gt"]
        results = {}
        def load_model_m(name):
            ck = torch.load(os.path.join(args.runs, name, "best.pt"),
                            map_location="cpu", weights_only=False)
            a = ck["args"]
            m = Elevate3D(d=a["d"], layers=a["layers"], heads=a["heads"],
                          use_tiers=not a["no_tiers"],
                          use_tier_bias=not a["no_tier_bias"]).to(dev)
            m.load_state_dict(ck["model"]); m.eval(); return m
        for meth, run, kw in (("ours", "m2_full", {}),
                              ("no_bias", "m2_no_bias", {}),
                              ("flatten", "m2_no_tiers", {"flatten": True})):
            if meth not in want:
                continue
            m = load_model_m(run)
            rg = np.random.default_rng(1)
            results[meth] = evaluate(
                [sample_scene(m, r["elev"], dev, rng=rg, **kw) for r in val_recs],
                ccn_n=len(val_recs))
            print(f"{meth} done", flush=True)
        if "per_tier" in want:
            m = load_model_m("m2_no_tiers")
            rg = np.random.default_rng(1)
            results["per_tier"] = evaluate(
                [per_tier_sample(m, r["elev"], dev, rg) for r in val_recs],
                ccn_n=len(val_recs))
            print("per_tier done", flush=True)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({"n": len(val_recs), "set": "mp3d_elev",
                       "results": results}, fh, indent=1)
        keys = ["overhang_rate", "embedded_rate", "straddling_rate",
                "datum_rate", "step_blocked_scene_rate", "headroom_rate",
                "any_violation_scene_rate", "tier_use_area_frac",
                "tier_use_objects_per_scene", "ccn_sweeping_robot",
                "ccn_wheelchair", "ccn_adult", "ccn_spread",
                "objects_per_scene"]
        print(f"\n{'metric':28s}" + "".join(f"{k:>12s}" for k in results))
        for k in keys:
            print(f"{k:28s}" + "".join(
                f"{results[m].get(k, float('nan')):12.4f}" for m in results))
        print(f"\nwrote {args.out}")
        return

    import glob
    recs = []
    for f in sorted(glob.glob(os.path.join(args.corpus, "*.jsonl.gz"))):
        with gzip.open(f, "rt") as fh:
            recs += [json.loads(l) for l in fh]
    # same split rule as training
    rng = np.random.default_rng(0)
    keys = np.array([r["flat"]["scene_id"] for r in recs])
    uniq = np.unique(keys)
    rng.shuffle(uniq)
    val = set(uniq[:max(1, int(len(uniq) * 0.1))].tolist())
    val_recs = [r for r, k in zip(recs, keys) if k in val][:args.n]
    print(f"evaluating on {len(val_recs)} held-out rooms", flush=True)

    def load_model(name, **kw):
        p = os.path.join(args.runs, name, "best.pt")
        ck = torch.load(p, map_location="cpu", weights_only=False)
        a = ck["args"]
        m = Elevate3D(d=a["d"], layers=a["layers"], heads=a["heads"],
                      use_tiers=not a["no_tiers"],
                      use_tier_bias=not a["no_tier_bias"]).to(dev)
        m.load_state_dict(ck["model"])
        m.eval()
        return m

    want = [w.strip() for w in args.methods.split(",") if w.strip()]
    results = {}

    if "gt" in want:
        results["gt"] = evaluate([load_scene(r["elev"]) for r in val_recs])
        print("gt done", flush=True)

    if "ours" in want:
        m = load_model("m2_full")
        rg = np.random.default_rng(1)
        results["ours"] = evaluate(
            [sample_scene(m, r["elev"], dev, rng=rg) for r in val_recs])
        print("ours done", flush=True)

    if "no_bias" in want:
        m = load_model("m2_no_bias")
        rg = np.random.default_rng(1)
        results["no_bias"] = evaluate(
            [sample_scene(m, r["elev"], dev, rng=rg) for r in val_recs])
        print("no_bias done", flush=True)

    if "flatten" in want:
        m = load_model("m2_no_tiers")
        rg = np.random.default_rng(1)
        results["flatten"] = evaluate(
            [sample_scene(m, r["elev"], dev, rng=rg, flatten=True)
             for r in val_recs])
        print("flatten done", flush=True)

    if "per_tier" in want:
        m = load_model("m2_no_tiers")
        rg = np.random.default_rng(1)
        results["per_tier"] = evaluate(
            [per_tier_sample(m, r["elev"], dev, rg) for r in val_recs])
        print("per_tier done", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"n": len(val_recs), "results": results}, fh, indent=1)

    keys = ["overhang_rate", "embedded_rate", "straddling_rate", "datum_rate",
            "step_blocked_scene_rate", "headroom_rate", "any_violation_scene_rate",
            "tier_use_area_frac", "tier_use_objects_per_scene",
            "ccn_sweeping_robot", "ccn_wheelchair", "ccn_adult", "ccn_spread",
            "objects_per_scene"]
    print(f"\n{'metric':28s}" + "".join(f"{k:>12s}" for k in results))
    for k in keys:
        print(f"{k:28s}" + "".join(
            f"{results[m].get(k, float('nan')):12.4f}" for m in results))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
