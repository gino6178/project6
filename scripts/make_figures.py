"""Render every 3D figure the paper needs.

Two stages, because the two Python environments on this machine are disjoint:
`bpy` lives in the infinigen env and `torch` does not. Stage `sample` runs under
the torch env and writes each panel's scene to a plan file; stage `render` runs
under the infinigen env, renders the plan, and composes the figure. Neither
stage imports the other's dependency.

    python scripts/make_figures.py sample  teaser qualitative mp3d
    <infinigen>/python scripts/make_figures.py render teaser qualitative mp3d
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
os.environ.setdefault("REROOM_ROOT", "/home/gino/project/project5")
import elevate3d  # noqa: F401,E402

PANELS = os.path.join(ROOT, "outputs", "panels")
ASSETS = os.path.join(ROOT, "assets")
CORPUS = "/home/gino/data/elevate3d/frontelev5"
MP3D = "/home/gino/data/elevate3d/mp3d_elev/rooms.jsonl"
RUNS = "/home/gino/data/elevate3d/runs6"
# The reported runs. Every table in the paper is sampled at a *calibrated*
# operating point -- stop_p chosen so the mean object count matches ground
# truth, tier_t so tier use does -- and those thresholds differ per method.
# Hard-coding a round number here would put the figures at an operating point
# no table describes, which is how the first version of Figure 5 ended up with
# 0.63 floor coverage against ground truth's 0.34.
EVAL_FRONT = os.path.join(ROOT, "outputs", "evalG_frontelev.json")
EVAL_MP3D = os.path.join(ROOT, "outputs", "evalI_mp3d.json")


def operating_point(eval_json, method, default_stop=0.5, default_t=0.0):
    d = json.load(open(eval_json))
    stop = (d.get("stop_thresholds") or {}).get(method, default_stop)
    t = (d.get("tier_temps") or {}).get(method, default_t)
    return float(stop), float(t)


def corpus(n_files: int = 2, limit: int = 600):
    out = []
    for f in sorted(glob.glob(os.path.join(CORPUS, "*.jsonl.gz")))[:n_files]:
        with gzip.open(f, "rt") as fh:
            for line in fh:
                out.append(json.loads(line))
                if len(out) >= limit:
                    return out
    return out


def held_out(recs):
    """The same 10 % split every evaluation in this repo uses."""
    rng = np.random.default_rng(0)
    keys = np.array([r["flat"]["scene_id"] for r in recs])
    uniq = np.unique(keys)
    rng.shuffle(uniq)
    val = set(uniq[:max(1, int(len(uniq) * 0.1))].tolist())
    return [r for r, k in zip(recs, keys) if k in val]


_BANK = None


def bank():
    """3D-FUTURE indexed by canonical category, for retrieval.

    A generated object is a category and a box; it has no asset id, so there is
    nothing to render. Retrieval is how every scene-synthesis paper turns its
    boxes into geometry, and project5 already implements it -- eq. (30) of
    ReRoom, size distance in log space with a hard cap so a retrieved sofa can
    never be larger than the box the model asked for.
    """
    global _BANK
    if _BANK is None:
        from reroom.data.asset_bank import FutureBank
        roots = sorted(glob.glob(
            "/home/gino/data/reroom/3D-FUTURE/3D-FUTURE-model-part*"))
        _BANK = FutureBank.from_dir(
            roots, bbox_cache="/home/gino/data/reroom/future_bboxes.json")
        print(f"asset bank: {len(_BANK.assets)} models", flush=True)
    return _BANK


def retrofit_jids(d: dict, seed: int = 0) -> dict:
    """Give every jid-less object in a scene dict a size-matched real mesh."""
    b = bank()
    rng = np.random.default_rng(seed)
    used, miss = set(), 0
    for o in d["objects"]:
        if o.get("jid"):
            continue
        # No `exclude`: retrieval is deterministic in the requested box, so two
        # objects of the same category and size get the same mesh. Forcing a
        # distinct model per object gave rooms four different dining chairs
        # around one table, which reads as clutter rather than as a layout.
        got = b.retrieve(o["category"], np.asarray(o["size"], float),
                         topk=1, rng=rng,
                         max_size=np.asarray(o["size"], float) * 1.02)
        if got:
            o["jid"] = got[0][0].aid
            used.add(got[0][0].aid)
        else:
            miss += 1
    if miss:
        print(f"    {miss}/{len(d['objects'])} objects had no asset "
              f"in their category", flush=True)
    return d


def region_fraction(d: dict) -> float:
    """Share of the room floor that is not at the datum."""
    from shapely.geometry import Polygon
    room = Polygon(np.asarray(d["room"]["polygon"], float)).buffer(0)
    if room.area <= 0:
        return 0.0
    up = sum(Polygon(np.asarray(t["polygon"], float)).buffer(0).area
             for t in d["field"]["tiers"] if abs(t["height"]) > 1e-6)
    return float(up / room.area)


def _plan_path(name):
    return os.path.join(PANELS, f"{name}.plan.json")


def write_plan(name, panels, layout):
    os.makedirs(PANELS, exist_ok=True)
    json.dump({"panels": panels, "layout": layout},
              open(_plan_path(name), "w"))
    print(f"wrote {_plan_path(name)}  ({len(panels)} panels)", flush=True)


# ================================================================= sampling
def sample_teaser():
    cand = [r for r in corpus()
            if len(r["elev"]["objects"]) >= 10
            and abs(r["elev"]["meta"]["rise"]) > 0.3]
    r = cand[0]
    rise = r["elev"]["meta"]["rise"]
    prog = r["elev"]["meta"].get("program", "sunken lounge").replace("_", " ")
    write_plan("teaser",
               [{"name": "teaser_flat", "scene": r["flat"], "res": [900, 620]},
                {"name": "teaser_elev", "scene": r["elev"], "res": [900, 620]}],
               {"cols": 2, "cell": [560, 300], "share_crop": True,
                "col_titles": ["K = 1  (every current method)",
                               "K = 2  (elevation field)"],
                "captions": ["one floor plane, every object on it",
                             f"{prog}, rise {rise:+.2f} m, "
                             f"support tree resolves z"],
                "out": "fig_teaser.png"})


def sample_qualitative(device, n_rooms=3):
    import torch
    from sample_elevate import load_model_generic, per_tier_sample, sample_scene

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    cand, seen = [], set()
    for r in held_out(corpus()):
        # one lift per source room: consecutive records are the same 3D-FRONT
        # room lifted under different programs, and three of those in a column
        # read as three near-identical rooms
        if r["flat"]["scene_id"] in seen:
            continue
        if len(r["elev"]["objects"]) < 9 or abs(r["elev"]["meta"]["rise"]) <= 0.25:
            continue
        # The calibrated corpus median puts the elevated region at a small
        # fraction of the room, which is right for the statistics and useless
        # for a figure -- a 6 % band against a wall is invisible at print size.
        # Panels are chosen from the upper half of the region-fraction
        # distribution; the numbers in every table use the whole of it.
        if region_fraction(r["elev"]) < 0.20:
            continue
        seen.add(r["flat"]["scene_id"])
        cand.append(r)
        if len(cand) >= n_rooms:
            break
    print(f"qualitative on {len(cand)} held-out rooms", flush=True)

    ours = load_model_generic(RUNS, "m6_ours", dev)
    flat = load_model_generic(RUNS, "m6_no_tiers", dev)
    op = {m: operating_point(EVAL_FRONT, m)
          for m in ("ours", "flatten", "per_tier")}
    print("operating point, from " + os.path.basename(EVAL_FRONT) + ": "
          + ", ".join(f"{m} stop={p} tier_t={t}" for m, (p, t) in op.items()),
          flush=True)

    panels, caps = [], []
    for i, r in enumerate(cand):
        rg = np.random.default_rng(1)
        scenes = [
            ("gt", r["elev"]),
            ("ours", retrofit_jids(sample_scene(
                ours, r["elev"], dev, rng=rg, stop_p=op["ours"][0],
                tier_t=op["ours"][1]).to_dict())),
            ("flatten", retrofit_jids(sample_scene(
                flat, r["elev"], dev, rng=rg, stop_p=op["flatten"][0],
                tier_t=op["flatten"][1], flatten=True).to_dict())),
            ("pertier", retrofit_jids(per_tier_sample(
                flat, r["elev"], dev, rg, *op["per_tier"]).to_dict())),
        ]
        for tag, sc in scenes:
            panels.append({"name": f"qual{i}_{tag}", "scene": sc,
                           "res": [760, 540]})
            caps.append(f"{len(sc['objects'])} obj")
    write_plan("qualitative", panels,
               {"cols": 4, "cell": [380, 250], "share_crop": True,
                "captions": caps,
                "col_titles": ["ground truth", "Elevate3D (ours)",
                               "flatten baseline", "per-tier baseline"],
                "row_titles": [
                    f"room {i + 1}\n{r['elev']['meta']['rise']:+.2f} m"
                    for i, r in enumerate(cand)],
                "out": "fig_qualitative.png"})


def sample_mp3d(device, n=4):
    import torch
    from sample_elevate import load_model_generic, sample_scene

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    rooms = [json.loads(l) for l in open(MP3D)]
    rooms = [r for r in rooms if len(r.get("field", {}).get("tiers", [])) >= 2]
    seen, pick = set(), []
    for r in rooms:                     # spread across buildings, not within one
        if r["building"] in seen:
            continue
        seen.add(r["building"])
        pick.append(r)
        if len(pick) >= n:
            break
    print(f"mp3d on {len(pick)} rooms from {len(seen)} buildings", flush=True)

    model = load_model_generic(RUNS, "m6_ours", dev)
    stop, tier_t = operating_point(EVAL_MP3D, "ours", default_t=1.8)
    print(f"operating point: stop={stop} tier_t={tier_t}", flush=True)
    panels, caps = [], []
    for i, r in enumerate(pick):
        rg = np.random.default_rng(1)
        sc = sample_scene(model, r, dev, rng=rg, stop_p=stop, tier_t=tier_t)
        panels.append({"name": f"mp3d{i}", "scene": retrofit_jids(sc.to_dict()),
                       "res": [760, 540]})
        hs = sorted({round(t["height"], 2) for t in r["field"]["tiers"]})
        caps.append(f"{r['building'][:11]}  |  tiers at "
                    + ", ".join(f"{h:+.2f}" for h in hs) + " m")
    write_plan("mp3d", panels,
               {"cols": 2, "cell": [430, 300], "captions": caps,
                "out": "fig_mp3d.png",
                "title": "Elevate3D on MP3D-Elev: real elevation fields, "
                         "never seen in training"})


# ================================================================ rendering
def render_plan(name, idx):
    from figure import grid
    from plot_scenes import load as load_scene
    from render3d import render_scene

    plan = json.load(open(_plan_path(name)))
    paths = []
    for p in plan["panels"]:
        out = os.path.join(PANELS, f"{p['name']}.png")
        n = render_scene(load_scene(p["scene"]), out, idx,
                         res=tuple(p.get("res", [900, 620])))
        print(f"  {p['name']}: {n} objects", flush=True)
        paths.append(out)
    lay = dict(plan["layout"])
    out = os.path.join(ASSETS, lay.pop("out"))
    lay["cell"] = tuple(lay["cell"])
    grid(paths, out, **lay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sample", "render"])
    ap.add_argument("figs", nargs="*",
                    default=["teaser", "qualitative", "mp3d"])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    figs = args.figs or ["teaser", "qualitative", "mp3d"]

    if args.stage == "render":
        from render3d import asset_index
        idx = asset_index()
        print(f"{len(idx)} 3D-FUTURE models indexed", flush=True)

    for f in figs:
        print(f"\n=== {args.stage} {f} ===", flush=True)
        if args.stage == "render":
            render_plan(f, idx)
        elif f == "teaser":
            sample_teaser()
        elif f == "qualitative":
            sample_qualitative(args.device)
        elif f == "mp3d":
            sample_mp3d(args.device)
        else:
            raise SystemExit(f"unknown figure {f}")


if __name__ == "__main__":
    main()
