"""A distributional appearance metric, since every number so far is geometric.

Every metric in this work counts a geometric violation. None of them asks
whether a generated room *looks* like a real one, which is the question FID
answers for every other scene-synthesis paper. That gap is real and a reviewer
is right to press it.

What is reported here is **not image FID**. There is no renderer on this machine,
so there are no photographs to compare. Instead each layout is rasterised to a
top-down semantic map — the room, the tiers shaded by height, and every object
footprint coloured by category — and the Fréchet distance is taken between real
and generated feature distributions under a frozen DINOv3 ViT-L/16.

What it actually measures was checked before any result was reported, by
corrupting one side of a real-vs-real comparison (120 rooms each, disjoint):

    B set condition                            Frechet vs A (real)
    real, disjoint rooms                                      1.57   <- the floor
    + every object moved down to the datum tier               1.65
    + categories permuted                                     1.63
    + objects dropped by half                                 1.82
    + positions jittered by 0.5 m                             1.88
    + positions randomised inside the room                    2.30

Read that honestly. It separates **in-plane composition** clearly — random
placement sits 0.7 above the floor. It sees **which tier an object stands on
barely**: flattening the ground truth moves it 0.08, against a between-sample
floor of 1.57. And it is nearly blind to **which category is where**.

So this is a composition metric. It is not a substitute for `elevation_f1`,
which is what measures use of the elevation, and a method could score well here
while putting a bed in the kitchen. It says nothing about materials, lighting or
geometry detail, and it is not comparable to an image FID from any other paper.
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

from elevate3d.core.scene import TIER
from elevate3d.eval.violations import footprint
from elevate3d.gen.dataset import CATEGORIES

IMG = 224
CAT_ID = {c: i for i, c in enumerate(CATEGORIES)}


def _palette(n: int) -> np.ndarray:
    """Distinct, stable colours; a category must always look the same."""
    i = np.arange(n)
    h = (i * 0.61803398875) % 1.0
    s = 0.55 + 0.35 * ((i % 3) / 2.0)
    v = 0.55 + 0.40 * ((i % 2))
    import colorsys
    return np.array([colorsys.hsv_to_rgb(hh, ss, vv)
                     for hh, ss, vv in zip(h, s, v)], dtype=np.float32)


PALETTE = _palette(len(CATEGORIES) + 1)


def rasterise(es, size: int = IMG) -> np.ndarray:
    """Top-down map with height as its own channel.

    R carries the object category, G the height of the surface a pixel belongs
    to, B the occupancy. The first version painted category colour over the tier
    shading, which made an object on a platform pixel-identical to the same
    object on the datum -- the raster was blind to the one thing being evaluated,
    and the metric duly scored the flat baseline *better* than real-against-real.
    """
    from shapely import contains_xy
    from shapely.geometry import Polygon

    room = np.asarray(es.room.polygon, float)
    c = room.mean(0)
    r = float(max(np.abs(room - c).max(), 1e-3)) * 1.08
    gx, gy = np.meshgrid(np.linspace(c[0] - r, c[0] + r, size),
                         np.linspace(c[1] + r, c[1] - r, size))
    img = np.zeros((size, size, 3), dtype=np.float32)

    rp = Polygon(room)
    if not rp.is_valid:
        rp = rp.buffer(0)
    img[..., 2] = 0.25 * contains_xy(rp, gx.ravel(), gy.ravel()).reshape(size, size)

    hs = [t.height for t in es.field.tiers]
    lo, hi = (min(hs), max(hs)) if hs else (0.0, 0.0)
    span = max(hi - lo, 1e-6)
    for t in es.field.tiers:
        m = contains_xy(t.shp, gx.ravel(), gy.ravel()).reshape(size, size)
        img[..., 1][m] = 0.15 + 0.85 * (t.height - lo) / span

    for o in es.objects:
        sup = es.support_of(o.oid)
        if sup.kind != TIER or es.dz.get(o.oid, 0.0) > 0.05:
            continue
        fp = footprint(o)
        if fp.is_empty or fp.area < 1e-9:
            continue
        m = contains_xy(fp, gx.ravel(), gy.ravel()).reshape(size, size)
        try:
            h = es.field.tier(sup.ref).height
        except KeyError:
            h = 0.0
        img[..., 0][m] = 0.2 + 0.8 * CAT_ID.get(
            o.category, len(CATEGORIES)) / len(CATEGORIES)
        img[..., 1][m] = 0.15 + 0.85 * (h - lo) / span
        img[..., 2][m] = 1.0

    for tr in es.field.transitions:
        band = tr.line.buffer(0.09, cap_style=2)
        m = contains_xy(band, gx.ravel(), gy.ravel()).reshape(size, size)
        img[..., 0][m] = 1.0
        img[..., 2][m] = 0.6
    return img


@torch.no_grad()
def features(images, model, device, bs: int = 32) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(images), bs):
        x = torch.from_numpy(np.stack(images[i:i + bs])).to(device)
        x = x.permute(0, 3, 1, 2)
        x = (x - mean) / std
        f = model(pixel_values=x).last_hidden_state
        out.append(f.mean(dim=1).float().cpu().numpy())
    return np.concatenate(out, 0)


def frechet(a: np.ndarray, b: np.ndarray) -> float:
    from scipy import linalg
    mu1, mu2 = a.mean(0), b.mean(0)
    s1 = np.cov(a, rowvar=False)
    s2 = np.cov(b, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/home/gino/data/elevate3d/frontelev5")
    ap.add_argument("--runs", default="/home/gino/data/elevate3d/runs6")
    ap.add_argument("--dino", default="/home/gino/data/reroom/dinov3-vitl16")
    ap.add_argument("--out", default="outputs/frechet_layout.json")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--methods", default="ours,no_height,flatten,per_tier")
    args = ap.parse_args()

    from plot_scenes import load as load_scene_dict
    from sample_elevate import load_model_generic, per_tier_sample, sample_scene

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
    train_recs = [r for r, k in zip(recs, keys) if k not in val][:args.n]
    print(f"{len(val_recs)} held-out, {len(train_recs)} train-side rooms",
          flush=True)

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    from transformers import AutoModel
    dino = AutoModel.from_pretrained(args.dino).to(dev).eval()

    sets = {}
    sets["real (held out)"] = [rasterise(load_scene_dict(r["elev"]))
                               for r in val_recs]
    # a floor for the metric: real rooms against other real rooms
    sets["real (train side)"] = [rasterise(load_scene_dict(r["elev"]))
                                 for r in train_recs]

    runs = {"ours": "m6_ours", "no_height": "m6_no_height",
            "flatten": "m6_no_tiers", "per_tier": "m6_no_tiers"}
    want = [m.strip() for m in args.methods.split(",") if m.strip()]
    for name in want:
        model = load_model_generic(args.runs, runs[name], dev)
        rg = np.random.default_rng(1)
        imgs = []
        for r in val_recs:
            if name == "per_tier":
                sc = per_tier_sample(model, r["elev"], dev, rg, 0.5, 0.8)
            else:
                sc = sample_scene(model, r["elev"], dev, rng=rg, stop_p=0.5,
                                  tier_t=0.0 if name == "flatten" else 1.0,
                                  flatten=(name == "flatten"))
            imgs.append(rasterise(sc))
        sets[name] = imgs
        print(f"{name} rasterised", flush=True)

    feats = {k: features(v, dino, dev) for k, v in sets.items()}
    ref = feats["real (held out)"]
    res = {k: frechet(ref, v) for k, v in feats.items() if k != "real (held out)"}

    print(f"\nFrechet DINOv3 distance on top-down semantic maps "
          f"(not image FID; see module docstring)")
    print(f"{'against real held-out':28s}{'distance':>10s}")
    for k, v in sorted(res.items(), key=lambda kv: kv[1]):
        print(f"{k:28s}{v:10.2f}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"n": len(val_recs), "metric": "frechet_dinov3_topdown",
               "note": "not comparable to image FID from other papers",
               "results": res}, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
