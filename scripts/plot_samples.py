"""What the methods actually produce, side by side.

Violation rates say a layout is wrong; they do not say how. This renders the
ground truth next to each method on the same room, with violating objects
outlined so the failure mode is visible rather than inferred.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPoly

import torch

from elevate3d.core.scene import TIER
from elevate3d.eval.navigability import ccn_profile
from elevate3d.eval.violations import footprint, violations
from elevate3d.gen.model import Elevate3D

from sample_elevate import load_scene, per_tier_sample, sample_scene, _R

TIER_COLOURS = ["#dfe7ec", "#b9d2e0", "#e8d3ae", "#cbbdd8", "#c4ddc4"]
BAD = {"overhang": "#c0392b", "straddling": "#8e44ad",
       "datum": "#d35400", "headroom": "#16a085"}


def draw(ax, es, title):
    room = np.asarray(es.room.polygon, float)
    for i, t in enumerate(sorted(es.field.tiers, key=lambda t: t.height)):
        ax.add_patch(MPoly(t.polygon, closed=True,
                           facecolor=TIER_COLOURS[i % len(TIER_COLOURS)],
                           edgecolor="#7d8f9a", lw=0.9, zorder=1))
        c = t.shp.representative_point()
        ax.text(c.x, c.y, f"{t.height:+.2f}", ha="center", va="center",
                fontsize=6.5, color="#41525c", zorder=6)
    ax.add_patch(MPoly(room, closed=True, fill=False,
                       edgecolor="#2b3a43", lw=1.6, zorder=5))

    flagged = {}
    for v in violations(es):
        flagged.setdefault(v.oid, v.kind)

    for o in es.objects:
        if es.support_of(o.oid).kind != TIER or es.dz.get(o.oid, 0.0) > 0.05:
            continue
        fp = np.asarray(footprint(o).exterior.coords)[:-1]
        kind = flagged.get(o.oid)
        ax.add_patch(MPoly(fp, closed=True, facecolor="#8a6f4e",
                           alpha=0.30 if kind else 0.55,
                           edgecolor=BAD.get(kind, "#5d4a33"),
                           lw=1.8 if kind else 0.6, zorder=3))

    for tr in es.field.transitions:
        ax.plot([tr.p0[0], tr.p1[0]], [tr.p0[1], tr.p1[1]],
                color="#b4472f", lw=2.4, solid_capstyle="butt", zorder=4)

    n_bad = len(flagged)
    pad = 0.05 * max(np.ptp(room[:, 0]), np.ptp(room[:, 1]), 1.0)
    ax.set_xlim(room[:, 0].min() - pad, room[:, 0].max() + pad)
    ax.set_ylim(room[:, 1].min() - pad, room[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.set_title(f"{title}\n{len(es.objects)} objects · {n_bad} flagged",
                 fontsize=7.5, pad=5)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def load_model(runs, name, dev):
    ck = torch.load(os.path.join(runs, name, "best.pt"),
                    map_location="cpu", weights_only=False)
    c = ck.get("cfg") or {
        "d": ck["args"]["d"], "layers": ck["args"]["layers"],
        "heads": ck["args"]["heads"],
        "use_tiers": not ck["args"]["no_tiers"],
        "use_tier_bias": not ck["args"].get("no_tier_bias", True)}
    m = Elevate3D(d=c["d"], layers=c["layers"], heads=c["heads"],
                  use_tiers=c["use_tiers"], use_tier_bias=c["use_tier_bias"]).to(dev)
    m.load_state_dict(ck["model"]); m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/home/gino/data/elevate3d/frontelev")
    ap.add_argument("--mp3d", default="")
    ap.add_argument("--runs", default="/home/gino/data/elevate3d/runs2")
    ap.add_argument("--out", default="outputs/samples.png")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--ours-run", default="m3_ours")
    ap.add_argument("--flat-run", default="m3_no_tiers")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.mp3d:
        recs = [{"elev": json.loads(l)} for l in open(args.mp3d)]
        has_gt = False
    else:
        recs = []
        for f in sorted(glob.glob(os.path.join(args.corpus, "*.jsonl.gz"))):
            with gzip.open(f, "rt") as fh:
                recs += [json.loads(l) for l in fh]
        rng = np.random.default_rng(0)
        keys = np.array([r["flat"]["scene_id"] for r in recs])
        uniq = np.unique(keys); rng.shuffle(uniq)
        val = set(uniq[:max(1, int(len(uniq) * 0.1))].tolist())
        recs = [r for r, k in zip(recs, keys) if k in val]
        has_gt = True

    rng = np.random.default_rng(3)
    picked = [recs[i] for i in rng.choice(len(recs),
                                          min(args.n, len(recs)), replace=False)]

    ours = load_model(args.runs, args.ours_run, dev)
    flat = load_model(args.runs, args.flat_run, dev)

    cols = ["ground truth"] if has_gt else []
    cols += ["ours (tier-aware)", "flatten (single floor)", "per-tier"]
    fig, axes = plt.subplots(len(picked), len(cols),
                             figsize=(3.1 * len(cols), 3.3 * len(picked)))
    axes = np.atleast_2d(axes)
    for r, rec in enumerate(picked):
        col = 0
        if has_gt:
            draw(axes[r, col], load_scene(rec["elev"]), "ground truth"); col += 1
        g = np.random.default_rng(7)
        draw(axes[r, col], sample_scene(ours, rec["elev"], dev, rng=g),
             "ours (tier-aware)"); col += 1
        g = np.random.default_rng(7)
        draw(axes[r, col], sample_scene(flat, rec["elev"], dev, rng=g,
                                        flatten=True),
             "flatten (single floor)"); col += 1
        g = np.random.default_rng(7)
        draw(axes[r, col], per_tier_sample(flat, rec["elev"], dev, g),
             "per-tier")
    fig.suptitle("outlined objects violate an elevation constraint  ·  "
                 "red overhang · purple straddling · orange wrong tier",
                 fontsize=8, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985), h_pad=1.4)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
