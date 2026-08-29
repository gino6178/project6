"""Cluster-robust reading of the MP3D-Elev result.

The out-of-distribution set is 72 rooms, but they come from 9 buildings and one
building contributes 24 of them. Rooms inside a building share its architect,
its wall angles and its storey heights, so treating 72 rooms as 72 independent
samples overstates the precision of every OOD number in the paper.

This reports the same metrics three ways: pooled over rooms (what was reported),
averaged over buildings with equal weight, and leave-one-building-out, which
shows how much any single building is carrying the result.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import elevate3d  # noqa: F401

import torch

from elevate3d.eval.violations import tier_utilisation, violations
from sample_elevate import load_model_generic, sample_scene

METRICS = ("overhang", "straddling", "datum", "embedded")


def per_room(scene) -> dict:
    n = max(len(scene.objects), 1)
    kinds = [v.kind for v in violations(scene)]
    out = {m: sum(1 for k in kinds if k == m) / n for m in METRICS}
    out["any"] = 1.0 if kinds else 0.0
    out["tier_use"] = float(tier_utilisation(scene)["non_datum_objects"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mp3d", default="/home/gino/data/elevate3d/mp3d_elev/rooms.jsonl")
    ap.add_argument("--runs", default="/home/gino/data/elevate3d/runs6")
    ap.add_argument("--run", default="m6_ours")
    ap.add_argument("--flat-run", default="m6_no_tiers")
    ap.add_argument("--out", default="outputs/ood_cluster.json")
    ap.add_argument("--stop-p", type=float, default=0.5)
    ap.add_argument("--tier-t", type=float, default=1.0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    rooms = [json.loads(l) for l in open(args.mp3d)]
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    counts = defaultdict(int)
    for r in rooms:
        counts[r["building"]] += 1
    print(f"{len(rooms)} rooms from {len(counts)} buildings; "
          f"largest contributes {max(counts.values())} "
          f"({max(counts.values())/len(rooms):.0%})", flush=True)

    results = {}
    for name, run, flat in (("ours", args.run, False),
                            ("flatten", args.flat_run, True)):
        model = load_model_generic(args.runs, run, dev)
        rg = np.random.default_rng(1)
        rows = []
        for r in rooms:
            sc = sample_scene(model, r, dev, rng=rg, stop_p=args.stop_p,
                              tier_t=0.0 if flat else args.tier_t, flatten=flat)
            d = per_room(sc)
            d["building"] = r["building"]
            rows.append(d)
        results[name] = rows
        print(f"{name} sampled", flush=True)

    keys = list(METRICS) + ["any", "tier_use"]
    report = {}
    for name, rows in results.items():
        by_b = defaultdict(list)
        for r in rows:
            by_b[r["building"]].append(r)
        entry = {}
        for k in keys:
            pooled = float(np.mean([r[k] for r in rows]))
            per_building = np.array([np.mean([r[k] for r in v])
                                     for v in by_b.values()])
            # leave-one-building-out: the pooled value with each building removed
            lobo = []
            for b in by_b:
                rest = [r for r in rows if r["building"] != b]
                if rest:
                    lobo.append(float(np.mean([r[k] for r in rest])))
            lobo = np.array(lobo)
            entry[k] = {
                "pooled": pooled,
                "building_mean": float(per_building.mean()),
                "building_sd": float(per_building.std(ddof=1))
                if len(per_building) > 1 else float("nan"),
                "lobo_min": float(lobo.min()) if len(lobo) else float("nan"),
                "lobo_max": float(lobo.max()) if len(lobo) else float("nan"),
            }
        report[name] = entry

    print(f"\n{'metric':12s}{'method':9s}{'pooled':>9s}{'bldg mean':>11s}"
          f"{'bldg sd':>9s}{'LOBO range':>20s}")
    for k in keys:
        for name in report:
            e = report[name][k]
            print(f"{k:12s}{name:9s}{e['pooled']:9.4f}{e['building_mean']:11.4f}"
                  f"{e['building_sd']:9.4f}"
                  f"   [{e['lobo_min']:.4f}, {e['lobo_max']:.4f}]")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"n_rooms": len(rooms), "buildings": dict(counts),
               "report": report}, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
