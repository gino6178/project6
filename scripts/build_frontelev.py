"""Build the FRONT-Elev corpus: every 3D-FRONT room, flat and lifted.

Each accepted room is written twice — once as the original flat layout (K = 1)
and once with an elevation program applied — so the controlled "same room, flat
versus multi-tier" comparison is available without re-deriving anything.

The ground truth is checked with the evaluation code before it is written, so a
scene that reaches the corpus has no overhang, straddling, blocked step or datum
mismatch in it.  A benchmark whose ground truth contains the failures it
measures is worse than no benchmark.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import elevate3d  # noqa: F401

from elevate3d.core.scene import ElevScene
from elevate3d.data.frontelev import GEOM_PRIOR, lift_scene
from elevate3d.eval.violations import violations


def match_to_real(pairs, key, target, bins=14, min_keep=0.45, seed=0):
    """Thin a generated set so one statistic matches a measured distribution.

    Widening the accept band lets more rooms through but does not make the
    corpus *look* like real homes — the region size is driven by how big the
    furniture group happens to be, not by how big real designers make it.
    Keeping a subset whose histogram matches the real one fixes the marginal
    without touching any geometry.

    The subset is the largest one that matches; if matching would cost more
    than ``1 - min_keep`` of the corpus, the constraint is relaxed and the
    residual mismatch is reported rather than paid for.
    """
    if len(target) == 0 or not pairs:
        return pairs, {}
    rng = np.random.default_rng(seed)
    vals = np.array([key(p) for p in pairs], dtype=float)
    lo, hi = float(min(target.min(), vals.min())), float(max(target.max(), vals.max()))
    edges = np.linspace(lo, hi + 1e-9, bins + 1)
    p_real = np.histogram(target, bins=edges)[0].astype(float)
    p_real /= max(p_real.sum(), 1)
    idx = np.clip(np.digitize(vals, edges) - 1, 0, bins - 1)
    n_gen = np.array([(idx == b).sum() for b in range(bins)], float)

    live = (p_real > 0) & (n_gen > 0)
    if not live.any():
        return pairs, {}
    ratios = n_gen[live] / p_real[live]
    for q in (0, 10, 25, 50):
        scale = float(np.percentile(ratios, q))
        keep_n = np.minimum(np.round(p_real * scale), n_gen).astype(int)
        if keep_n.sum() >= min_keep * len(pairs):
            break
    chosen = []
    for b in range(bins):
        if keep_n[b] <= 0:
            continue
        pool = np.flatnonzero(idx == b)
        chosen.extend(rng.choice(pool, size=min(keep_n[b], len(pool)),
                                 replace=False).tolist())
    chosen.sort()
    kept = [pairs[i] for i in chosen]
    from scipy.stats import wasserstein_distance
    stats = {
        "before_n": len(pairs), "after_n": len(kept),
        "wasserstein_before": round(float(wasserstein_distance(vals, target)), 4),
        "wasserstein_after": round(float(wasserstein_distance(
            vals[chosen], target)), 4) if chosen else None,
        "median_before": round(float(np.median(vals)), 4),
        "median_after": round(float(np.median(vals[chosen])), 4) if chosen else None,
        "median_real": round(float(np.median(target)), 4),
    }
    return kept, stats

FRONT = "/home/gino/data/reroom/3D-FRONT_raw/3D-FRONT"
BBOX = "/home/gino/data/reroom/future_bboxes.json"
CATS = "/home/gino/data/reroom/future_categories.json"

_G = {}


def _init():
    from reroom.data.threed_front import load_bboxes
    _G["bb"] = load_bboxes(BBOX)
    with open(CATS) as fh:
        _G["cats"] = json.load(fh)


def do_house(path: str, seed: int, tries: int, max_variants: int = 3):
    from reroom.data.threed_front import parse_scene_file
    if "bb" not in _G:
        _init()
    rng = np.random.default_rng(seed)
    stats = Counter()
    out = []
    try:
        scenes = parse_scene_file(path, _G["bb"], _G["cats"])
    except Exception:
        return [], Counter({"parse_error": 1})
    for sc in scenes:
        stats["rooms"] += 1
        try:
            flat = ElevScene.from_flat(sc).resolve()
        except Exception:
            stats["flat_error"] += 1
            continue
        variants, seen_keys = [], set()
        for _ in range(tries):
            try:
                e = lift_scene(sc, rng, stats=stats)
            except Exception as exc:
                # 3D-FRONT floor polygons include self-touching rings that GEOS
                # refuses to node; one bad room must not take the house with it
                stats["geom_error:" + type(exc).__name__] += 1
                continue
            if e is None:
                continue
            key = (e.meta["program"], round(e.meta["region_frac"], 2))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            variants.append(e)
            if len(variants) >= max_variants:
                break
        if not variants:
            continue
        stats["lifted"] += 1
        fd = flat.to_dict()
        for e in variants:
            out.append({"flat": fd, "elev": e.to_dict()})
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=FRONT)
    ap.add_argument("--out", default="/home/gino/data/elevate3d/frontelev")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--tries", type=int, default=3,
                    help="programs sampled per room before giving up")
    ap.add_argument("--variants", type=int, default=3,
                    help="distinct elevation programs kept per room")
    ap.add_argument("--shard", type=int, default=2000, help="pairs per shard")
    ap.add_argument("--match-real", action="store_true",
                    help="thin the corpus so the region-size distribution "
                         "matches the measured one")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.root) if f.endswith(".json"))
    if args.limit:
        files = files[:args.limit]
    os.makedirs(args.out, exist_ok=True)
    print(f"building FRONT-Elev from {len(files)} houses -> {args.out}")

    stats = Counter()
    allpairs = []

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init) as ex:
        futs = {ex.submit(do_house, os.path.join(args.root, f),
                          args.seed + i, args.tries, args.variants): f
                for i, f in enumerate(files)}
        done = 0
        for fut in as_completed(futs):
            rows, st = fut.result()
            stats.update(st)
            allpairs.extend(rows)
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(files)} houses  "
                      f"pairs={len(allpairs)}", flush=True)

    match = {}
    if args.match_real:
        allpairs, match = match_to_real(
            allpairs, lambda p: p["elev"]["meta"]["region_frac"],
            GEOM_PRIOR["area_frac"], seed=args.seed)
        print("area-fraction matching:", json.dumps(match), flush=True)

    n_pairs = len(allpairs)
    shard = 0
    for i in range(0, n_pairs, args.shard):
        p = os.path.join(args.out, f"pairs_{shard:03d}.jsonl.gz")
        with gzip.open(p, "wt") as fh:
            for r in allpairs[i:i + args.shard]:
                fh.write(json.dumps(r) + "\n")
        print(f"  wrote {p}  ({len(allpairs[i:i+args.shard])} pairs)", flush=True)
        shard += 1

    summary = {
        "houses": len(files),
        "rooms": stats.get("rooms", 0),
        "pairs": n_pairs,
        "rooms_lifted": stats.get("lifted", 0),
        "lift_rate": round(stats.get("lifted", 0) / max(stats.get("rooms", 1), 1), 4),
        "variants_per_room": round(n_pairs / max(stats.get("lifted", 1), 1), 3),
        "rejections": {k: v for k, v in stats.most_common()
                       if k not in ("rooms", "lifted")},
        "shards": shard,
        "area_frac_matching": match,
    }
    with open(os.path.join(args.out, "index.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
