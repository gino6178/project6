"""FRONT-Elev as tensors.

The one structural decision here is what the network is asked to emit. It never
emits an absolute ``z``: it points at a support — a tier or an already-placed
object — and gives an offset above it, and ``ElevScene.resolve`` computes the
height. Floating and vertical interpenetration are therefore not in the output
space at all, which is the difference from guidance-based methods that emit a
free ``z`` and repair it afterwards.

Everything is normalised into a room frame: centred on the room centroid, scaled
by its half-diagonal, so a 4 m bedroom and a 9 m living room look alike to the
attention. Heights are *not* scaled — a 0.4 m step is 0.4 m in any room, and
dividing it by the room size would destroy the quantity the model exists to
learn.
"""
from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["ElevCorpus", "CATEGORIES", "MAX_OBJECTS", "MAX_TIERS",
           "BOUNDARY_POINTS", "collate", "SUPPORT_CEILING", "scene_to_arrays"]

MAX_OBJECTS = 40
MAX_TIERS = 6
BOUNDARY_POINTS = 32
HEIGHT_SCALE = 1.0        # heights stay in metres, deliberately

# ReRoom's canonical vocabulary, in frequency order so ids are stable
CATEGORIES = (
    "dining_chair", "pendant_lamp", "nightstand", "wardrobe", "double_bed",
    "coffee_table", "tv_stand", "side_table", "dining_table", "sideboard",
    "armchair", "sofa", "office_chair", "plant", "drawer_chest", "floor_lamp",
    "desk", "ceiling_lamp", "dressing_table", "loveseat", "l_sofa",
    "decoration", "single_bed", "misc", "cabinet", "kids_bed", "wine_cabinet",
    "shelf", "barstool", "lounge_chair", "bunk_bed", "shoe_cabinet",
    "bookshelf", "stool", "tv", "mirror", "rug", "curtain",
)
CAT_ID = {c: i for i, c in enumerate(CATEGORIES)}
N_CATEGORIES = len(CATEGORIES) + 1          # + unknown
UNKNOWN_CAT = len(CATEGORIES)

# support slots: [0 .. MAX_TIERS-1] tiers, then already-placed objects, then
# this one for ceiling-hung things
SUPPORT_CEILING = MAX_TIERS + MAX_OBJECTS


def _resample(poly: np.ndarray, n: int) -> np.ndarray:
    """Arc-length resampling of a closed polygon to ``n`` points."""
    p = np.asarray(poly, dtype=float)
    p = np.vstack([p, p[:1]])
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total < 1e-9:
        return np.repeat(p[:1], n, axis=0)
    t = np.linspace(0.0, total, n, endpoint=False)
    out = np.empty((n, 2))
    for k, tt in enumerate(t):
        i = int(np.searchsorted(cum, tt, side="right") - 1)
        i = min(max(i, 0), len(seg) - 1)
        u = (tt - cum[i]) / max(seg[i], 1e-9)
        out[k] = p[i] * (1 - u) + p[i + 1] * u
    return out


def _poly_stats(poly: np.ndarray, c: np.ndarray, s: float) -> np.ndarray:
    """Centroid, extent and area of a tier, in the room frame."""
    q = (np.asarray(poly, dtype=float) - c) / s
    lo, hi = q.min(0), q.max(0)
    # shoelace
    x, y = q[:, 0], q[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    return np.array([*q.mean(0), *(hi - lo), area], dtype=np.float32)


def scene_to_arrays(d: dict, rng: np.random.Generator | None = None) -> dict:
    """One corpus record -> flat arrays. ``d`` is the ``elev`` (or ``flat``) dict."""
    room = np.asarray(d["room"]["polygon"], dtype=float)
    c = room.mean(0)
    s = float(max(np.linalg.norm(room - c, axis=1).max(), 1e-3))

    bnd = (_resample(room, BOUNDARY_POINTS) - c) / s

    field = d["field"]
    tiers = field["tiers"][:MAX_TIERS]
    tid_slot = {t["tid"]: i for i, t in enumerate(tiers)}
    n_tiers = len(tiers)

    # per-tier: geometry + height + the smallest riser leaving it, which is what
    # decides who can get on
    up = {t["tid"]: [] for t in tiers}
    down = {t["tid"]: [] for t in tiers}
    for tr in field.get("transitions", []):
        per = tr["rise"] / max(tr.get("n_tread", 1), 1)
        if tr["lo"] in up:
            up[tr["lo"]].append(per)
        if tr["hi"] in down:
            down[tr["hi"]].append(per)

    T = np.zeros((MAX_TIERS, 9), dtype=np.float32)
    tmask = np.zeros(MAX_TIERS, dtype=bool)
    areas = []
    for i, t in enumerate(tiers):
        st = _poly_stats(np.asarray(t["polygon"], float), c, s)
        areas.append(st[4])
        T[i, :5] = st
        T[i, 5] = t["height"] * HEIGHT_SCALE
        T[i, 6] = min(up[t["tid"]], default=0.0)
        T[i, 7] = min(down[t["tid"]], default=0.0)
        tmask[i] = True
    if areas:
        T[np.argmax(areas), 8] = 1.0                    # datum flag

    # MP3D-Elev rooms are unfurnished: they are fields to generate into
    objs = d.get("objects", [])[:MAX_OBJECTS]
    oid_slot = {o["oid"]: i for i, o in enumerate(objs)}
    n = len(objs)

    cat = np.full(MAX_OBJECTS, UNKNOWN_CAT, dtype=np.int64)
    box = np.zeros((MAX_OBJECTS, 7), dtype=np.float32)   # xy, cos, sin, size3
    sup = np.zeros(MAX_OBJECTS, dtype=np.int64)
    dz = np.zeros(MAX_OBJECTS, dtype=np.float32)
    omask = np.zeros(MAX_OBJECTS, dtype=bool)

    for i, o in enumerate(objs):
        cat[i] = CAT_ID.get(o["category"], UNKNOWN_CAT)
        xy = (np.asarray(o["xy"], float) - c) / s
        yaw = float(o["yaw"])
        sz = np.asarray(o["size"], float)
        box[i] = [xy[0], xy[1], np.cos(yaw), np.sin(yaw),
                  sz[0] / s, sz[1] / s, sz[2] * HEIGHT_SCALE]
        kind, _, ref = str(o["support"]).partition(":")
        if kind == "tier":
            sup[i] = tid_slot.get(int(ref), 0)
        elif kind == "obj":
            sup[i] = MAX_TIERS + oid_slot.get(ref, 0)
        elif kind == "ceiling":
            sup[i] = SUPPORT_CEILING
        else:
            sup[i] = 0
        dz[i] = float(o["dz"]) * HEIGHT_SCALE
        omask[i] = True

    return {
        "boundary": bnd.astype(np.float32),
        "room": np.array([s, float(d["room"].get("height", 2.8))],
                         dtype=np.float32),
        "tiers": T, "tier_mask": tmask, "n_tiers": np.int64(n_tiers),
        "cat": cat, "box": box, "sup": sup, "dz": dz, "obj_mask": omask,
        "n_obj": np.int64(n),
    }


class ElevCorpus(Dataset):
    """FRONT-Elev pairs. ``arm`` selects the flat or the elevated half."""

    def __init__(self, root: str, arm: str = "elev", split: str = "train",
                 val_frac: float = 0.1, seed: int = 0,
                 order: str = "shuffle"):
        import glob
        self.arm = arm
        self.order = order
        recs = []
        for f in sorted(glob.glob(os.path.join(root, "*.jsonl.gz"))):
            with gzip.open(f, "rt") as fh:
                for line in fh:
                    recs.append(line)
        if not recs:
            raise FileNotFoundError(f"no shards under {root}")
        # split by the *flat* scene id so a room never straddles the split
        rng = np.random.default_rng(seed)
        keys = np.array([json.loads(r)["flat"]["scene_id"] for r in recs])
        uniq = np.unique(keys)
        rng.shuffle(uniq)
        n_val = max(1, int(len(uniq) * val_frac))
        val = set(uniq[:n_val].tolist())
        keep = [i for i, k in enumerate(keys)
                if (k in val) == (split == "val")]
        self.recs = [recs[i] for i in keep]
        self.split = split

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        d = json.loads(self.recs[i])[self.arm]
        rng = np.random.default_rng(i)
        a = scene_to_arrays(d, rng)
        if self.order == "shuffle" and int(a["n_obj"]) > 1:
            # ATISS trains on random permutations so the model is a set model;
            # supports point backwards, so the permutation has to keep parents
            # before children
            a = _permute_respecting_supports(a, rng)
        return {k: torch.as_tensor(v) for k, v in a.items()}


def _permute_respecting_supports(a: dict, rng) -> dict:
    n = int(a["n_obj"])
    parent = {}
    for i in range(n):
        s = int(a["sup"][i])
        if MAX_TIERS <= s < SUPPORT_CEILING:
            parent[i] = s - MAX_TIERS
    order, placed = [], set()
    pool = list(rng.permutation(n))
    guard = 0
    while pool and guard < 4 * n + 16:
        guard += 1
        nxt = []
        for i in pool:
            p = parent.get(i)
            if p is None or p in placed or p >= n:
                order.append(i)
                placed.add(i)
            else:
                nxt.append(i)
        if len(nxt) == len(pool):
            order.extend(nxt)          # a cycle; keep the original order
            break
        pool = nxt
    if len(order) < n:
        order += [i for i in range(n) if i not in placed]
    inv = {o: k for k, o in enumerate(order)}

    out = dict(a)
    for key in ("cat", "dz"):
        out[key] = a[key].copy()
        out[key][:n] = a[key][list(order)]
    out["box"] = a["box"].copy()
    out["box"][:n] = a["box"][list(order)]
    sup = a["sup"].copy()
    new = sup.copy()
    for k, o in enumerate(order):
        s = int(sup[o])
        if MAX_TIERS <= s < SUPPORT_CEILING:
            p = s - MAX_TIERS
            new[k] = MAX_TIERS + inv.get(p, k)
        else:
            new[k] = s
    out["sup"] = new
    return out


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
