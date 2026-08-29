"""M1 — generating the elevation field itself.

Everything up to here conditions on a field that was given: the corpus's ground
truth in distribution, HouseLayout3D's annotations out of it. That makes the
layout comparison clean but leaves the system unable to produce a multi-elevation
room on its own — the formalism's ``{(R_k, h_k)}`` and ``T`` were inputs, not
outputs.

The architecture-driven region rule made this tractable. A field reduces to a
program label, a convex region in the room's principal frame, and a rise.

The region was an axis-aligned box at first, which left a gap against the
formalism's "a tier is any ``R_k ⊂ P``". Measured, that gap is real but smaller
than it looks: 52 % of real elevated floors are non-rectangular, yet a box
clipped to the room still round-trips them at 0.905 median IoU, because the room
boundary does part of the shaping. Where the box actually fails is the tail —
the regions following a wall that is not orthogonal to the others. The region is
therefore a **support function** sampled at 12 fixed directions
(``geom/support_poly.py``), which contains the box as a special case, is convex
and valid by construction, and lifts the 10th-percentile IoU from 0.471 to
0.626.

Those parameters are *recovered* from the corpus rather than recorded during
generation, so the target is exactly what a reader could recompute from the
released data.

The flat half of each corpus pair supplies the negatives: a model that always
proposes an elevation is not a model of where elevations belong.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon

from ..geom.elevation import ElevationField, Tier, make_transition
from ..geom.support_poly import (N_DIRS, offsets_from_poly, poly_from_offsets)

__all__ = ["PROGRAMS", "FieldParams", "params_from_scene", "field_from_params",
           "room_frame"]

# same order as elevate3d.data.frontelev.PROGRAMS, plus "none" for a flat room
PROGRAMS = ("none", "sunken_lounge", "tatami_platform", "study_dais",
            "dining_tier")
PROGRAM_ID = {p: i for i, p in enumerate(PROGRAMS)}


def room_frame(poly: np.ndarray):
    """Yaw, origin and extent of the room's minimum rotated rectangle.

    Everything M1 predicts lives in this frame, so a room's shape rather than
    its position in the world decides the answer.
    """
    p = Polygon(np.asarray(poly, dtype=float))
    if not p.is_valid:
        p = p.buffer(0)
    mrr = p.minimum_rotated_rectangle
    c = np.asarray(mrr.exterior.coords)[:-1]
    e = c[1] - c[0]
    if np.linalg.norm(c[2] - c[1]) > np.linalg.norm(e):
        e = c[2] - c[1]
    yaw = float(math.atan2(e[1], e[0]))
    R = np.array([[math.cos(-yaw), -math.sin(-yaw)],
                  [math.sin(-yaw), math.cos(-yaw)]])
    q = np.asarray(poly, dtype=float) @ R.T
    return yaw, q.min(0), q.max(0) - q.min(0)


@dataclass
class FieldParams:
    """What M1 emits. ``program == 0`` means the room stays flat."""

    program: int
    offsets: np.ndarray    # (N_DIRS,) support function in the room frame
    rise: float            # metres, signed

    def to_dict(self) -> dict:
        return {"program": int(self.program),
                "offsets": [round(float(x), 4) for x in self.offsets],
                "rise": round(float(self.rise), 4)}


def params_from_scene(d: dict) -> FieldParams:
    """Recover M1's target from a corpus record (either arm)."""
    room = np.asarray(d["room"]["polygon"], dtype=float)
    yaw, origin, extent = room_frame(room)
    field = d["field"]
    tiers = field["tiers"]
    prog = PROGRAM_ID.get((d.get("meta") or {}).get("program", "none"), 0)
    if len(tiers) < 2 or prog == 0:
        return FieldParams(0, np.zeros(N_DIRS, dtype=np.float32), 0.0)

    # the elevated region is the tier whose height is furthest from the datum
    datum = max(tiers, key=lambda t: Polygon(t["polygon"]).area)
    region = max(tiers, key=lambda t: abs(t["height"] - datum["height"]))
    R = np.array([[math.cos(-yaw), -math.sin(-yaw)],
                  [math.sin(-yaw), math.cos(-yaw)]])
    q = Polygon(np.asarray(region["polygon"], dtype=float) @ R.T)
    h = offsets_from_poly(q, origin, extent)
    return FieldParams(prog, h, float(region["height"] - datum["height"]))


def field_from_params(room: np.ndarray, p: FieldParams,
                      min_datum_frac: float = 0.25,
                      min_share: float = 0.6) -> ElevationField | None:
    """Turn M1's numbers back into a field, or refuse.

    A predicted box can be degenerate, swallow the room, or share too little
    edge with the datum to be walked across. Returning ``None`` for those is the
    honest behaviour: the model failed to propose a buildable field, and the
    caller falls back to a flat room rather than being handed a broken one.
    """
    room = np.asarray(room, dtype=float)
    rp = Polygon(room)
    if not rp.is_valid:
        rp = rp.buffer(0)
    if p.program == 0 or abs(p.rise) < 0.06:
        return ElevationField.flat(room)

    yaw, origin, extent = room_frame(room)
    R = np.array([[math.cos(-yaw), -math.sin(-yaw)],
                  [math.sin(-yaw), math.cos(-yaw)]])
    Rb = np.array([[math.cos(yaw), -math.sin(yaw)],
                   [math.sin(yaw), math.cos(yaw)]])
    room_q = Polygon(np.asarray(rp.exterior.coords)[:-1] @ R.T)
    reg_q = poly_from_offsets(p.offsets, origin, extent, clip=room_q,
                              min_area=1.2)
    if reg_q is None:
        return None
    region = Polygon(np.asarray(reg_q.exterior.coords)[:-1] @ Rb.T)
    if not region.is_valid:
        region = region.buffer(0)
    region = region.intersection(rp)
    if region.is_empty or region.geom_type != "Polygon" or region.area < 1.2:
        return None
    rest = rp.difference(region)
    lobes = [g for g in getattr(rest, "geoms", [rest])
             if getattr(g, "geom_type", "") == "Polygon" and g.area > 0.5]
    if not lobes:
        return None
    lobes.sort(key=lambda g: -g.area)
    if sum(g.area for g in lobes) / rp.area < min_datum_frac:
        return None

    def _t(tid, g, h):
        return Tier(tid, np.asarray(g.exterior.coords)[:-1], h,
                    holes=[np.asarray(r.coords)[:-1] for r in g.interiors])

    rise = float(p.rise)
    t_datum = _t(0, lobes[0], 0.0)
    t_region = _t(1, region, rise)
    extra = [_t(2 + i, g, 0.0) for i, g in enumerate(lobes[1:])]

    shared = region.boundary.difference(rp.boundary.buffer(0.05))
    if getattr(shared, "length", 0.0) < min_share:
        return None
    best = None
    for g in getattr(shared, "geoms", [shared]):
        if getattr(g, "geom_type", "") != "LineString":
            continue
        c = np.asarray(g.coords)
        if best is None or np.linalg.norm(c[-1] - c[0]) > np.linalg.norm(best[-1] - best[0]):
            best = c
    if best is None or len(best) < 2:
        return None

    lo_t, hi_t = (t_region, t_datum) if rise < 0 else (t_datum, t_region)
    trans = [make_transition(lo_t, hi_t, best[0], best[-1])]
    for t in extra:
        elo, ehi = (t_region, t) if rise < 0 else (t, t_region)
        trans.append(make_transition(elo, ehi, best[0], best[-1]))

    f = ElevationField(np.asarray(rp.exterior.coords)[:-1],
                       [t_datum, t_region] + extra, trans)
    return None if f.validate() else f
