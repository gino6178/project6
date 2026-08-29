"""Failure modes that only exist once the floor stops being one plane.

Collision rate and floating rate — the metrics PhyScene introduced and everyone
reports — are blind here by construction: they were defined on scenes where
every object stands on the same plane, so none of them can see a sofa hanging
half over the edge of a sunken lounge.  These five can.

F1a overhang    the object hangs past its tier's edge, over lower ground
F1b embedded    it hangs past its tier's edge into *higher* ground, so it
                passes through the platform next to it
F2 straddling   the object spans two tiers and cannot rest on either
F3 step blocked the treads of a transition are occupied
F4 headroom     a standing surface has a tier above it too low to stand under
F5 datum        the declared support tier is not the tier under the object

F1 is split in two because a single-sided version is not a fair measure. A
method that puts everything on the lowest tier — which is what a flat-floor
method does — can never overhang anything, since there is nothing below it to
hang over. It can only fail by driving furniture into the platform beside it,
and without F1b that failure goes uncounted while the same error the other way
round is penalised.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon, box

from ..core.scene import OBJ, TIER, ElevScene

__all__ = ["VIOLATIONS", "footprint", "violations", "violation_rates",
           "tier_utilisation", "STANDING_HEADROOM"]

VIOLATIONS = ("overhang", "embedded", "straddling", "step_blocked",
              "headroom", "datum")

# Below this a person cannot stand under a soffit.  Residential codes put
# habitable headroom at 2.0-2.3 m; 1.9 m is the permissive reading.
STANDING_HEADROOM = 1.90


def footprint(o, inflate: float = 0.0) -> Polygon:
    sx, sy = float(o.size[0]), float(o.size[1])
    p = box(-sx / 2 - inflate, -sy / 2 - inflate,
            sx / 2 + inflate, sy / 2 + inflate)
    p = rotate(p, float(o.yaw), origin=(0, 0), use_radians=True)
    return translate(p, float(o.xy[0]), float(o.xy[1]))


def _grounded(es: ElevScene, o) -> bool:
    """Does this object rest on a tier, rather than on another object?"""
    return es.support_of(o.oid).kind == TIER and es.dz.get(o.oid, 0.0) < 0.05


@dataclass
class Violation:
    kind: str
    oid: str
    value: float          # how bad, in the natural unit of the failure
    detail: str = ""

    def __repr__(self) -> str:
        return f"<{self.kind} {self.oid} {self.value:.3f} {self.detail}>"


def violations(es: ElevScene, *,
               overhang_tol: float = 0.05,
               straddle_tol: float = 0.10,
               step_tol: float = 0.10,
               headroom: float = STANDING_HEADROOM) -> list[Violation]:
    """Every elevation-specific failure in one scene."""
    out: list[Violation] = []
    field = es.field
    ceiling = float(getattr(es.room, "height", 2.8))

    tier_shapes = {t.tid: t.shp for t in field.tiers}

    for o in es.objects:
        s = es.support_of(o.oid)
        if s.kind == OBJ:
            continue                      # governed by its parent, not the floor
        fp = footprint(o)
        if fp.area < 1e-9:
            continue

        # -- F2 straddling: meaningful area on more than one tier ----------
        shares = {tid: sh.intersection(fp).area / fp.area
                  for tid, sh in tier_shapes.items()}
        big = {tid: f for tid, f in shares.items() if f > straddle_tol}
        if len(big) > 1:
            hs = [field.tier(tid).height for tid in big]
            if max(hs) - min(hs) > 0.02:
                out.append(Violation("straddling", o.oid,
                                     float(sorted(big.values())[-2]),
                                     f"tiers {sorted(big)}"))

        if s.kind != TIER:
            continue
        try:
            own = field.tier(s.ref)
        except KeyError:
            out.append(Violation("datum", o.oid, 1.0, "support tier missing"))
            continue

        # -- F5 datum mismatch: standing somewhere other than declared ------
        under = field.tier_at(o.xy)
        if under is not None and under.tid != own.tid:
            out.append(Violation("datum", o.oid,
                                 abs(under.height - own.height),
                                 f"declared {own.tid}, stands on {under.tid}"))

        # -- F1 the footprint leaves its own tier ---------------------------
        outside = fp.difference(own.shp)
        if not outside.is_empty and outside.area / fp.area > overhang_tol:
            over_low = sum(t.shp.intersection(outside).area
                           for t in field.tiers if t.height < own.height - 0.02)
            over_high = sum(t.shp.intersection(outside).area
                            for t in field.tiers if t.height > own.height + 0.02)
            if over_low / fp.area > overhang_tol:
                out.append(Violation("overhang", o.oid,
                                     float(over_low / fp.area),
                                     f"tier {own.tid}"))
            if over_high / fp.area > overhang_tol:
                out.append(Violation("embedded", o.oid,
                                     float(over_high / fp.area),
                                     f"tier {own.tid}"))

        # -- F4 headroom: is there a soffit over where this object stands? --
        clear = field.headroom_at(o.xy, ceiling)
        if clear < headroom - 1e-6 and float(o.size[2]) > clear:
            out.append(Violation("headroom", o.oid, float(clear),
                                 f"needs {float(o.size[2]):.2f} m"))

    # -- F3 step blocked: the treads have to be walkable -------------------
    from shapely.ops import unary_union
    blocked = unary_union([footprint(o) for o in es.objects
                           if _grounded(es, o)]) if es.objects else None
    for i, tr in enumerate(es.field.transitions):
        if not tr.traversable:
            continue
        try:
            band = tr.footprint(field.tier(tr.lo))
        except KeyError:
            continue
        if band is None or band.area < 1e-6:
            continue
        taken = band.intersection(blocked).area if blocked is not None else 0.0
        frac = taken / band.area
        if frac > step_tol:
            out.append(Violation("step_blocked", f"transition/{i}",
                                 float(frac),
                                 f"tiers {tr.lo}->{tr.hi}"))
    return out


def tier_utilisation(es: ElevScene) -> dict:
    """How much of the raised or sunken floor is actually furnished.

    A method can score well on every violation above by simply never putting
    anything on a non-datum tier — the failures are all about objects being in
    the wrong place, and an empty tier has no objects to be wrong.  Measuring
    what fraction of the non-datum floor carries furniture is what separates
    "used the tiers correctly" from "declined to use them".
    """
    from shapely.ops import unary_union
    field = es.field
    datum = field.datum.tid
    others = [t for t in field.tiers if t.tid != datum]
    if not others:
        return {"non_datum_area": 0.0, "non_datum_covered": 0.0,
                "non_datum_objects": 0}
    area = sum(t.area for t in others)
    fps = [footprint(o) for o in es.objects if _grounded(es, o)]
    covered, n = 0.0, 0
    if fps:
        u = unary_union(fps)
        for t in others:
            covered += t.shp.intersection(u).area
        for o in es.objects:
            if not _grounded(es, o):
                continue
            s = es.support_of(o.oid)
            if s.kind == TIER and s.ref != datum:
                n += 1
    return {"non_datum_area": float(area),
            "non_datum_covered": float(covered),
            "non_datum_objects": int(n)}


def violation_rates(scenes) -> dict:
    """Per-object rates for F1/F2/F4/F5 and a per-scene rate for F3.

    F3 is per-scene because a blocked step is a property of the room, not of
    any one object; averaging it per object would hide it.
    """
    n_obj = n_scene = 0
    per_obj = {k: 0 for k in VIOLATIONS if k != "step_blocked"}
    n_step_scenes = 0
    any_scene = 0
    for es in scenes:
        n_scene += 1
        n_obj += len(es.objects)
        vs = violations(es)
        kinds = {v.kind for v in vs}
        for v in vs:
            if v.kind in per_obj:
                per_obj[v.kind] += 1
        if "step_blocked" in kinds:
            n_step_scenes += 1
        if vs:
            any_scene += 1
    util_a = util_c = 0.0
    util_n = 0
    for es in scenes:
        u = tier_utilisation(es)
        util_a += u["non_datum_area"]
        util_c += u["non_datum_covered"]
        util_n += u["non_datum_objects"]

    out = {f"{k}_rate": (per_obj[k] / n_obj if n_obj else 0.0) for k in per_obj}
    out["tier_use_area_frac"] = util_c / util_a if util_a > 0 else 0.0
    out["tier_use_objects_per_scene"] = util_n / max(n_scene, 1)
    out["step_blocked_scene_rate"] = n_step_scenes / n_scene if n_scene else 0.0
    out["any_violation_scene_rate"] = any_scene / n_scene if n_scene else 0.0
    out["n_scenes"] = n_scene
    out["n_objects"] = n_obj
    return out
