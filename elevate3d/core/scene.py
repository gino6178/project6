"""Layouts whose vertical coordinate is derived, not predicted.

PhyScene and SPREAD let a network emit an absolute ``z`` and then push it back
onto a surface with a differentiable guidance term.  That leaves floating and
vertical interpenetration in the output space, to be repaired.

Here an object stores *what it stands on* and *how far above that surface it
sits*, and ``z`` is computed forward through the support tree.  An object
resting on a surface has ``dz = 0`` and is exactly in contact by construction,
so the contact loss of the original proposal is identically zero and does not
need to exist.  What remains are horizontal failures — overhang, straddling,
blocked steps, headroom — which is what ``elevate3d.eval`` measures.

The object type is ReRoom's ``ObjectInstance`` (project5), so the 3D-FRONT
parser, category table and asset bank are reused rather than reimplemented.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from ..geom.elevation import ElevationField, Tier

__all__ = ["Support", "ElevScene", "TIER", "OBJ", "WALL", "CEILING",
           "parse_support", "SupportCycle"]

TIER = "tier"
OBJ = "obj"
WALL = "wall"
CEILING = "ceiling"


class SupportCycle(RuntimeError):
    """The support graph is not a tree."""


@dataclass(frozen=True)
class Support:
    """What an object stands on.

    ``kind`` is one of ``tier``/``obj``/``wall``/``ceiling``; ``ref`` is the tier
    id or the parent object's ``oid`` and is unused for wall and ceiling.
    """

    kind: str
    ref: Any = None

    def __str__(self) -> str:
        return self.kind if self.ref is None else f"{self.kind}:{self.ref}"

    def to_json(self) -> str:
        return str(self)


def parse_support(s: str | Support) -> Support:
    if isinstance(s, Support):
        return s
    if ":" in s:
        k, r = s.split(":", 1)
        return Support(k, int(r) if k == TIER else r)
    return Support(s)


@dataclass
class ElevScene:
    """A room with an elevation field and a support-parameterised layout."""

    scene_id: str
    room: Any                                   # reroom.core.scene.Room
    field: ElevationField
    objects: list = field(default_factory=list)  # reroom ObjectInstance
    supports: dict = field(default_factory=dict)  # oid -> Support
    dz: dict = field(default_factory=dict)        # oid -> float
    source: str = ""
    meta: dict = field(default_factory=dict)

    # -- support tree ------------------------------------------------------
    def support_of(self, oid: str) -> Support:
        return self.supports.get(oid, Support(TIER, self.field.datum.tid))

    def by_id(self) -> dict:
        return {o.oid: o for o in self.objects}

    def tier_of(self, obj) -> Tier | None:
        """The tier an object ultimately stands over."""
        s = self.support_of(obj.oid)
        if s.kind == TIER:
            try:
                return self.field.tier(s.ref)
            except KeyError:
                return None
        if s.kind == OBJ:
            parent = self.by_id().get(s.ref)
            return self.tier_of(parent) if parent is not None else None
        return self.field.tier_at(obj.xy)

    def children_of(self, oid: str) -> list:
        return [o for o in self.objects
                if self.support_of(o.oid).kind == OBJ
                and self.support_of(o.oid).ref == oid]

    def roots(self) -> list:
        return [o for o in self.objects if self.support_of(o.oid).kind != OBJ]

    # -- the forward pass --------------------------------------------------
    def resolve(self) -> "ElevScene":
        """Write the derived ``z`` (bottom height) into every object.

        ``z`` is never a free parameter, so contact with the support surface is
        exact and floating cannot be represented.
        """
        objs = self.by_id()
        done: dict[str, float] = {}
        ceiling = float(getattr(self.room, "height", 2.8))

        def top_of(oid: str, stack: frozenset) -> float:
            if oid in done:
                o = objs[oid]
                return done[oid] + float(o.size[2])
            if oid in stack:
                raise SupportCycle(f"support cycle through {oid!r}")
            z = bottom(oid, stack | {oid})
            o = objs[oid]
            return z + float(o.size[2])

        def bottom(oid: str, stack: frozenset = frozenset()) -> float:
            if oid in done:
                return done[oid]
            o = objs[oid]
            s = self.support_of(oid)
            d = float(self.dz.get(oid, 0.0))
            if s.kind == TIER:
                try:
                    base = self.field.tier(s.ref).height
                except KeyError:
                    base = self.field.height_at(o.xy)
            elif s.kind == OBJ:
                if s.ref not in objs:
                    base = self.field.height_at(o.xy)
                else:
                    base = top_of(s.ref, stack | {oid})
            elif s.kind == WALL:
                base = self.field.height_at(o.xy)
            elif s.kind == CEILING:
                base = ceiling - float(o.size[2]) - d
                d = 0.0
            else:
                base = self.field.height_at(o.xy)
            z = base + d
            done[oid] = z
            return z

        for o in self.objects:
            o.position[2] = bottom(o.oid)
        return self

    # -- construction ------------------------------------------------------
    @classmethod
    def from_flat(cls, scene, tol: float = 0.02) -> "ElevScene":
        """Lift a conventional flat ReRoom ``Scene`` into this representation.

        Objects sitting on the floor become children of the single tier; objects
        whose bottom coincides with another object's top become children of that
        object; the rest keep their offset above the floor.  This is the inverse
        of what every parser does today, and it is lossless for K = 1.
        """
        f = ElevationField.flat(scene.room.polygon)
        es = cls(scene_id=scene.scene_id, room=scene.room, field=f,
                 objects=list(scene.objects), source=getattr(scene, "source", ""),
                 meta=dict(getattr(scene, "meta", {}) or {}))
        tid = f.datum.tid
        tops = sorted(((float(o.position[2] + o.size[2]), o) for o in es.objects),
                      key=lambda t: -t[0])
        for o in es.objects:
            z = float(o.position[2])
            if z <= tol:
                es.supports[o.oid] = Support(TIER, tid)
                es.dz[o.oid] = 0.0
                continue
            parent = None
            for top, cand in tops:
                if cand.oid == o.oid or top > z + tol:
                    continue
                if abs(top - z) <= tol and _overlaps(o, cand):
                    parent = cand
                    break
            if parent is not None:
                es.supports[o.oid] = Support(OBJ, parent.oid)
                es.dz[o.oid] = 0.0
            else:
                es.supports[o.oid] = Support(TIER, tid)
                es.dz[o.oid] = z
        return es

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> dict:
        from dataclasses import asdict
        return {
            "scene_id": self.scene_id,
            "source": self.source,
            "room": {"polygon": np.round(np.asarray(self.room.polygon), 4).tolist(),
                     "height": float(getattr(self.room, "height", 2.8)),
                     "room_type": getattr(self.room, "room_type", "")},
            "field": self.field.to_dict(),
            "objects": [{
                "oid": o.oid, "category": o.category, "jid": o.jid,
                "xy": np.round(o.xy, 4).tolist(),
                "yaw": round(float(o.yaw), 5),
                "size": np.round(np.asarray(o.size), 4).tolist(),
                "support": str(self.support_of(o.oid)),
                "dz": round(float(self.dz.get(o.oid, 0.0)), 4),
                "z": round(float(o.position[2]), 4),
            } for o in self.objects],
            "meta": self.meta,
        }

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), **kw)

    def __repr__(self) -> str:
        return (f"ElevScene({self.scene_id!r}, {len(self.objects)} objects, "
                f"{self.field!r})")


def _overlaps(a, b, frac: float = 0.15) -> bool:
    """Do two footprints overlap enough for one to rest on the other?"""
    from shapely.geometry import box
    from shapely.affinity import rotate

    def fp(o):
        sx, sy = float(o.size[0]), float(o.size[1])
        p = box(-sx / 2, -sy / 2, sx / 2, sy / 2)
        p = rotate(p, float(o.yaw), origin=(0, 0), use_radians=True)
        x, y = o.xy
        from shapely.affinity import translate
        return translate(p, x, y)

    pa, pb = fp(a), fp(b)
    if pa.area < 1e-9:
        return False
    return pa.intersection(pb).area >= frac * pa.area
