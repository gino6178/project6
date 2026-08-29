"""The elevation field: a room floor that is not one plane.

Every layout dataset in use stores a room as a 2D polygon and puts every object
at ``z = 0``.  ``scripts/scan_front_elevation.py`` measured how literally true
that is for 3D-FRONT: across 37,529 rooms the height spread of the floor mesh is
exactly zero.  This module is the representation that removes the assumption.

A room floor becomes a *piecewise-constant height field*: a partition of the
room outline into tiers, each with its own height, plus the transitions (steps,
stairs, ramps, ledges) that connect them.  A single tier at height 0 reproduces
the conventional setting exactly, so everything downstream is a strict
generalisation rather than a separate code path.

Coordinates follow ReRoom: the floor is ``xy``, ``z`` is up, and heights are
metres relative to the room datum (the largest tier, by convention ``0.0``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union

__all__ = [
    "Tier", "Transition", "ElevationField",
    "STEP_MAX_RISE", "TRANSITION_KINDS",
]

TRANSITION_KINDS = ("step", "stair", "ramp", "ledge")

# A single step a person takes in one stride; above this a transition needs
# treads and is a "stair".  Building codes cluster around 0.18-0.20 m per riser.
STEP_MAX_RISE = 0.22


def _as_poly(p) -> Polygon:
    if isinstance(p, Polygon):
        return p
    a = np.asarray(p, dtype=float)
    return Polygon(a)


def _as_valid(poly):
    """Repair a ring without throwing any of it away.

    ``_clean`` keeps only the largest part, which is right when a single region
    is wanted and wrong when measuring coverage: a tier that repairs into two
    lobes still covers both, and dropping one reports a hole in the floor that
    is not there.
    """
    if poly is None or poly.is_empty:
        return None
    return poly if poly.is_valid else poly.buffer(0)


def _clean(poly: Polygon) -> Polygon | None:
    if poly.is_empty:
        return None
    if not poly.is_valid:
        poly = poly.buffer(0)
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area)
    if poly.is_empty or poly.geom_type != "Polygon" or poly.area < 1e-9:
        return None
    return poly


@dataclass
class Tier:
    """One horizontal region of the floor.

    ``holes`` matters more than it looks: a sunken lounge in the middle of a
    room leaves the datum tier as an annulus, and a tier that could only be a
    simple ring would have to overlap the region it surrounds.
    """

    tid: int
    polygon: np.ndarray          # (N, 2) exterior ring, no repeated last point
    height: float                # metres relative to the room datum
    holes: list = field(default_factory=list)
    _shp_cache: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.polygon = np.asarray(self.polygon, dtype=float).reshape(-1, 2)
        self.height = float(self.height)
        self.tid = int(self.tid)
        self.holes = [np.asarray(h, dtype=float).reshape(-1, 2)
                      for h in (self.holes or [])]
        self._shp_cache = None

    @property
    def shp(self):
        """The tier as a geometry, repaired if the ring needs it.

        Rings arrive here from three places — the generator, a JSON round trip
        that rounds coordinates to 4 decimals, and a model's output — and the
        last two both produce self-touching rings that GEOS refuses to
        intersect.  Repairing at the accessor keeps every caller from having to
        remember.  The repair can yield a MultiPolygon; that is left alone,
        because dropping a lobe would silently shrink the tier.
        """
        g = self._shp_cache
        if g is None:
            g = Polygon(self.polygon, [h for h in self.holes if len(h) >= 3])
            if not g.is_valid:
                g = g.buffer(0)
            self._shp_cache = g
        return g

    @property
    def area(self) -> float:
        return float(self.shp.area)

    def contains(self, xy) -> bool:
        return self.shp.covers(Point(float(xy[0]), float(xy[1])))

    def to_dict(self) -> dict:
        d = {"tid": self.tid, "height": round(self.height, 4),
             "polygon": np.round(self.polygon, 4).tolist()}
        if self.holes:
            d["holes"] = [np.round(h, 4).tolist() for h in self.holes]
        return d

    @staticmethod
    def from_dict(d: dict) -> "Tier":
        return Tier(tid=d["tid"], polygon=d["polygon"], height=d["height"],
                    holes=d.get("holes", []))


@dataclass
class Transition:
    """How two tiers meet: the traversable part of the elevation change.

    ``p0``/``p1`` are the endpoints of the shared boundary segment.  ``rise`` is
    signed from ``lo`` to ``hi`` and therefore always positive; ``run`` is the
    horizontal depth the transition occupies on the *lower* tier.
    """

    kind: str
    lo: int
    hi: int
    p0: np.ndarray
    p1: np.ndarray
    rise: float
    run: float = 0.0
    n_tread: int = 0

    def __post_init__(self) -> None:
        if self.kind not in TRANSITION_KINDS:
            raise ValueError(f"unknown transition kind {self.kind!r}")
        self.p0 = np.asarray(self.p0, dtype=float).reshape(2)
        self.p1 = np.asarray(self.p1, dtype=float).reshape(2)
        self.rise = abs(float(self.rise))
        self.run = float(self.run)
        self.n_tread = int(self.n_tread)

    @property
    def width(self) -> float:
        return float(np.linalg.norm(self.p1 - self.p0))

    @property
    def traversable(self) -> bool:
        """A ledge is a drop with no way down; everything else can be used."""
        return self.kind != "ledge"

    @property
    def line(self) -> LineString:
        return LineString([self.p0, self.p1])

    def footprint(self, tier_lo: Tier) -> Polygon | None:
        """The area on the lower tier that the treads occupy."""
        if self.run <= 1e-6:
            return None
        seg = self.line
        d = self.p1 - self.p0
        n = np.array([-d[1], d[0]])
        ln = np.linalg.norm(n)
        if ln < 1e-9:
            return None
        n = n / ln
        for s in (1.0, -1.0):
            band = Polygon([self.p0, self.p1,
                            self.p1 + s * n * self.run,
                            self.p0 + s * n * self.run])
            band = _clean(band)
            if band is None:
                continue
            inter = band.intersection(tier_lo.shp)
            if not inter.is_empty and inter.area > 0.5 * band.area:
                return _clean(inter if inter.geom_type == "Polygon"
                              else max(inter.geoms, key=lambda g: g.area))
        return None

    def to_dict(self) -> dict:
        return {"kind": self.kind, "lo": self.lo, "hi": self.hi,
                "p0": np.round(self.p0, 4).tolist(),
                "p1": np.round(self.p1, 4).tolist(),
                "rise": round(self.rise, 4), "run": round(self.run, 4),
                "n_tread": self.n_tread}

    @staticmethod
    def from_dict(d: dict) -> "Transition":
        return Transition(**d)


@dataclass
class ElevationField:
    """``S = (P, {(R_k, h_k)}, T)`` — the room floor as a height field."""

    boundary: np.ndarray                  # room outline P
    tiers: list[Tier] = field(default_factory=list)
    transitions: list[Transition] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.boundary = np.asarray(self.boundary, dtype=float).reshape(-1, 2)
        if not self.tiers:
            self.tiers = [Tier(0, self.boundary, 0.0)]

    # -- construction ------------------------------------------------------
    @classmethod
    def flat(cls, boundary) -> "ElevationField":
        """The degenerate K = 1 field every current dataset implies."""
        b = np.asarray(boundary, dtype=float).reshape(-1, 2)
        return cls(boundary=b, tiers=[Tier(0, b, 0.0)], transitions=[])

    # -- queries -----------------------------------------------------------
    @property
    def K(self) -> int:
        return len(self.tiers)

    @property
    def is_flat(self) -> bool:
        return self.K == 1 or max(abs(t.height) for t in self.tiers) < 1e-6

    @property
    def datum(self) -> Tier:
        """The reference tier: the largest one."""
        return max(self.tiers, key=lambda t: t.area)

    @property
    def relief(self) -> float:
        """Total vertical extent of the floor."""
        hs = [t.height for t in self.tiers]
        return float(max(hs) - min(hs)) if hs else 0.0

    def tier(self, tid: int) -> Tier:
        for t in self.tiers:
            if t.tid == tid:
                return t
        raise KeyError(f"no tier {tid}")

    def tier_at(self, xy) -> Tier | None:
        """Which tier a plan position stands on (nearest if on a seam)."""
        p = Point(float(xy[0]), float(xy[1]))
        hit = [t for t in self.tiers if t.shp.covers(p)]
        if hit:
            return max(hit, key=lambda t: t.area)
        near = min(self.tiers, key=lambda t: t.shp.distance(p))
        return near if near.shp.distance(p) < 0.05 else None

    def height_at(self, xy) -> float:
        t = self.tier_at(xy)
        return t.height if t is not None else 0.0

    def transitions_between(self, a: int, b: int) -> list[Transition]:
        return [t for t in self.transitions
                if {t.lo, t.hi} == {a, b}]

    def neighbours(self, tid: int) -> list[int]:
        out = set()
        for t in self.transitions:
            if t.lo == tid:
                out.add(t.hi)
            elif t.hi == tid:
                out.add(t.lo)
        return sorted(out)

    def reachable_tiers(self, start: int, max_step: float) -> set[int]:
        """Tiers an agent that can climb ``max_step`` in one go can get to.

        A stair is passable when every individual riser is within reach, which
        is why ``n_tread`` matters and the total rise does not.
        """
        seen, stack = {start}, [start]
        while stack:
            cur = stack.pop()
            for tr in self.transitions:
                if cur not in (tr.lo, tr.hi) or not tr.traversable:
                    continue
                nxt = tr.hi if tr.lo == cur else tr.lo
                if nxt in seen:
                    continue
                per = tr.rise / max(tr.n_tread, 1)
                if tr.kind == "ramp" or per <= max_step + 1e-9:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    def headroom_at(self, xy, ceiling: float) -> float:
        """Clear height above a plan position.

        A tier whose polygon covers ``xy`` and sits *above* the standing surface
        is a soffit — a mezzanine slab or the underside of a raised platform.
        """
        t = self.tier_at(xy)
        if t is None:
            return 0.0
        above = [o.height for o in self.tiers
                 if o.tid != t.tid and o.height > t.height + 0.5
                 and o.shp.covers(Point(float(xy[0]), float(xy[1])))]
        return float(min(above) if above else ceiling) - t.height

    # -- validation --------------------------------------------------------
    def validate(self, tol: float = 0.02) -> list[str]:
        """Structural problems with the field itself, not with a layout."""
        errs = []
        bp = _clean(_as_poly(self.boundary))
        if bp is None:
            return ["boundary is not a valid polygon"]
        ids = [t.tid for t in self.tiers]
        if len(set(ids)) != len(ids):
            errs.append("duplicate tier ids")

        polys = []
        for t in self.tiers:
            p = t.shp
            if p is None or p.is_empty or p.area < 1e-9:
                errs.append(f"tier {t.tid} is not a valid polygon")
                continue
            polys.append((t.tid, p))

        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                inter = polys[i][1].intersection(polys[j][1]).area
                if inter > tol * min(polys[i][1].area, polys[j][1].area):
                    errs.append(f"tiers {polys[i][0]} and {polys[j][0]} overlap "
                                f"by {inter:.3f} m2")

        if polys:
            cover = unary_union([p for _, p in polys])
            gap = bp.difference(cover).area
            if gap > tol * bp.area:
                errs.append(f"tiers leave {gap:.3f} m2 of the room uncovered")
            spill = cover.difference(bp).area
            if spill > tol * bp.area:
                errs.append(f"tiers spill {spill:.3f} m2 outside the room")

        for tr in self.transitions:
            if tr.lo == tr.hi:
                errs.append("transition connects a tier to itself")
                continue
            try:
                dh = self.tier(tr.hi).height - self.tier(tr.lo).height
            except KeyError:
                errs.append(f"transition references a missing tier "
                            f"({tr.lo}, {tr.hi})")
                continue
            if dh < 0:
                errs.append(f"transition lo={tr.lo} hi={tr.hi} has hi below lo")
            elif abs(dh - tr.rise) > 0.02:
                errs.append(f"transition rise {tr.rise:.3f} disagrees with the "
                            f"tier heights ({dh:.3f})")
            if tr.kind == "step" and tr.rise > STEP_MAX_RISE + 1e-6:
                errs.append(f"step of {tr.rise:.3f} m exceeds one riser; it "
                            f"should be a stair")

        # every tier must be reachable by someone, or the field is nonsense
        if self.K > 1:
            big = self.datum.tid
            unreach = set(t.tid for t in self.tiers) - self.reachable_tiers(big, 0.60)
            if unreach:
                errs.append(f"tiers unreachable even for a person: {sorted(unreach)}")
        return errs

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> dict:
        return {"boundary": np.round(self.boundary, 4).tolist(),
                "tiers": [t.to_dict() for t in self.tiers],
                "transitions": [t.to_dict() for t in self.transitions]}

    @staticmethod
    def from_dict(d: dict) -> "ElevationField":
        return ElevationField(
            boundary=d["boundary"],
            tiers=[Tier.from_dict(t) for t in d["tiers"]],
            transitions=[Transition.from_dict(t) for t in d.get("transitions", [])])

    def __repr__(self) -> str:
        hs = ", ".join(f"{t.height:+.2f}" for t in
                       sorted(self.tiers, key=lambda t: t.height))
        return (f"ElevationField(K={self.K}, heights=[{hs}], "
                f"relief={self.relief:.2f}m, "
                f"transitions={len(self.transitions)})")


def make_transition(lo: Tier, hi: Tier, p0, p1, *, kind: str | None = None,
                    tread: float = 0.30) -> Transition:
    """Build a transition between two tiers, choosing the kind from the rise."""
    rise = hi.height - lo.height
    if rise < 0:
        lo, hi = hi, lo
        rise = -rise
    if kind is None:
        kind = "step" if rise <= STEP_MAX_RISE else "stair"
    n = 1 if kind in ("step", "ledge", "ramp") else max(
        1, int(math.ceil(rise / 0.18)))
    run = 0.0 if kind == "ledge" else (tread * n if kind != "ramp"
                                       else rise / 0.083)   # 1:12 accessible
    return Transition(kind=kind, lo=lo.tid, hi=hi.tid, p0=p0, p1=p1,
                      rise=rise, run=run, n_tread=n)
