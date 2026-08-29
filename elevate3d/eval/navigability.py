"""Who can actually get where: capability-conditioned navigability.

Walkable-area and reachability metrics in the layout literature are computed on
one plane, so they answer "is there floor here" — a question whose answer does
not change when the floor drops 45 cm.  CapNav (CVPR 2026) makes the point for
real scenes: a door sill or a floor height difference stops a wheelchair user
and a sweeping robot while a quadruped walks over it.  This module measures the
same thing on a generated scene.

The signal is the *spread* across agents.  A flat room gives every agent the
same number.  A room with a sunken lounge gives an adult ~100 % and a sweeping
robot the datum tier only, and a generator that has learned the elevation field
but not what it does to circulation will produce rooms whose spread does not
look like the real one.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from ..core.scene import TIER, ElevScene
from .violations import footprint

__all__ = ["Agent", "AGENTS", "navigability", "ccn_profile"]


@dataclass(frozen=True)
class Agent:
    """A mobility profile.

    ``max_step`` is the tallest single riser the agent can take; a stair is
    passable when *each* riser is within it, which is why a 45 cm rise in three
    treads is fine for a quadruped and a 45 cm ledge is not.
    """

    name: str
    max_step: float        # m, per riser
    radius: float          # m, half the body width
    height: float          # m, needed headroom
    ramp_only: bool = False


# Ranges follow CapNav's agent set and the wheeled-legged locomotion literature
# (a rough-terrain policy climbs a 24 cm step, a flat-terrain one 10 cm).
AGENTS = (
    Agent("sweeping_robot", max_step=0.02, radius=0.17, height=0.12),
    Agent("wheelchair",     max_step=0.00, radius=0.35, height=1.30,
          ramp_only=True),
    Agent("wheeled_robot",  max_step=0.05, radius=0.25, height=1.20),
    Agent("quadruped",      max_step=0.20, radius=0.20, height=0.70),
    Agent("adult",          max_step=0.45, radius=0.23, height=1.90),
)
AGENT_BY_NAME = {a.name: a for a in AGENTS}


def _rasterise(poly: Polygon, res: float, x0: float, y0: float,
               nx: int, ny: int) -> np.ndarray:
    """Cell-centre containment test for one polygon."""
    xs = x0 + (np.arange(nx) + 0.5) * res
    ys = y0 + (np.arange(ny) + 0.5) * res
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    from shapely import contains_xy
    return contains_xy(poly, gx.ravel(), gy.ravel()).reshape(nx, ny)


def _gate_polys(es: ElevScene, agent: Agent) -> list:
    """Where the agent may change tier, and between which pair."""
    out = []
    for tr in es.field.transitions:
        if not tr.traversable:
            continue
        if agent.ramp_only and tr.kind != "ramp":
            continue
        per = tr.rise / max(tr.n_tread, 1)
        if tr.kind != "ramp" and per > agent.max_step + 1e-9:
            continue
        if tr.width < 2 * agent.radius:
            continue                       # too narrow to fit through
        band = tr.line.buffer(tr.run + 0.20, cap_style=2)
        out.append((tr.lo, tr.hi, band))
    return out


def navigability(es: ElevScene, agent: Agent, res: float = 0.06) -> dict:
    """Fraction of the room's standable floor the agent can reach.

    Free space is eroded by the agent's radius, so a 35 cm-wide gap is not
    passable by a 70 cm-wide wheelchair; connectivity is then flood-filled from
    the doors, crossing tiers only through gates the agent can use.
    """
    room = Polygon(np.asarray(es.room.polygon, dtype=float))
    if not room.is_valid:
        room = room.buffer(0)
    minx, miny, maxx, maxy = room.bounds
    nx = max(2, int(np.ceil((maxx - minx) / res)))
    ny = max(2, int(np.ceil((maxy - miny) / res)))

    blockers = [footprint(o, inflate=0.0) for o in es.objects
                if es.support_of(o.oid).kind == TIER
                and es.dz.get(o.oid, 0.0) < 0.05
                and float(o.size[2]) > agent.max_step]
    solid = unary_union(blockers) if blockers else None

    walk = room if solid is None else room.difference(solid)
    if walk.is_empty:
        return {"agent": agent.name, "reachable": 0.0, "free_area": 0.0}
    # body width: the agent's centre must stay this far from any obstacle
    walk = walk.buffer(-agent.radius)
    if walk.is_empty:
        return {"agent": agent.name, "reachable": 0.0, "free_area": 0.0}

    free = _rasterise(walk if walk.geom_type == "Polygon" else
                      unary_union(walk), res, minx, miny, nx, ny)
    if not free.any():
        return {"agent": agent.name, "reachable": 0.0, "free_area": 0.0}

    tier_id = np.full((nx, ny), -1, dtype=np.int16)
    for t in es.field.tiers:
        m = _rasterise(t.shp, res, minx, miny, nx, ny)
        tier_id[m] = t.tid

    # headroom: a soffit lower than the agent is a ceiling, not a floor
    for t in es.field.tiers:
        above = [o for o in es.field.tiers
                 if o.tid != t.tid and o.height > t.height + 0.5]
        for o in above:
            m = _rasterise(o.shp, res, minx, miny, nx, ny) & (tier_id == t.tid)
            if (o.height - t.height) < agent.height:
                free &= ~m

    gates = _gate_polys(es, agent)
    gate_masks = [(lo, hi, _rasterise(g, res, minx, miny, nx, ny))
                  for lo, hi, g in gates]

    # seeds: just inside each door, else the largest free patch on the datum
    seeds = []
    for op in getattr(es.room, "openings", []) or []:
        if getattr(op, "kind", "") != "door":
            continue
        c = (np.asarray(op.p0, float)[:2] + np.asarray(op.p1, float)[:2]) / 2.0
        i = int(np.clip((c[0] - minx) / res, 0, nx - 1))
        j = int(np.clip((c[1] - miny) / res, 0, ny - 1))
        best, bd = None, 1e9
        for di in range(-8, 9):
            for dj in range(-8, 9):
                a, b = i + di, j + dj
                if 0 <= a < nx and 0 <= b < ny and free[a, b]:
                    d = di * di + dj * dj
                    if d < bd:
                        best, bd = (a, b), d
        if best:
            seeds.append(best)
    if not seeds:
        datum = es.field.datum.tid
        cand = np.argwhere(free & (tier_id == datum))
        if not len(cand):
            cand = np.argwhere(free)
        if not len(cand):
            return {"agent": agent.name, "reachable": 0.0, "free_area": 0.0}
        seeds = [tuple(cand[len(cand) // 2])]

    seen = np.zeros((nx, ny), dtype=bool)
    q = deque()
    for s in seeds:
        if free[s]:
            seen[s] = True
            q.append(s)

    def can_cross(a, b):
        ta, tb = tier_id[a], tier_id[b]
        if ta == tb:
            return True
        if ta < 0 or tb < 0:
            return False
        for lo, hi, m in gate_masks:
            if {lo, hi} == {int(ta), int(tb)} and m[a] and m[b]:
                return True
        return False

    while q:
        i, j = q.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if not (0 <= a < nx and 0 <= b < ny):
                continue
            if seen[a, b] or not free[a, b]:
                continue
            if not can_cross((i, j), (a, b)):
                continue
            seen[a, b] = True
            q.append((a, b))

    cell = res * res
    return {
        "agent": agent.name,
        "reachable": float(seen.sum() / free.sum()),
        "free_area": float(free.sum() * cell),
        "reachable_area": float(seen.sum() * cell),
    }


def ccn_profile(es: ElevScene, agents=AGENTS, res: float = 0.06) -> dict:
    """The whole curve — this is the number the paper reports."""
    return {a.name: navigability(es, a, res)["reachable"] for a in agents}


def ccn_spread(profile: dict) -> float:
    """How differently the room treats its most and least able visitor."""
    if not profile:
        return 0.0
    return float(max(profile.values()) - min(profile.values()))
