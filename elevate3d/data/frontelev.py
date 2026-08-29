"""FRONT-Elev — lifting real flat layouts onto real elevation programs.

3D-FRONT has no multi-elevation rooms at all (37,529 of 37,529 have a perfectly
planar floor), so it cannot supervise this task directly.  What it does have is
37,529 *professionally designed layouts*: real room outlines, real furniture
groupings, real assets.  This module keeps all of that and changes only the
floor underneath it.

The idea that makes the result usable as supervision is to **grow the elevated
region out of a furniture group rather than stamping a shape onto the room**.  A
real sunken lounge is sized to the conversation group that sits in it and a real
tatami platform is sized to the bed on top of it, so growing from the group both
matches how the feature is designed and guarantees the region boundary never
cuts through an object — which is exactly the failure (tier straddling) the
evaluation is meant to catch in *generated* layouts, not in ground truth.

Each room yields a paired sample: the original flat scene and its elevated
counterpart with the same objects.  That pairing is what makes the controlled
"same room, flat vs multi-tier" comparison possible.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from shapely.affinity import rotate, translate
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union

from ..core.scene import ElevScene, Support, OBJ, TIER
from ..geom.elevation import (STEP_MAX_RISE, ElevationField, Tier, Transition,
                              make_transition)

__all__ = ["PROGRAMS", "ElevationProgram", "lift_scene", "program_for",
           "RISE_PRIOR", "sample_rise"]

# ---------------------------------------------------------------------------
# The vocabulary.  Rises are metres; the ranges come from residential practice
# (a riser is 0.15-0.20 m, a tatami/ji-dai platform is one or two of them, an
# accessible mezzanine needs 2.0 m of headroom underneath).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ElevationProgram:
    name: str
    seed_categories: tuple                # furniture the region is grown from
    rise: tuple                           # (lo, hi), signed
    area_frac: tuple                      # region share of the room
    room_types: tuple                     # which rooms it belongs in
    wall_backed: bool = True              # must the region touch a wall?
    min_room_area: float = 0.0
    min_room_height: float = 0.0
    min_seeds: int = 2                    # a bed alone is enough for a platform
    anchor_categories: tuple = ()         # a single one of these also suffices


PROGRAMS = (
    ElevationProgram(
        "sunken_lounge",
        seed_categories=("sofa", "l_sofa", "loveseat", "coffee_table",
                         "tv_stand", "armchair", "lounge_chair"),
        rise=(-0.60, -0.12), area_frac=(0.04, 0.62),
        room_types=("living_room", "living_dining_room"),
        wall_backed=False, min_room_area=12.0,
        anchor_categories=("sofa", "l_sofa")),
    ElevationProgram(
        "tatami_platform",
        seed_categories=("double_bed", "single_bed", "kids_bed", "bunk_bed",
                         "nightstand"),
        rise=(0.12, 0.55), area_frac=(0.04, 0.66),
        room_types=("bedroom", "kids_room", "elderly_room", "second_bedroom",
                    "master_bedroom"),
        wall_backed=True, min_room_area=8.0, min_seeds=1,
        anchor_categories=("double_bed", "single_bed", "kids_bed", "bunk_bed")),
    ElevationProgram(
        "study_dais",
        seed_categories=("desk", "shelf", "office_chair", "dressing_table",
                         "drawer_chest"),
        rise=(0.12, 0.40), area_frac=(0.04, 0.55),
        room_types=("library", "study"),
        wall_backed=True, min_room_area=7.0,
        anchor_categories=("desk",)),
    ElevationProgram(
        "dining_tier",
        seed_categories=("dining_table", "dining_chair", "sideboard"),
        rise=(0.12, 0.45), area_frac=(0.04, 0.60),
        room_types=("dining_room", "living_dining_room"),
        wall_backed=False, min_room_area=11.0,
        anchor_categories=("dining_table",)),
)

PROGRAM_BY_NAME = {p.name: p for p in PROGRAMS}


def _load_rise_prior():
    """Real floor height differences, measured rather than invented.

    ``scripts/scan_houselayout_elevation.py`` finds 202 adjacent floor pairs at
    different heights inside a storey across 12 of the 16 real Matterport3D
    buildings HouseLayout3D annotates.  Sampling magnitudes from that empirical
    distribution instead of a hand-picked range takes one arbitrary choice out
    of the corpus — it is the parameter a reviewer would otherwise ask where
    the numbers came from.
    """
    path = os.path.join(os.path.dirname(__file__), "rise_prior.json")
    try:
        with open(path) as fh:
            return np.asarray(json.load(fh)["rises"], dtype=float)
    except Exception:
        return np.zeros(0)


RISE_PRIOR = _load_rise_prior()


def _load_geometry_prior():
    """Measured area fraction and shared-edge width from real homes.

    ``scripts/measure_real_geometry.py`` produces it. Only the two quantities
    that are genuinely observable are here; transitions-per-room is not, because
    MP3D-Elev constructs exactly one per room and calibrating against that would
    be fitting to my own builder.
    """
    path = os.path.join(os.path.dirname(__file__), "geometry_prior.json")
    try:
        with open(path) as fh:
            d = json.load(fh)
        return {"area_frac": np.asarray(d["area_frac"]["values"], float),
                "shared_edge": np.asarray(d["shared_edge_m"]["values"], float),
                "wall_backed": float(d["wall_backed_frac"])}
    except Exception:
        return {"area_frac": np.zeros(0), "shared_edge": np.zeros(0),
                "wall_backed": 0.73}


GEOM_PRIOR = _load_geometry_prior()


def sample_rise(program: "ElevationProgram", rng: np.random.Generator) -> float:
    """A magnitude from the measured distribution, clamped to the program.

    Falls back to the program's own range when the prior is unavailable, so the
    generator still runs without the HouseLayout3D scan.
    """
    lo, hi = sorted(abs(v) for v in program.rise)
    sign = -1.0 if min(program.rise) < 0 else 1.0
    if len(RISE_PRIOR):
        band = RISE_PRIOR[(RISE_PRIOR >= lo) & (RISE_PRIOR <= hi)]
        pool = band if len(band) >= 8 else RISE_PRIOR
        v = float(rng.choice(pool))
        # smooth the 202-sample empirical CDF so the corpus is not quantised
        # onto the exact values a handful of buildings happened to have
        v += float(rng.normal(0.0, 0.02))
        return sign * float(np.clip(abs(v), lo, hi))
    return sign * float(rng.uniform(lo, hi))


def program_for(room_type: str, area: float, height: float,
                rng: np.random.Generator) -> ElevationProgram | None:
    ok = [p for p in PROGRAMS
          if any(r in (room_type or "").lower() for r in p.room_types)
          and area >= p.min_room_area and height >= p.min_room_height]
    if not ok:
        return None
    return ok[int(rng.integers(len(ok)))]


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def _footprint(o, inflate: float = 0.0) -> Polygon:
    sx, sy = float(o.size[0]), float(o.size[1])
    p = box(-sx / 2 - inflate, -sy / 2 - inflate,
            sx / 2 + inflate, sy / 2 + inflate)
    p = rotate(p, float(o.yaw), origin=(0, 0), use_radians=True)
    return translate(p, float(o.xy[0]), float(o.xy[1]))


def _largest(g):
    if g is None or g.is_empty:
        return None
    if isinstance(g, MultiPolygon):
        g = max(g.geoms, key=lambda x: x.area)
    return g if getattr(g, "geom_type", "") == "Polygon" else None


def _principal_axes(room_poly: Polygon) -> float:
    """Yaw of the room's minimum rotated rectangle."""
    mrr = room_poly.minimum_rotated_rectangle
    c = np.asarray(mrr.exterior.coords)[:-1]
    e = c[1] - c[0]
    if np.linalg.norm(c[2] - c[1]) > np.linalg.norm(e):
        e = c[2] - c[1]
    return float(math.atan2(e[1], e[0]))


def _axis_rect(pts: np.ndarray, yaw: float, pad: float = 0.0) -> Polygon:
    """Bounding rectangle of ``pts`` in the frame rotated by ``yaw``."""
    c, s = math.cos(-yaw), math.sin(-yaw)
    R = np.array([[c, -s], [s, c]])
    q = pts @ R.T
    lo, hi = q.min(0) - pad, q.max(0) + pad
    rect = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    Rb = np.array([[math.cos(yaw), -math.sin(yaw)],
                   [math.sin(yaw), math.cos(yaw)]])
    return Polygon(rect @ Rb.T)


def _snap_to_walls(region: Polygon, room: Polygon, yaw: float,
                   tol: float = 0.55, max_edges: int = 2) -> Polygon:
    """Push the region's nearest edges against the wall they almost touch.

    A platform that stops 12 cm short of the wall is a modelling artefact, not a
    design.  But snapping *every* near edge inflates the region to fill the
    room: real elevated floors sit against one or two walls, not four, and the
    measured median is 0.27 of the floor against the 0.41 this produced when it
    snapped all four.  Only the ``max_edges`` closest edges move.
    """
    c, s = math.cos(-yaw), math.sin(-yaw)
    R = np.array([[c, -s], [s, c]])
    Rb = np.array([[math.cos(yaw), -math.sin(yaw)],
                   [math.sin(yaw), math.cos(yaw)]])
    rq = np.asarray(room.exterior.coords)[:-1] @ R.T
    gq = np.asarray(region.exterior.coords)[:-1] @ R.T
    lo, hi = gq.min(0), gq.max(0)
    rlo, rhi = rq.min(0), rq.max(0)
    cands = []
    for k in (0, 1):
        d = lo[k] - rlo[k]
        if 0 <= d < tol:
            cands.append((d, k, "lo"))
        d = rhi[k] - hi[k]
        if 0 <= d < tol:
            cands.append((d, k, "hi"))
    cands.sort()
    for _, k, side in cands[:max_edges]:
        if side == "lo":
            lo[k] = rlo[k]
        else:
            hi[k] = rhi[k]
    rect = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    return _largest(Polygon(rect @ Rb.T).intersection(room))


def _rot(yaw):
    c, sn = math.cos(-yaw), math.sin(-yaw)
    fwd = np.array([[c, -sn], [sn, c]])
    c2, s2 = math.cos(yaw), math.sin(yaw)
    return fwd, np.array([[c2, -s2], [s2, c2]])


def _iv(poly: Polygon, R: np.ndarray) -> np.ndarray:
    """Axis-aligned interval box of a polygon in the rotated frame."""
    q = np.asarray(poly.exterior.coords)[:-1] @ R.T
    return np.stack([q.min(0), q.max(0)])          # (2, 2): [lo; hi]


def _rect(lo, hi, Rb: np.ndarray) -> Polygon:
    r = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    return Polygon(r @ Rb.T)


def _free_coords(boxes: list, k: int, lo_o: float, hi_o: float,
                 span: tuple, eps: float = 0.01) -> list:
    """Coordinates on axis ``k`` where a boundary can sit without cutting.

    Only objects that overlap the region on the *other* axis can be cut by a
    boundary on this one, so the occupancy is built from those alone.
    """
    j = 1 - k
    occ = []
    for b in boxes:
        if b[1][j] <= lo_o + eps or b[0][j] >= hi_o - eps:
            continue                                # misses on the other axis
        occ.append((b[0][k] - eps, b[1][k] + eps))
    occ.sort()
    free, cur = [], span[0]
    for a, b in occ:
        if a > cur:
            free.append((cur, a))
        cur = max(cur, b)
    if cur < span[1]:
        free.append((cur, span[1]))
    return free


def _snap_into_free(v: float, free: list, span: tuple, outward: int,
                    policy: str = "nearest") -> float | None:
    """Move a boundary onto the nearest coordinate no object straddles.

    This was outward-only at first, to stop a boundary shrinking past a seed.
    But outward-only inflates every region: a nightstand's rectangle grows until
    it reaches the next free coordinate, which in a furnished room can be the
    far wall, and the corpus ended up with a median region of 0.41 of the floor
    against a measured 0.27.  Nearest-first is the right rule now that
    ``cut_lost_seed`` catches the case it was guarding against.
    """
    for a, b in free:
        if a - 1e-9 <= v <= b + 1e-9:
            return v
    below = [b for a, b in free if b <= v + 1e-9]
    above = [a for a, b in free if a >= v - 1e-9]
    if policy == "outward":
        if outward < 0:
            return max(below) if below else v
        return min(above) if above else v
    cands = []
    if below:
        cands.append(max(below))
    if above:
        cands.append(min(above))
    return min(cands, key=lambda c: abs(c - v)) if cands else v


def _note(stats, why):
    if stats is not None:
        stats[why] = stats.get(why, 0) + 1
    return None


def _resolve_cuts(region: Polygon, objs: Sequence, yaw: float,
                  room: Polygon, max_iter: int = 4,
                  stats: dict | None = None,
                  seeds: Sequence = (),
                  policy: str = "nearest") -> Polygon | None:
    """Move the region's edges into the gaps between objects.

    A tier boundary that slices a sofa in half is not a design, it is an
    artefact — and it would hand the evaluation a straddling object in the
    ground truth, which is the one thing this corpus must never contain.  The
    boundaries are therefore snapped onto coordinates where no object's
    footprint is crossed, which is a 1-D interval problem per axis.
    """
    R, Rb = _rot(yaw)
    boxes = [_iv(_footprint(o), R) for o in objs
             if _footprint(o).area > 1e-9]
    rb = _iv(room, R)
    span = [(rb[0][0], rb[1][0]), (rb[0][1], rb[1][1])]
    iv = _iv(region, R)
    lo, hi = iv[0].copy(), iv[1].copy()

    for _ in range(max_iter):
        moved = False
        for k in (0, 1):
            free = _free_coords(boxes, k, lo[1 - k], hi[1 - k], span[k])
            if not free:
                return _note(stats, 'cut_no_free')
            a = _snap_into_free(lo[k], free, span[k], -1, policy)
            b = _snap_into_free(hi[k], free, span[k], +1, policy)
            if a is None or b is None or b - a < 0.8:
                return _note(stats, 'cut_no_free_span')
            if abs(a - lo[k]) > 1e-6 or abs(b - hi[k]) > 1e-6:
                moved = True
            lo[k], hi[k] = a, b
        if not moved:
            break

    out = _largest(_rect(lo, hi, Rb).intersection(room))
    if out is None or out.area < 1.5:
        return _note(stats, 'cut_region_tiny')
    for o in seeds:
        fp = _footprint(o)
        if fp.area > 1e-9 and out.intersection(fp).area / fp.area < 0.5:
            return _note(stats, 'cut_lost_seed')
    return out


def cut_objects(region: Polygon, objs: Sequence) -> list:
    """Objects the tier boundary passes through."""
    out = []
    for o in objs:
        fp = _footprint(o)
        if fp.area < 1e-9:
            continue
        f = region.intersection(fp).area / fp.area
        if 0.05 < f < 0.95:
            out.append((o, f, fp))
    return out


def nudge_off_boundary(region: Polygon, objs: Sequence, room: Polygon,
                       max_shift: float = 0.70,
                       stats: dict | None = None) -> bool:
    """Slide straddling objects fully onto one tier.

    Snapping the boundary into a gap handles the easy rooms; the rest need the
    object to move, which is what a designer would do — a nightstand half over
    the edge of a tatami platform gets pushed on or off, not left hanging.  The
    shift is the smallest one that clears the edge, and it is rejected if it
    would leave the room or hit a neighbour.
    """
    others = list(objs)
    for o, f, fp in cut_objects(region, objs):
        best = None
        for target in (True, False):
            # direction: towards the region interior (True) or away from it
            ref = region if target else room.difference(region)
            ref = _largest(ref)
            if ref is None:
                continue
            d = np.asarray(ref.centroid.coords[0]) - np.asarray(fp.centroid.coords[0])
            n = np.linalg.norm(d)
            if n < 1e-9:
                continue
            u = d / n
            for step in np.arange(0.05, max_shift + 1e-9, 0.05):
                cand = translate(fp, float(u[0] * step), float(u[1] * step))
                if not room.covers(cand.buffer(-0.01)):
                    continue
                cf = region.intersection(cand).area / cand.area
                if not (cf <= 0.02 or cf >= 0.98):
                    continue
                clash = any(cand.intersection(_footprint(q)).area > 0.02
                            for q in others if q is not o)
                if clash:
                    continue
                if best is None or step < best[0]:
                    best = (step, u)
                break
        if best is None:
            if stats is not None:
                stats['nudge_failed'] = stats.get('nudge_failed', 0) + 1
            return False
        step, u = best
        o.position[0] += float(u[0] * step)
        o.position[1] += float(u[1] * step)
    return True


def _door_polys(room_obj, depth: float = 0.75) -> list:
    """Clearance in front of each door: an elevation change must not land here."""
    out = []
    for op in getattr(room_obj, "openings", []) or []:
        if getattr(op, "kind", "") != "door":
            continue
        p0 = np.asarray(op.p0, dtype=float)[:2]
        p1 = np.asarray(op.p1, dtype=float)[:2]
        d = p1 - p0
        n = np.array([-d[1], d[0]])
        ln = np.linalg.norm(n)
        if ln < 1e-9:
            continue
        n = n / ln * depth
        out.append(Polygon([p0 - n, p1 - n, p1 + n, p0 + n]))
    return out


def _clear_runs(a: np.ndarray, b: np.ndarray, lo: Tier, hi: Tier,
                blocked, min_width: float, stride: float = 0.15) -> list:
    """Sub-segments of an edge whose treads land on clear floor.

    A step whose landing is under a wardrobe is not a step.  The span is walked
    at ``stride`` and the maximal runs whose tread band is unobstructed are
    returned, longest first.
    """
    L = float(np.linalg.norm(b - a))
    if L < min_width:
        return []
    u = (b - a) / L
    n = max(2, int(L / stride) + 1)
    ts = np.linspace(0.0, L, n)
    free = []
    for i in range(len(ts) - 1):
        p0, p1 = a + u * ts[i], a + u * ts[i + 1]
        probe = make_transition(lo, hi, p0, p1)
        band = probe.footprint(lo)
        if band is None or band.area < 1e-9:
            free.append(False)
            continue
        taken = band.intersection(blocked).area if blocked is not None else 0.0
        free.append(taken / band.area <= 0.10)
    runs, i = [], 0
    while i < len(free):
        if not free[i]:
            i += 1
            continue
        j = i
        while j < len(free) and free[j]:
            j += 1
        if ts[j] - ts[i] >= min_width:
            runs.append((a + u * ts[i], a + u * ts[j]))
        i = j
    runs.sort(key=lambda r: -float(np.linalg.norm(r[1] - r[0])))
    return runs


def _transitions_for(region: Polygon, room_boundary, lo: Tier, hi: Tier,
                     blocked=None, min_width: float = 0.60) -> list[Transition]:
    """One transition per stretch of shared edge long enough to walk over.

    The shared boundary comes back as a pile of collinear fragments, so it is
    merged first; otherwise a 3 m step reads as six 0.5 m ones and every
    candidate falls under ``min_width``.
    """
    from shapely.ops import linemerge

    # the part of the region's edge that does not lie on a wall; a boolean
    # intersection of the two tier boundaries is unreliable because the datum
    # is built by a buffered difference and the two rings no longer touch
    shared = region.boundary.difference(room_boundary.buffer(0.05))
    if shared.is_empty:
        return []
    lines = [g for g in getattr(shared, "geoms", [shared])
             if getattr(g, "geom_type", "") == "LineString" and g.length > 1e-6]
    if not lines:
        return []
    merged = linemerge(lines)
    segs = []
    for g in getattr(merged, "geoms", [merged]):
        if getattr(g, "geom_type", "") != "LineString":
            continue
        c = np.asarray(g.coords)
        # collapse the run into its straight spans
        i = 0
        while i < len(c) - 1:
            j = i + 1
            d0 = c[i + 1] - c[i]
            n0 = np.linalg.norm(d0)
            if n0 < 1e-9:
                i += 1
                continue
            d0 = d0 / n0
            while j < len(c) - 1:
                d1 = c[j + 1] - c[j]
                n1 = np.linalg.norm(d1)
                if n1 < 1e-9 or abs(float(np.cross(d0, d1 / n1))) > 1e-6:
                    break
                j += 1
            if np.linalg.norm(c[j] - c[i]) >= min_width:
                segs.append((c[i], c[j]))
            i = j
    if not segs:
        return []
    runs = []
    for a, b in segs:
        runs += _clear_runs(a, b, lo, hi, blocked, min_width)
    if not runs:
        return []
    runs.sort(key=lambda r: -float(np.linalg.norm(r[1] - r[0])))
    keep = [runs[0]]
    widest = float(np.linalg.norm(runs[0][1] - runs[0][0]))
    for a, b in runs[1:]:
        w = float(np.linalg.norm(b - a))
        # a second way across only counts if it is a real one, not a sliver
        if w >= max(1.2, 0.5 * widest):
            keep.append((a, b))
        if len(keep) == 2:
            break
    return [make_transition(lo, hi, a, b) for a, b in keep]


# ---------------------------------------------------------------------------
# the lift
# ---------------------------------------------------------------------------

def lift_scene(scene, rng: np.random.Generator,
               program: ElevationProgram | None = None,
               min_datum_frac: float = 0.30,
               stats: dict | None = None) -> ElevScene | None:
    """Turn a flat ReRoom ``Scene`` into a multi-elevation ``ElevScene``.

    Returns ``None`` when the room admits no plausible elevation program — a
    corridor with one wardrobe in it has nowhere for a sunken lounge to go, and
    inventing one would poison the training distribution.  Pass ``stats`` to
    collect why rooms were rejected; the mix matters when tuning the corpus.
    """
    def drop(why):
        if stats is not None:
            stats[why] = stats.get(why, 0) + 1
        return None
    room_poly = Polygon(np.asarray(scene.room.polygon, dtype=float))
    if not room_poly.is_valid:
        from shapely.validation import make_valid
        room_poly = make_valid(room_poly)
    # a hair of buffer noding removes the self-touching vertices 3D-FRONT floor
    # rings carry, which GEOS otherwise refuses to intersect
    room_poly = _largest(_largest(room_poly).buffer(1e-6).buffer(-1e-6)
                         if _largest(room_poly) is not None else None)
    if room_poly is None or room_poly.area < 6.0:
        return drop('room_too_small')

    rtype = getattr(scene.room, "room_type", "") or ""
    height = float(getattr(scene.room, "height", 2.8))
    program = program or program_for(rtype, room_poly.area, height, rng)
    if program is None:
        return drop('no_program')

    import copy
    flat = ElevScene.from_flat(copy.deepcopy(scene))
    grounded = [o for o in flat.objects
                if flat.support_of(o.oid).kind == TIER
                and flat.dz.get(o.oid, 0.0) < 0.05
                and _footprint(o).area > 0.06]

    seeds = [o for o in grounded if o.category in program.seed_categories]
    anchored = [o for o in seeds if o.category in program.anchor_categories]
    if len(seeds) < program.min_seeds and not anchored:
        return drop('too_few_seeds')
    if not seeds:
        return drop('too_few_seeds')

    # Growing from the whole furniture group makes the region as large as the
    # group, which in a 20 m2 3D-FRONT room is most of the floor — the measured
    # real median is 0.27 of the room and a tenth of real cases are under 0.06.
    # Real platforms are often under the bed alone.  Drawing a subset of the
    # group reaches those sizes, and gives distinct variants of the same room
    # instead of the one deterministic region the full group always produced.
    if len(seeds) > 1:
        order = list(rng.permutation(len(seeds)))
        k = int(rng.integers(1, len(seeds) + 1))
        chosen = [seeds[i] for i in order[:k]]
        if anchored and not any(o in anchored for o in chosen):
            chosen[0] = anchored[int(rng.integers(len(anchored)))]
        seeds = chosen

    yaw = _principal_axes(room_poly)
    pts = np.vstack([np.asarray(_footprint(o, 0.12).exterior.coords)[:-1]
                     for o in seeds])
    region = _largest(_axis_rect(pts, yaw, 0.0).intersection(room_poly))
    if region is None:
        return drop('seed_rect_empty')
    # 73 % of real elevated floors reach the outer wall, so this is a draw
    # rather than a property of the program
    if program.wall_backed or rng.random() < GEOM_PRIOR["wall_backed"]:
        region = _snap_to_walls(region, room_poly, yaw)
        if region is None:
            return drop('wall_snap_empty')

    # Draw the size real designers would give this feature, then let the two
    # snapping policies compete for it.  "outward" always yields a region and
    # tends to be too large; "nearest" matches the real distribution but can
    # shrink past a seed and fail.  Choosing between them per room gets the
    # measured distribution without paying "nearest"'s rejection rate.
    target = float(rng.choice(GEOM_PRIOR["area_frac"])) \
        if len(GEOM_PRIOR["area_frac"]) else 0.27
    cands = []
    for pol in ("nearest", "outward"):
        r = _resolve_cuts(region, grounded, yaw, room_poly,
                          stats=stats if pol == "outward" else None,
                          seeds=seeds, policy=pol)
        if r is not None:
            cands.append((abs(r.area / room_poly.area - target), r))
    if not cands:
        return drop('cuts_unresolvable')
    region = min(cands, key=lambda t: t[0])[1]

    if not nudge_off_boundary(region, grounded, room_poly, stats=stats):
        return drop('nudge_failed')

    frac = region.area / room_poly.area
    if not (program.area_frac[0] <= frac <= program.area_frac[1]):
        return drop('area_frac_out_of_range')

    rest = room_poly.difference(region)
    lobes = [g for g in getattr(rest, "geoms", [rest])
             if getattr(g, "geom_type", "") == "Polygon" and g.area > 0.5]
    if not lobes:
        return drop('datum_too_small')
    lobes.sort(key=lambda g: -g.area)
    if sum(g.area for g in lobes) / room_poly.area < min_datum_frac:
        return drop('datum_too_small')
    datum = lobes[0]

    # the datum tier has to keep every door usable, and the step must not land
    # in the swing
    doors = _door_polys(scene.room)
    for d in doors:
        if region.intersection(d).area > 0.15 * d.area:
            return drop('door_blocked')

    rise = sample_rise(program, rng)
    # quantise to whole risers so the geometry is buildable
    n = max(1, int(round(abs(rise) / 0.15)))
    rise = math.copysign(round(abs(rise) / n, 3) * n, rise)

    def _tier(tid, g, h):
        return Tier(tid, np.asarray(g.exterior.coords)[:-1], h,
                    holes=[np.asarray(r.coords)[:-1] for r in g.interiors])

    t_datum = _tier(0, datum, 0.0)
    # a region that splits the room leaves further datum lobes; they are the
    # same height but not the same tier, and the reachability metric depends on
    # keeping them apart
    extra = [_tier(2 + i, g, 0.0) for i, g in enumerate(lobes[1:])]
    t_region = _tier(1, region, rise)
    lo, hi = (t_region, t_datum) if rise < 0 else (t_datum, t_region)
    from shapely.ops import unary_union
    blocked = unary_union([_footprint(o) for o in grounded]) if grounded else None
    trans = _transitions_for(region, room_poly.boundary, lo, hi, blocked)
    if not trans:
        return drop('no_transition')

    for t in extra:
        elo, ehi = (t_region, t) if rise < 0 else (t, t_region)
        trans += _transitions_for(region, room_poly.boundary, elo, ehi, blocked)
    field = ElevationField(np.asarray(room_poly.exterior.coords)[:-1],
                           [t_datum, t_region] + extra, trans)
    errs = field.validate()
    if errs:
        return drop('field_invalid:' + errs[0].split()[0] + ' ' + errs[0].split()[1])

    # -- carry the layout over --------------------------------------------
    es = ElevScene(scene_id=scene.scene_id + f"__{program.name}",
                   room=scene.room, field=field, objects=flat.objects,
                   source=(getattr(scene, "source", "") or "3D-FRONT") + "-Elev",
                   meta={**dict(getattr(scene, "meta", {}) or {}),
                         "program": program.name,
                         "rise": rise,
                         "region_frac": round(float(frac), 4),
                         "flat_id": scene.scene_id})

    for o in es.objects:
        s = flat.support_of(o.oid)
        if s.kind == OBJ:
            es.supports[o.oid] = s                # keep object-on-object
            es.dz[o.oid] = flat.dz.get(o.oid, 0.0)
            continue
        fp = _footprint(o)
        inside = region.intersection(fp).area / fp.area if fp.area > 1e-9 else 0.0
        if inside >= 0.5:
            tid = t_region.tid
        else:
            tid = max([t_datum] + extra,
                      key=lambda t: t.shp.intersection(fp).area).tid
        es.supports[o.oid] = Support(TIER, tid)
        es.dz[o.oid] = flat.dz.get(o.oid, 0.0)

    es.resolve()

    # The ground truth must not contain the failures the benchmark exists to
    # detect, so the scene is checked with the same code the evaluation uses.
    from ..eval.violations import violations
    bad = [v for v in violations(es)
           if v.kind in ("overhang", "straddling", "datum", "step_blocked")]
    if bad:
        return drop("gt_violation:" + bad[0].kind)
    return es
