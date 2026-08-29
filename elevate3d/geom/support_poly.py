"""Convex regions as support functions, so the generator is not stuck on boxes.

M1's first version emitted an axis-aligned rectangle. That was the price of
making the field predictable, and it left a gap between the formalism — which
says a tier is any ``R_k ⊂ P`` — and what the system could actually produce.
Measured against HouseLayout3D, the gap matters: **52 % of real elevated floors
have a rectangularity below 0.90**, and their median is 0.888. They are not
boxes.

They are also not arbitrary. Only 15 % are non-convex, and the median has 5
exterior vertices (90th percentile 7). A real elevated floor is a trapezoid or a
wedge following walls that are not orthogonal to each other — a bounded convex
polygon.

The support function is the natural representation for exactly that class. A
convex body ``K`` is determined by ``h(u) = max_{x∈K} <x, u>``; sampling ``h`` at
``n`` fixed directions and intersecting the resulting half-planes gives a convex
polygon with at most ``n`` sides. Two properties make it the right choice here:

* **It contains the box.** Axis-aligned rectangles are the special case where the
  diagonal offsets are slack, so nothing that worked before stops working.
* **Every prediction is valid.** An intersection of half-planes is convex and
  non-self-intersecting by construction, so a model cannot emit a tangled ring
  the way it could with free vertex coordinates.

Offsets are stored in the room's principal frame and normalised by its extent,
so a model sees a shape rather than a location and a size.
"""
from __future__ import annotations

import numpy as np
from shapely.geometry import MultiPolygon, Polygon

__all__ = ["N_DIRS", "DIRECTIONS", "offsets_from_poly", "poly_from_offsets",
           "rectangularity"]

# Chosen by measurement, not by argument.  Round-tripping the 80 real regions
# that have a usable room boundary, clipped to that room and in its principal
# frame:
#
#   directions   IoU median   IoU 10th pct
#            4        0.905          0.471     <- the box, i.e. M1 v1
#            8        0.930          0.551
#           12        0.936          0.626
#           16        0.943          0.626
#
# Two things follow.  The box is better than the raw rectangularity statistic
# suggests, because clipping to the room does part of the shaping.  And the gain
# from more directions is in the *tail*, not the median: it fixes the worst
# quarter, which are the regions following walls that are not orthogonal.
# Twelve is where the tail stops improving.
N_DIRS = 12
DIRECTIONS = np.stack([np.cos(np.arange(N_DIRS) * 2 * np.pi / N_DIRS),
                       np.sin(np.arange(N_DIRS) * 2 * np.pi / N_DIRS)], axis=1)


def offsets_from_poly(poly, origin, extent) -> np.ndarray:
    """Support function of ``poly`` at the fixed directions, normalised.

    Non-convex regions are represented by their convex hull; 15 % of real ones
    are non-convex and that residual is a stated limitation rather than a
    silent approximation.
    """
    p = poly if isinstance(poly, Polygon) else Polygon(np.asarray(poly, float))
    if not p.is_valid:
        p = p.buffer(0)
    if isinstance(p, MultiPolygon):
        p = max(p.geoms, key=lambda g: g.area)
    v = np.asarray(p.convex_hull.exterior.coords)[:-1]
    if len(v) < 3:
        return np.zeros(N_DIRS, dtype=np.float32)
    s = float(max(np.max(extent), 1e-6))
    q = (v - np.asarray(origin)) / s
    return (q @ DIRECTIONS.T).max(axis=0).astype(np.float32)


def poly_from_offsets(offsets, origin, extent, clip=None,
                      min_area: float = 0.5):
    """Intersect the half-planes back into a polygon, clipped to the room.

    Returns ``None`` when the offsets describe nothing usable — a model that has
    proposed an empty or hair-thin region should be refused, not rounded up into
    something plausible.
    """
    h = np.asarray(offsets, dtype=float).reshape(-1)
    if h.shape[0] != N_DIRS or not np.all(np.isfinite(h)):
        return None
    s = float(max(np.max(extent), 1e-6))
    o = np.asarray(origin, dtype=float)

    # start from a box big enough to contain any half-plane set, then cut
    r = float(np.max(np.abs(h))) + 1.0
    poly = Polygon([(-r, -r), (r, -r), (r, r), (-r, r)])
    for u, hi in zip(DIRECTIONS, h):
        # the half-plane <x, u> <= hi, as a large rectangle rotated to u
        n = np.array([-u[1], u[0]])
        a = u * hi + n * (3 * r)
        b = u * hi - n * (3 * r)
        cut = Polygon([a, b, b - u * (6 * r), a - u * (6 * r)])
        poly = poly.intersection(cut)
        if poly.is_empty or poly.geom_type != "Polygon":
            return None

    coords = np.asarray(poly.exterior.coords)[:-1] * s + o
    out = Polygon(coords)
    if not out.is_valid:
        out = out.buffer(0)
    if clip is not None:
        out = out.intersection(clip)
    if isinstance(out, MultiPolygon):
        out = max(out.geoms, key=lambda g: g.area)
    if out.is_empty or out.geom_type != "Polygon" or out.area < min_area:
        return None
    return out


def rectangularity(poly) -> float:
    """Area over the area of the minimum rotated rectangle. 1.0 is a box."""
    p = poly if isinstance(poly, Polygon) else Polygon(np.asarray(poly, float))
    if not p.is_valid:
        p = p.buffer(0)
    if p.is_empty or p.area < 1e-9:
        return 0.0
    return float(p.area / max(p.minimum_rotated_rectangle.area, 1e-9))
