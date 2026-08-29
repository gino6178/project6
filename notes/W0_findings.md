# Week 0 — go / no-go results

Run 2026-08-29. All numbers are from code in this repo; scripts named below
reproduce them.

## G1 — is 3D-FRONT flat?  **Yes, exactly.**

`scripts/scan_front_elevation.py` accumulates floor-mesh triangle area against
height for every room in all 6,813 3D-FRONT houses.

| | |
|---|---|
| houses scanned | 6,813 |
| rooms with a `Floor` mesh | 37,529 |
| rooms with **any** floor height variation | **0** |
| max height spread over all rooms | **0.0000 m** |
| rooms with spread > 5 cm | 0 |

Not "mostly flat" — the spread is identically zero at every percentile. The
premise of the paper holds more strongly than the proposal assumed, and this is
a one-line motivation number.

## G1b — the elevation 3D-FRONT does carry, and why nobody sees it

The dataset has 3,925 `CustomizedPlatform` meshes. Every layout parser filters
on `type == "Floor"`, so none of them are ever read. ReRoom's parser
(`project5/reroom/data/threed_front.py:_floor_polygon`) goes further and drops
the height component of the floor vertices outright — the elevation is
projected away at parse time.

`scripts/scan_front_platforms.py` censused them: **428 rooms** across the
dataset have at least one platform. But the type is a grab-bag — the rise
distribution has a median of 0.58 m and a 90th percentile of 2.6 m, so most are
dropped ceilings, bay-window sills and full-height joinery, not floor platforms.
Filtering to a plausible floor platform (rise 0.08–0.80 m, top face ≥ 1 m²)
leaves ~20 per 300 houses, and of those **almost none carry furniture**: of 20
inspected, 1 had a floor-level object overlapping it, i.e. embedded in it.

**Conclusion: 3D-FRONT supplies zero multi-elevation layout supervision.** The
platforms are decorative geometry, not supported surfaces.

## Contact in 3D-FRONT is not exact

Measured in `tests/test_representation.py::test_flat_3dfront_rooms_round_trip_losslessly`:

| | share of objects |
|---|---|
| exactly flush with their support | 66.4 % |
| off by under 2 cm | 18.9 % |
| genuinely below the surface (z < −0.02) | 0.25 % |

So a third of the "ground truth" everyone trains on is not in contact. The
support-tree parameterisation makes all of it exact by construction.

## G1c — do *real* homes have intra-storey elevation changes?  **Yes, 75 % of them.**

The HuggingFace mirror suggested for Matterport3D turned out to be 492 MB of
panoramic images with two class labels — no meshes, no `.house` files, nothing
that carries floor geometry.  What works instead is **HouseLayout3D** (NeurIPS
2025, MIT, ungated, **40 MB**): vectorised structural annotations of 16 real
Matterport3D buildings — walls, floors, ceilings, doors, windows, stairs.

Entity classes are not labelled (the per-entity colours are random), so
`scripts/scan_houselayout_elevation.py` classifies geometrically: a horizontal
slab with a matching slab a room-height above it is a floor.  Floors are grouped
into storeys, and two floors in one storey that touch in plan but differ in
height are an elevation change.

| | |
|---|---|
| buildings | 16 |
| **buildings with an intra-storey elevation change** | **12 (75 %)** |
| floor slabs found | 309 |
| adjacent floor pairs at different heights | **218** |
| rise median / 10th / 90th percentile | **0.38 m** / 0.14 / 0.53 |

This clears the G1 gate (≥ 150 real multi-elevation rooms) for 40 MB instead of
the 15 GB the plan budgeted.

Caveat: these are adjacent floor-slab pairs, not verified sunken lounges. The
16 pairs under 10 cm are probably annotation jitter and are excluded downstream.

## The generator's rise magnitudes are now measured, not invented

The first corpus used hand-picked rise ranges. Comparing them against the real
distribution above showed they were too narrow and slightly too low, so
`elevate3d/data/rise_prior.json` now holds the 202 measured rises and
`sample_rise()` draws from them (smoothed by 2 cm, clamped per program).

| percentile | real | generated |
|---|---|---|
| 10 % | 0.198 | 0.180 |
| 50 % | 0.384 | 0.370 |
| 90 % | 0.526 | 0.498 |
| mean | 0.386 | 0.354 |

Wasserstein distance **0.071 m → 0.032 m**. The elevation magnitudes in the
corpus are no longer a guess; they follow real homes to within ~3 cm.

## G2 — Infinigen

Not installed, and neither is Blender. Reading the paper and its constraint API:
rooms are 2D polygons extruded to a constant floor height, and the object
relations in the DSL are `SupportedBy` / `StableAgainst` only. There is no
intra-room elevation concept to extend — adding one is a new primitive plus
solver support, not a configuration change. **Deferred**; FRONT-Elev (below)
removes the dependency on it for a first result.

## G3 — the data plan, revised

MP3D is not on this machine and the disk is at 97 % (33 GB free), so the real
evaluation set from the proposal is not available today. What replaced it:

**FRONT-Elev** (`elevate3d/data/frontelev.py`, `scripts/build_frontelev.py`) —
real 3D-FRONT layouts lifted onto real elevation programs. Four programs so far:
`sunken_lounge`, `tatami_platform`, `dining_tier`, `study_dais`.

The mechanism that makes it usable as ground truth: **the elevated region is
grown out of a furniture group, not stamped onto the room**. A sunken lounge is
sized to the conversation group; a tatami platform to the bed. Boundaries are
then snapped into the gaps between objects, and anything still straddling is
nudged the smallest distance that clears the edge.

Final build: **5,792 pairs from 17,007 parseable rooms (34.1 %)**, 54,424
objects. Programs: tatami 2,807 · sunken lounge 2,509 · study dais 261 ·
dining tier 215. Tier counts: K=2 for 5,000, K=3 for 778, K=4 for 14 (a region
that splits the datum leaves lobes, and each is its own tier).

Ground-truth violation rates over all 5,792, measured with the same code the
evaluation uses (`elevate3d/eval/violations.py`):

| failure | rate |
|---|---|
| F1 overhang | 0.00000 |
| F2 straddling | 0.00000 |
| F3 step blocked (per scene) | 0.00000 |
| F5 datum mismatch | 0.00000 |
| F4 headroom | 0.00094 |

Every one of the 5,792 elevation fields passes `validate()`.

A scene is checked before it is written, so the corpus cannot contain the
failures the benchmark measures.

## The metric separates agents — which is the whole point

`elevate3d/eval/navigability.py`, 400 sampled flat/elevated pairs from the final
corpus, reachable floor fraction per agent:

| agent | flat | elevated | Δ |
|---|---|---|---|
| sweeping robot | 0.965 | 0.574 | −0.392 |
| wheelchair | 0.900 | 0.618 | −0.282 |
| wheeled robot | 0.930 | 0.584 | −0.346 |
| quadruped | 0.947 | 0.907 | −0.039 |
| adult | 0.944 | 0.913 | −0.031 |

Capability spread (best agent − worst) goes from **0.128 flat to 0.431
elevated — 3.4×**. On one plane the metric is nearly constant across agents,
which is why nobody reports it; the elevation is what gives it discriminative
power.

## Open risk

Still the circularity: the corpus is procedurally derived, so a model trained on
it and evaluated on it is partly learning the generator.  Two of its parameters
are no longer arbitrary — the rise magnitudes come from real homes, and the
region shapes come from real furniture groups — but the *placement* of the
region and the tier assignment rules are still mine.

What would close it: an evaluation set whose elevation fields the generator
never produced.  HouseLayout3D gives 218 real ones; turning them into furnished
test rooms is the obvious next step and is not done.
