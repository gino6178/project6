# What a plausible change of floor level looks like

Measured on the 72 annotated MP3D-Elev fields (real Matterport buildings, 9 of
them) against 1,200 regions from the FRONT-Elev generator. Everything is
computed in the room's own principal frame; a boundary within 10 cm of a wall
counts as being on the wall.

`scripts/what_is_plausible.py`, figure `assets/fig_plausible.png`.

## The measurement

|                                              |  real  |  ours  |
|----------------------------------------------|-------:|-------:|
| drop edge, share of region perimeter (median) |  0.19  |  0.46  |
| p90 of that share                             |  0.36  |  0.49  |
| **more than half the perimeter is a drop**    | **0 / 72** | 5.8 % |
| exactly one drop edge                         | 85 %   | 61 %   |
| **two or more separate drop edges**           | **1.4 %** | **38.8 %** |
| no drop edge inside the room at all           | 18.1 % |  0.0 % |
| region is essentially the whole room (>90 %)  | 16.7 % |  0.0 % |
| region area / room (median)                   |  0.43  |  0.24  |
| edges on the wall grid (median)               |  0.85  |  1.00  |
| perfect rectangle                             |  2.8 % | 69.9 % |
| rectangularity (median)                       |  0.77  |  1.00  |
| stair on the region's drop edge               | 97 %   | 100 %  |
| rise, m (median [p10, p90])                   | 0.39 [0.23, 0.51] | 0.15 [-0.44, 0.43] |

## What it says

**A real level change is a shelf, not an island.** Four fifths of a real
region's perimeter is the building's own wall; the remaining fifth is a single
edge you can step off, and the stair is on it 97 % of the time. Not one of the
72 has a drop around more than half its perimeter. Ours puts a drop around 46 %
of the perimeter, and 39 % of the time in two separate places -- a plinth in the
middle of the floor with two different edges to fall off. No building has a
reason to construct that.

**In a sixth of real cases the level change is not inside a room at all.** 18 %
of real regions have no drop edge within the room: the change coincides exactly
with the room's own boundary, so the step is at the doorway. 17 % are the whole
room. The generator cannot express either -- it always carves a sub-region out
of one room -- so the most common real form of "a storey with more than one
floor level" is absent from the corpus by construction.

Counting the other way: only **40 %** of the real rooms are a genuine
within-room split with both levels at 25 % of the floor or more. The premise
"a room contains an elevation change" describes a minority of the phenomenon.

**Real regions are not rectangles.** 2.8 % are, against our 70 %. Real ones
follow the wall grid (0.85 of edge length aligned) while inheriting the room's
irregular corners; ours are axis-aligned boxes and diagonal corner cuts, which
is a different thing from following the structure.

## The specification this implies

An elevated region is plausible when:

1. it has **exactly one** drop edge, never two;
2. that edge is **at most about a third** of its perimeter (real p90 = 0.36),
   and the rest of its boundary is the room's own wall;
3. the transition sits **on** that edge;
4. its edges follow the wall grid but it is **not** a rectangle -- it is what
   remains when a wall-to-wall cut meets the room's real corners;
5. the rise is **0.23-0.51 m** (real p10-p90, median 0.39);
6. and the region is allowed to be the **entire room**, with the step at the
   doorway, which is a sixth of real cases and 0 % of ours.

Points 1, 2 and 6 are not satisfiable by patching `architectural_region()`.
(1) and (2) require cutting from a wall to a wall rather than insetting a band,
and (6) requires the unit of generation to be a storey rather than a room.
