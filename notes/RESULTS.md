# First end-to-end results

Model: 27.8 M-parameter autoregressive layout transformer, 400 epochs on 5,213
FRONT-Elev rooms, ~12 min per run on one L40. Three runs differing only in what
the model is told about the floor. Sampling, rejection and metrics are shared by
every row, so the comparison isolates that one variable.

## Held-out FRONT-Elev — 300 rooms

| metric | GT | **ours** | no-bias | flatten | per-tier |
|---|---|---|---|---|---|
| overhang | 0.000 | **0.032** | 0.027 | 0.059 | 0.078 |
| embedded | 0.000 | **0.031** | 0.040 | 0.077 | 0.095 |
| straddling | 0.000 | **0.039** | 0.042 | 0.095 | 0.117 |
| wrong tier (datum) | 0.000 | **0.010** | 0.014 | 0.051 | 0.067 |
| step blocked (scene) | 0.000 | 0.653 | 0.697 | **0.537** | 0.907 |
| headroom | 0.001 | **0.007** | 0.007 | 0.012 | 0.014 |
| **tier use, area** | 0.384 | **0.356** | 0.359 | 0.077 | 0.408 |
| **tier use, objects/scene** | 3.53 | **3.60** | 3.92 | **0.00** | 5.70 |
| CCN sweeping robot | 0.561 | **0.501** | 0.484 | 0.240 | 0.389 |
| CCN wheelchair | 0.607 | **0.505** | 0.498 | 0.217 | 0.451 |
| CCN adult | 0.909 | **0.775** | 0.770 | 0.674 | 0.543 |
| objects/scene | 9.34 | 9.48 | 9.70 | 9.44 | 14.58 |

Tier conditioning roughly halves every placement violation against the flat
baseline and beats the per-tier decomposition by 2–3×, and it is the only method
whose use of the raised and sunken floor matches the ground truth (3.60 objects
per scene against 3.53).

## MP3D-Elev — 72 real rooms the generator never produced

| metric | **ours** | no-bias | flatten | per-tier |
|---|---|---|---|---|
| overhang | 0.096 | 0.112 | **0.025** | 0.057 |
| embedded | 0.063 | 0.059 | **0.026** | 0.013 |
| straddling | 0.088 | 0.072 | **0.029** | 0.039 |
| wrong tier | 0.079 | 0.092 | **0.024** | 0.027 |
| step blocked (scene) | 0.542 | **0.431** | 0.444 | 0.583 |
| **tier use, area** | 0.337 | 0.291 | **0.031** | 0.364 |
| **tier use, objects/scene** | 3.64 | 3.03 | **0.00** | 4.26 |
| CCN sweeping robot | 0.636 | 0.620 | 0.484 | **0.664** |
| CCN wheelchair | 0.618 | 0.615 | 0.424 | **0.620** |

**On real fields our method does not beat the flat baseline on the violation
metrics.** The advantage measured in distribution does not transfer.

## What the numbers actually say

**The flat baseline is safe by abstention.** It places *zero* objects on any
non-datum tier — 0.00 objects per scene on both test sets, covering 7.7 % and
3.1 % of the raised floor against a ground truth of 38.4 %. Every placement
violation is about an object being on the wrong tier, and an empty tier has no
objects to be wrong. Its low violation rates are the price it charges for not
using the elevation at all, and the cost shows up in reachability: a sweeping
robot reaches 0.240 of a flattened scene against 0.501 of ours and 0.561 of the
ground truth. Reporting violations without tier utilisation would have made this
baseline look like the winner.

The overhang check was one-sided at first and had the same problem: an object on
the lowest tier can never hang over anything lower, so a method that puts
everything on the datum was structurally immune. `embedded` (F1b) is the missing
half and is now reported alongside it.

**The tier-relative attention bias does nothing.** `ours` and `no-bias` are
within noise of each other on every metric, on both sets. One of the three
proposed modules is not supported by evidence and should be dropped from the
paper rather than defended.

**Nobody keeps the steps clear.** 0.54–0.91 of generated scenes block a
transition, against 0.000 in the ground truth. This is the largest single gap
and no method addresses it; the sampler's rejection term scores tier containment
and object overlap, and does not know that the treads must stay walkable.

**Absolute quality is not there yet.** 76 % of generated scenes contain at least
one violation, against 1 % of ground truth, and the qualitative figures show
loose, overlapping placements next to a clean ground truth. 5.2 k training rooms
is small, and the corpus is the binding constraint.

## The domain gap, measured

| statistic | FRONT-Elev (train) | MP3D-Elev (real) |
|---|---|---|
| room area (m²) | 19.97 | 34.19 |
| relief (m) | 0.37 | 0.39 |
| smallest tier, share of floor | 0.40 | 0.23 |
| transitions per room | 2.0 | 1.0 |
| transition width (m) | 1.84 | 2.93 |

Rise transferred well — that is the parameter calibrated against real data.
The three that were not calibrated all differ: the generator builds elevated
regions that are proportionally larger than real ones, and reaches them by two
narrow steps where a real room has one wide one. Recalibrating those the way the
rise was calibrated is the obvious next experiment, and would test directly
whether the OOD gap is a distribution problem or a method problem.

## Next, in order of expected value

1. Recalibrate region fraction, transition count and width against the
   HouseLayout3D statistics above, rebuild, retrain. Cheap, and it tests the
   most likely explanation for the OOD result.
2. Add a step-clearance term to the sampler's rejection cost. The one failure
   every method shares.
3. Drop the tier-relative bias, or find a formulation that earns its place.
4. Grow the corpus. Yield is 34 %; `nudge_failed` and `area_frac_out_of_range`
   are the two largest rejection buckets.
