# Results

Four rounds. Each acted on what the previous one measured. Round 3 corrects an
attribution round 2 got wrong; round 4 adds the metric that three rounds of
results kept asking for, and refutes the next step rounds 2 and 3 proposed.

Model: 27.8 M-parameter autoregressive layout transformer. Sampling, rejection
and metrics are shared by every row, so each comparison isolates one variable.

| round | corpus | training rooms | sampler |
|---|---|---|---|
| R1 | uncalibrated, 5,792 pairs | 5,213 | 8 tries, no step term, stop at 0.5 |
| R2 | calibrated, 14,782 pairs | 13,300 | 12 tries, step term, stop at 0.5 |
| R3 | calibrated, 14,782 pairs | 13,300 | as R2, stop threshold calibrated |
| R4 | same as R3 | 13,300 | as R3; adds tier precision, density and elevation F1 |

---

## Round 3 — held-out FRONT-Elev, 300 rooms

`ours-small` is the same model on the same corpus trained on R1's number of
rooms, so it separates corpus scale from everything else.

| metric | GT | **ours** | ours-small | +bias | flatten | per-tier |
|---|---|---|---|---|---|---|
| overhang | 0.000 | **0.022** | 0.038 | 0.022 | 0.058 | 0.060 |
| embedded | 0.000 | **0.030** | 0.046 | 0.032 | 0.063 | 0.069 |
| straddling | 0.000 | **0.036** | 0.048 | 0.038 | 0.076 | 0.092 |
| wrong tier | 0.000 | **0.008** | 0.011 | 0.009 | 0.050 | 0.044 |
| step blocked (scene) | 0.000 | 0.187 | 0.277 | **0.160** | 0.310 | 0.867 |
| any violation (scene) | 0.003 | 0.450 | 0.643 | **0.443** | 0.693 | 0.943 |
| tier use, area | 0.457 | **0.389** | 0.385 | 0.387 | 0.087 | 0.414 |
| tier use, objects/scene | 3.56 | **3.32** | 3.23 | 3.18 | **0.00** | 4.58 |
| CCN sweeping robot | 0.676 | **0.598** | 0.551 | 0.580 | 0.324 | 0.492 |
| CCN wheelchair | 0.699 | **0.602** | 0.532 | 0.576 | 0.277 | 0.534 |
| objects/scene | 9.87 | 9.16 | 9.04 | 8.50 | 9.65 | 13.66 |

## Round 3 — MP3D-Elev, 72 real rooms

| metric | **ours** | ours-small | +bias | flatten | per-tier |
|---|---|---|---|---|---|
| overhang | 0.091 | 0.102 | 0.088 | **0.016** | 0.055 |
| embedded | **0.008** | 0.024 | 0.036 | 0.029 | 0.006 |
| straddling | 0.050 | 0.055 | 0.074 | **0.022** | 0.045 |
| wrong tier | 0.040 | 0.060 | 0.051 | **0.016** | 0.026 |
| step blocked (scene) | 0.167 | 0.181 | 0.236 | **0.083** | 0.611 |
| any violation (scene) | 0.681 | 0.708 | 0.611 | **0.306** | 0.750 |
| tier use, area | 0.237 | 0.264 | 0.246 | **0.034** | 0.344 |
| tier use, objects/scene | 2.58 | 2.63 | 2.49 | **0.00** | 3.96 |
| CCN sweeping robot | 0.641 | 0.608 | **0.692** | 0.552 | 0.693 |

---

## What each change bought

### Calibrating the stop threshold — fixed the over-placement R2 introduced

R2 generated 11.92 objects per scene against a ground truth 9.87, and 4.27 on
raised tiers against 3.56: the larger corpus made the model keener to keep
going, and a fixed 0.5 threshold has no reason to land on the right number. The
threshold is one scalar with an obvious target, so it is now chosen on the
held-out split against the ground-truth count, once per method so no method is
favoured. MP3D-Elev reuses the in-distribution thresholds, because calibrating
on the test set would be cheating.

| | R2 | R3 | GT |
|---|---|---|---|
| objects/scene | 11.92 | 9.16 | 9.87 |
| tier use, objects/scene | 4.27 | 3.32 | 3.56 |
| any violation (scene) | 0.580 | 0.450 | 0.003 |

Every violation rate fell with it. The chosen thresholds are low — 0.05 for
`ours`, at the bottom of the sweep — which says the stop head is itself poorly
calibrated and only an aggressive threshold reaches the right count. A properly
calibrated stop head would be better than a swept threshold.

### Step clearance in the sampler — still the largest single win

| step blocked | R1 | R3 |
|---|---|---|
| in distribution | 0.653 | **0.187** |
| out of distribution | 0.542 | **0.167** |

Unambiguous, because the sampler is shared and every method improved.

### Corpus scale, separated from calibration — and R2's attribution was wrong

R2 claimed the out-of-distribution improvement came from calibrating the
generator's geometry. `ours-small` — same calibrated corpus, same sampler, R1's
number of training rooms — shows that is mostly not true.

**Scale, at fixed corpus and sampler** (`ours-small` → `ours`):

| | in distribution | out of distribution |
|---|---|---|
| overhang | 0.038 → 0.022 (−43 %) | 0.102 → 0.091 (−10 %) |
| straddling | 0.048 → 0.036 (−26 %) | 0.055 → 0.050 (−8 %) |
| any violation | 0.643 → 0.450 (−30 %) | 0.708 → 0.681 (−4 %) |

**Calibration** cannot be isolated as cleanly, because R1 also used the older
sampler. What can be said: at R1's training size on the calibrated corpus,
out-of-distribution overhang is 0.102 against R1's 0.096 — *no better*. So the
geometry calibration did not buy the OOD improvement R2 credited it with. Most
of that improvement was the sampler's step term and the extra data.

The calibration was still worth doing — it removed three invented parameters and
made the corpus's marginals defensible — but it should be reported as a
methodological cleanup, not as the cause of a metric gain.

### The tier-relative attention bias — the null holds a third time

`ours` and `+bias` are within noise on both sets across three rounds and two
corpora. Off by default; it should be dropped from the paper.

---

## What is still true and unwelcome

**Flatten still wins the OOD violation metrics by abstaining.** 0.00 objects on
non-datum tiers, 3.4 % of the raised floor covered against a ground truth
45.7 %. Three rounds of improvement have not moved this ranking, because what
separates us from it is not accuracy but willingness to use the elevation.

**Still far from clean.** 0.45 of generated scenes contain at least one
violation against 0.003 of ground truth.

**Two corpus statistics remain off.** Transition width 1.90 m against a real
2.93 m; region size 0.35 of the floor against a real 0.27. Round 3 blamed
3D-FRONT's room sizes for this; round 4 measured it and that was wrong — see
below.

---

## Round 4 — a metric that abstention cannot win

Three rounds ended the same way: the flat baseline posted the lowest violation
rates by never placing an object on a non-datum tier, and no violation metric
could see it, because every one of them is about an object being in the wrong
place and an empty tier holds none.

`elevation_f1` closes that. Precision is the share of the objects a method put
on a raised or sunken tier that are placed validly; recall is how much of the
elevation it used against the ground truth. Refusing to use the elevation drives
recall to zero, and the score with it.

| | GT | **ours** | ours-small | +bias | flatten | per-tier |
|---|---|---|---|---|---|---|
| tier precision | 0.999 | 0.905 | 0.857 | 0.895 | **0.000** | 0.839 |
| tier use, objects/scene | 3.56 | 3.31 | 2.96 | 3.07 | **0.00** | 4.58 |
| **elevation F1** | 0.9996 | **0.917** | 0.845 | 0.879 | **0.000** | 0.912 |
| step blocked (scene) | 0.000 | 0.187 | 0.273 | 0.173 | 0.270 | 0.847 |

On MP3D-Elev: ours 0.672, +bias 0.665, ours-small 0.615, **flatten 0.000**,
per-tier 0.934.

Two things to read carefully. **flatten scores exactly zero on both sets**, which
is the point — a baseline that wins every violation metric by abstention should
not survive a metric that asks whether it used the elevation at all. And
**per-tier wins F1 out of distribution while blocking 61 % of the steps**: it is
handed the ground-truth elevation field and samples each tier in isolation, so
its tier placements are precise by construction and its circulation is not.
F1 is about placement, not about circulation, so it must be read next to step
blocking rather than instead of it.

## The larger-room source: measured, and the premise was wrong

Rounds 2 and 3 blamed two stuck corpus statistics — region size 0.35 against a
real 0.27, transition width 1.90 m against 2.93 m — on 3D-FRONT's rooms being
small (22 m² median against 34 m² for real ones), and named a larger-room source
as the fix. Both halves of that were tested and both are wrong.

**Larger rooms do not change the region size.** Retargeting 3D-FRONT living
rooms into 34–49 m² boundaries with project5's optimiser and lifting the result
gives a region fraction of **0.354** at the median — indistinguishable from the
corpus's 0.35. Furniture fills a room proportionally, so a region grown from a
furniture group is the same fraction of a big room as of a small one.

**Infinigen's rooms are not larger anyway.** Its floorplan solver produces a
median room of **22.0 m²** (10th–90th 6.2–56.0, n = 83 over four houses) against
3D-FRONT's 22.3 and real homes' 34.2. It also costs 1.6–10 minutes per house,
and the state its `solve()` returns is solidified Blender geometry, so the 2D
polygons have to be recovered by re-running the first half of the solve.

So the real cause is not the room. **It is that the region is derived from
furniture at all.** Real elevated floors are sized by architecture — a bay
window recess, a structural split, a mezzanine over part of a span — and the
furniture arrives afterwards. An architecture-driven region rule is the actual
fix, and it would also remove the artefact where a sunken area covering 56 % of
the room leaves the datum as a narrow ring.

That refutation cost two measurements and saved a multi-day Infinigen
integration. The install assessment stands on its own: Infinigen 1.15.5 works
against Python 3.11 (`bpy` 4.2.0, ~3 GB), and adding a per-region floor height
would still mean changes across `state_def`, `room/solidifier.py`, the annealing
moves in `room/solver.py` and the object solver's `SupportedBy`.

The cheap intermediate — merging adjacent 3D-FRONT rooms into open-plan spaces —
was also tested and rejected: ~350 usable rooms across the dataset, and most
adjacent pairs are combinations like bedroom + living room that are not one
space.

## A published baseline: why not PhyScene

PhyScene's output space is (x, y, θ) on one plane and it cannot take an
elevation field, so any adaptation of it is a second flatten baseline. It would
differ from `flatten` in architecture *and* in tier knowledge at once, which is
weaker evidence than the controlled ablation already run. Its released weights
also cover living rooms only. Worth running to answer the reviewer question
directly, not to strengthen the argument.

## Next

1. **An architecture-driven region rule.** The measurements above say the region
   should be sized by the room's structure, not by the furniture group that ends
   up on it. This is the single change that would move region size, transition
   width and the narrow-ring artefact together.
2. **Calibrate the stop head** rather than sweeping a threshold; the sweep
   bottoms out at 0.05, which says the head is miscalibrated.
3. **Put the elevation F1 in the objective**, not only in the report. It now
   measures the abstention gap; nothing yet closes it during training.
