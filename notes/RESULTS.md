# Results

Three rounds. Each acted on what the previous one measured, and round 3 corrects
an attribution round 2 got wrong.

Model: 27.8 M-parameter autoregressive layout transformer. Sampling, rejection
and metrics are shared by every row, so each comparison isolates one variable.

| round | corpus | training rooms | sampler |
|---|---|---|---|
| R1 | uncalibrated, 5,792 pairs | 5,213 | 8 tries, no step term, stop at 0.5 |
| R2 | calibrated, 14,782 pairs | 13,300 | 12 tries, step term, stop at 0.5 |
| R3 | calibrated, 14,782 pairs | 13,300 | as R2, stop threshold calibrated |

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
2.93 m; region size 0.35 of the floor against a real 0.27. Both are bounded by
3D-FRONT's room sizes — 22 m² median against 34 m² for real rooms.

---

## The larger-room source: assessed, not built

§9 named extending Infinigen Indoors with an elevation primitive as the way out
of the room-size bound. That assessment is now concrete rather than speculative.
Infinigen 1.15.5 installs cleanly against Python 3.11 (`bpy` 4.2.0, ~3 GB), its
constraint graph builds, and its floorplan solver runs. Two findings:

* **The elevation primitive is deep surgery.** Rooms are shapely polygons
  extruded by a constant `constants.wall_height`; the floor mesh is built from
  the 2D contour in `room/solidifier.py`. Per-region floor heights would need
  changes across `state_def`, `solidifier`, the annealing moves in
  `room/solver.py`, and the object solver's `SupportedBy` — four modules of
  someone else's codebase, then a validation pass on the result.
* **Even using it only as a room-shape source is a pipeline, not a patch.** The
  floorplan solve is simulated annealing and takes minutes per house; the rooms
  would then have to be furnished by retargeting 3D-FRONT layouts into them
  (project5 does this), and only then could the elevation programs run.

Both are multi-day. They are the right next step and they are out of scope for
the current pass; the intermediate cheap option — merging adjacent 3D-FRONT
rooms into open-plan spaces — was tested and rejected: it yields roughly 350
usable rooms across the dataset, and most adjacent pairs are combinations like
bedroom + living room that are not one space.

## A published baseline: why not PhyScene

PhyScene's output space is (x, y, θ) on one plane and it cannot take an
elevation field, so any adaptation of it is a second flatten baseline. It would
differ from `flatten` in architecture *and* in tier knowledge at once, which is
weaker evidence than the controlled ablation already run. Its released weights
also cover living rooms only. Worth running to answer the reviewer question
directly, not to strengthen the argument.

## Next

1. **Calibrate the stop head** rather than sweeping a threshold; the sweep bottoms
   out at 0.05, which says the head is miscalibrated.
2. **The larger-room source**, at the scope described above.
3. **Close the abstention gap**: a metric or a training signal that rewards using
   the elevation correctly, rather than only punishing using it wrongly. Tier
   utilisation reports the gap but nothing in the objective closes it.
