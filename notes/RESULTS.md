# Results

Two rounds. Round 1 established the pipeline and produced a mixed result; round
2 acted on what round 1 measured. Both are here because the second only makes
sense against the first.

Model: 27.8 M-parameter autoregressive layout transformer. Round 1 trained 400
epochs on 5,213 rooms; round 2 trained 250 epochs on 13,300. Sampling, rejection
and metrics are shared by every row, so each comparison isolates one variable.

---

## Round 2 — held-out FRONT-Elev, 300 rooms

| metric | GT | **ours** | +bias | flatten | per-tier | *ours R1* |
|---|---|---|---|---|---|---|
| overhang | 0.000 | **0.024** | 0.023 | 0.054 | 0.064 | *0.032* |
| embedded | 0.000 | **0.033** | 0.038 | 0.048 | 0.071 | *0.031* |
| straddling | 0.000 | **0.037** | 0.039 | 0.061 | 0.092 | *0.039* |
| wrong tier | 0.000 | **0.012** | 0.011 | 0.045 | 0.049 | *0.010* |
| **step blocked** (scene) | 0.000 | **0.213** | 0.213 | 0.330 | 0.877 | *0.653* |
| headroom | 0.000 | 0.007 | 0.005 | 0.012 | 0.009 | *0.007* |
| any violation (scene) | 0.003 | **0.580** | 0.550 | 0.710 | 0.950 | *0.763* |
| tier use, area | 0.457 | **0.429** | 0.427 | 0.097 | 0.450 | *0.356* |
| tier use, objects/scene | 3.56 | 4.27 | 3.88 | **0.00** | 5.22 | *3.60* |
| CCN sweeping robot | 0.676 | **0.576** | 0.565 | 0.300 | 0.433 | *0.501* |
| CCN wheelchair | 0.699 | **0.562** | 0.568 | 0.264 | 0.502 | *0.505* |
| objects/scene | 9.87 | 11.92 | 10.88 | 11.56 | 16.09 | *9.48* |

## Round 2 — MP3D-Elev, 72 real rooms the generator never produced

| metric | **ours** | +bias | flatten | per-tier | *ours R1* |
|---|---|---|---|---|---|
| overhang | 0.073 | 0.077 | **0.015** | 0.055 | *0.096* |
| embedded | 0.018 | 0.025 | 0.014 | **0.008** | *0.063* |
| straddling | 0.046 | 0.052 | **0.015** | 0.042 | *0.088* |
| wrong tier | 0.045 | 0.056 | **0.007** | 0.028 | *0.079* |
| step blocked (scene) | 0.264 | 0.236 | **0.083** | 0.667 | *0.542* |
| any violation (scene) | 0.722 | 0.694 | **0.306** | 0.764 | *0.889* |
| tier use, area | 0.292 | 0.296 | **0.018** | 0.384 | *0.337* |
| tier use, objects/scene | 3.40 | 3.31 | **0.00** | 4.85 | *3.64* |
| CCN sweeping robot | 0.567 | 0.617 | 0.498 | **0.662** | *0.636* |

---

## What round 2 changed, and what each change bought

### Step clearance in the sampler — the largest single win

Round 1's worst number was step blocking: 0.65 of generated scenes obstructed a
transition against 0.000 in ground truth, and no method addressed it. The
rejection cost scored tier containment and object overlap but had no idea the
treads had to stay walkable. Adding tread occupancy to that cost, weighted three
times an object overlap:

| | round 1 | round 2 |
|---|---|---|
| in distribution | 0.653 | **0.213** |
| out of distribution | 0.542 | **0.264** |

A two-thirds reduction in distribution and a halving out of it. This is the one
change whose effect is unambiguous, because the sampler is shared by every
method and every method improved.

### Calibrating the generator's geometry — helped OOD, as predicted

Round 1 measured a domain gap and observed that rise, the one statistic
calibrated against real homes, was the one that transferred. Round 2 measured
two more (region size, shared-edge width) and rebuilt. Every OOD violation rate
fell:

| OOD metric | round 1 | round 2 | change |
|---|---|---|---|
| straddling | 0.088 | 0.046 | −48 % |
| embedded | 0.063 | 0.018 | −71 % |
| wrong tier | 0.079 | 0.045 | −43 % |
| overhang | 0.096 | 0.073 | −24 % |
| any violation (scene) | 0.889 | 0.722 | −19 % |

Corpus went from 5,792 to 14,782 pairs at the same time, so this is calibration
and scale together, not calibration alone. Separating them would need a run on a
2.55×-scaled but uncalibrated corpus, which has not been done.

### The tier-relative attention bias — the null replicates

`ours` and `+bias` remain within noise of each other on every metric of both
test sets, now on a 2.55× larger corpus:

| | ours | +bias |
|---|---|---|
| overhang, in dist. | 0.0243 | 0.0230 |
| overhang, OOD | 0.0729 | 0.0771 |
| straddling, OOD | 0.0456 | 0.0522 |

Two independent corpora, same answer. The module is off by default and should
be dropped from the paper.

---

## What is still true and unwelcome

**Flatten still wins the OOD violation metrics, and still by abstaining.** It
places 0.00 objects on non-datum tiers on both test sets and covers 1.8 % of the
raised floor against a ground truth of 45.7 %. Every violation is about an
object being on the wrong tier and it never puts one there. Our absolute numbers
improved by 24–71 % out of distribution and the ranking did not move, because
the thing separating us is not accuracy but willingness to use the elevation.
`samples3_mp3d.png` shows it directly: in two of four rooms the raised tier is
entirely empty in the flatten column.

**We now over-place.** 11.92 objects per scene against a ground truth of 9.87,
and 4.27 on raised tiers against 3.56. Round 1 matched ground truth here (9.48
and 3.60) and round 2 does not — the larger corpus made the model more willing
to keep going. `ccn_spread` at 0.391 against a ground truth 0.313 says the same
thing: our rooms are more obstructed than real ones.

**Still far from clean.** 0.58 of generated scenes contain at least one
violation against 0.003 of ground truth.

**Two calibration targets did not move.** Transition width is 1.90 m against a
real 2.93 m, and region size is 0.35 of the floor against a real 0.27. Both are
bounded by 3D-FRONT itself: its rooms are 22 m² at the median against 34 m² for
real ones, and a region grown from a furniture group cannot be a small fraction
of a small room. The visible consequence is in `corpus_v2_examples.png`, rows 2
and 6, where a sunken area covering 56 % of the room leaves the datum as a
narrow ring — not a design anyone would draw.

---

## Next

1. **Cap the object count** against the ground-truth distribution, or calibrate
   the stop head. The over-placement is new and cheap to fix.
2. **A larger-room source.** Both stuck statistics trace to 3D-FRONT's room
   sizes. Infinigen Indoors generates arbitrary rooms and its constraint solver
   is already hierarchical; extending it with an elevation primitive is the
   obvious way out, and was deferred in week 0.
3. **Separate scale from calibration** with a run on an uncalibrated corpus of
   the same size, so the OOD improvement can be attributed.
4. **A published baseline.** PhyScene cannot take an elevation field, so any
   adaptation of it is a second flatten. Worth doing only to answer the
   reviewer question directly, and it changes two variables at once.
