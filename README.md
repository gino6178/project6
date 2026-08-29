# Elevate3D

Indoor scene synthesis where the floor is **not** one plane — sunken lounges,
tatami platforms, split levels and mezzanines as a generation target.

**[Method and results →](https://gino6178.github.io/project6/)**

Every layout dataset in use stores a room as a 2D polygon and puts every object
at `z = 0`. Measured on 3D-FRONT: across all 6,813 houses and 37,529 rooms the
floor-mesh height spread is **exactly zero**. Measured on real Matterport3D
scans via HouseLayout3D: **12 of 16 buildings** have an intra-storey floor
height change, median 0.38 m.

## What is here

| | |
|---|---|
| `elevate3d/geom/elevation.py` | the elevation field: tiers, transitions, per-riser reachability |
| `elevate3d/core/scene.py` | support-tree parameterisation — `z` is derived, never predicted |
| `elevate3d/data/frontelev.py` | lifts real 3D-FRONT layouts onto real elevation programs |
| `elevate3d/data/houselayout.py` | real elevation fields from HouseLayout3D / Matterport3D |
| `elevate3d/eval/violations.py` | F1a overhang · F1b embedded · F2 straddling · F3 step blocked · F4 headroom · F5 datum, plus tier utilisation |
| `elevate3d/eval/navigability.py` | capability-conditioned reachability (CapNav agent profiles) |
| `elevate3d/gen/` | 27.8M autoregressive layout transformer with a support pointer |
| `scripts/measure_real_geometry.py` | the measured priors the generator is calibrated against |

`notes/W0_findings.md` records the dataset measurements; `notes/RESULTS.md`
records the first end-to-end numbers, including the negative ones.

## State

Three rounds, each acting on what the previous measured.

In distribution, tier conditioning cuts every placement violation to a third or
a sixth of the flat baseline, and object counts and raised-floor use both match
ground truth. Scoring tread occupancy in the shared sampler cut step blocking
from 0.65 to **0.19**; calibrating the stop threshold fixed the over-placement
that the larger corpus introduced.

Round 3 **corrects an attribution round 2 got wrong**: a scale ablation (same
corpus and sampler, round-1 training size) shows the out-of-distribution gain
came from the sampler and the extra data, not from calibrating the generator's
geometry. The calibration was still worth doing — it removed three invented
parameters — but it is a methodological cleanup, not a metric gain.

Still open: **the flat baseline keeps lower out-of-distribution violation rates
by abstaining** — 0.00 objects on non-datum tiers, 3.4 % of the raised floor
covered against a ground truth 45.7 %. Nothing in the objective rewards using
the elevation correctly; it only punishes using it wrongly. The tier-relative
attention bias measured as having no effect across three rounds and two corpora
and is off by default.

`notes/W0_findings.md` records the dataset measurements, `notes/RESULTS.md` all
three rounds including the negative results and the Infinigen assessment.

## Reproduce

Depends on [project5 (ReRoom)](https://github.com/gino6178/project5) for the
3D-FRONT parser and geometry helpers — imported, not vendored. Override the path
with `REROOM_ROOT`.

```bash
pytest tests/                       # 13 tests, including L_contact == 0
python scripts/scan_front_elevation.py
python scripts/build_frontelev.py --workers 12 --tries 8
python scripts/build_mp3d_elev.py
python scripts/train_elevate.py --out runs/m2_full --epochs 400
python scripts/sample_elevate.py --runs runs --out outputs/eval_frontelev.json
```

Full command list in §10 of the page.
