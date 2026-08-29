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

Two rounds. In distribution, tier conditioning roughly halves every placement
violation against the flat baseline and is the only method whose use of the
raised floor matches ground truth.

Round 2 acted on what round 1 measured. Scoring tread occupancy in the sampler
cut step blocking from 0.65 to **0.21** in distribution and 0.54 to **0.26** out
of it — the largest single win, and it applies to every method since the sampler
is shared. Calibrating the generator's geometry against real homes, together
with a 2.55x larger corpus, cut every out-of-distribution violation rate by
24–71 %.

Still open: **the flat baseline keeps lower OOD violation rates by abstaining** —
it places 0.00 objects on non-datum tiers and covers 1.8 % of the raised floor
against a ground truth 45.7 %. The tier-relative attention bias measured as
having no effect on two independent corpora and is off by default. See §8–§9 of
the page.

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
