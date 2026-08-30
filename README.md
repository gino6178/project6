# Elevate3D

Indoor scene synthesis where the floor is **not** one plane — sunken lounges,
tatami platforms, split levels and mezzanines as a generation target.

**[Paper →](https://gino6178.github.io/project6/paper.html)** ·
**[Research log →](https://gino6178.github.io/project6/)**

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

Seven rounds, each acting on what the previous one measured. Two stages:
**M1** (3.5 M) generates the elevation field, **M2** (27.8 M) lays out on it.

### End to end

| | GT | **M1 + M2** | flatten |
|---|---|---|---|
| elevation F1 | 0.994 | **0.813** | 0.000 |
| raised-floor use, obj/scene | 1.66 | **1.58** | 0.00 |
| CCN sweeping robot | 0.660 | **0.655** | 0.583 |

M1 proposes an elevation for 58 % of rooms and refuses on 42 %. Generating the
field costs about a tenth of the F1 against being handed one (0.813 vs 0.892);
in round 6 it cost a quarter.

### What each round established

- **R3 corrected R2**: the out-of-distribution gain came from the sampler and
  the extra data, not from calibrating the generator's geometry.
- **R4 added `elevation_f1`**, which scores the flat baseline **0.000** — it had
  been winning every violation metric by never placing an object on a raised
  tier — and **refuted R2–R3's proposed next step**: retargeting into 34–49 m²
  rooms leaves the region fraction unchanged, and Infinigen's rooms are no
  larger than 3D-FRONT's.
- **R5 acted on that**: regions are taken off a wall at a measured depth.
  Region fraction 0.410 → **0.278** (real 0.271), Wasserstein 0.164 → **0.022**.
  On the 72 real MP3D-Elev fields, violations fell 56–78 %.
- **R6** built M1; **R7** closed the last four items. The stop head is now
  trained calibrated (its threshold moved from the grid floor 0.05 to 0.5), M1
  sees the room type (program accuracy 0.53 → 0.650), and scoring the step's
  landing lifted corpus yield 52.7 % → **73.7 %**.

### Negative results, kept

- The tier-relative attention bias measured as having no effect in every round.
- Upweighting the non-datum tiers in the support loss **overshoots** — it uses
  the elevation more than ground truth and loses precision (0.755 vs 0.883). A
  calibrated sampling temperature achieves the same end without the cost.
- Clearing the step's landing by moving furniture widens transitions to 2.8 m
  but halves the yield and breaks ground-truth cleanliness. Transition width
  (1.67 m against a real 2.93) is a property of 3D-FRONT's furniture density,
  and both available routes have now been measured.

`notes/W0_findings.md` records the dataset measurements, `notes/RESULTS.md` all
seven rounds including every negative result and every reverted change.

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
