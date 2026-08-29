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

Six rounds, each acting on what the previous one measured. It is now a two-stage
system: **M1** (3.5 M) generates the elevation field, **M2** (27.8 M) lays out
on it.

**Round 4 refuted rounds 2-3's proposed next step** — the region fraction was
stuck because the region was derived from furniture, not because 3D-FRONT's
rooms are small. **Round 5 acted on that**: the region is now taken off a wall
at a depth drawn from the measured distribution.

| corpus statistic | R1 | R3 | **R5** | real |
|---|---|---|---|---|
| region fraction | 0.46 | 0.41 | **0.278** | 0.271 |
| Wasserstein | 0.164 | 0.099 | **0.022** | — |

**The out-of-distribution gap has essentially closed.** On the 72 real
MP3D-Elev fields, ours went from 0.569 to **0.250** any-violation between
rounds 3 and 6 against flatten's 0.208, and now *beats* flatten on straddling
while placing 1.28 objects per scene on the raised floor to flatten's 0.00.

**Round 6 is the first end-to-end result.** M1 proposes a field for 60 % of
rooms and refuses on 40 %. Elevation F1: 0.817 with a given field, **0.594**
end-to-end, 0.000 for the flat baseline.

The tier-relative attention bias measured as having no effect across every round
and is off by default.

`notes/W0_findings.md` records the dataset measurements, `notes/RESULTS.md` all
six rounds including every negative result and every reverted change.

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
