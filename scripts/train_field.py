"""Train M1, the elevation-field generator.

Both arms of every corpus pair are used: the elevated one teaches where a
platform belongs and how big it is, the flat one teaches that most rooms do not
get one. A model trained only on the elevated arm would propose an elevation for
every room it ever saw.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import elevate3d  # noqa: F401

from elevate3d.gen.field import PROGRAMS, params_from_scene
from elevate3d.gen.field_model import FieldNet, encode_room, field_losses


class FieldCorpus(Dataset):
    def __init__(self, root, split="train", val_frac=0.1, seed=0,
                 flat_ratio=1.0):
        recs = []
        for f in sorted(glob.glob(os.path.join(root, "*.jsonl.gz"))):
            with gzip.open(f, "rt") as fh:
                recs += [json.loads(l) for l in fh]
        if not recs:
            raise FileNotFoundError(root)
        rng = np.random.default_rng(seed)
        keys = np.array([r["flat"]["scene_id"] for r in recs])
        uniq = np.unique(keys)
        rng.shuffle(uniq)
        val = set(uniq[:max(1, int(len(uniq) * val_frac))].tolist())
        keep = [r for r, k in zip(recs, keys) if (k in val) == (split == "val")]

        self.items = []
        seen_flat = set()
        for r in keep:
            self.items.append(r["elev"])
            # one flat negative per distinct room, not per variant, or the
            # negatives get duplicated by the variant count
            sid = r["flat"]["scene_id"]
            if sid not in seen_flat and rng.random() < flat_ratio:
                seen_flat.add(sid)
                self.items.append(r["flat"])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        d = self.items[i]
        enc = encode_room(d["room"]["polygon"], d["room"].get("height", 2.8))
        p = params_from_scene(d)
        return {
            "boundary": torch.as_tensor(enc["boundary"]),
            "scalars": torch.as_tensor(enc["scalars"]),
            "program": torch.tensor(p.program, dtype=torch.long),
            "box": torch.as_tensor(np.asarray(p.box, dtype=np.float32)),
            "rise": torch.tensor(float(p.rise), dtype=torch.float32),
        }


def collate(b):
    return {k: torch.stack([x[k] for x in b]) for k in b[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/home/gino/data/elevate3d/frontelev3")
    ap.add_argument("--out", default="/home/gino/data/elevate3d/runs/m1")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tr = FieldCorpus(args.corpus, "train")
    va = FieldCorpus(args.corpus, "val")
    n_pos = sum(1 for i in tr.items if (i.get("meta") or {}).get("program"))
    print(f"train {len(tr)} ({n_pos} elevated)  val {len(va)}  device {dev}",
          flush=True)

    dtr = DataLoader(tr, batch_size=args.bs, shuffle=True, drop_last=True,
                     num_workers=args.workers, collate_fn=collate,
                     persistent_workers=args.workers > 0)
    dva = DataLoader(va, batch_size=args.bs, num_workers=2, collate_fn=collate)

    model = FieldNet(d=args.d, layers=args.layers).to(dev)
    print(f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters",
          flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.02)
    steps = args.epochs * max(len(dtr), 1)
    warm = min(500, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / max(warm, 1) if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(steps - warm, 1))))

    best, hist, t0 = float("inf"), [], time.time()
    for ep in range(args.epochs):
        model.train()
        run = {}
        for b in dtr:
            b = {k: v.to(dev) for k, v in b.items()}
            loss, parts = field_losses(model, b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            for k, v in parts.items():
                run[k] = run.get(k, 0.0) + v
        if ep % 10 == 0 or ep == args.epochs - 1:
            model.eval()
            agg, n = {}, 0
            with torch.no_grad():
                for b in dva:
                    b = {k: v.to(dev) for k, v in b.items()}
                    _, parts = field_losses(model, b)
                    for k, v in parts.items():
                        agg[k] = agg.get(k, 0.0) + v
                    n += 1
            vl = {k: v / max(n, 1) for k, v in agg.items()}
            score = vl.get("box", 9e9) + vl.get("prog", 0.0)
            hist.append({"epoch": ep, "val": vl})
            print(f"ep {ep:3d} {time.time()-t0:5.0f}s  val prog={vl.get('prog',0):.3f} "
                  f"acc={vl.get('prog_acc',0):.3f} box={vl.get('box',0):.3f} "
                  f"rise={vl.get('rise',0):.3f}", flush=True)
            if score < best:
                best = score
                torch.save({"model": model.state_dict(), "args": vars(args),
                            "cfg": {"d": args.d, "layers": args.layers},
                            "epoch": ep, "val": vl},
                           os.path.join(args.out, "best.pt"))
            json.dump(hist, open(os.path.join(args.out, "history.json"), "w"),
                      indent=1)
    print(f"done in {time.time()-t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
