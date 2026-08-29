"""Train the layout transformer on FRONT-Elev.

Teacher-forced autoregressive: for each scene one prefix length is sampled, the
model sees the tiers plus the objects already placed, and predicts the next one
— whether to stop, its category, then (conditioned on that category) its size,
plan position, yaw, support and offset.

The support term is the one that matters. It is a cross-entropy over a pointer
into [tiers | placed objects | ceiling], so getting the floor wrong is a
classification error the model is directly penalised for, rather than a few
centimetres of regression error on a z it was free to invent.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import elevate3d  # noqa: F401

from elevate3d.gen.dataset import (ElevCorpus, MAX_OBJECTS, MAX_TIERS,
                                   N_CATEGORIES, SUPPORT_CEILING, collate)
from elevate3d.gen.model import Elevate3D, model_size


def losses(model, batch, device, rng):
    B = batch["cat"].shape[0]
    n = batch["n_obj"]
    # one prefix per scene; including n itself teaches the stop decision
    k = (torch.rand(B, device=device) * (n.float() + 1)).long().clamp(max=MAX_OBJECTS)
    n_ctx = int(k.max().item())

    # mask the context to each scene's own prefix
    b = {kk: v.clone() for kk, v in batch.items()}
    if n_ctx > 0:
        idx = torch.arange(n_ctx, device=device).unsqueeze(0)
        b["obj_mask"] = batch["obj_mask"][:, :n_ctx] & (idx < k.unsqueeze(1))
        for f in ("cat", "box", "dz", "sup"):
            b[f] = batch[f][:, :n_ctx] if f != "sup" else batch[f][:, :n_ctx]
    out, x, m, iob, q = model(b, n_ctx)

    stopping = (k >= n).float()
    l_stop = F.binary_cross_entropy_with_logits(out["stop"], stopping)

    live = stopping < 0.5
    if live.sum() == 0:
        return l_stop, {"stop": l_stop.item()}

    gather = k.clamp(max=MAX_OBJECTS - 1)
    tgt_cat = batch["cat"].gather(1, gather.unsqueeze(1)).squeeze(1)
    tgt_box = batch["box"].gather(
        1, gather.view(-1, 1, 1).expand(-1, 1, 7)).squeeze(1)
    tgt_sup = batch["sup"].gather(1, gather.unsqueeze(1)).squeeze(1)
    tgt_dz = batch["dz"].gather(1, gather.unsqueeze(1)).squeeze(1)

    if not model.use_tiers:
        # The ablation is "a model that does not know the floor has tiers", and
        # such a model has exactly one floor to point at — which is what every
        # current method assumes.  Collapsing the tier targets onto slot 0 is
        # that assumption stated in the loss; leaving them pointing at masked
        # slots would just make the loss infinite.
        tgt_sup = torch.where(tgt_sup < MAX_TIERS,
                              torch.zeros_like(tgt_sup), tgt_sup)

    l_cat = F.cross_entropy(out["cat"][live], tgt_cat[live])

    onehot = F.one_hot(tgt_cat, N_CATEGORIES).float()
    g = model.predict_geometry(q, x, m, iob, onehot, n_ctx)

    h = g["h"][live]
    l_size = g["size"].nll(h, tgt_box[live, 4:7])
    l_xy = g["xy"].nll(h, tgt_box[live, 0:2])
    l_yaw = g["yaw"].nll(h, tgt_box[live, 2:4])
    l_dz = g["dz"].nll(h, tgt_dz[live].unsqueeze(-1))
    l_sup = F.cross_entropy(g["support"][live], tgt_sup[live])

    # the NLLs live on a different scale from the cross-entropies, so they are
    # down-weighted rather than left to dominate the support term
    total = (l_stop + l_cat + 2.0 * l_sup + 0.2 * l_xy + 0.1 * l_size
             + 0.1 * l_yaw + 0.1 * l_dz)
    with torch.no_grad():
        acc = (g["support"][live].argmax(-1) == tgt_sup[live]).float().mean()
        cacc = (out["cat"][live].argmax(-1) == tgt_cat[live]).float().mean()
    return total, {"stop": l_stop.item(), "cat": l_cat.item(),
                   "sup": l_sup.item(), "xy": l_xy.item(),
                   "size": l_size.item(), "yaw": l_yaw.item(),
                   "dz": l_dz.item(), "sup_acc": acc.item(),
                   "cat_acc": cacc.item()}


def run_val(model, dl, device):
    model.eval()
    agg, n = {}, 0
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            _, parts = losses(model, batch, device, rng)
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
            n += 1
    model.train()
    return {k: v / max(n, 1) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/home/gino/data/elevate3d/frontelev")
    ap.add_argument("--arm", default="elev", choices=("elev", "flat"))
    ap.add_argument("--out", default="/home/gino/data/elevate3d/runs/m2")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-tiers", action="store_true",
                    help="ablation: hide the elevation field from the model")
    ap.add_argument("--no-tier-bias", action="store_true",
                    help="ablation: keep tier tokens, drop the attention bias")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tr = ElevCorpus(args.corpus, arm=args.arm, split="train")
    va = ElevCorpus(args.corpus, arm=args.arm, split="val")
    print(f"train {len(tr)}  val {len(va)}  device {dev}", flush=True)

    dtr = DataLoader(tr, batch_size=args.bs, shuffle=True, drop_last=True,
                     num_workers=args.workers, collate_fn=collate,
                     pin_memory=True, persistent_workers=args.workers > 0)
    dva = DataLoader(va, batch_size=args.bs, shuffle=False,
                     num_workers=2, collate_fn=collate)

    model = Elevate3D(d=args.d, layers=args.layers, heads=args.heads,
                      use_tiers=not args.no_tiers,
                      use_tier_bias=not args.no_tier_bias).to(dev)
    print(model_size(model), flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wd, betas=(0.9, 0.95))
    steps = args.epochs * max(len(dtr), 1)
    warm = min(1000, steps // 20)

    def lr_at(s):
        if s < warm:
            return s / max(warm, 1)
        p = (s - warm) / max(steps - warm, 1)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

    rng = np.random.default_rng(args.seed)
    best = float("inf")
    hist = []
    step = 0
    t0 = time.time()
    for ep in range(args.epochs):
        run = {}
        for batch in dtr:
            batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=dev.type == "cuda",
                                    dtype=torch.bfloat16):
                loss, parts = losses(model, batch, dev, rng)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            for k, v in parts.items():
                run[k] = run.get(k, 0.0) + v
        nb = max(len(dtr), 1)
        tr_line = {k: v / nb for k, v in run.items()}
        if ep % 5 == 0 or ep == args.epochs - 1:
            vl = run_val(model, dva, dev)
            score = vl.get("sup", 9e9) + vl.get("xy", 0.0)
            hist.append({"epoch": ep, "train": tr_line, "val": vl,
                         "lr": sched.get_last_lr()[0]})
            print(f"ep {ep:3d}  {time.time()-t0:6.0f}s  "
                  f"train sup={tr_line.get('sup',0):.3f} xy={tr_line.get('xy',0):.4f} "
                  f"| val sup={vl.get('sup',0):.3f} "
                  f"sup_acc={vl.get('sup_acc',0):.3f} "
                  f"xy={vl.get('xy',0):.4f} cat_acc={vl.get('cat_acc',0):.3f}",
                  flush=True)
            if score < best:
                best = score
                torch.save({"model": model.state_dict(), "args": vars(args),
                            "epoch": ep, "val": vl},
                           os.path.join(args.out, "best.pt"))
            with open(os.path.join(args.out, "history.json"), "w") as fh:
                json.dump(hist, fh, indent=1)
    torch.save({"model": model.state_dict(), "args": vars(args),
                "epoch": args.epochs - 1}, os.path.join(args.out, "last.pt"))
    print(f"done in {time.time()-t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
