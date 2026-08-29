"""Elevate3D's layout transformer.

Autoregressive over objects, in the ATISS mould, with three changes that follow
from the floor no longer being one plane:

* **Tier tokens.** The elevation field enters the context as one token per
  tier — geometry, height, and the smallest riser leaving it — so the attention
  can see both where the tiers are and who can get onto them.
* **A support pointer instead of a z regression.** The support head scores the
  tier tokens and the already-placed object tokens and points at one of them.
  ``z`` is then computed by ``ElevScene.resolve``, so contact is exact and
  floating is unrepresentable.
* **A tier-relative attention bias.** Three learned scalars — same tier, across
  tiers, parent/child — added to the attention logits. The original proposal
  had a bias on continuous Δz; that is nearly always zero within a tier and
  carries no signal, whereas the discrete relation is exactly what matters.

``use_tiers=False`` strips the tier tokens and the bias, which is the ablation:
the same model told nothing about the floor.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import (BOUNDARY_POINTS, MAX_OBJECTS, MAX_TIERS, N_CATEGORIES,
                      SUPPORT_CEILING)

__all__ = ["Elevate3D", "model_size"]


class MixtureHead(nn.Module):
    """A mixture density head, because a regression head cannot place furniture.

    Trained with smooth L1, ``xy`` converges to the conditional mean of where a
    sofa goes, which is the middle of the room — so every sofa lands in the same
    place and the layout is a pile.  The placement distribution is multimodal
    (against this wall, or that one), and only a mixture can represent it.
    """

    def __init__(self, d_in: int, dim: int, k: int = 10, hidden: int = 512):
        super().__init__()
        self.dim, self.k = dim, k
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(),
                                 nn.Linear(hidden, k * (1 + 2 * dim)))

    def params(self, h):
        o = self.net(h)
        k, dim = self.k, self.dim
        pi = o[..., :k]
        mu = o[..., k:k + k * dim].view(*o.shape[:-1], k, dim)
        ls = o[..., k + k * dim:].view(*o.shape[:-1], k, dim)
        return pi, mu, ls.clamp(-7.0, 2.0)

    def nll(self, h, target):
        pi, mu, ls = self.params(h)
        t = target.unsqueeze(-2)
        # diagonal Gaussian components, shared weights across dims
        lp = (-0.5 * ((t - mu) * torch.exp(-ls)) ** 2 - ls
              - 0.5 * math.log(2 * math.pi)).sum(-1)
        return -(torch.logsumexp(F.log_softmax(pi, -1) + lp, -1)).mean()

    def sample(self, h, temperature: float = 1.0, generator=None):
        pi, mu, ls = self.params(h)
        idx = torch.distributions.Categorical(
            logits=pi / max(temperature, 1e-3)).sample()
        g = idx.view(*idx.shape, 1, 1).expand(*idx.shape, 1, self.dim)
        m = mu.gather(-2, g).squeeze(-2)
        s = ls.gather(-2, g).squeeze(-2).exp()
        return m + s * torch.randn_like(m) * temperature

    def mode(self, h):
        pi, mu, _ = self.params(h)
        idx = pi.argmax(-1)
        g = idx.view(*idx.shape, 1, 1).expand(*idx.shape, 1, self.dim)
        return mu.gather(-2, g).squeeze(-2)


class TierBiasedBlock(nn.Module):
    """Pre-norm encoder block whose attention takes an additive bias."""

    def __init__(self, d: int, heads: int, ff: int, drop: float):
        super().__init__()
        self.h = heads
        self.dk = d // heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, ff), nn.GELU(),
                                 nn.Linear(ff, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x, key_mask, bias=None):
        B, L, D = x.shape
        h = self.n1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(B, L, self.h, self.dk).transpose(1, 2)
        k = k.view(B, L, self.h, self.dk).transpose(1, 2)
        v = v.view(B, L, self.h, self.dk).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        if bias is not None:
            att = att + bias.unsqueeze(1)
        att = att.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        att = att.softmax(-1)
        out = (att @ v).transpose(1, 2).reshape(B, L, D)
        x = x + self.drop(self.proj(out))
        x = x + self.drop(self.mlp(self.n2(x)))
        return x


class Elevate3D(nn.Module):
    def __init__(self, d: int = 512, layers: int = 8, heads: int = 8,
                 ff: int = 2048, drop: float = 0.1, use_tiers: bool = True,
                 use_tier_bias: bool = True):
        super().__init__()
        self.d = d
        self.use_tiers = use_tiers
        self.use_tier_bias = use_tier_bias and use_tiers

        self.bnd = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        self.bnd_pos = nn.Parameter(torch.randn(BOUNDARY_POINTS, d) * 0.02)
        self.room = nn.Linear(2, d)
        self.tier = nn.Sequential(nn.Linear(9, d), nn.GELU(), nn.Linear(d, d))
        self.tier_pos = nn.Parameter(torch.randn(MAX_TIERS, d) * 0.02)

        self.cat_emb = nn.Embedding(N_CATEGORIES, d)
        self.obj = nn.Sequential(nn.Linear(7 + 1, d), nn.GELU(), nn.Linear(d, d))
        self.query = nn.Parameter(torch.randn(d) * 0.02)

        self.kind = nn.Parameter(torch.randn(4, d) * 0.02)  # bnd/room/tier/obj

        self.blocks = nn.ModuleList(
            [TierBiasedBlock(d, heads, ff, drop) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)

        # three learned scalars per head: same tier, across tiers, parent/child
        self.rel_bias = nn.Parameter(torch.zeros(3))

        self.head_stop = nn.Linear(d, 1)
        self.head_cat = nn.Linear(d, N_CATEGORIES)
        self.head_size = MixtureHead(d + N_CATEGORIES, 3, k=8, hidden=d)
        self.head_xy = MixtureHead(d + N_CATEGORIES, 2, k=16, hidden=d)
        self.head_yaw = MixtureHead(d + N_CATEGORIES, 2, k=8, hidden=d)
        self.head_dz = MixtureHead(d + N_CATEGORIES, 1, k=4, hidden=d)
        # support is a pointer, so it is scored against the candidate tokens
        self.sup_q = nn.Linear(d + N_CATEGORIES, d)
        self.sup_k = nn.Linear(d, d)
        self.ceil_tok = nn.Parameter(torch.randn(d) * 0.02)

    # -- context ----------------------------------------------------------
    def encode(self, batch, n_ctx: int):
        """Tokens = boundary + room + tiers + the first ``n_ctx`` objects + query."""
        B = batch["boundary"].shape[0]
        dev = batch["boundary"].device
        toks, mask, tier_of, is_obj = [], [], [], []

        b = self.bnd(batch["boundary"]) + self.bnd_pos + self.kind[0]
        toks.append(b)
        mask.append(torch.ones(B, BOUNDARY_POINTS, dtype=torch.bool, device=dev))
        tier_of.append(torch.full((B, BOUNDARY_POINTS), -1, device=dev,
                                  dtype=torch.long))
        is_obj.append(torch.zeros(B, BOUNDARY_POINTS, dtype=torch.bool, device=dev))

        r = (self.room(batch["room"]) + self.kind[1]).unsqueeze(1)
        toks.append(r)
        mask.append(torch.ones(B, 1, dtype=torch.bool, device=dev))
        tier_of.append(torch.full((B, 1), -1, device=dev, dtype=torch.long))
        is_obj.append(torch.zeros(B, 1, dtype=torch.bool, device=dev))

        if self.use_tiers:
            t = self.tier(batch["tiers"]) + self.tier_pos + self.kind[2]
            toks.append(t)
            mask.append(batch["tier_mask"])
            tier_of.append(
                torch.arange(MAX_TIERS, device=dev).unsqueeze(0).expand(B, -1))
            is_obj.append(torch.zeros(B, MAX_TIERS, dtype=torch.bool, device=dev))

        if n_ctx > 0:
            cat = batch["cat"][:, :n_ctx]
            box = batch["box"][:, :n_ctx]
            dz = batch["dz"][:, :n_ctx].unsqueeze(-1)
            o = self.obj(torch.cat([box, dz], -1)) + self.cat_emb(cat) + self.kind[3]
            toks.append(o)
            mask.append(batch["obj_mask"][:, :n_ctx])
            sup = batch["sup"][:, :n_ctx]
            # which tier does each placed object ultimately sit over?  a
            # one-step lookup is enough: parents are placed before children
            tt = torch.where(sup < MAX_TIERS, sup, torch.zeros_like(sup))
            tier_of.append(tt)
            is_obj.append(torch.ones(B, n_ctx, dtype=torch.bool, device=dev))

        q = self.query.view(1, 1, -1).expand(B, 1, -1)
        toks.append(q)
        mask.append(torch.ones(B, 1, dtype=torch.bool, device=dev))
        tier_of.append(torch.full((B, 1), -1, device=dev, dtype=torch.long))
        is_obj.append(torch.zeros(B, 1, dtype=torch.bool, device=dev))

        x = torch.cat(toks, 1)
        m = torch.cat(mask, 1)
        tof = torch.cat(tier_of, 1)
        iob = torch.cat(is_obj, 1)

        bias = None
        if self.use_tier_bias:
            same = (tof[:, :, None] == tof[:, None, :]) & (tof[:, :, None] >= 0)
            both = (tof[:, :, None] >= 0) & (tof[:, None, :] >= 0)
            cross = both & ~same
            pc = iob[:, :, None] & iob[:, None, :] & cross
            bias = (self.rel_bias[0] * same.float()
                    + self.rel_bias[1] * cross.float()
                    + self.rel_bias[2] * pc.float())

        for blk in self.blocks:
            x = blk(x, m, bias)
        return self.norm(x), m, iob

    # -- prediction -------------------------------------------------------
    def forward(self, batch, n_ctx: int):
        x, m, iob = self.encode(batch, n_ctx)
        q = x[:, -1]                                     # the query token
        out = {"stop": self.head_stop(q).squeeze(-1),
               "cat": self.head_cat(q)}
        return out, x, m, iob, q

    def predict_geometry(self, q, x, m, iob, cat_onehot, n_ctx: int):
        """Everything that is conditioned on the category being placed."""
        h = torch.cat([q, cat_onehot], -1)
        d = {"h": h,
             "size": self.head_size, "xy": self.head_xy,
             "yaw": self.head_yaw, "dz": self.head_dz}
        # support pointer over [tiers | placed objects | ceiling]
        B = x.shape[0]
        off = BOUNDARY_POINTS + 1
        cands, cmask = [], []
        if self.use_tiers:
            cands.append(x[:, off:off + MAX_TIERS])
            cmask.append(m[:, off:off + MAX_TIERS])
            off += MAX_TIERS
        else:
            cands.append(x.new_zeros(B, MAX_TIERS, self.d))
            cmask.append(torch.zeros(B, MAX_TIERS, dtype=torch.bool,
                                     device=x.device))
            cmask[-1][:, 0] = True                       # a single implicit floor
        obj_slots = x.new_zeros(B, MAX_OBJECTS, self.d)
        obj_ok = torch.zeros(B, MAX_OBJECTS, dtype=torch.bool, device=x.device)
        if n_ctx > 0:
            obj_slots[:, :n_ctx] = x[:, off:off + n_ctx]
            obj_ok[:, :n_ctx] = m[:, off:off + n_ctx]
        cands.append(obj_slots)
        cmask.append(obj_ok)
        cands.append(self.ceil_tok.view(1, 1, -1).expand(B, 1, -1))
        cmask.append(torch.ones(B, 1, dtype=torch.bool, device=x.device))

        K = torch.cat(cands, 1)
        KM = torch.cat(cmask, 1)
        logits = (self.sup_k(K) @ self.sup_q(h).unsqueeze(-1)).squeeze(-1)
        logits = logits / math.sqrt(self.d)
        d["support"] = logits.masked_fill(~KM, float("-inf"))
        return d


def model_size(m: nn.Module) -> str:
    n = sum(p.numel() for p in m.parameters())
    return f"{n/1e6:.1f}M parameters"
