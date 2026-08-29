"""M1's network: room shape in, elevation field out.

Small on purpose. The architecture-driven region rule reduced a field to a
program label, an interval box in the room's principal frame and a rise, so the
model has a handful of numbers to predict from the boundary — there is no
sequence to roll out and no polygon to draw.

The box and the rise get mixture heads for the same reason the layout model
does: where a platform goes is multimodal (this wall or that one), and a
regression head would answer with the middle of the room.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import BOUNDARY_POINTS
from .field import PROGRAMS
from .model import MixtureHead, TierBiasedBlock

__all__ = ["FieldNet", "encode_room"]

N_PROGRAMS = len(PROGRAMS)


def encode_room(poly: np.ndarray, height: float, room_type: str = "") -> dict:
    """Boundary points in the room's principal frame, plus scale scalars."""
    from .field import room_frame
    import math
    from shapely.geometry import Polygon

    poly = np.asarray(poly, dtype=float)
    yaw, origin, extent = room_frame(poly)
    R = np.array([[math.cos(-yaw), -math.sin(-yaw)],
                  [math.sin(-yaw), math.cos(-yaw)]])
    q = (poly @ R.T - origin) / np.maximum(extent, 1e-6)

    # arc-length resample so a 4-gon and a 12-gon look alike to the encoder
    p = np.vstack([q, q[:1]])
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = max(cum[-1], 1e-9)
    t = np.linspace(0.0, total, BOUNDARY_POINTS, endpoint=False)
    out = np.empty((BOUNDARY_POINTS, 2))
    for k, tt in enumerate(t):
        i = int(np.clip(np.searchsorted(cum, tt, side="right") - 1, 0, len(seg) - 1))
        u = (tt - cum[i]) / max(seg[i], 1e-9)
        out[k] = p[i] * (1 - u) + p[i + 1] * u

    area = float(Polygon(poly).area)
    return {
        "boundary": out.astype(np.float32),
        "scalars": np.array([extent[0], extent[1], area, float(height),
                             extent[0] / max(extent[1], 1e-6)],
                            dtype=np.float32),
    }


class FieldNet(nn.Module):
    def __init__(self, d: int = 256, layers: int = 4, heads: int = 8,
                 ff: int = 1024, drop: float = 0.1):
        super().__init__()
        self.d = d
        self.bnd = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        self.pos = nn.Parameter(torch.randn(BOUNDARY_POINTS, d) * 0.02)
        self.sca = nn.Sequential(nn.Linear(5, d), nn.GELU(), nn.Linear(d, d))
        self.query = nn.Parameter(torch.randn(d) * 0.02)
        self.blocks = nn.ModuleList(
            [TierBiasedBlock(d, heads, ff, drop) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)

        self.head_prog = nn.Linear(d, N_PROGRAMS)
        self.head_box = MixtureHead(d + N_PROGRAMS, 4, k=12, hidden=d)
        self.head_rise = MixtureHead(d + N_PROGRAMS, 1, k=8, hidden=d)

    def encode(self, batch):
        B = batch["boundary"].shape[0]
        dev = batch["boundary"].device
        x = torch.cat([
            self.bnd(batch["boundary"]) + self.pos,
            self.sca(batch["scalars"]).unsqueeze(1),
            self.query.view(1, 1, -1).expand(B, 1, -1),
        ], dim=1)
        m = torch.ones(x.shape[:2], dtype=torch.bool, device=dev)
        for blk in self.blocks:
            x = blk(x, m, None)
        return self.norm(x)[:, -1]

    def forward(self, batch):
        q = self.encode(batch)
        return {"q": q, "prog": self.head_prog(q)}

    def geometry(self, q, prog_onehot):
        h = torch.cat([q, prog_onehot], -1)
        return h

    @torch.no_grad()
    def sample(self, batch, temperature: float = 1.0, rng=None):
        """A program, a box and a rise per room."""
        out = self.forward(batch)
        p = F.softmax(out["prog"] / max(temperature, 1e-3), -1)
        idx = torch.multinomial(p, 1).squeeze(-1)
        oh = F.one_hot(idx, N_PROGRAMS).float()
        h = self.geometry(out["q"], oh)
        box = self.head_box.sample(h, temperature).clamp(0.0, 1.0)
        rise = self.head_rise.sample(h, temperature).squeeze(-1)
        return idx, box, rise


def field_losses(model, batch):
    out = model(batch)
    l_prog = F.cross_entropy(out["prog"], batch["program"])
    live = batch["program"] > 0
    parts = {"prog": l_prog.item(),
             "prog_acc": (out["prog"].argmax(-1) == batch["program"]).float()
             .mean().item()}
    total = l_prog
    if live.any():
        oh = F.one_hot(batch["program"], N_PROGRAMS).float()
        h = model.geometry(out["q"], oh)[live]
        l_box = model.head_box.nll(h, batch["box"][live])
        l_rise = model.head_rise.nll(h, batch["rise"][live].unsqueeze(-1))
        total = total + 0.3 * l_box + 0.3 * l_rise
        parts.update(box=l_box.item(), rise=l_rise.item())
    return total, parts
