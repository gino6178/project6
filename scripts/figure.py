"""Compose rendered panels into paper figures.

The renderer writes RGBA with a transparent film, because a room is an L or a
T and its bounding box is mostly empty air. Cropping to the alpha bbox before
laying panels out is what makes a four-column comparison readable at column
width -- laid out uncropped, the geometry occupies about a third of each cell.
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BG = (255, 255, 255)
INK = (28, 30, 34)
MUTED = (110, 116, 124)


def _font(size: int, bold: bool = False):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf"
              % ("-Bold" if bold else ""),):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def crop_alpha(img: Image.Image, pad: int = 6) -> Image.Image:
    """Trim to what was actually drawn, keeping a small margin."""
    a = np.asarray(img.convert("RGBA"))[..., 3]
    ys, xs = np.where(a > 4)
    if not len(ys):
        return img
    y0, y1 = max(0, ys.min() - pad), min(img.height, ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(img.width, xs.max() + pad + 1)
    return img.crop((x0, y0, x1, y1))


def on_white(img: Image.Image) -> Image.Image:
    bg = Image.new("RGB", img.size, BG)
    img = img.convert("RGBA")
    bg.paste(img, (0, 0), img)
    return bg


def load(path: str, pad: int = 6) -> Image.Image:
    return on_white(crop_alpha(Image.open(path), pad))


def load_group(paths, pad: int = 6):
    """Crop a set of panels to one shared box.

    Panels in a comparison row show the same room under different methods. Crop
    each to its own content and the room silently changes size from column to
    column, which reads as a difference between the methods when it is only a
    difference in what happened to be drawn. A shared box keeps the scale
    honest -- provided the panels were rendered at the same resolution, which is
    checked here rather than assumed.
    """
    imgs = [Image.open(p).convert("RGBA") for p in paths]
    if len({im.size for im in imgs}) != 1:
        return [on_white(crop_alpha(im, pad)) for im in imgs]
    box = None
    for im in imgs:
        a = np.asarray(im)[..., 3]
        ys, xs = np.where(a > 4)
        if not len(ys):
            continue
        b = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        box = b if box is None else (min(box[0], b[0]), min(box[1], b[1]),
                                     max(box[2], b[2]), max(box[3], b[3]))
    if box is None:
        return [on_white(im) for im in imgs]
    w, h = imgs[0].size
    box = (max(0, box[0] - pad), max(0, box[1] - pad),
           min(w, box[2] + pad), min(h, box[3] + pad))
    return [on_white(im.crop(box)) for im in imgs]


def fit(img: Image.Image, w: int, h: int) -> Image.Image:
    """Scale to fit inside w x h without distortion, centred on white."""
    s = min(w / img.width, h / img.height)
    im = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))),
                    Image.LANCZOS)
    cell = Image.new("RGB", (w, h), BG)
    cell.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return cell


def grid(paths, out: str, cols: int, cell=(430, 300), col_titles=None,
         row_titles=None, captions=None, title=None, gap=10,
         share_crop=False):
    """Lay panels out in a grid with optional column / row / per-cell labels."""
    n = len(paths)
    rows = (n + cols - 1) // cols
    cw, ch = cell
    f_col = _font(17, True)
    f_row = _font(15, True)
    f_cap = _font(14)
    f_ttl = _font(19, True)

    lm = 0 if not row_titles else 96
    tm = 0 if not col_titles else 30
    tm += 0 if not title else 30
    cap_h = 0 if not captions else 22

    W = lm + cols * cw + (cols - 1) * gap
    H = tm + rows * (ch + cap_h) + (rows - 1) * gap
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)

    if title:
        d.text((lm, 6), title, font=f_ttl, fill=INK)
    if col_titles:
        y = 8 + (30 if title else 0)
        for c, t in enumerate(col_titles[:cols]):
            x = lm + c * (cw + gap)
            w = d.textlength(t, font=f_col)
            d.text((x + (cw - w) / 2, y), t, font=f_col, fill=INK)

    imgs = load_group(paths) if share_crop else [load(p) for p in paths]
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        x = lm + c * (cw + gap)
        y = tm + r * (ch + cap_h + gap)
        canvas.paste(fit(im, cw, ch), (x, y))
        if captions and i < len(captions) and captions[i]:
            t = captions[i]
            w = d.textlength(t, font=f_cap)
            d.text((x + (cw - w) / 2, y + ch + 3), t, font=f_cap, fill=MUTED)

    if row_titles:
        for r, t in enumerate(row_titles[:rows]):
            y = tm + r * (ch + cap_h + gap) + ch / 2 - 8
            for j, line in enumerate(t.split("\n")):
                d.text((8, y + j * 17), line, font=f_row, fill=INK)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    canvas.save(out)
    print(f"wrote {out} {canvas.size}")
    return out
