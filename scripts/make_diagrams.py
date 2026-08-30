"""The two schematic figures: the six failure modes, and the method.

These are drawn rather than rendered. A section view states what "overhang"
means in one glance; a photorealistic render of an overhanging sofa is a
picture of a sofa. They are emitted as SVG so they stay crisp at print size and
inherit the page's theme through `currentColor`.
"""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")

INK = "#1e2227"
MUTED = "#6b7280"
SLAB = "#d9d2c4"
SLAB_E = "#b9ae99"
TIER = "#a8c3d4"
TIER_E = "#6f93a8"
OBJ = "#8a6f4e"
OBJ_E = "#5d4a33"
BAD = "#c0392b"
GOOD = "#2e7d5b"

HEAD = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        'width="{w}" height="{h}" font-family="Helvetica,Arial,sans-serif">'
        '<style>'
        '.t{{font-size:12.5px;fill:%s}}'
        '.s{{font-size:11px;fill:%s}}'
        '.k{{font-size:12px;font-weight:700;fill:%s}}'
        '.b{{font-size:11.5px;font-weight:700;fill:%s}}'
        '.m{{font-size:10.5px;fill:%s;font-style:italic}}'
        '</style>' % (INK, MUTED, INK, BAD, MUTED))


def rect(x, y, w, h, fill, stroke, sw=1.2, rx=0):
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def text(x, y, s, cls="t", anchor="start"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" '
            f'text-anchor="{anchor}">{s}</text>')


# ============================================================== violations
def violations_svg(out):
    """Six section views. Ground is drawn thick, the offending object in red."""
    CW, CH = 250, 158          # cell
    cols, rows = 3, 2
    pad_x, pad_y, top = 14, 46, 34
    W = cols * CW + pad_x
    H = top + rows * (CH + pad_y)
    p = [HEAD.format(w=W, h=H)]
    p.append(f'<rect width="{W}" height="{H}" fill="none"/>')
    p.append(text(4, 20, "Failure modes that only exist once the floor stops "
                         "being one plane", "k"))

    def cell(i, key, name, rule, draw):
        cx = (i % cols) * CW + 10
        cy = top + (i // cols) * (CH + pad_y)
        p.append(text(cx, cy + 14, f"{key}  {name}", "b"))
        p.append(text(cx, cy + 29, rule, "s"))
        p.append(f'<g transform="translate({cx},{cy + 36})">')
        draw()
        p.append('</g>')

    G = 88          # ground line inside the cell group
    def slab(x, w, top_y, label=None, lx=None):
        p.append(rect(x, top_y, w, G - top_y + 16, SLAB, SLAB_E))
        if label:
            p.append(text(lx if lx is not None else x + w / 2, top_y - 5,
                          label, "m", "middle"))

    def raised(x, w, top_y, label=None):
        p.append(rect(x, top_y, w, G - top_y + 16, TIER, TIER_E))
        if label:
            p.append(text(x + w / 2, top_y - 5, label, "m", "middle"))

    # -- F1a overhang ----------------------------------------------------
    def f1a():
        slab(0, 118, G)                      # low ground
        raised(118, 110, G - 26, "+0.4 m")
        p.append(rect(150, G - 26 - 34, 96, 34, OBJ, BAD, 2.0))
        p.append(f'<path d="M228,{G-26} L228,{G}" stroke="{BAD}" '
                 f'stroke-width="1.6" stroke-dasharray="3 3"/>')
        p.append(f'<path d="M228,{G-26-34} L246,{G-26-34} L246,{G-26} '
                 f'L228,{G-26} Z" fill="{BAD}" fill-opacity="0.18" '
                 f'stroke="none"/>')
        p.append(text(114, G + 30, "hangs over the drop", "m", "middle"))

    # -- F1b embedded ----------------------------------------------------
    def f1b():
        slab(0, 130, G, "datum")
        raised(130, 98, G - 30, "+0.5 m")
        p.append(rect(86, G - 34, 92, 34, OBJ, BAD, 2.0))
        p.append(f'<path d="M130,{G-34} L178,{G-34} L178,{G-30} L130,{G-30} Z" '
                 f'fill="{BAD}" fill-opacity="0.25" stroke="none"/>')
        p.append(text(114, G + 30, "drives into the platform beside it",
                      "m", "middle"))

    # -- F2 straddling ---------------------------------------------------
    def f2():
        slab(0, 108, G)
        raised(108, 106, G - 28)
        p.append(text(228, G - 8, "+0.45 m", "m", "end"))
        p.append(rect(58, G - 28 - 30, 108, 30, OBJ, BAD, 2.0))
        p.append(f'<path d="M58,{G-28} L108,{G-28} L108,{G} L58,{G} Z" '
                 f'fill="{BAD}" fill-opacity="0.18" stroke="none"/>')
        p.append(f'<path d="M108,{G-58} L108,{G+8}" stroke="{BAD}" '
                 f'stroke-width="1.6" stroke-dasharray="3 3"/>')
        p.append(text(114, G + 30, "rests on neither", "m", "middle"))

    # -- F3 step blocked -------------------------------------------------
    def f3():
        slab(0, 96, G)
        p.append(rect(96, G - 12, 40, 28, SLAB, SLAB_E))
        p.append(rect(136, G - 24, 40, 40, SLAB, SLAB_E))
        raised(176, 52, G - 36, "+0.54 m")
        p.append(text(136, G - 56, "2 treads", "m", "middle"))
        p.append(rect(104, G - 12 - 26, 62, 26, OBJ, BAD, 2.0))
        p.append(text(114, G + 30, "furniture on the treads", "m", "middle"))

    # -- F4 headroom -----------------------------------------------------
    def f4():
        slab(0, 228, G)
        # the mezzanine is a plate carried overhead, and the violation is the
        # space it leaves underneath -- drawing it as a solid block to the
        # ground removes the very gap being measured
        p.append(rect(96, G - 46, 132, 11, TIER, TIER_E))
        p.append(rect(217, G - 46, 11, 46, TIER, TIER_E))
        p.append(text(150, G - 52, "mezzanine  +1.55 m", "m", "middle"))
        p.append(f'<path d="M140,{G-35} L140,{G}" stroke="{BAD}" '
                 f'stroke-width="1.6"/>')
        p.append(f'<path d="M134,{G-35} L146,{G-35} M134,{G} L146,{G}" '
                 f'stroke="{BAD}" stroke-width="1.6"/>')
        p.append(text(150, G - 14, "1.55 m &lt; 1.90", "m"))
        p.append(text(114, G + 30, "cannot stand under it", "m", "middle"))

    # -- F5 datum --------------------------------------------------------
    def f5():
        slab(0, 116, G, "tier 0")
        raised(116, 112, G - 28, "tier 1")
        p.append(rect(140, G - 28 - 32, 74, 32, OBJ, BAD, 2.0))
        p.append(text(177, G - 68, 'declared "tier 0"', "m", "middle"))
        p.append(f'<path d="M177,{G-64} L177,{G-32}" stroke="{BAD}" '
                 f'stroke-width="1.4" marker-end="url(#a)"/>')
        p.append(text(114, G + 30, "stands on a tier it did not declare",
                      "m", "middle"))

    p.append(f'<defs><marker id="a" viewBox="0 0 8 8" refX="6" refY="4" '
             f'markerWidth="5" markerHeight="5" orient="auto">'
             f'<path d="M0,1 L6,4 L0,7 z" fill="{BAD}"/></marker></defs>')

    for i, (k, n, r, d) in enumerate([
            ("F1a", "overhang", "footprint leaves its tier, over lower ground", f1a),
            ("F1b", "embedded", "footprint leaves its tier, into higher ground", f1b),
            ("F2", "straddling", "&gt;10 % of the footprint on two tiers", f2),
            ("F3", "step blocked", "a transition tread is occupied", f3),
            ("F4", "headroom", "clearance under a tier above &lt; 1.90 m", f4),
            ("F5", "datum", "declared support &#8800; tier underneath", f5)]):
        cell(i, k, n, r, d)

    p.append('</svg>')
    os.makedirs(ASSETS, exist_ok=True)
    open(out, "w").write("".join(p))
    print(f"wrote {out}")


# ================================================================ pipeline
def pipeline_svg(out):
    W, H = 1060, 322
    p = [HEAD.format(w=W, h=H)]
    BOXF, BOXE = "#f4f2ee", "#c8c3b8"
    ACC = "#3f6d8c"

    def sub(t):
        return f'<tspan baseline-shift="-24%" font-size="8.5">{t}</tspan>'

    def box(x, y, w, h, title, lines, edge=BOXE):
        p.append(rect(x, y, w, h, BOXF, edge, 1.4, 7))
        p.append(text(x + 11, y + 20, title, "b" if edge == ACC else "k"))
        for i, ln in enumerate(lines):
            p.append(text(x + 11, y + 39 + i * 15, ln, "s"))

    def arrow(x0, x1, y, label):
        p.append(f'<path d="M{x0},{y} L{x1},{y}" stroke="{ACC}" '
                 f'stroke-width="1.6" marker-end="url(#p)"/>')
        p.append(text((x0 + x1) / 2, y - 8, label, "m", "middle"))

    p.append(f'<defs><marker id="p" viewBox="0 0 8 8" refX="7" refY="4" '
             f'markerWidth="6" markerHeight="6" orient="auto">'
             f'<path d="M0,1 L7,4 L0,7 z" fill="{ACC}"/></marker></defs>')
    p.append(text(6, 20, "Elevate3D: two stages over one representation", "k"))

    y0, bh = 40, 112
    box(6, y0, 190, bh, "input",
        ["room polygon P", "room type", "ceiling height"])
    box(258, y0, 218, bh, "M1  FieldNet  (3.5 M)",
        ["program, 1 of 4", "12 support offsets h(u)",
         "rise \u0394h", "\u2192 tiers, transitions"], edge=ACC)
    box(538, y0, 238, bh, "M2  Elevate3D  (27.8 M)",
        ["autoregressive over objects", "tier-biased attention",
         "mixture heads: x, y, yaw, size",
         "support pointer, not z"], edge=ACC)
    box(838, y0, 216, bh, "resolve",
        [f"z{sub('i')} = top(\u03c0(i)) + \u0394z{sub('i')}",
         "one pass over the support tree",
         "L_contact \u2261 0, by construction"])

    ym = y0 + bh / 2
    arrow(196, 256, ym, "P, type")
    arrow(476, 536, ym, "field S")
    arrow(776, 836, ym, f"\u03c0(i), \u0394z{sub('i')}")
    p.append(text(6, y0 + bh + 18,
                  f"S = (P, {{(R{sub('k')}, h{sub('k')})}}{sub('k=1..K')}, T)"
                  "   \u2014   a piecewise-constant floor: K convex regions, "
                  "their heights, and the transitions between them. K = 1 is "
                  "the flat floor every current method assumes.", "s"))

    # ---- the one design decision the paper turns on --------------------
    yb = y0 + bh + 44
    p.append(f'<path d="M6,{yb} L1054,{yb}" stroke="{BOXE}" '
             f'stroke-width="1"/>')
    yb += 20
    p.append(text(6, yb, "why a support pointer and not a height", "k"))
    for i, ln in enumerate([
            "Regressing z makes contact a soft penalty. The loss is smallest",
            "near the mean of the tier heights, so an object that could go on",
            "either of two tiers lands between them, touching neither.",
            "Pointing at a support makes contact exact and unlearnable-away."]):
        p.append(text(6, yb + 20 + i * 15, ln, "s"))

    gx, gy = 470, yb + 34
    p.append(rect(gx, gy, 150, 20, SLAB, SLAB_E))
    p.append(text(gx + 75, gy + 14, "datum   h = 0", "m", "middle"))
    p.append(rect(gx + 150, gy - 18, 130, 38, TIER, TIER_E))
    p.append(text(gx + 215, gy + 4, "tier 1   h = +0.42", "m", "middle"))
    p.append(rect(gx + 178, gy - 50, 74, 32, OBJ, OBJ_E, 1.4))
    p.append(text(gx + 215, gy - 56, "\u03c0(sofa) = tier 1", "m", "middle"))
    p.append(text(gx + 140, gy + 40,
                  "z is read off the support, never predicted", "m", "middle"))

    p.append(text(790, yb + 20,
                  "Three consequences the tables measure:", "b"))
    for i, ln in enumerate([
            "\u2022  contact violations cannot occur at all",
            "\u2022  the model spends no capacity on z",
            "\u2022  K = 1 degenerates to the flat setting,",
            "    so the formalism is a strict superset"]):
        p.append(text(790, yb + 38 + i * 15, ln, "s"))

    p.append('</svg>')
    open(out, "w").write("".join(p))
    print(f"wrote {out}")


if __name__ == "__main__":
    os.makedirs(ASSETS, exist_ok=True)
    violations_svg(os.path.join(ASSETS, "fig_violations.svg"))
    pipeline_svg(os.path.join(ASSETS, "fig_pipeline.svg"))
