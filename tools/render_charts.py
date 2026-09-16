#!/usr/bin/env python3
"""Render the README charts as light/dark SVG pairs.

The SVG is emitted directly rather than through a plotting library: these are
simple horizontal bars, and writing the geometry by hand is what makes the mark
spec exact -- a 4px radius on the data end only, square at the baseline, and a
fixed 2px of surface between bars. It also keeps the output small and diffable.

Every chart is a single series of named categories, so identity comes from the
axis label and colour carries emphasis only: the shipped configuration in the
series hue, the thing it is measured against in the second hue, the rest in
recessive ink. Every bar is direct-labelled, so nothing depends on reading a
length against a grid.

Run from the repository root:

    python3 tools/render_charts.py
"""

from __future__ import annotations

import html
from pathlib import Path

# Colour tokens. The dark column is stepped for the dark surface rather than
# flipped from the light one; both were checked for contrast against their own
# surface and for separation under simulated colour-vision deficiency.
THEMES = {
    "light": dict(surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e",
                  baseline="#c3c2b7", series="#2a78d6", other="#eb6834",
                  recessive="#b9b8b1"),
    "dark": dict(surface="#1a1a19", primary="#ffffff", secondary="#c3c2b7",
                 baseline="#383835", series="#3987e5", other="#d95926",
                 recessive="#55544f"),
}

FONT = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', "
        "Helvetica, Arial, sans-serif")

WIDTH = 760
GUTTER = 210       # room for the category labels
VALUE_PAD = 10     # gap between a bar's end and its value
VALUE_ROOM = 86    # room for the value text
BAR = 20           # bar thickness, under the 24px cap
ROW = 30           # bar plus the surface gap between rows
RADIUS = 4


def bar_path(x: float, y: float, length: float, height: float, radius: float) -> str:
    """A bar rounded on the data end and square at the baseline."""
    radius = min(radius, max(length, 0.0))
    if radius <= 0:
        return f"M{x},{y}h{length:.2f}v{height}h{-length:.2f}z"
    straight = length - radius
    return (f"M{x},{y}"
            f"h{straight:.2f}"
            f"a{radius},{radius} 0 0 1 {radius},{radius}"
            f"v{height - 2 * radius}"
            f"a{radius},{radius} 0 0 1 {-radius},{radius}"
            f"h{-straight:.2f}"
            f"z")


def chart(title: str, subtitle: str, rows, theme: dict, fmt) -> str:
    """rows: [(label, value, role)] where role indexes a colour token."""
    top = 64
    height = top + ROW * len(rows) + 14
    scale = (WIDTH - GUTTER - VALUE_PAD - VALUE_ROOM) / max(v for _, v, _ in rows)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
        f'viewBox="0 0 {WIDTH} {height}" role="img" '
        f'aria-label="{html.escape(title)}. {html.escape(subtitle)}">',
        f'<rect width="{WIDTH}" height="{height}" fill="{theme["surface"]}"/>',
        f'<g font-family="{FONT}">',
        f'<text x="16" y="28" font-size="15" font-weight="600" '
        f'fill="{theme["primary"]}">{html.escape(title)}</text>',
        f'<text x="16" y="48" font-size="11.5" fill="{theme["secondary"]}">'
        f'{html.escape(subtitle)}</text>',
        f'<line x1="{GUTTER}" y1="{top - 8}" x2="{GUTTER}" y2="{height - 12}" '
        f'stroke="{theme["baseline"]}" stroke-width="1"/>',
    ]

    for index, (label, value, role) in enumerate(rows):
        y = top + index * ROW
        length = value * scale
        out.append(f'<text x="{GUTTER - 10}" y="{y + BAR - 6}" font-size="11.5" '
                   f'text-anchor="end" fill="{theme["secondary"]}">'
                   f'{html.escape(label)}</text>')
        out.append(f'<path d="{bar_path(GUTTER, y, length, BAR, RADIUS)}" '
                   f'fill="{theme[role]}"/>')
        out.append(f'<text x="{GUTTER + length + VALUE_PAD:.2f}" y="{y + BAR - 6}" '
                   f'font-size="11.5" fill="{theme["primary"]}">'
                   f'{html.escape(fmt(value))}</text>')

    out.append("</g></svg>")
    return "\n".join(out) + "\n"


# Total GEMM time per forward pass, summed over the eight shapes the model
# issues and weighted by how often each runs.
TILE_SWEEP = [
    ("64x32  (shipped)", 1738, "series"),
    ("64x32, K tile 8", 1923, "recessive"),
    ("96x32", 2192, "recessive"),
    ("128x64", 2387, "recessive"),
    ("128x32", 2410, "recessive"),
    ("64x64  (previous)", 2484, "other"),
    ("32x32", 2530, "recessive"),
]

LATENCY = [
    ("PyTorch, cuBLAS + cuDNN", 1.691, "other"),
    ("Cu-Vit 0.3.0", 1.887, "series"),
    ("Cu-Vit 0.2.0", 2.363, "recessive"),
]

# GPU time per stage, from nsys on the shipped build.
BREAKDOWN = [
    ("proj + fc2 + patch embed", 1084.7, "series"),
    ("mlp fc1", 426.1, "series"),
    ("qkv projection", 301.3, "series"),
    ("attention × V", 181.4, "series"),
    ("Q × Kᵀ", 157.2, "series"),
    ("softmax", 72.3, "recessive"),
    ("layer norm", 71.6, "recessive"),
    ("classifier head", 15.2, "series"),
    ("patch gather", 4.9, "recessive"),
    ("positional add", 1.8, "recessive"),
]

CHARTS = [
    ("tile-sweep", "GEMM tile sweep",
     "Total GEMM time per forward pass, lower is better. Eight shapes, weighted "
     "by how often each runs.", TILE_SWEEP, lambda v: f"{v:,.0f} µs"),
    ("latency", "Forward pass, batch size one",
     "ViT-Tiny/16 at 224×224, FP32, RTX 4070 Laptop GPU. Median of seven runs "
     "of 400 iterations.", LATENCY, lambda v: f"{v:.3f} ms"),
    ("breakdown", "Where a forward pass goes",
     "GPU time per stage, from nsys. The six GEMM rows are 93% of it, across 113 "
     "kernel launches.", BREAKDOWN, lambda v: f"{v:,.1f} µs"),
]



# ---------------------------------------------------------------------------
# The packed QKV buffer, and the slices attention reads out of it in place.

def qkv_layout(theme: dict) -> str:
    """One row of the [tokens, 3, heads, head_dim] projection, and how it is read."""
    width, height = 760, 258
    left, top = 24, 92
    cell, gap = 76, 2
    group = 3 * cell + 2 * gap

    groups = [("Q", theme["series"], 0), ("K", theme["other"], 192),
              ("V", theme["recessive"], 384)]
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="One row of the packed '
        f'Q K V projection, 576 values wide, split into three groups of three heads. '
        f'Attention reads each head as a column slice in place, with a leading '
        f'dimension of 576 and an offset of the head index times 64.">',
        f'<rect width="{width}" height="{height}" fill="{theme["surface"]}"/>',
        f'<g font-family="{FONT}">',
        f'<text x="{left}" y="28" font-size="15" font-weight="600" '
        f'fill="{theme["primary"]}">One row of the packed projection</text>',
        f'<text x="{left}" y="49" font-size="11.5" fill="{theme["secondary"]}">'
        f'A single GEMM writes 576 values per token — three parts of three heads.'
        f'</text>',
        f'<text x="{left}" y="66" font-size="11.5" fill="{theme["secondary"]}">'
        f'Nothing is gathered afterwards: every head is a column slice read in place.'
        f'</text>',
    ]

    for index, (name, colour, offset) in enumerate(groups):
        base = left + index * (group + 18)
        out.append(f'<text x="{base}" y="{top - 8}" font-size="12" font-weight="600" '
                   f'fill="{theme["primary"]}">{name}'
                   f'<tspan font-weight="400" font-size="10.5" '
                   f'fill="{theme["secondary"]}">   offset {offset}</tspan></text>')
        for head in range(3):
            x = base + head * (cell + gap)
            out.append(f'<rect x="{x}" y="{top}" width="{cell}" height="34" rx="3" '
                       f'fill="{colour}"/>')
            out.append(f'<text x="{x + cell / 2}" y="{top + 22}" font-size="10.5" '
                       f'text-anchor="middle" fill="{theme["surface"]}">head {head}</text>')

    # Mark the one slice the callout describes.
    marked = left + cell + gap
    out.append(f'<path d="M{marked + cell / 2 - 5},{top + 40} l5,6 l5,-6 z" '
               f'fill="{theme["primary"]}"/>')

    body = top + 74
    out.append(f'<text x="{left}" y="{body}" font-size="12" font-weight="600" '
               f'fill="{theme["primary"]}">Reading Q for head 1</text>')
    for index, line in enumerate([
            "base = qkv + 0·192 + 1·64",
            "lda = 576   \u2014 skip to the next token",
            "extent = 197 × 64",
    ]):
        out.append(f'<text x="{left}" y="{body + 22 + index * 18}" font-size="11.5" '
                   f'font-family="ui-monospace, SFMono-Regular, Menlo, monospace" '
                   f'fill="{theme["secondary"]}">{line}</text>')

    out.append(f'<text x="{left + 330}" y="{body}" font-size="12" font-weight="600" '
               f'fill="{theme["primary"]}">What the two GEMMs do</text>')
    for index, (bold, rest) in enumerate([
            ("Q × Kᵀ", "reads two column slices, one GEMM per head"),
            ("× V", "writes into [tokens, heads · head_dim] — the"),
            ("", "layout the output projection already expects"),
    ]):
        y = body + 22 + index * 18
        out.append(f'<text x="{left + 330}" y="{y}" font-size="11.5" '
                   f'fill="{theme["primary"]}" font-weight="600">{bold}</text>')
        out.append(f'<text x="{left + 330 + (48 if bold else 0)}" y="{y}" '
                   f'font-size="11.5" fill="{theme["secondary"]}">{rest}</text>')

    out.append("</g></svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    destination = Path("docs")
    destination.mkdir(exist_ok=True)
    for name, title, subtitle, rows, fmt in CHARTS:
        for mode, theme in THEMES.items():
            path = destination / f"{name}-{mode}.svg"
            path.write_text(chart(title, subtitle, rows, theme, fmt), encoding="utf-8")
            print(f"wrote {path}")
    for mode, theme in THEMES.items():
        path = destination / f"qkv-layout-{mode}.svg"
        path.write_text(qkv_layout(theme), encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
