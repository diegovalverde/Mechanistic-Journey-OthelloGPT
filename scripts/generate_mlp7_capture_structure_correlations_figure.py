#!/usr/bin/env python3
"""Regenerate the MLP7 capture-structure correlation heatmap from measured JSON."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
DATA_PATH = FIG / "mlp7_capture_structure_correlations.json"
OUT_PATH = FIG / "mlp7_capture_structure_correlations.svg"

STYLE = """<style>
svg{color-scheme:light dark;--text:#172033;--muted:#5c667a;--grid:#d8dde8}
@media(prefers-color-scheme:dark){svg{--text:#eef2ff;--muted:#b7bfd1;--grid:#384153}}
.title{font:700 23px system-ui,-apple-system,Segoe UI,sans-serif;fill:var(--text)}.label{font:600 12px system-ui,-apple-system,Segoe UI,sans-serif;fill:var(--text)}.small{font:11px system-ui,-apple-system,Segoe UI,sans-serif;fill:var(--muted)}
.cell{stroke:var(--grid);stroke-width:1}.axis{stroke:var(--grid);stroke-width:1.1}
</style>"""


def color(value: float) -> str:
    lim = 0.08
    t = max(-1.0, min(1.0, value / lim))
    if t >= 0:
        r = int(245 - 190 * t)
        g = int(247 - 70 * t)
        b = int(249 - 120 * t)
    else:
        t = -t
        r = int(245 - 30 * t)
        g = int(247 - 105 * t)
        b = int(249 - 150 * t)
    return f"rgb({r},{g},{b})"


def main() -> None:
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    rows = payload["correlations"]
    features = payload["features"]
    labels = {
        "longest_capture_line": "longest line",
        "num_capture_directions": "directions",
        "total_flipped": "flipped",
    }
    x0, y0, cw, ch = 180, 92, 150, 24
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 670" role="img" aria-labelledby="title desc">',
        '<title id="title">MLP7 capture-structure correlations</title>',
        '<desc id="desc">Zero-centered heatmap of small Pearson correlations.</desc>',
        STYLE,
        '<text class="title" x="42" y="44">Line structure correlations are weak</text>',
        f'<text class="small" x="42" y="68">Pearson r over {payload["examples"]} valid-condition examples; color scale is centered at zero.</text>',
    ]
    for j, feature in enumerate(features):
        parts.append(f'<text class="label" x="{x0+j*cw+cw/2}" y="84" text-anchor="middle">{labels[feature]}</text>')
    for i, row in enumerate(rows):
        y = y0 + i * ch
        parts.append(f'<text class="small" x="{x0-12}" y="{y+16}" text-anchor="end">{row["neuron"]}</text>')
        for j, feature in enumerate(features):
            x = x0 + j * cw
            value = row[feature]
            parts.append(f'<rect class="cell" x="{x}" y="{y}" width="{cw}" height="{ch}" fill="{color(value)}"/>')
            parts.append(f'<text class="small" x="{x+cw/2}" y="{y+16}" text-anchor="middle">{value:+.3f}</text>')
    parts.extend([
        '<text class="small" x="180" y="615">blue-ish = negative; green-ish = positive; all displayed magnitudes are small.</text>',
        f'<metadata>{json.dumps({"source": payload["notebook_section"], "notebook": payload["notebook"], "branch": payload["branch"]})}</metadata>',
        '</svg>\n',
    ])
    OUT_PATH.write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
