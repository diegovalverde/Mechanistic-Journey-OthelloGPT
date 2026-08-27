#!/usr/bin/env python3
"""Regenerate the Chapter 7 layer-sweep figure from measured JSON.

Input:
    docs/figures/chapter07_layer_sweep.json

Output:
    docs/figures/chapter07_layer_sweep.svg

The data were copied from executed outputs in:
    diegovalverde/TransformerLens
    branch: othello-jspace-analysis
    notebook: demos/Othello_GPT_Jacobian_Lens.ipynb
    section: 17. Which layer computes legality?
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
DATA_PATH = FIG / "chapter07_layer_sweep.json"
OUT_PATH = FIG / "chapter07_layer_sweep.svg"


STYLE = """<style>
svg { color-scheme: light dark; --text:#172033; --muted:#5c667a; --grid:#d8dde8; --fill:#f7f9fc; --blue:#246bfe; --teal:#0f8f7f; --gold:#b98113; --red:#cc4b37; }
@media (prefers-color-scheme: dark) { svg { --text:#eef2ff; --muted:#b7bfd1; --grid:#384153; --fill:#1f2633; --blue:#86a9ff; --teal:#5fd4c7; --gold:#ffd27a; --red:#ff8f7d; } }
.title{font:700 24px system-ui,-apple-system,Segoe UI,sans-serif;fill:var(--text)}
.label{font:600 14px system-ui,-apple-system,Segoe UI,sans-serif;fill:var(--text)}
.small{font:12px system-ui,-apple-system,Segoe UI,sans-serif;fill:var(--muted)}
.axis{stroke:var(--grid);stroke-width:1.2}.bar{fill:var(--blue)}.acc{fill:var(--teal)}.zero{stroke:var(--red);stroke-width:1.5;stroke-dasharray:5 5}.box{fill:var(--fill);stroke:var(--grid);stroke-width:1.2;rx:8}
</style>"""


def sx(i: int) -> float:
    return 120 + i * 125


def main() -> None:
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    rows = payload["data"]
    ratio_max = 2.5
    acc_max = 1.0
    top = 100
    bottom = 350
    height = bottom - top

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 470" role="img" aria-labelledby="title desc">',
        '<title id="title">Layer sweep: capture-line legality enrichment</title>',
        '<desc id="desc">Measured capture-vs-unrelated ratio and probe validation accuracy for layers 2, 4, 6, and 7.</desc>',
        STYLE,
        '<text class="title" x="42" y="48">Layer sweep: legality enrichment appears late</text>',
        '<text class="small" x="42" y="72">Measured notebook output, section 17. Ratio axis starts at 0; neutral ratio = 1.0.</text>',
        f'<line class="axis" x1="88" y1="{bottom}" x2="660" y2="{bottom}"/>',
        f'<line class="axis" x1="88" y1="{top}" x2="88" y2="{bottom}"/>',
    ]
    for tick in [0, 0.5, 1.0, 1.5, 2.0, 2.5]:
        y = bottom - tick / ratio_max * height
        cls = "zero" if tick == 1.0 else "axis"
        parts.append(f'<line class="{cls}" x1="88" y1="{y:.1f}" x2="660" y2="{y:.1f}"/>')
        parts.append(f'<text class="small" x="74" y="{y+4:.1f}" text-anchor="end">{tick:.1f}</text>')
    parts.append('<text class="small" x="30" y="226" transform="rotate(-90 30 226)" text-anchor="middle">capture / unrelated ratio</text>')

    for i, row in enumerate(rows):
        x = sx(i)
        bar_h = row["capture_vs_unrelated_ratio"] / ratio_max * height
        y = bottom - bar_h
        parts.append(f'<rect class="bar" x="{x-26}" y="{y:.1f}" width="52" height="{bar_h:.1f}" rx="4"/>')
        parts.append(f'<text class="label" x="{x}" y="{y-8:.1f}" text-anchor="middle">{row["capture_vs_unrelated_ratio"]:.3f}</text>')
        acc_h = row["probe_validation_accuracy"] / acc_max * 90
        parts.append(f'<rect class="acc" x="{x-20}" y="{405-acc_h:.1f}" width="40" height="{acc_h:.1f}" rx="4"/>')
        parts.append(f'<text class="small" x="{x}" y="430" text-anchor="middle">L{row["layer"]}</text>')
        parts.append(f'<text class="small" x="{x}" y="390" text-anchor="middle">acc {row["probe_validation_accuracy"]:.3f}</text>')

    parts.extend([
        '<rect class="box" x="520" y="112" width="184" height="78"/>',
        '<rect class="bar" x="538" y="132" width="18" height="18" rx="3"/><text class="small" x="566" y="146">enrichment ratio</text>',
        '<rect class="acc" x="538" y="160" width="18" height="18" rx="3"/><text class="small" x="566" y="174">probe accuracy</text>',
        '<text class="small" x="92" y="452">Probe accuracy is shown separately below each layer; it is not plotted on a second y-axis.</text>',
        f'<metadata>{json.dumps({"source": payload["notebook_section"], "notebook": payload["notebook"], "branch": payload["branch"]})}</metadata>',
        '</svg>\n',
    ])
    OUT_PATH.write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
