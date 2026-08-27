#!/usr/bin/env python3
"""Regenerate the MLP7 end-to-end neuron test figure from measured JSON."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
DATA_PATH = FIG / "mlp7_end_to_end_neuron_test.json"
OUT_PATH = FIG / "mlp7_end_to_end_neuron_test.svg"

STYLE = """<style>
svg{color-scheme:light dark;--text:#172033;--muted:#5c667a;--grid:#d8dde8;--sel:#cc4b37;--ctrl:#0f8f7f;--rnd:#246bfe}
@media(prefers-color-scheme:dark){svg{--text:#eef2ff;--muted:#b7bfd1;--grid:#384153;--sel:#ff8f7d;--ctrl:#5fd4c7;--rnd:#86a9ff}}
.title{font:700 23px system-ui,-apple-system,Segoe UI,sans-serif;fill:var(--text)}.label{font:600 12px system-ui,-apple-system,Segoe UI,sans-serif;fill:var(--text)}.small{font:11px system-ui,-apple-system,Segoe UI,sans-serif;fill:var(--muted)}
.axis{stroke:var(--grid);stroke-width:1.1}.zero{stroke:var(--grid);stroke-width:1.6;stroke-dasharray:5 5}.sel{fill:var(--sel)}.ctrl{fill:var(--ctrl)}.rnd{fill:var(--rnd)}
</style>"""


def main() -> None:
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    rows = payload["groups"]
    xmin, xmax = -0.023, 0.004
    left, right = 270, 730
    top = 120

    def x(value: float) -> float:
        return left + (value - xmin) / (xmax - xmin) * (right - left)

    cls_for = {
        "combined-evidence top neurons": "sel",
        "low-attribution controls": "ctrl",
        "random controls": "rnd",
    }
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 820 330" role="img" aria-labelledby="title desc">',
        '<title id="title">MLP7 end-to-end neuron test</title>',
        '<desc id="desc">Combined-evidence neurons have a larger but small mean effect than controls.</desc>',
        STYLE,
        '<text class="title" x="42" y="44">Selected neurons have a small selective causal effect</text>',
        '<text class="small" x="42" y="68">Mean-replacement ablations on preferred relational-condition examples; sign convention is L_clean - L_ablate.</text>',
        f'<line class="axis" x1="{left}" y1="246" x2="{right}" y2="246"/>',
        f'<line class="zero" x1="{x(0):.1f}" y1="100" x2="{x(0):.1f}" y2="258"/>',
    ]
    for tick in [-0.02, -0.01, 0.0]:
        parts.append(f'<text class="small" x="{x(tick):.1f}" y="272" text-anchor="middle">{tick:+.2f}</text>')
    for i, row in enumerate(rows):
        y = top + i * 44
        value = row["mean_legality_degradation"]
        cls = cls_for[row["group"]]
        parts.append(f'<text class="small" x="256" y="{y+17}" text-anchor="end">{row["group"]}</text>')
        parts.append(f'<rect class="{cls}" x="{min(x(0), x(value)):.1f}" y="{y}" width="{abs(x(value)-x(0)):.1f}" height="24"/>')
        parts.append(f'<text class="small" x="{x(value)-6:.1f}" y="{y+17}" text-anchor="end">{value:+.6f}</text>')
    parts.extend([
        '<text class="small" x="500" y="310" text-anchor="middle">mean legality degradation</text>',
        f'<metadata>{json.dumps({"source": payload["notebook_section"], "notebook": payload["notebook"], "branch": payload["branch"]})}</metadata>',
        '</svg>\n',
    ])
    OUT_PATH.write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
