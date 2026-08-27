#!/usr/bin/env python3
"""Regenerate the MLP7 valid-vs-invalid selectivity figure from measured JSON."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
DATA_PATH = FIG / "mlp7_valid_invalid_selectivity.json"
OUT_PATH = FIG / "mlp7_valid_invalid_selectivity.svg"

STYLE = """<style>
svg{color-scheme:light dark;--text:#172033;--muted:#5c667a;--grid:#d8dde8;--pos:#0f8f7f;--neg:#cc4b37;--fill:#f7f9fc}
@media(prefers-color-scheme:dark){svg{--text:#eef2ff;--muted:#b7bfd1;--grid:#384153;--pos:#5fd4c7;--neg:#ff8f7d;--fill:#1f2633}}
.title{font:700 23px system-ui,-apple-system,Segoe UI,sans-serif;fill:var(--text)}.label{font:600 12px system-ui,-apple-system,Segoe UI,sans-serif;fill:var(--text)}.small{font:11px system-ui,-apple-system,Segoe UI,sans-serif;fill:var(--muted)}
.axis{stroke:var(--grid);stroke-width:1.1}.zero{stroke:var(--grid);stroke-width:1.7;stroke-dasharray:5 5}.pos{fill:var(--pos)}.neg{fill:var(--neg)}
</style>"""


def main() -> None:
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    rows = payload["neurons"]
    ymin, ymax = -0.14, 0.12
    top, bottom = 95, 350
    left, right = 74, 760
    step = (right - left) / len(rows)

    def y(value: float) -> float:
        return bottom - (value - ymin) / (ymax - ymin) * (bottom - top)

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 840 440" role="img" aria-labelledby="title desc">',
        '<title id="title">MLP7 valid-vs-invalid selectivity</title>',
        '<desc id="desc">Small mixed unpaired selectivity values among fixed candidate neurons.</desc>',
        STYLE,
        '<text class="title" x="42" y="44">Valid-vs-invalid activation selectivity is small and mixed</text>',
        '<text class="small" x="42" y="68">Unpaired standardized difference: valid capture minus opponent-run without friendly terminator.</text>',
        f'<line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>',
        f'<line class="zero" x1="{left}" y1="{y(0):.1f}" x2="{right}" y2="{y(0):.1f}"/>',
    ]
    for tick in [-0.1, 0.0, 0.1]:
        parts.append(f'<text class="small" x="{left-10}" y="{y(tick)+4:.1f}" text-anchor="end">{tick:.1f}</text>')
    for i, row in enumerate(rows):
        x = left + i * step + step * 0.15
        width = step * 0.7
        value = row["valid_vs_opponent_no_terminator_selectivity"]
        y0 = y(0)
        yv = y(value)
        cls = "pos" if value >= 0 else "neg"
        parts.append(f'<rect class="{cls}" x="{x:.1f}" y="{min(y0, yv):.1f}" width="{width:.1f}" height="{abs(yv-y0):.1f}"/>')
        parts.append(f'<text class="small" x="{x+width/2:.1f}" y="374" text-anchor="middle" transform="rotate(60 {x+width/2:.1f} 374)">{row["neuron"]}</text>')
    parts.extend([
        '<text class="small" x="415" y="423" text-anchor="middle">fixed attribution-ranked candidate neuron ID</text>',
        '<text class="small" x="26" y="226" transform="rotate(-90 26 226)" text-anchor="middle">standardized selectivity</text>',
        '<rect class="pos" x="624" y="88" width="12" height="12"/><text class="small" x="642" y="98">higher on valid capture</text>',
        '<rect class="neg" x="624" y="110" width="12" height="12"/><text class="small" x="642" y="120">higher on invalid control</text>',
        f'<metadata>{json.dumps({"source": payload["notebook_section"], "notebook": payload["notebook"], "branch": payload["branch"]})}</metadata>',
        '</svg>\n',
    ])
    OUT_PATH.write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
