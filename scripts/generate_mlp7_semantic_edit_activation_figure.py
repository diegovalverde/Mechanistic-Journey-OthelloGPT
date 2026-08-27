#!/usr/bin/env python3
"""Regenerate the MLP7 semantic-edit activation figure from measured JSON."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
DATA_PATH = FIG / "mlp7_semantic_edit_activation.json"
OUT_PATH = FIG / "mlp7_semantic_edit_activation.svg"

STYLE = """<style>
svg{color-scheme:light dark;--text:#172033;--muted:#5c667a;--grid:#d8dde8;--pos:#0f8f7f;--neg:#cc4b37}
@media(prefers-color-scheme:dark){svg{--text:#eef2ff;--muted:#b7bfd1;--grid:#384153;--pos:#5fd4c7;--neg:#ff8f7d}}
.title{font:700 23px system-ui,-apple-system,Segoe UI,sans-serif;fill:var(--text)}.label{font:600 12px system-ui,-apple-system,Segoe UI,sans-serif;fill:var(--text)}.small{font:11px system-ui,-apple-system,Segoe UI,sans-serif;fill:var(--muted)}
.axis{stroke:var(--grid);stroke-width:1.1}.zero{stroke:var(--grid);stroke-width:1.6;stroke-dasharray:5 5}.pos{fill:var(--pos)}.neg{fill:var(--neg)}
</style>"""


def main() -> None:
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    rows = payload["rows"]
    xmin, xmax = -0.022, 0.022
    left, right = 250, 735
    top = 105

    def x(value: float) -> float:
        return left + (value - xmin) / (xmax - xmin) * (right - left)

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 820 340" role="img" aria-labelledby="title desc">',
        '<title id="title">MLP7 semantic edit activation deltas</title>',
        '<desc id="desc">Largest sparse activation deltas from example semantic residual edits.</desc>',
        STYLE,
        '<text class="title" x="42" y="44">Semantic edits can move some candidate activations</text>',
        '<text class="small" x="42" y="68">Displayed rows are the largest measured example-level effects; this is not a dataset distribution.</text>',
        f'<line class="axis" x1="{left}" y1="274" x2="{right}" y2="274"/>',
        f'<line class="zero" x1="{x(0):.1f}" y1="92" x2="{x(0):.1f}" y2="286"/>',
    ]
    for tick in [-0.02, -0.01, 0.0, 0.01, 0.02]:
        parts.append(f'<text class="small" x="{x(tick):.1f}" y="298" text-anchor="middle">{tick:+.2f}</text>')
    for i, row in enumerate(rows):
        y = top + i * 34
        value = row["mean_delta_activation"]
        cls = "pos" if value >= 0 else "neg"
        label = f'{row["semantic_edit"]} / {row["sign_label"]} / n{row["neuron"]}'
        parts.append(f'<text class="small" x="236" y="{y+16}" text-anchor="end">{label}</text>')
        parts.append(f'<rect class="{cls}" x="{min(x(0), x(value)):.1f}" y="{y}" width="{abs(x(value)-x(0)):.1f}" height="22"/>')
        parts.append(f'<text class="small" x="{x(value)+(6 if value>=0 else -6):.1f}" y="{y+15}" text-anchor="{"start" if value>=0 else "end"}">{value:+.6f}</text>')
    parts.extend([
        '<text class="small" x="493" y="326" text-anchor="middle">mean change in post-GELU activation</text>',
        f'<metadata>{json.dumps({"source": payload["notebook_section"], "notebook": payload["notebook"], "branch": payload["branch"]})}</metadata>',
        '</svg>\n',
    ])
    OUT_PATH.write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
