#!/usr/bin/env python3
"""Regenerate the MLP7 component-attribution figure from measured JSON."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
DATA_PATH = FIG / "mlp7_component_attribution.json"
OUT_PATH = FIG / "mlp7_component_attribution.svg"

STYLE = """<style>
svg{color-scheme:light dark;--text:#172033;--muted:#5c667a;--grid:#d8dde8;--blue:#246bfe;--teal:#0f8f7f;--red:#cc4b37}
@media(prefers-color-scheme:dark){svg{--text:#eef2ff;--muted:#b7bfd1;--grid:#384153;--blue:#86a9ff;--teal:#5fd4c7;--red:#ff8f7d}}
.title{font:700 24px system-ui,-apple-system,Segoe UI,sans-serif;fill:var(--text)}.label{font:600 13px system-ui,-apple-system,Segoe UI,sans-serif;fill:var(--text)}.small{font:12px system-ui,-apple-system,Segoe UI,sans-serif;fill:var(--muted)}
.axis{stroke:var(--grid);stroke-width:1.1}.bar{fill:var(--blue)}.signed{fill:var(--teal)}.neg{fill:var(--red)}
</style>"""


def main() -> None:
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    rows = sorted(payload["aggregate"], key=lambda r: r["rank"])
    max_abs = max(r["mean_absolute_attribution"] for r in rows)
    scale = 430 / max_abs
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 820 520" role="img" aria-labelledby="title desc">',
        '<title id="title">MLP7 component attribution</title>',
        '<desc id="desc">Mean absolute legality attribution for each layer-7 attention head and MLP7.</desc>',
        STYLE,
        '<text class="title" x="42" y="48">Component attribution inside layer 7</text>',
        '<text class="small" x="42" y="72">A_c = gradient dot component output. Attribution is a local alignment measure, not an ablation.</text>',
        '<line class="axis" x1="200" y1="98" x2="200" y2="430"/>',
    ]
    for i, row in enumerate(rows):
        y = 112 + i * 34
        w = row["mean_absolute_attribution"] * scale
        parts.append(f'<text class="label" x="180" y="{y+17}" text-anchor="end">{row["component"]}</text>')
        parts.append(f'<rect class="bar" x="200" y="{y}" width="{w:.1f}" height="22" rx="4"/>')
        parts.append(f'<text class="small" x="{208+w:.1f}" y="{y+16}">{row["mean_absolute_attribution"]:.6f}</text>')
    parts.append('<text class="label" x="200" y="470">Mean signed attribution</text>')
    sx0 = 390
    signed_scale = 260 / max(abs(r["mean_legality_attribution"]) for r in rows)
    parts.append(f'<line class="axis" x1="{sx0}" y1="476" x2="{sx0}" y2="504"/>')
    for row in rows[:5]:
        x = sx0 + row["mean_legality_attribution"] * signed_scale
        cls = "signed" if row["mean_legality_attribution"] >= 0 else "neg"
        parts.append(f'<line class="{cls}" x1="{sx0}" y1="{478+rows.index(row)*5}" x2="{x:.1f}" y2="{478+rows.index(row)*5}" stroke-width="4"/>')
    parts.extend([
        '<text class="small" x="42" y="492">Main bars show mean absolute attribution across 30 positions. Signed means are kept separate below.</text>',
        f'<metadata>{json.dumps({"source": payload["notebook_section"], "notebook": payload["notebook"], "branch": payload["branch"]})}</metadata>',
        '</svg>\n',
    ])
    OUT_PATH.write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
