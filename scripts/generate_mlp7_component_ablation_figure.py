#!/usr/bin/env python3
"""Regenerate the MLP7 component-ablation figure from measured JSON."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
DATA_PATH = FIG / "mlp7_component_ablation.json"
OUT_PATH = FIG / "mlp7_component_ablation.svg"

STYLE = """<style>
svg{color-scheme:light dark;--text:#172033;--muted:#5c667a;--grid:#d8dde8;--blue:#246bfe;--teal:#0f8f7f;--red:#cc4b37}
@media(prefers-color-scheme:dark){svg{--text:#eef2ff;--muted:#b7bfd1;--grid:#384153;--blue:#86a9ff;--teal:#5fd4c7;--red:#ff8f7d}}
.title{font:700 24px system-ui,-apple-system,Segoe UI,sans-serif;fill:var(--text)}.label{font:600 13px system-ui,-apple-system,Segoe UI,sans-serif;fill:var(--text)}.small{font:12px system-ui,-apple-system,Segoe UI,sans-serif;fill:var(--muted)}
.axis{stroke:var(--grid);stroke-width:1.1}.bar{fill:var(--blue)}.neg{fill:var(--red)}.pos{fill:var(--teal)}
</style>"""


def main() -> None:
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    rows = sorted(payload["components"], key=lambda r: r["rank"])
    max_abs = max(r["mean_absolute_effect"] for r in rows)
    scale = 430 / max_abs
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 840 520" role="img" aria-labelledby="title desc">',
        '<title id="title">MLP7 component ablation</title>',
        '<desc id="desc">Mean absolute ablation effect for each layer-7 attention head and MLP7.</desc>',
        STYLE,
        '<text class="title" x="42" y="48">Component ablation inside layer 7</text>',
        '<text class="small" x="42" y="72">Delta L = L_ablate - L_clean. Negative signed values mean removal reduced the legality contrast.</text>',
        '<line class="axis" x1="200" y1="98" x2="200" y2="430"/>',
    ]
    for i, row in enumerate(rows):
        y = 112 + i * 34
        w = row["mean_absolute_effect"] * scale
        parts.append(f'<text class="label" x="180" y="{y+17}" text-anchor="end">{row["component"]}</text>')
        parts.append(f'<rect class="bar" x="200" y="{y}" width="{w:.1f}" height="22" rx="4"/>')
        parts.append(f'<text class="small" x="{208+w:.1f}" y="{y+16}">|dL| {row["mean_absolute_effect"]:.6f}; signed {row["mean_signed_effect"]:.6f}</text>')
    parts.extend([
        '<text class="small" x="42" y="474">Components were replaced at the final token with mean activations from the 30-position component set, then logits were recomputed.</text>',
        f'<metadata>{json.dumps({"source": payload["notebook_section"], "notebook": payload["notebook"], "branch": payload["branch"]})}</metadata>',
        '</svg>\n',
    ])
    OUT_PATH.write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
