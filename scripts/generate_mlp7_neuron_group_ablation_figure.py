#!/usr/bin/env python3
"""Regenerate the MLP7 neuron-group ablation figure from measured JSON."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
DATA_PATH = FIG / "mlp7_neuron_group_ablation.json"
OUT_PATH = FIG / "mlp7_neuron_group_ablation.svg"

STYLE = """<style>
svg{color-scheme:light dark;--text:#172033;--muted:#5c667a;--grid:#d8dde8;--blue:#246bfe;--teal:#0f8f7f;--red:#cc4b37}
@media(prefers-color-scheme:dark){svg{--text:#eef2ff;--muted:#b7bfd1;--grid:#384153;--blue:#86a9ff;--teal:#5fd4c7;--red:#ff8f7d}}
.title{font:700 24px system-ui,-apple-system,Segoe UI,sans-serif;fill:var(--text)}.label{font:600 13px system-ui,-apple-system,Segoe UI,sans-serif;fill:var(--text)}.small{font:12px system-ui,-apple-system,Segoe UI,sans-serif;fill:var(--muted)}
.axis{stroke:var(--grid);stroke-width:1.1}.zero{stroke:var(--grid);stroke-width:1.6;stroke-dasharray:5 5}.top{stroke:var(--red);stroke-width:3.5;fill:none}.rnd{stroke:var(--teal);stroke-width:3.5;fill:none}.dotTop{fill:var(--red)}.dotRnd{fill:var(--teal)}
</style>"""


def main() -> None:
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    top = {r["group_size"]: r for r in payload["groups"] if r["group_kind"] == "top neurons"}
    rnd = {r["group_size"]: r for r in payload["groups"] if r["group_kind"] == "random mean"}
    sizes = [1, 2, 5, 10, 20]
    xmap = {1: 110, 2: 220, 5: 330, 10: 470, 20: 640}
    y_top, y_bot = 92, 350
    ymin, ymax = -0.6, 0.05
    def y(v: float) -> float:
        return y_bot - (v - ymin) / (ymax - ymin) * (y_bot - y_top)
    def line(rows: dict[int, dict], cls: str) -> str:
        return " ".join(f"{xmap[s]},{y(rows[s]['mean_legality_degradation']):.1f}" for s in sizes)

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 780 460" role="img" aria-labelledby="title desc">',
        '<title id="title">MLP7 neuron-group ablation</title>',
        '<desc id="desc">Top-attribution MLP7 neuron groups compared with same-size random group means.</desc>',
        STYLE,
        '<text class="title" x="42" y="48">Selected MLP7 neurons differ from random groups</text>',
        '<text class="small" x="42" y="72">Measured top-N and random mean group interventions; no error bars are shown because only means were preserved here.</text>',
        f'<line class="axis" x1="82" y1="{y_bot}" x2="680" y2="{y_bot}"/>',
        f'<line class="axis" x1="82" y1="{y_top}" x2="82" y2="{y_bot}"/>',
        f'<line class="zero" x1="82" y1="{y(0):.1f}" x2="680" y2="{y(0):.1f}"/>',
        f'<polyline class="top" points="{line(top, "top")}"/>',
        f'<polyline class="rnd" points="{line(rnd, "rnd")}"/>',
    ]
    for s in sizes:
        parts.append(f'<circle class="dotTop" cx="{xmap[s]}" cy="{y(top[s]["mean_legality_degradation"]):.1f}" r="5"/>')
        parts.append(f'<circle class="dotRnd" cx="{xmap[s]}" cy="{y(rnd[s]["mean_legality_degradation"]):.1f}" r="5"/>')
        parts.append(f'<text class="small" x="{xmap[s]}" y="378" text-anchor="middle">{s}</text>')
    for tick in [-0.6, -0.4, -0.2, 0.0]:
        parts.append(f'<text class="small" x="70" y="{y(tick)+4:.1f}" text-anchor="end">{tick:.1f}</text>')
    parts.extend([
        '<text class="small" x="330" y="414" text-anchor="middle">number of selected neurons</text>',
        '<text class="small" x="30" y="228" transform="rotate(-90 30 228)" text-anchor="middle">mean legality degradation</text>',
        '<circle class="dotTop" cx="590" cy="104" r="5"/><text class="small" x="604" y="108">top attribution group</text>',
        '<circle class="dotRnd" cx="590" cy="128" r="5"/><text class="small" x="604" y="132">random same-size mean</text>',
        f'<metadata>{json.dumps({"source": payload["notebook_section"], "notebook": payload["notebook"], "branch": payload["branch"]})}</metadata>',
        '</svg>\n',
    ])
    OUT_PATH.write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
