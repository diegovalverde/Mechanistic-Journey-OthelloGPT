#!/usr/bin/env python3
"""Regenerate the small MLP7 semantic-mediation diagnostic figure."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
DATA_PATH = FIG / "mlp7_semantic_mediation.json"
OUT_PATH = FIG / "mlp7_semantic_mediation.svg"

STYLE = """<style>
svg{color-scheme:light dark;--text:#172033;--muted:#5c667a;--grid:#d8dde8;--blue:#246bfe;--red:#cc4b37;--teal:#0f8f7f}
@media(prefers-color-scheme:dark){svg{--text:#eef2ff;--muted:#b7bfd1;--grid:#384153;--blue:#86a9ff;--red:#ff8f7d;--teal:#5fd4c7}}
.title{font:700 24px system-ui,-apple-system,Segoe UI,sans-serif;fill:var(--text)}.label{font:600 13px system-ui,-apple-system,Segoe UI,sans-serif;fill:var(--text)}.small{font:12px system-ui,-apple-system,Segoe UI,sans-serif;fill:var(--muted)}
.axis{stroke:var(--grid);stroke-width:1.1}.pos{fill:var(--teal)}.neg{fill:var(--red)}
</style>"""


def main() -> None:
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    rows = payload["rows"]
    x0 = 360
    scale = 420 / 0.08
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 840 360" role="img" aria-labelledby="title desc">',
        '<title id="title">MLP7 semantic mediation diagnostic</title>',
        '<desc id="desc">Example-level mediation-like effects for two capture-line and two unrelated semantic edits.</desc>',
        STYLE,
        '<text class="title" x="42" y="48">Semantic edit x MLP7 ablation diagnostic</text>',
        '<text class="small" x="42" y="72">M = delta L normal - delta L with MLP7 ablated. Example-level diagnostic, not a dataset distribution.</text>',
        f'<line class="axis" x1="{x0}" y1="100" x2="{x0}" y2="250"/>',
    ]
    for i, row in enumerate(rows):
        y = 110 + i * 36
        v = row["mediation_like_effect"]
        w = abs(v) * scale
        x = x0 if v >= 0 else x0 - w
        cls = "pos" if v >= 0 else "neg"
        parts.append(f'<text class="label" x="330" y="{y+16}" text-anchor="end">{row["square"]} {row["edit_group"].split("-")[0]}</text>')
        parts.append(f'<rect class="{cls}" x="{x:.1f}" y="{y}" width="{w:.1f}" height="22" rx="4"/>')
        parts.append(f'<text class="small" x="{x0 + (w if v >= 0 else -w) + (8 if v >= 0 else -8):.1f}" y="{y+16}" text-anchor="{"start" if v >= 0 else "end"}">{v:.6f}</text>')
    parts.extend([
        f'<text class="small" x="42" y="300">Mean capture-edit M = {payload["summary"]["mean_mediation_like_effect_capture_edits"]:.6f}; mean unrelated-edit M = {payload["summary"]["mean_mediation_like_effect_unrelated_edits"]:.6f}.</text>',
        f'<metadata>{json.dumps({"source": payload["notebook_section"], "notebook": payload["notebook"], "branch": payload["branch"]})}</metadata>',
        '</svg>\n',
    ])
    OUT_PATH.write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
