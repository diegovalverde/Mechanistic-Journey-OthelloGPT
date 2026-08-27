#!/usr/bin/env python3
"""Regenerate the Chapter 7 layer-7 validation figure from measured JSON."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
DATA_PATH = FIG / "chapter07_layer7_validation.json"
OUT_PATH = FIG / "chapter07_layer7_validation.svg"

STYLE = """<style>
svg{color-scheme:light dark;--text:#172033;--muted:#5c667a;--grid:#d8dde8;--fill:#f7f9fc;--blue:#246bfe;--teal:#0f8f7f;--gold:#b98113;--red:#cc4b37}
@media(prefers-color-scheme:dark){svg{--text:#eef2ff;--muted:#b7bfd1;--grid:#384153;--fill:#1f2633;--blue:#86a9ff;--teal:#5fd4c7;--gold:#ffd27a;--red:#ff8f7d}}
.title{font:700 24px system-ui,-apple-system,Segoe UI,sans-serif;fill:var(--text)}.label{font:600 14px system-ui,-apple-system,Segoe UI,sans-serif;fill:var(--text)}.small{font:12px system-ui,-apple-system,Segoe UI,sans-serif;fill:var(--muted)}
.axis{stroke:var(--grid);stroke-width:1.2}.bar1{fill:var(--blue)}.bar2{fill:var(--teal)}.null{stroke:var(--gold);stroke-width:3}.obs{stroke:var(--red);stroke-width:3}.box{fill:var(--fill);stroke:var(--grid);stroke-width:1.2;rx:8}
</style>"""


def main() -> None:
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    v = payload["validation"]
    top = 94
    bottom = 322
    h = bottom - top
    max_mean = 0.07
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 780 450" role="img" aria-labelledby="title desc">',
        '<title id="title">Layer-7 capture-opponent enrichment validation</title>',
        '<desc id="desc">Measured capture mean, unrelated occupied mean, bootstrap intervals, and shuffled-ratio null summary.</desc>',
        STYLE,
        '<text class="title" x="42" y="48">Layer-7 validation: capture opponents stand out</text>',
        '<text class="small" x="42" y="72">Bootstrap resamples positions. Shuffled null permutes square labels inside sensitivity maps.</text>',
        f'<line class="axis" x1="92" y1="{bottom}" x2="345" y2="{bottom}"/>',
        f'<line class="axis" x1="92" y1="{top}" x2="92" y2="{bottom}"/>',
    ]
    for tick in [0, 0.02, 0.04, 0.06]:
        y = bottom - tick / max_mean * h
        parts.append(f'<line class="axis" x1="92" y1="{y:.1f}" x2="345" y2="{y:.1f}"/>')
        parts.append(f'<text class="small" x="80" y="{y+4:.1f}" text-anchor="end">{tick:.2f}</text>')
    bars = [
        ("capture", v["capture_mean"], "bar1", 160),
        ("unrelated", v["unrelated_occupied_mean"], "bar2", 270),
    ]
    for label, value, cls, x in bars:
        bh = value / max_mean * h
        y = bottom - bh
        parts.append(f'<rect class="{cls}" x="{x-32}" y="{y:.1f}" width="64" height="{bh:.1f}" rx="4"/>')
        parts.append(f'<text class="label" x="{x}" y="{y-8:.1f}" text-anchor="middle">{value:.6f}</text>')
        parts.append(f'<text class="small" x="{x}" y="346" text-anchor="middle">{label}</text>')

    rx0, rx1 = 430, 700
    rtop, rbot = 128, 308
    rmax = 3.0
    def rx(value: float) -> float:
        return rx0 + value / rmax * (rx1 - rx0)

    parts.extend([
        f'<line class="axis" x1="{rx0}" y1="{rbot}" x2="{rx1}" y2="{rbot}"/>',
        f'<line class="axis" x1="{rx(1.0):.1f}" y1="{rtop}" x2="{rx(1.0):.1f}" y2="{rbot}"/>',
        f'<text class="small" x="{rx(1.0):.1f}" y="{rbot+22}" text-anchor="middle">1.0</text>',
        f'<line class="null" x1="{rx(v["shuffled_null_mean_ratio"]):.1f}" y1="176" x2="{rx(v["shuffled_null_mean_ratio"]):.1f}" y2="278"/>',
        f'<line class="null" x1="{rx(v["shuffled_null_95th_percentile"]):.1f}" y1="176" x2="{rx(v["shuffled_null_95th_percentile"]):.1f}" y2="278"/>',
        f'<line class="obs" x1="{rx(v["ratio"]):.1f}" y1="130" x2="{rx(v["ratio"]):.1f}" y2="308"/>',
        f'<text class="label" x="{rx(v["ratio"]):.1f}" y="116" text-anchor="middle">observed {v["ratio"]:.3f}</text>',
        f'<text class="small" x="{rx(v["shuffled_null_mean_ratio"]):.1f}" y="168" text-anchor="middle">null mean</text>',
        f'<text class="small" x="{rx(v["shuffled_null_95th_percentile"]):.1f}" y="296" text-anchor="middle">null p95</text>',
        '<text class="label" x="430" y="100">Ratio validation</text>',
        f'<text class="small" x="430" y="354">difference CI [{v["bootstrap_difference_95_ci"][0]:.6f}, {v["bootstrap_difference_95_ci"][1]:.6f}]</text>',
        f'<text class="small" x="430" y="374">ratio CI [{v["bootstrap_ratio_95_ci"][0]:.6f}, {v["bootstrap_ratio_95_ci"][1]:.6f}]</text>',
        f'<text class="small" x="430" y="394">empirical p = {v["empirical_permutation_p"]:.6f}</text>',
        f'<metadata>{json.dumps({"source": payload["notebook_section"], "notebook": payload["notebook"], "branch": payload["branch"]})}</metadata>',
        '</svg>\n',
    ])
    OUT_PATH.write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
