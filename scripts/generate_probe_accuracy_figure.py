#!/usr/bin/env python3
"""Generate the strict-split board-probe accuracy figure.

Source:
    diegovalverde/TransformerLens, branch othello-jspace-analysis
    demos/Othello_GPT_Jacobian_Lens.ipynb
    section 7. Train a linear mine / theirs / empty board probe

Output:
    docs/figures/board_probe_accuracy.svg
"""

from __future__ import annotations

from pathlib import Path


ACCURACIES = {
    "Overall": 0.9796,
    "Empty": 0.9976,
    "Mine": 0.9561,
    "Theirs": 0.9703,
}

OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "figures" / "board_probe_accuracy.svg"


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    width = 820
    height = 520
    margin_left = 96
    margin_right = 42
    chart_top = 92
    chart_bottom = 382
    chart_width = width - margin_left - margin_right
    chart_height = chart_bottom - chart_top
    bar_width = 92
    gap = (chart_width - bar_width * len(ACCURACIES)) / (len(ACCURACIES) - 1)

    colors = {
        "Overall": "#2563eb",
        "Empty": "#0f766e",
        "Mine": "#7c3aed",
        "Theirs": "#dc2626",
    }

    bars = []
    for index, (label, value) in enumerate(ACCURACIES.items()):
        x = margin_left + index * (bar_width + gap)
        bar_height = value * chart_height
        y = chart_bottom - bar_height
        bars.append(
            f"""
  <g>
    <rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" rx="5" class="bar" fill="{colors[label]}"/>
    <text x="{x + bar_width / 2:.1f}" y="{y - 12:.1f}" text-anchor="middle" class="value">{pct(value)}</text>
    <text x="{x + bar_width / 2:.1f}" y="{chart_bottom + 34}" text-anchor="middle" class="label">{label}</text>
  </g>"""
        )

    grid_lines = []
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        y = chart_bottom - tick * chart_height
        grid_lines.append(
            f"""
  <line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" class="grid"/>
  <text x="{margin_left - 16}" y="{y + 5:.1f}" text-anchor="end" class="tick">{int(tick * 100)}%</text>"""
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">Strict game-level board-probe validation accuracy</title>
  <desc id="desc">A bar chart showing board-probe validation accuracy from zero to one hundred percent: overall 97.96 percent, empty 99.76 percent, mine 95.61 percent, and theirs 97.03 percent.</desc>
  <defs>
    <style>
      :root {{
        --bg: #ffffff;
        --ink: #172033;
        --muted: #475569;
        --grid: #cbd5e1;
        --axis: #64748b;
      }}
      @media (prefers-color-scheme: dark) {{
        :root {{
          --bg: #111827;
          --ink: #f8fafc;
          --muted: #cbd5e1;
          --grid: #334155;
          --axis: #94a3b8;
        }}
      }}
      .title {{ font: 700 24px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: var(--ink); }}
      .subtitle {{ font: 500 14px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: var(--muted); }}
      .label {{ font: 650 15px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: var(--ink); }}
      .value {{ font: 700 16px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: var(--ink); }}
      .tick {{ font: 13px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: var(--muted); }}
      .axis-title {{ font: 600 13px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: var(--muted); }}
      .grid {{ stroke: var(--grid); stroke-width: 1; }}
      .axis {{ stroke: var(--axis); stroke-width: 1.6; }}
      .bar {{ stroke: rgba(15, 23, 42, 0.16); stroke-width: 1; }}
      .note {{ font: 13px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: var(--muted); }}
    </style>
  </defs>
  <rect width="{width}" height="{height}" fill="var(--bg)" rx="10"/>
  <text x="{width / 2}" y="38" text-anchor="middle" class="title">Board Probe Accuracy</text>
  <text x="{width / 2}" y="64" text-anchor="middle" class="subtitle">Strict game-level held-out validation, 330 positions</text>
  {''.join(grid_lines)}
  <line x1="{margin_left}" y1="{chart_top}" x2="{margin_left}" y2="{chart_bottom}" class="axis"/>
  <line x1="{margin_left}" y1="{chart_bottom}" x2="{width - margin_right}" y2="{chart_bottom}" class="axis"/>
  <text x="24" y="{(chart_top + chart_bottom) / 2}" class="axis-title" transform="rotate(-90 24 {(chart_top + chart_bottom) / 2})" text-anchor="middle">validation accuracy</text>
  {''.join(bars)}
  <text x="{width / 2}" y="450" text-anchor="middle" class="note">Source: Othello_GPT_Jacobian_Lens.ipynb, section 7.</text>
  <text x="{width / 2}" y="474" text-anchor="middle" class="note">Y-axis begins at 0; values are measured validation accuracies, not illustrative estimates.</text>
</svg>
"""
    OUTPUT.write_text(svg, encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    for label, value in ACCURACIES.items():
        print(f"{label}: {value:.4f} ({pct(value)})")


if __name__ == "__main__":
    main()
