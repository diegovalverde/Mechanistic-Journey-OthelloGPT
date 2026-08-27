#!/usr/bin/env python3
"""Generate the Chapter 4 measured Jacobian validation figure.

Inputs are constants copied from executed outputs in:

    https://github.com/diegovalverde/TransformerLens
    branch: othello-jspace-analysis
    notebook: demos/Othello_GPT_Jacobian_Lens.ipynb
    section: 9. Jacobian prediction vs actual board-state intervention

The script does not rerun the notebook. It preserves the provenance chain:

    executed notebook -> research memory -> book figure
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "docs" / "figures"
JSON_PATH = FIGURE_DIR / "jacobian_prediction_vs_intervention.json"
SVG_PATH = FIGURE_DIR / "jacobian_prediction_vs_intervention.svg"


ROWS = [
    {
        "alpha": -0.1,
        "predicted_delta_logit": -0.003090,
        "actual_delta_logit": -0.003156,
        "abs_error": 0.000066057,
    },
    {
        "alpha": -0.03,
        "predicted_delta_logit": -0.000927,
        "actual_delta_logit": -0.000932,
        "abs_error": 0.000004844,
    },
    {
        "alpha": -0.01,
        "predicted_delta_logit": -0.000309,
        "actual_delta_logit": -0.000308,
        "abs_error": 0.000000928,
    },
    {
        "alpha": -0.003,
        "predicted_delta_logit": -0.000093,
        "actual_delta_logit": -0.000095,
        "abs_error": 0.000002678,
    },
    {
        "alpha": 0.003,
        "predicted_delta_logit": 0.000093,
        "actual_delta_logit": 0.000092,
        "abs_error": 0.000001137,
    },
    {
        "alpha": 0.01,
        "predicted_delta_logit": 0.000309,
        "actual_delta_logit": 0.000308,
        "abs_error": 0.000000928,
    },
    {
        "alpha": 0.03,
        "predicted_delta_logit": 0.000927,
        "actual_delta_logit": 0.000919,
        "abs_error": 0.000007553,
    },
    {
        "alpha": 0.1,
        "predicted_delta_logit": 0.003090,
        "actual_delta_logit": 0.003023,
        "abs_error": 0.000066504,
    },
]


METADATA = {
    "source_repository": "diegovalverde/TransformerLens",
    "source_repository_url": "https://github.com/diegovalverde/TransformerLens",
    "branch": "othello-jspace-analysis",
    "notebook": "demos/Othello_GPT_Jacobian_Lens.ipynb",
    "section": "9. Jacobian prediction vs actual board-state intervention",
    "model_dimensions": {"d_model": 512, "d_vocab": 61, "d_vocab_out": 61},
    "layer": 4,
    "hook": "blocks.4.hook_resid_post",
    "source_position": 27,
    "target_position": 27,
    "prefix_length": 28,
    "prefix_token_ids": [
        20,
        19,
        18,
        10,
        2,
        1,
        27,
        3,
        41,
        42,
        34,
        12,
        4,
        40,
        11,
        29,
        43,
        13,
        48,
        56,
        33,
        39,
        22,
        44,
        24,
        5,
        46,
        6,
    ],
    "selected_move": {"label": "E3", "token_id": 21},
    "model_favorite_move": {"label": "E8", "token_id": 57},
    "selected_move_baseline_logit": 8.940763473510742,
    "selected_move_capture_lines": [["D3", "C3"]],
    "direction": {
        "type": "normalized probe direction",
        "semantic": "mine-vs-theirs",
        "square": "G6",
        "square_index": 46,
    },
    "directional_derivative_v_dot_g": 0.030897,
    "max_abs_prediction_error": 0.000067,
    "note": "Numerical rows are copied from the executed notebook display output.",
}


def write_json() -> None:
    payload = {"metadata": METADATA, "rows": ROWS}
    JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def scale(value: float, domain_min: float, domain_max: float, range_min: float, range_max: float) -> float:
    fraction = (value - domain_min) / (domain_max - domain_min)
    return range_min + fraction * (range_max - range_min)


def write_svg() -> None:
    width = 760
    height = 560
    left = 92
    right = 46
    top = 54
    bottom = 86
    plot_w = width - left - right
    plot_h = height - top - bottom

    all_values = [
        value
        for row in ROWS
        for value in (row["predicted_delta_logit"], row["actual_delta_logit"])
    ]
    limit = max(abs(value) for value in all_values) * 1.15
    domain_min = -limit
    domain_max = limit

    def x_pos(value: float) -> float:
        return scale(value, domain_min, domain_max, left, left + plot_w)

    def y_pos(value: float) -> float:
        return scale(value, domain_min, domain_max, top + plot_h, top)

    ticks = [-0.003, -0.0015, 0.0, 0.0015, 0.003]
    grid_lines = []
    for tick in ticks:
        x = x_pos(tick)
        y = y_pos(tick)
        label = f"{tick:+.4f}".replace("+", "")
        grid_lines.append(
            f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}"/>'
            f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}"/>'
            f'<text class="tick x" x="{x:.2f}" y="{top + plot_h + 28}">{label}</text>'
            f'<text class="tick y" x="{left - 14}" y="{y + 4:.2f}">{label}</text>'
        )

    diagonal = (
        f'<line class="diagonal" x1="{x_pos(domain_min):.2f}" y1="{y_pos(domain_min):.2f}" '
        f'x2="{x_pos(domain_max):.2f}" y2="{y_pos(domain_max):.2f}"/>'
    )

    points = []
    for row in ROWS:
        x = x_pos(row["predicted_delta_logit"])
        y = y_pos(row["actual_delta_logit"])
        points.append(
            f'<circle class="point" cx="{x:.2f}" cy="{y:.2f}" r="6">'
            f'<title>alpha={row["alpha"]}; predicted={row["predicted_delta_logit"]:.6f}; '
            f'actual={row["actual_delta_logit"]:.6f}; abs error={row["abs_error"]:.9f}</title>'
            "</circle>"
        )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">Jacobian prediction versus actual intervention</title>
  <desc id="desc">Measured predicted and actual E3 logit deltas for a G6 mine-vs-theirs residual intervention lie close to the y equals x diagonal.</desc>
  <style>
    svg {{
      color-scheme: light dark;
      --bg: transparent;
      --text: #172033;
      --muted: #5c667a;
      --grid: #d8dde8;
      --axis: #2c3446;
      --diag: #7a8496;
      --point: #0f8f7f;
      --point-stroke: #ffffff;
    }}
    @media (prefers-color-scheme: dark) {{
      svg {{
        --text: #eef2ff;
        --muted: #b7bfd1;
        --grid: #3a4254;
        --axis: #dce4f4;
        --diag: #98a2b3;
        --point: #3dd6c6;
        --point-stroke: #101522;
      }}
    }}
    .title {{ fill: var(--text); font: 700 22px system-ui, -apple-system, Segoe UI, sans-serif; }}
    .subtitle {{ fill: var(--muted); font: 14px system-ui, -apple-system, Segoe UI, sans-serif; }}
    .axis {{ stroke: var(--axis); stroke-width: 1.5; }}
    .grid {{ stroke: var(--grid); stroke-width: 1; }}
    .tick {{ fill: var(--muted); font: 12px system-ui, -apple-system, Segoe UI, sans-serif; }}
    .tick.x {{ text-anchor: middle; }}
    .tick.y {{ text-anchor: end; }}
    .label {{ fill: var(--text); font: 600 14px system-ui, -apple-system, Segoe UI, sans-serif; }}
    .diagonal {{ stroke: var(--diag); stroke-width: 2; stroke-dasharray: 7 7; }}
    .point {{ fill: var(--point); stroke: var(--point-stroke); stroke-width: 2; }}
    .note {{ fill: var(--muted); font: 12px system-ui, -apple-system, Segoe UI, sans-serif; }}
  </style>

  <text class="title" x="{left}" y="30">Jacobian prediction vs actual intervention</text>
  <text class="subtitle" x="{left}" y="50">Measured E3 logit deltas from a layer-4 G6 mine-vs-theirs residual edit</text>

  {''.join(grid_lines)}
  {diagonal}
  <line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>
  <line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>
  {''.join(points)}

  <text class="label" x="{left + plot_w / 2:.2f}" y="{height - 30}" text-anchor="middle">Jacobian-predicted delta logit</text>
  <text class="label" x="22" y="{top + plot_h / 2:.2f}" transform="rotate(-90 22 {top + plot_h / 2:.2f})" text-anchor="middle">Actual intervention delta logit</text>
  <text class="note" x="{left + plot_w - 4}" y="{top + 18}" text-anchor="end">dashed line: y = x</text>
  <text class="note" x="{left}" y="{height - 10}">Source: executed notebook section 9; max absolute prediction error = 0.000067.</text>
</svg>
'''
    SVG_PATH.write_text(svg, encoding="utf-8")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    write_json()
    write_svg()
    print(f"Wrote {JSON_PATH}")
    print(f"Wrote {SVG_PATH}")


if __name__ == "__main__":
    main()
