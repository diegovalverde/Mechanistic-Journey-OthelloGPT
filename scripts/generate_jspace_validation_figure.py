#!/usr/bin/env python3
"""Generate Chapter 5 J-space figures.

Inputs are constants copied from executed outputs in:

    /Users/diegovalverdegarro/workspace/projects/TransformerLens
    branch: othello-jspace-analysis
    notebook: demos/Othello_GPT_Jacobian_Lens.ipynb
    section: 10. Local J-space vs averaged J-space

The script does not rerun the notebook. It preserves the provenance chain:

    executed notebook -> research memory -> book figure
"""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "docs" / "figures"


METRICS = {
    "source_repository": "diegovalverde/TransformerLens",
    "source_checkout": "/Users/diegovalverdegarro/workspace/projects/TransformerLens",
    "branch": "othello-jspace-analysis",
    "notebook": "demos/Othello_GPT_Jacobian_Lens.ipynb",
    "section": "10. Local J-space vs averaged J-space",
    "model_dimensions": {"d_model": 512, "d_vocab": 61, "d_vocab_out": 61},
    "source": {
        "layer": 4,
        "hook": "blocks.4.hook_resid_post",
        "position": 27,
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
    },
    "target": {
        "representation": "final residual stream immediately before final layer norm and unembedding",
        "position": 27,
        "dimension": 512,
    },
    "semantic_direction": {
        "square": "G6",
        "square_index": 46,
        "contrast": "mine-vs-theirs",
        "construction": "normalized layer-4 linear-probe weight difference W[q,mine] - W[q,theirs]",
    },
    "jvp": {
        "implementation": "torch.autograd.functional.jvp over final_resid_from_delta",
        "finite_difference": "central difference (F(h + eps v) - F(h - eps v)) / (2 eps)",
        "finite_difference_epsilon": 0.001,
        "local_jvp_finite_difference_cosine": 0.999944,
        "local_jvp_finite_difference_relative_error": 0.010651,
    },
    "averaging": {
        "num_positions": 100,
        "prefix_len_min": 12,
        "prefix_len_mean": 29.14,
        "prefix_len_max": 45,
        "random_seed": "PROBE_RANDOM_SEED + 1",
        "sampling": (
            "Generate random legal Othello games, choose one unique prefix per sampled "
            "position with length uniformly sampled from the valid range, require at "
            "least one legal move, and record the legal move that maximizes flipped "
            "pieces, then capture lines, then lower token id."
        ),
        "first_five_sampled_moves": [
            {"prefix_len": 30, "chosen_move": "F3", "num_flipped": 5},
            {"prefix_len": 29, "chosen_move": "A3", "num_flipped": 3},
            {"prefix_len": 28, "chosen_move": "G4", "num_flipped": 5},
            {"prefix_len": 16, "chosen_move": "B4", "num_flipped": 4},
            {"prefix_len": 16, "chosen_move": "D1", "num_flipped": 5},
        ],
    },
    "results": {
        "source_space_derivative_v_dot_grad_z_E3": 0.030897,
        "final_readout_effect_local_J_local_v": 0.030897,
        "final_readout_effect_averaged_J_avg_v": 0.018023,
        "norm_J_local_v": 1.496970,
        "norm_J_avg_v": 0.819020,
        "local_vs_average_cosine": 0.617840,
        "local_vs_average_angle_degrees": math.degrees(math.acos(0.617840)),
    },
    "figure_note": "Measured annotations come from executed notebook output; schematic arrow geometry is illustrative unless explicitly stated otherwise.",
}


STYLE = """<style>
  svg {
    color-scheme: light dark;
    --text: #172033;
    --muted: #5c667a;
    --line: #253044;
    --grid: #d8dde8;
    --blue: #246bfe;
    --green: #0f8f7f;
    --red: #cc4b37;
    --gold: #b98113;
    --fill: #f7f9fc;
  }
  @media (prefers-color-scheme: dark) {
    svg {
      --text: #eef2ff;
      --muted: #b7bfd1;
      --line: #dce3f7;
      --grid: #384153;
      --blue: #82a7ff;
      --green: #60d4c6;
      --red: #ff8a76;
      --gold: #ffd27a;
      --fill: #1f2633;
    }
  }
  .title { font: 700 24px system-ui, -apple-system, Segoe UI, sans-serif; fill: var(--text); }
  .label { font: 600 15px system-ui, -apple-system, Segoe UI, sans-serif; fill: var(--text); }
  .small { font: 13px system-ui, -apple-system, Segoe UI, sans-serif; fill: var(--muted); }
  .tiny { font: 11px system-ui, -apple-system, Segoe UI, sans-serif; fill: var(--muted); }
  .axis { stroke: var(--grid); stroke-width: 1.2; }
  .line { stroke: var(--line); stroke-width: 2.2; fill: none; }
  .surface { stroke: var(--grid); fill: none; stroke-width: 1.4; }
  .box { fill: var(--fill); stroke: var(--grid); stroke-width: 1.4; rx: 8; }
  .arrow-v { stroke: var(--blue); stroke-width: 4; fill: none; marker-end: url(#arrowBlue); }
  .arrow-local { stroke: var(--green); stroke-width: 4; fill: none; marker-end: url(#arrowGreen); }
  .arrow-avg { stroke: var(--gold); stroke-width: 4; fill: none; marker-end: url(#arrowGold); }
  .arrow-red { stroke: var(--red); stroke-width: 4; fill: none; marker-end: url(#arrowRed); }
  .node { fill: var(--fill); stroke: var(--line); stroke-width: 1.8; }
</style>"""


DEFS = """<defs>
  <marker id="arrowBlue" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#246bfe"/></marker>
  <marker id="arrowGreen" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#0f8f7f"/></marker>
  <marker id="arrowGold" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#b98113"/></marker>
  <marker id="arrowRed" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#cc4b37"/></marker>
</defs>"""


def write(path: str, body: str) -> None:
    (FIGURE_DIR / path).write_text(body, encoding="utf-8")


def svg_start(width: int, height: int, title: str, desc: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">{title}</title>
  <desc id="desc">{desc}</desc>
  {STYLE}
  {DEFS}
'''


def context_dependent_jacobian() -> None:
    body = svg_start(
        820,
        470,
        "Context-dependent Jacobian",
        "The same source arrow at two different points on a nonlinear field is transformed into different target arrows.",
    )
    curves = "\n".join(
        f'<path class="surface" d="M80 {y} C220 {y - 45}, 340 {y + 35}, 500 {y - 10} S680 {y - 5}, 760 {y - 42}"/>'
        for y in [150, 210, 270, 330]
    )
    body += f'''
  <text class="title" x="36" y="48">A Jacobian is attached to a point</text>
  {curves}
  <circle class="node" cx="215" cy="245" r="7"/>
  <circle class="node" cx="585" cy="220" r="7"/>
  <path class="arrow-v" d="M215 245 L280 205"/>
  <path class="arrow-v" d="M585 220 L650 180"/>
  <path class="arrow-local" d="M215 245 L255 310"/>
  <path class="arrow-red" d="M585 220 L665 238"/>
  <text class="label" x="154" y="272">context x1</text>
  <text class="small" x="245" y="200">same v</text>
  <text class="small" x="260" y="320">J(x1)v</text>
  <text class="label" x="525" y="247">context x2</text>
  <text class="small" x="622" y="175">same v</text>
  <text class="small" x="670" y="244">J(x2)v</text>
  <text class="small" x="92" y="405">Same semantic question. Different local context. Different first-order answer.</text>
</svg>
'''
    write("context_dependent_jacobian.svg", body)


def jacobian_direction_transport() -> None:
    body = svg_start(
        820,
        420,
        "Jacobian direction transport",
        "A semantic direction in source residual space is mapped by a local Jacobian into a final residual-space direction.",
    )
    body += '''
  <text class="title" x="36" y="48">First-order transport of a direction</text>
  <rect class="box" x="70" y="92" width="250" height="210"/>
  <rect class="box" x="500" y="92" width="250" height="210"/>
  <text class="label" x="108" y="128">source residual space</text>
  <text class="small" x="108" y="154">layer 4, final token</text>
  <text class="label" x="536" y="128">target residual space</text>
  <text class="small" x="536" y="154">final residual, same token</text>
  <path class="axis" d="M115 250 L280 250 M150 285 L150 140"/>
  <path class="axis" d="M545 250 L710 250 M580 285 L580 140"/>
  <path class="arrow-v" d="M150 250 L245 190"/>
  <path class="arrow-local" d="M580 250 L675 214"/>
  <text class="small" x="170" y="185">v = G6 mine-vs-theirs</text>
  <text class="small" x="600" y="207">J_x v</text>
  <path class="line" d="M335 197 L482 197" marker-end="url(#arrowGreen)"/>
  <text class="label" x="382" y="184">F_x</text>
  <text class="small" x="330" y="235">linearized by J_x at this context</text>
  <text class="tiny" x="116" y="352">The arrow is not physically carried through the model; this is the local derivative of the downstream function.</text>
</svg>
'''
    write("jacobian_direction_transport.svg", body)


def probe_space_vs_jspace() -> None:
    body = svg_start(
        860,
        440,
        "Probe space versus J-space",
        "Probe geometry asks what can be decoded; J-space asks how a semantic direction is locally transformed downstream.",
    )
    body += '''
  <text class="title" x="36" y="48">Probe space and J-space ask different questions</text>
  <rect class="box" x="58" y="92" width="340" height="245"/>
  <rect class="box" x="462" y="92" width="340" height="245"/>
  <text class="label" x="92" y="130">Probe space</text>
  <circle class="node" cx="126" cy="205" r="28"/>
  <path class="arrow-v" d="M158 205 L252 205"/>
  <circle class="node" cx="292" cy="205" r="28"/>
  <text class="small" x="100" y="210">h</text>
  <text class="small" x="192" y="194">probe v</text>
  <text class="small" x="265" y="210">label</text>
  <text class="small" x="92" y="285">Question: what board fact can be decoded here?</text>
  <text class="label" x="496" y="130">J-space</text>
  <circle class="node" cx="530" cy="205" r="28"/>
  <path class="arrow-v" d="M562 205 L640 205"/>
  <circle class="node" cx="680" cy="205" r="28"/>
  <path class="arrow-local" d="M705 224 L760 268"/>
  <text class="small" x="525" y="210">v</text>
  <text class="small" x="588" y="194">J_x</text>
  <text class="small" x="657" y="210">J_x v</text>
  <text class="small" x="496" y="285">Question: how does downstream computation transform it?</text>
</svg>
'''
    write("probe_space_vs_jspace.svg", body)


def jspace_context_cloud() -> None:
    body = svg_start(
        820,
        450,
        "J-space context cloud",
        "One source semantic direction has many local transformed versions across sampled Othello contexts.",
    )
    arrows = [
        (545, 255, 625, 206, "arrow-local", "J1 v"),
        (545, 255, 642, 250, "arrow-local", "J2 v"),
        (545, 255, 608, 315, "arrow-red", "J3 v"),
        (545, 255, 590, 188, "arrow-local", "J4 v"),
        (545, 255, 664, 282, "arrow-avg", "mean"),
    ]
    body += '''
  <text class="title" x="36" y="48">One semantic direction, many local images</text>
  <rect class="box" x="60" y="96" width="250" height="250"/>
  <rect class="box" x="475" y="96" width="275" height="250"/>
  <path class="axis" d="M112 280 L280 280 M146 312 L146 145"/>
  <path class="axis" d="M520 280 L705 280 M545 318 L545 135"/>
  <path class="arrow-v" d="M146 280 L235 220"/>
  <text class="small" x="156" y="213">v</text>
  <text class="small" x="92" y="130">source direction</text>
  <path class="line" d="M328 220 C380 165, 430 165, 462 220" marker-end="url(#arrowGreen)"/>
  <path class="line" d="M328 245 C384 245, 420 245, 462 245" marker-end="url(#arrowGreen)"/>
  <path class="line" d="M328 270 C380 325, 430 325, 462 270" marker-end="url(#arrowGreen)"/>
  <text class="small" x="356" y="150">different contexts x_i</text>
'''
    for x1, y1, x2, y2, cls, label in arrows:
        body += f'  <path class="{cls}" d="M{x1} {y1} L{x2} {y2}"/>\n'
        body += f'  <text class="tiny" x="{x2 + 5}" y="{y2 + 4}">{label}</text>\n'
    body += '''
  <text class="small" x="505" y="130">target-space cloud</text>
  <text class="tiny" x="508" y="370">Schematic: arrows cluster partly, but not perfectly, around a shared direction.</text>
</svg>
'''
    write("jspace_context_cloud.svg", body)


def local_vs_average_jspace() -> None:
    angle = METRICS["results"]["local_vs_average_angle_degrees"]
    body = svg_start(
        760,
        520,
        "Local versus averaged J-space",
        "Schematic arrows show the measured cosine between local and averaged transformed G6 mine-vs-theirs directions.",
    )
    body += f'''
  <text class="title" x="36" y="48">Local vs averaged transformed direction</text>
  <path class="axis" d="M110 390 L650 390 M160 430 L160 105"/>
  <circle class="node" cx="160" cy="390" r="6"/>
  <path class="arrow-local" d="M160 390 L365 155"/>
  <path class="arrow-avg" d="M160 390 L522 315"/>
  <path class="surface" d="M247 291 A150 150 0 0 1 307 359"/>
  <text class="label" x="374" y="151">local J_x v</text>
  <text class="label" x="532" y="320">average E[J_x v]</text>
  <text class="label" x="304" y="266">cosine = 0.617840</text>
  <text class="small" x="315" y="287">angle approx {angle:.1f} degrees</text>
  <text class="small" x="76" y="455">Arrow lengths and layout are schematic. The cosine annotation is measured from the executed notebook.</text>
</svg>
'''
    write("local_vs_average_jspace.svg", body)


def jspace_bridge() -> None:
    body = svg_start(
        820,
        560,
        "J-space as a bridge",
        "J-space connects semantic directions from probes to local Jacobian transport and later component hypotheses.",
    )
    steps = [
        ("semantic direction", "G6 mine-vs-theirs"),
        ("local Jacobian", "context-specific J_x"),
        ("transformed direction", "J_x v"),
        ("compare contexts", "local vs average"),
        ("candidate computation", "where is it produced?"),
    ]
    y = 105
    for i, (title, sub) in enumerate(steps):
        body += f'  <rect class="box" x="235" y="{y}" width="350" height="58"/>\n'
        body += f'  <text class="label" x="270" y="{y + 24}">{title}</text>\n'
        body += f'  <text class="small" x="270" y="{y + 45}">{sub}</text>\n'
        if i < len(steps) - 1:
            body += f'  <path class="line" d="M410 {y + 64} L410 {y + 94}" marker-end="url(#arrowGreen)"/>\n'
        y += 95
    body += '''
  <text class="title" x="36" y="48">J-space bridges representation and computation</text>
  <text class="small" x="76" y="520">Probe analysis supplies interpretable directions. Jacobian analysis asks what the downstream model does to them.</text>
</svg>
'''
    write("jspace_bridge.svg", body)


def jspace_jvp_validation() -> None:
    body = svg_start(
        760,
        470,
        "J-space JVP validation",
        "Measured finite-difference validation metrics for the local JVP are shown as compact gauges.",
    )
    cos = METRICS["jvp"]["local_jvp_finite_difference_cosine"]
    err = METRICS["jvp"]["local_jvp_finite_difference_relative_error"]
    local_norm = METRICS["results"]["norm_J_local_v"]
    avg_norm = METRICS["results"]["norm_J_avg_v"]
    body += f'''
  <text class="title" x="36" y="48">JVP validation and scale checks</text>
  <rect class="box" x="72" y="92" width="285" height="210"/>
  <text class="label" x="104" y="132">Finite-difference cosine</text>
  <text class="title" x="104" y="190">{cos:.6f}</text>
  <path class="axis" d="M104 232 L320 232"/>
  <path class="arrow-local" d="M104 232 L318 232"/>
  <text class="small" x="104" y="268">Central difference, epsilon = 0.001</text>
  <rect class="box" x="403" y="92" width="285" height="210"/>
  <text class="label" x="435" y="132">Relative error</text>
  <text class="title" x="435" y="190">{err:.6f}</text>
  <path class="axis" d="M435 232 L650 232"/>
  <path class="arrow-red" d="M435 232 L480 232"/>
  <text class="small" x="435" y="268">Direction nearly perfect; magnitude differs slightly.</text>
  <rect class="box" x="155" y="332" width="450" height="72"/>
  <text class="small" x="188" y="362">||J_local v|| = {local_norm:.6f}</text>
  <text class="small" x="188" y="385">||E[J_i v]|| = {avg_norm:.6f}; local-vs-average cosine = 0.617840</text>
  <text class="tiny" x="90" y="436">Measured values copied from executed notebook section 10.</text>
</svg>
'''
    write("jspace_jvp_validation.svg", body)


def write_json() -> None:
    (FIGURE_DIR / "jspace_jvp_validation.json").write_text(
        json.dumps(METRICS, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    context_dependent_jacobian()
    jacobian_direction_transport()
    probe_space_vs_jspace()
    jspace_context_cloud()
    local_vs_average_jspace()
    jspace_bridge()
    jspace_jvp_validation()
    write_json()


if __name__ == "__main__":
    main()
