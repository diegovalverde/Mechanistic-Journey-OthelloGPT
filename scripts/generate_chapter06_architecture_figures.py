#!/usr/bin/env python3
"""Generate Chapter 6 conceptual architecture figures.

These figures are not measured experimental plots. They document the verified
Othello-GPT architecture and the TransformerLens hook locations used by this
project.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "docs" / "figures"

STYLE = """<style>
  svg {
    color-scheme: light dark;
    --text: #172033;
    --muted: #5e687c;
    --line: #243045;
    --grid: #d7dde8;
    --fill: #f7f9fc;
    --blue: #246bfe;
    --teal: #0f8f7f;
    --gold: #b98113;
    --red: #cc4b37;
  }
  @media (prefers-color-scheme: dark) {
    svg {
      --text: #eef2ff;
      --muted: #b8c1d3;
      --line: #dce3f7;
      --grid: #3b4558;
      --fill: #1f2633;
      --blue: #86a9ff;
      --teal: #5fd4c7;
      --gold: #ffd27a;
      --red: #ff8f7d;
    }
  }
  .title { font: 700 24px system-ui, -apple-system, Segoe UI, sans-serif; fill: var(--text); }
  .label { font: 600 15px system-ui, -apple-system, Segoe UI, sans-serif; fill: var(--text); }
  .small { font: 13px system-ui, -apple-system, Segoe UI, sans-serif; fill: var(--muted); }
  .tiny { font: 11px system-ui, -apple-system, Segoe UI, sans-serif; fill: var(--muted); }
  .box { fill: var(--fill); stroke: var(--grid); stroke-width: 1.4; rx: 8; }
  .pill { fill: transparent; stroke: var(--grid); stroke-width: 1.4; rx: 18; }
  .line { stroke: var(--line); stroke-width: 2.2; fill: none; }
  .dash { stroke: var(--grid); stroke-width: 1.6; stroke-dasharray: 6 6; fill: none; }
  .arrow { stroke: var(--line); stroke-width: 2.2; fill: none; marker-end: url(#arrowLine); }
  .arrow-blue { stroke: var(--blue); stroke-width: 3; fill: none; marker-end: url(#arrowBlue); }
  .arrow-teal { stroke: var(--teal); stroke-width: 3; fill: none; marker-end: url(#arrowTeal); }
  .arrow-gold { stroke: var(--gold); stroke-width: 3; fill: none; marker-end: url(#arrowGold); }
  .arrow-red { stroke: var(--red); stroke-width: 3; fill: none; marker-end: url(#arrowRed); }
  .node { fill: var(--fill); stroke: var(--line); stroke-width: 1.6; }
</style>"""

DEFS = """<defs>
  <marker id="arrowLine" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#243045"/></marker>
  <marker id="arrowBlue" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#246bfe"/></marker>
  <marker id="arrowTeal" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#0f8f7f"/></marker>
  <marker id="arrowGold" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#b98113"/></marker>
  <marker id="arrowRed" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#cc4b37"/></marker>
</defs>"""


def start(width: int, height: int, title: str, desc: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'  <title id="title">{title}</title>',
        f'  <desc id="desc">{desc}</desc>',
        f"  {STYLE}",
        f"  {DEFS}",
    ]


def finish(name: str, parts: list[str]) -> None:
    parts.append("</svg>\n")
    (FIGURE_DIR / name).write_text("\n".join(parts), encoding="utf-8")


def rect(parts: list[str], x: int, y: int, w: int, h: int, label: str, sub: str = "") -> None:
    parts.append(f'  <rect class="box" x="{x}" y="{y}" width="{w}" height="{h}"/>')
    parts.append(f'  <text class="label" x="{x + w / 2}" y="{y + 28}" text-anchor="middle">{label}</text>')
    if sub:
        parts.append(f'  <text class="small" x="{x + w / 2}" y="{y + 50}" text-anchor="middle">{sub}</text>')


def arrow(parts: list[str], x1: int, y1: int, x2: int, y2: int, cls: str = "arrow") -> None:
    parts.append(f'  <path class="{cls}" d="M{x1} {y1} L{x2} {y2}"/>')


def othello_transformer_block() -> None:
    p = start(760, 780, "One Othello-GPT transformer block", "Pre-norm attention and MLP block with residual additions and TransformerLens hooks.")
    p.append('  <text class="title" x="42" y="44">One Othello-GPT block</text>')
    p.append('  <text class="small" x="42" y="70">512-D residual stream, block l</text>')
    ys = [105, 185, 265, 350, 430, 510, 595]
    items = [
        ("r_pre", "hook_resid_pre"),
        ("LNPre", "attention input normalization"),
        ("attention", "8 heads x 64"),
        ("attn_out", "hook_attn_out"),
        ("r_mid = r_pre + attn_out", "hook_resid_mid"),
        ("MLP after LNPre", "512 -> 2048 -> 512"),
        ("mlp_out", "hook_mlp_out"),
    ]
    for y, (label, sub) in zip(ys, items):
        rect(p, 250, y, 260, 55, label, sub)
    rect(p, 250, 675, 260, 55, "r_post = r_mid + mlp_out", "hook_resid_post")
    for y1, y2 in zip([160, 240, 320, 405, 485, 565, 650], [185, 265, 350, 430, 510, 595, 675]):
        arrow(p, 380, y1, 380, y2)
    arrow(p, 250, 132, 185, 132, "arrow-blue")
    arrow(p, 185, 132, 185, 378, "arrow-blue")
    arrow(p, 185, 378, 250, 378, "arrow-blue")
    p.append('  <text class="small" x="86" y="126">residual add</text>')
    arrow(p, 510, 458, 585, 458, "arrow-teal")
    arrow(p, 585, 458, 585, 702, "arrow-teal")
    arrow(p, 585, 702, 510, 702, "arrow-teal")
    p.append('  <text class="small" x="590" y="451">residual add</text>')
    p.append('  <text class="label" x="84" y="748">r_post(layer l) = r_pre(layer l+1)</text>')
    finish("othello_transformer_block.svg", p)


def residual_stream_sum() -> None:
    p = start(820, 430, "Residual stream as a sum", "Successive attention and MLP outputs are written into a common residual stream.")
    p.append('  <text class="title" x="42" y="44">Residual stream as accumulating updates</text>')
    p.append('  <rect class="box" x="80" y="185" width="660" height="70"/>')
    p.append('  <text class="label" x="410" y="225" text-anchor="middle">common 512-D residual channel</text>')
    labels = [("embed + pos", 120, "arrow-blue"), ("a0", 220, "arrow-teal"), ("m0", 300, "arrow-gold"), ("a1", 390, "arrow-teal"), ("m1", 470, "arrow-gold"), ("...", 555, "arrow-blue"), ("final", 650, "arrow-red")]
    for label, x, cls in labels:
        arrow(p, x, 100, x, 185, cls)
        p.append(f'  <text class="small" x="{x}" y="88" text-anchor="middle">{label}</text>')
    p.append('  <text class="label" x="96" y="320">schematically: r_final = r_initial + sum_l a_l + sum_l m_l</text>')
    p.append('  <text class="small" x="96" y="348">The outputs add in one space; the computations that produce them are sequential and nonlinear.</text>')
    finish("residual_stream_sum.svg", p)


def attention_moves_information() -> None:
    p = start(820, 430, "Attention can move information across positions", "Earlier move positions feed the current position through attention arrows.")
    p.append('  <text class="title" x="42" y="44">Attention can mix positions</text>')
    moves = ["D3", "C3", "B3", "B2", "B1", "A1", "current"]
    xs = [90, 200, 310, 420, 530, 640, 750]
    for x, m in zip(xs, moves):
        p.append(f'  <circle class="node" cx="{x}" cy="240" r="34"/>')
        p.append(f'  <text class="label" x="{x}" y="246" text-anchor="middle">{m}</text>')
        p.append(f'  <text class="tiny" x="{x}" y="298" text-anchor="middle">512-D row</text>')
    for x, cls in [(90, "arrow-blue"), (200, "arrow-teal"), (310, "arrow-gold"), (530, "arrow-red"), (640, "arrow-blue")]:
        p.append(f'  <path class="{cls}" d="M{x} 205 C{x + 70} 95, 690 95, 750 205"/>')
    p.append('  <text class="small" x="74" y="365">Architectural possibility: earlier positions can write information into the current residual position.</text>')
    p.append('  <text class="small" x="74" y="390">This is not a discovered Othello circuit.</text>')
    finish("attention_moves_information.svg", p)


def mlp_neuron_read_write() -> None:
    p = start(820, 440, "MLP neuron read gate write view", "One MLP neuron reads a direction, passes through GELU, and writes a vector back.")
    p.append('  <text class="title" x="42" y="44">One MLP neuron: detect, gate, write</text>')
    rect(p, 70, 180, 170, 70, "input x", "512-D")
    rect(p, 325, 180, 170, 70, "GELU(p_j)", "scalar activation")
    rect(p, 580, 180, 170, 70, "output vector", "a_j W_out[j]")
    arrow(p, 240, 215, 325, 215, "arrow-blue")
    arrow(p, 495, 215, 580, 215, "arrow-teal")
    p.append('  <text class="small" x="258" y="196">p_j = x W_in[:,j] + b_j</text>')
    p.append('  <text class="small" x="104" y="330">input direction: what activates?</text>')
    p.append('  <text class="small" x="336" y="330">nonlinearity: how much?</text>')
    p.append('  <text class="small" x="580" y="330">write direction: what is added?</text>')
    p.append('  <text class="small" x="78" y="385">Useful intuition, not a guarantee that one neuron equals one human-readable concept.</text>')
    finish("mlp_neuron_read_write.svg", p)


def one_block_information_flow() -> None:
    p = start(820, 520, "Information flow through one block", "Current residual state is updated by attention and then by a position-wise MLP.")
    p.append('  <text class="title" x="42" y="44">Information flow through one block</text>')
    rect(p, 70, 210, 160, 70, "move history", "token positions")
    rect(p, 325, 95, 180, 70, "attention", "mix positions")
    rect(p, 325, 325, 180, 70, "MLP", "position-wise")
    rect(p, 590, 210, 170, 70, "new residual", "512-D")
    rect(p, 300, 210, 230, 70, "current residual", "old + attn update")
    arrow(p, 230, 245, 300, 245, "arrow-blue")
    p.append('  <path class="arrow-teal" d="M185 210 C220 120, 280 115, 325 130"/>')
    p.append('  <path class="arrow-teal" d="M505 130 C575 145, 595 190, 530 225"/>')
    arrow(p, 415, 280, 415, 325, "arrow-gold")
    p.append('  <path class="arrow-red" d="M505 360 C590 350, 615 310, 620 280"/>')
    arrow(p, 530, 245, 590, 245)
    p.append('  <text class="small" x="78" y="455">Attention can bring earlier-position information into the current row; the MLP transforms the row it receives.</text>')
    finish("one_block_information_flow.svg", p)


def transformerlens_hook_map() -> None:
    p = start(820, 520, "TransformerLens hook map", "Hook points in a single pre-norm Othello-GPT block.")
    p.append('  <text class="title" x="42" y="44">TransformerLens hook map</text>')
    rows = [
        ("hook_resid_pre", "residual before attention"),
        ("hook_attn_out", "attention update added to residual"),
        ("hook_resid_mid", "residual after attention add"),
        ("hook_mlp_out", "MLP update added to residual"),
        ("hook_resid_post", "residual after full block"),
    ]
    y = 105
    for name, meaning in rows:
        rect(p, 80, y, 230, 55, name, "")
        p.append(f'  <text class="small" x="345" y="{y + 34}">{meaning}</text>')
        p.append(f'  <text class="tiny" x="675" y="{y + 34}" text-anchor="middle">[batch, pos, 512]</text>')
        y += 75
    p.append('  <text class="small" x="88" y="478">Hooks are measurement and intervention points. They are not extra components in ordinary model inference.</text>')
    finish("transformerlens_hook_map.svg", p)


def attribution_vs_ablation() -> None:
    p = start(820, 460, "Attribution versus ablation", "Attribution compares component outputs with gradients; ablation reruns after removing or replacing a component.")
    p.append('  <text class="title" x="42" y="44">Attribution and ablation ask different questions</text>')
    rect(p, 70, 120, 290, 210, "Attribution", "")
    p.append('  <text class="small" x="105" y="185">current component output c</text>')
    p.append('  <text class="small" x="105" y="220">downstream sensitivity g</text>')
    p.append('  <text class="label" x="105" y="265">score: g^T c</text>')
    p.append('  <text class="small" x="105" y="295">cheap local ranking signal</text>')
    rect(p, 460, 120, 290, 210, "Ablation", "")
    p.append('  <text class="small" x="495" y="185">remove, replace, or patch c</text>')
    p.append('  <text class="small" x="495" y="220">rerun downstream model</text>')
    p.append('  <text class="label" x="495" y="265">measure output change</text>')
    p.append('  <text class="small" x="495" y="295">interventional but baseline-dependent</text>')
    p.append('  <text class="small" x="92" y="390">Neither one alone identifies a complete algorithm.</text>')
    finish("attribution_vs_ablation.svg", p)


def layer_sweep_question() -> None:
    p = start(880, 430, "Layer sweep question", "Eight neutral transformer blocks with question marks at candidate checkpoints.")
    p.append('  <text class="title" x="42" y="44">Where does legality-relevant structure emerge?</text>')
    x0 = 70
    for i in range(8):
        x = x0 + i * 95
        rect(p, x, 170, 70, 62, str(i), "block")
        if i < 7:
            arrow(p, x + 70, 201, x + 95, 201)
    for i, text in [(2, "decodable?"), (4, "board state?"), (6, "sensitivity?"), (7, "rule evidence?")]:
        x = x0 + i * 95 + 35
        p.append(f'  <text class="title" x="{x}" y="132" text-anchor="middle">?</text>')
        p.append(f'  <text class="tiny" x="{x}" y="260" text-anchor="middle">{text}</text>')
    p.append('  <text class="label" x="86" y="335">Questions to test, not measured results.</text>')
    p.append('  <text class="small" x="86" y="365">All layers are shown neutrally here; Chapter 7 performs the sweep.</text>')
    finish("layer_sweep_question.svg", p)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    othello_transformer_block()
    residual_stream_sum()
    attention_moves_information()
    mlp_neuron_read_write()
    one_block_information_flow()
    transformerlens_hook_map()
    attribution_vs_ablation()
    layer_sweep_question()


if __name__ == "__main__":
    main()
