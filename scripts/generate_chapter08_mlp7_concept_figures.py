#!/usr/bin/env python3
"""Generate conceptual Chapter 8 MLP7 figures.

These SVGs are schematic. They document architecture and evidence boundaries;
they are not measured experimental plots.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"

STYLE = """<style>
svg{color-scheme:light dark;--text:#172033;--muted:#5c667a;--grid:#d8dde8;--fill:#f7f9fc;--blue:#246bfe;--teal:#0f8f7f;--gold:#b98113;--red:#cc4b37}
@media(prefers-color-scheme:dark){svg{--text:#eef2ff;--muted:#b7bfd1;--grid:#384153;--fill:#1f2633;--blue:#86a9ff;--teal:#5fd4c7;--gold:#ffd27a;--red:#ff8f7d}}
.title{font:700 24px system-ui,-apple-system,Segoe UI,sans-serif;fill:var(--text)}.label{font:600 14px system-ui,-apple-system,Segoe UI,sans-serif;fill:var(--text)}.small{font:12px system-ui,-apple-system,Segoe UI,sans-serif;fill:var(--muted)}
.box{fill:var(--fill);stroke:var(--grid);stroke-width:1.3;rx:8}.node{fill:var(--fill);stroke:var(--grid);stroke-width:1.3}.line{stroke:var(--grid);stroke-width:2;fill:none;marker-end:url(#arrow)}.blue{stroke:var(--blue);stroke-width:3;fill:none;marker-end:url(#arrowBlue)}.teal{stroke:var(--teal);stroke-width:3;fill:none;marker-end:url(#arrowTeal)}.gold{stroke:var(--gold);stroke-width:3;fill:none;marker-end:url(#arrowGold)}.red{stroke:var(--red);stroke-width:3;fill:none;marker-end:url(#arrowRed)}.frontier{stroke:var(--red);stroke-width:2.4;stroke-dasharray:7 5}
</style>
<defs>
<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#5c667a"/></marker>
<marker id="arrowBlue" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#246bfe"/></marker>
<marker id="arrowTeal" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#0f8f7f"/></marker>
<marker id="arrowGold" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#b98113"/></marker>
<marker id="arrowRed" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#cc4b37"/></marker>
</defs>"""


def write(name: str, body: str) -> None:
    (FIG / name).write_text(body, encoding="utf-8")


def layer7_component_map() -> None:
    head_boxes = []
    for i in range(8):
        x = 70 + i * 82
        head_boxes.append(f'<rect class="box" x="{x}" y="130" width="62" height="42"/><text class="small" x="{x+31}" y="156" text-anchor="middle">L7H{i}</text>')
        head_boxes.append(f'<path class="blue" d="M{x+31} 172 L380 230"/>')
    body = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 780 460" role="img" aria-labelledby="title desc">
<title id="title">Layer 7 component map</title>
<desc id="desc">Candidate components inside the final transformer block.</desc>
{STYLE}
<text class="title" x="42" y="48">Layer 7: one block, nine obvious components</text>
<text class="small" x="42" y="72">Candidate components inside the final transformer block.</text>
<rect class="box" x="300" y="92" width="160" height="54"/><text class="label" x="380" y="125" text-anchor="middle">r_pre</text>
{''.join(head_boxes)}
<rect class="box" x="300" y="232" width="160" height="54"/><text class="label" x="380" y="265" text-anchor="middle">attention update</text>
<path class="line" d="M380 286 L380 326"/>
<rect class="box" x="300" y="326" width="160" height="54"/><text class="label" x="380" y="359" text-anchor="middle">r_mid</text>
<path class="teal" d="M460 353 L565 353"/>
<rect class="box" x="565" y="326" width="122" height="54"/><text class="label" x="626" y="359" text-anchor="middle">MLP7</text>
<path class="gold" d="M626 380 L380 410"/>
<rect class="box" x="300" y="390" width="160" height="42"/><text class="label" x="380" y="416" text-anchor="middle">r_post</text>
</svg>
'''
    write("layer7_component_map.svg", body)


def attention_to_mlp7_hypothesis() -> None:
    body = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 820 430" role="img" aria-labelledby="title desc">
<title id="title">Attention to MLP7 hypothesis</title>
<desc id="desc">A schematic hypothesis where layer-7 attention helps form the state that MLP7 reads.</desc>
{STYLE}
<text class="title" x="42" y="48">Attention may help prepare what MLP7 reads</text>
<text class="small" x="42" y="72">Architectural hypothesis -- component results do not yet establish this causal path.</text>
<rect class="box" x="72" y="170" width="150" height="70"/><text class="label" x="147" y="198" text-anchor="middle">earlier residual</text><text class="small" x="147" y="220" text-anchor="middle">layers 0-6</text>
<path class="blue" d="M222 205 L325 205"/>
<rect class="box" x="325" y="150" width="170" height="110"/><text class="label" x="410" y="184" text-anchor="middle">L7 attention heads</text><text class="small" x="410" y="210" text-anchor="middle">L7H0 ... L7H7</text><text class="small" x="410" y="232" text-anchor="middle">possible information routing</text>
<path class="teal" d="M495 205 L585 205"/>
<rect class="box" x="585" y="170" width="150" height="70"/><text class="label" x="660" y="198" text-anchor="middle">resid_mid</text><text class="small" x="660" y="220" text-anchor="middle">MLP7 input state</text>
<path class="gold" d="M660 240 L660 305"/>
<rect class="box" x="585" y="305" width="150" height="70"/><text class="label" x="660" y="333" text-anchor="middle">MLP7 write</text><text class="small" x="660" y="355" text-anchor="middle">legality-relevant?</text>
<path class="red" d="M585 340 L475 340"/>
<text class="small" x="210" y="340">large MLP7 effect does not imply attention is irrelevant</text>
</svg>
'''
    write("attention_to_mlp7_hypothesis.svg", body)


def evidence_ladder_mlp7() -> None:
    steps = [
        ("Behavior", "established"),
        ("Board decodability", "established"),
        ("Semantic intervention", "strong"),
        ("J-space transport", "moderate"),
        ("Layer-7 capture enrichment", "strong"),
        ("MLP7 component localization", "strong evidence frontier"),
        ("Candidate neuron subpopulation", "candidate frontier"),
        ("Relational selectivity / mediation", "not established"),
        ("Mechanistic circuit", "open"),
    ]
    rows = []
    for i, (label, note) in enumerate(steps):
        y = 80 + i * 42
        cls = "box"
        rows.append(f'<rect class="{cls}" x="210" y="{y}" width="360" height="30"/><text class="label" x="390" y="{y+20}" text-anchor="middle">{label}</text><text class="small" x="590" y="{y+20}">{note}</text>')
        if i < len(steps) - 1:
            rows.append(f'<path class="line" d="M390 {y+30} L390 {y+42}"/>')
    body = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 840 500" role="img" aria-labelledby="title desc">
<title id="title">Evidence ladder through MLP7</title>
<desc id="desc">Evidence ladder showing MLP7 component localization as the current strong evidence frontier.</desc>
{STYLE}
<text class="title" x="42" y="44">Evidence ladder after Chapter 8</text>
{''.join(rows)}
<line class="frontier" x1="178" y1="292" x2="700" y2="292"/><text class="small" x="42" y="296">strong evidence frontier</text>
<line class="frontier" x1="178" y1="334" x2="700" y2="334"/><text class="small" x="42" y="338">candidate frontier</text>
</svg>
'''
    write("evidence_ladder_mlp7.svg", body)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    layer7_component_map()
    attention_to_mlp7_hypothesis()
    evidence_ladder_mlp7()


if __name__ == "__main__":
    main()
