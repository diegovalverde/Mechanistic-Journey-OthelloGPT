#!/usr/bin/env python3
"""Generate conceptual Chapter 10 summary figures.

These figures are schematic synthesis figures. They contain no measured
experimental data and should be read as evidence maps, workflows, or future
experimental programs.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"

STYLE = """<style>
svg{color-scheme:light dark;--text:#172033;--muted:#5c667a;--grid:#d8dde8;--fill:#f7f9fc;--blue:#246bfe;--teal:#0f8f7f;--gold:#b98113;--red:#cc4b37;--green:#2f8f46;--purple:#6a4bc3}
@media(prefers-color-scheme:dark){svg{--text:#eef2ff;--muted:#b7bfd1;--grid:#384153;--fill:#1f2633;--blue:#86a9ff;--teal:#5fd4c7;--gold:#ffd27a;--red:#ff8f7d;--green:#7bd88f;--purple:#b9a5ff}}
.title{font:700 23px system-ui,-apple-system,Segoe UI,sans-serif;fill:var(--text)}
.label{font:600 13px system-ui,-apple-system,Segoe UI,sans-serif;fill:var(--text)}
.small{font:12px system-ui,-apple-system,Segoe UI,sans-serif;fill:var(--muted)}
.tiny{font:11px system-ui,-apple-system,Segoe UI,sans-serif;fill:var(--muted)}
.box{fill:var(--fill);stroke:var(--grid);stroke-width:1.25;rx:8}
.pill{fill:var(--fill);stroke:var(--grid);stroke-width:1.2;rx:16}
.line{stroke:var(--grid);stroke-width:2;fill:none;marker-end:url(#arrow)}
.teal{stroke:var(--teal);stroke-width:3;fill:none;marker-end:url(#arrowTeal)}
.blue{stroke:var(--blue);stroke-width:3;fill:none;marker-end:url(#arrowBlue)}
.gold{stroke:var(--gold);stroke-width:3;fill:none;marker-end:url(#arrowGold)}
.red{stroke:var(--red);stroke-width:2.5;fill:none;marker-end:url(#arrowRed)}
.dash{stroke:var(--muted);stroke-width:2;stroke-dasharray:7 5;fill:none;marker-end:url(#arrow)}
.gate{fill:#fff8e5;stroke:var(--gold);stroke-width:1.2;rx:8}
@media(prefers-color-scheme:dark){.gate{fill:#332b1b}}
</style>
<defs>
<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#5c667a"/></marker>
<marker id="arrowTeal" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#0f8f7f"/></marker>
<marker id="arrowBlue" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#246bfe"/></marker>
<marker id="arrowGold" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#b98113"/></marker>
<marker id="arrowRed" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#cc4b37"/></marker>
</defs>"""


def write(name: str, body: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    (FIG / name).write_text(body, encoding="utf-8")


def mechanistic_workflow() -> None:
    steps = [
        ("Known latent variable", "B_t, legal moves, capture rays"),
        ("Probe", "Can we decode it?"),
        ("Semantic direction", "Can we get a handle?"),
        ("Causal intervention", "Does changing it matter?"),
        ("Jacobian / J-space", "Can local geometry predict transport?"),
        ("Task-specific score", "What computation are we isolating?"),
        ("Layer localization", "Where does the effect emerge?"),
        ("Component localization", "Which parts matter?"),
        ("Population characterization", "What do candidates read and write?"),
        ("Mediation / rescue", "Does the proposed path survive controls?"),
        ("Mechanistic claim", "Match claim strength to evidence"),
    ]
    rows = []
    for i, (label, note) in enumerate(steps):
        y = 72 + i * 50
        rows.append(
            f'<rect class="box" x="238" y="{y}" width="330" height="34"/>'
            f'<text class="label" x="403" y="{y+15}" text-anchor="middle">{label}</text>'
            f'<text class="tiny" x="403" y="{y+29}" text-anchor="middle">{note}</text>'
        )
        if i < len(steps) - 1:
            rows.append(f'<path class="line" d="M403 {y+34} L403 {y+50}"/>')
        if i in {0, 2, 4, 7, 9}:
            gates = {
                0: "Can we decode it?",
                2: "Does changing it matter?",
                4: "Where does its effect emerge?",
                7: "Which components matter?",
                9: "Survive controls?",
            }
            rows.append(
                f'<rect class="gate" x="590" y="{y+38}" width="170" height="28"/>'
                f'<text class="tiny" x="675" y="{y+56}" text-anchor="middle">{gates[i]}</text>'
            )
    body = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 820 660" role="img" aria-labelledby="title desc">
<title id="title">Mechanistic workflow</title>
<desc id="desc">A staged evidence workflow from known latent variables to mechanistic claims.</desc>
{STYLE}
<text class="title" x="42" y="42">A workflow, not a slogan</text>
<text class="small" x="42" y="64">Major transitions require evidence gates. The final claim should not be stronger than the weakest unsupported gate.</text>
{''.join(rows)}
</svg>
'''
    write("mechanistic_workflow.svg", body)


def why_othello_is_special() -> None:
    items = [
        ("Exact latent state", "board after every prefix"),
        ("Exact dynamics", "deterministic simulator"),
        ("Exact rules", "legal moves and capture rays"),
        ("Counterfactuals", "known board edits"),
        ("Dense labels", "64 square states per position"),
        ("Small model", "8 blocks, d_model 512"),
        ("Reproducible hooks", "probes, gradients, ablations"),
    ]
    boxes = []
    positions = [(56, 116), (300, 116), (544, 116), (56, 244), (300, 244), (544, 244), (300, 372)]
    for (label, note), (x, y) in zip(items, positions):
        boxes.append(
            f'<rect class="box" x="{x}" y="{y}" width="220" height="76"/>'
            f'<text class="label" x="{x+110}" y="{y+30}" text-anchor="middle">{label}</text>'
            f'<text class="small" x="{x+110}" y="{y+54}" text-anchor="middle">{note}</text>'
        )
    body = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 820 500" role="img" aria-labelledby="title desc">
<title id="title">Why Othello is special</title>
<desc id="desc">Seven properties that make Othello-GPT unusually favorable for mechanistic interpretability.</desc>
{STYLE}
<text class="title" x="42" y="50">Why Othello-GPT was unusually favorable</text>
<text class="small" x="42" y="74">The clean external state makes the investigation falsifiable. It does not make the internal mechanism simple.</text>
{''.join(boxes)}
</svg>
'''
    write("why_othello_is_special.svg", body)


def ground_truth_spectrum() -> None:
    body = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 860 430" role="img" aria-labelledby="title desc">
<title id="title">Ground truth spectrum</title>
<desc id="desc">A conceptual spectrum from formal environments to open-ended natural language.</desc>
{STYLE}
<text class="title" x="42" y="50">Ground truth becomes harder as worlds become richer</text>
<text class="small" x="42" y="74">This is a conceptual spectrum, not a strict one-dimensional law.</text>
<line x1="96" y1="300" x2="760" y2="300" stroke="var(--grid)" stroke-width="3"/>
<path class="blue" d="M120 318 L120 128"/><text class="small" x="100" y="342">higher latent-state certainty</text>
<path class="teal" d="M118 318 L732 318"/><text class="small" x="570" y="350">greater ecological realism</text>
<rect class="box" x="74" y="150" width="220" height="110"/><text class="label" x="184" y="178" text-anchor="middle">formal / synthetic</text><text class="small" x="184" y="204" text-anchor="middle">Othello</text><text class="small" x="184" y="225" text-anchor="middle">finite-state tasks</text><text class="small" x="184" y="246" text-anchor="middle">code interpreter</text>
<rect class="box" x="320" y="140" width="220" height="120"/><text class="label" x="430" y="168" text-anchor="middle">simulated worlds</text><text class="small" x="430" y="194" text-anchor="middle">navigation</text><text class="small" x="430" y="215" text-anchor="middle">robotics simulators</text><text class="small" x="430" y="236" text-anchor="middle">physics engines</text>
<rect class="box" x="566" y="128" width="220" height="132"/><text class="label" x="676" y="156" text-anchor="middle">open-ended language</text><text class="small" x="676" y="182" text-anchor="middle">conversation</text><text class="small" x="676" y="203" text-anchor="middle">beliefs and intent</text><text class="small" x="676" y="224" text-anchor="middle">world knowledge</text><text class="small" x="676" y="245" text-anchor="middle">theory-dependent labels</text>
</svg>
'''
    write("ground_truth_spectrum.svg", body)


def mechanistic_roles() -> None:
    body = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 820 420" role="img" aria-labelledby="title desc">
<title id="title">Mechanistic roles</title>
<desc id="desc">Components can be interpreted by what they read, route, transform, and write.</desc>
{STYLE}
<text class="title" x="42" y="50">Four roles a component or population can play</text>
<text class="small" x="42" y="74">The roles can overlap. They are analysis handles, not a claim of strict modularity.</text>
<rect class="box" x="66" y="165" width="138" height="84"/><text class="label" x="135" y="197" text-anchor="middle">read</text><text class="small" x="135" y="222" text-anchor="middle">input geometry</text>
<path class="blue" d="M204 207 L286 207"/>
<rect class="box" x="286" y="165" width="138" height="84"/><text class="label" x="355" y="197" text-anchor="middle">route</text><text class="small" x="355" y="222" text-anchor="middle">information path</text>
<path class="teal" d="M424 207 L506 207"/>
<rect class="box" x="506" y="165" width="138" height="84"/><text class="label" x="575" y="197" text-anchor="middle">transform</text><text class="small" x="575" y="222" text-anchor="middle">nonlinear map</text>
<path class="gold" d="M644 207 L716 207"/>
<rect class="box" x="686" y="165" width="96" height="84"/><text class="label" x="734" y="197" text-anchor="middle">write</text><text class="small" x="734" y="222" text-anchor="middle">output dir.</text>
<rect class="pill" x="236" y="300" width="350" height="44"/><text class="label" x="411" y="328" text-anchor="middle">population behavior can be the interpretable scale</text>
</svg>
'''
    write("mechanistic_roles.svg", body)


def othello_next_experiments() -> None:
    body = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 840 500" role="img" aria-labelledby="title desc">
<title id="title">Othello next experiments</title>
<desc id="desc">Open experimental program after the current Othello-GPT evidence frontier.</desc>
{STYLE}
<text class="title" x="42" y="50">Open experimental program for Othello-GPT</text>
<text class="small" x="42" y="74">Current evidence narrows the search. The dashed arrows mark tests not yet completed.</text>
<rect class="box" x="70" y="190" width="130" height="60"/><text class="label" x="135" y="214" text-anchor="middle">board</text><text class="small" x="135" y="235" text-anchor="middle">strong</text>
<path class="teal" d="M200 220 L285 220"/>
<rect class="box" x="285" y="190" width="130" height="60"/><text class="label" x="350" y="214" text-anchor="middle">layer 7</text><text class="small" x="350" y="235" text-anchor="middle">strong</text>
<path class="teal" d="M415 220 L500 220"/>
<rect class="box" x="500" y="190" width="130" height="60"/><text class="label" x="565" y="214" text-anchor="middle">MLP7</text><text class="small" x="565" y="235" text-anchor="middle">strong</text>
<path class="teal" d="M630 220 L715 220"/>
<rect class="box" x="715" y="190" width="90" height="60"/><text class="label" x="760" y="214" text-anchor="middle">candidates</text><text class="small" x="760" y="235" text-anchor="middle">mixed</text>
<path class="dash" d="M350 190 C350 118 450 118 500 178"/><text class="small" x="363" y="124">attention path?</text>
<path class="dash" d="M760 190 C744 116 642 112 592 178"/><text class="small" x="640" y="112">population subspace?</text>
<path class="dash" d="M760 250 C748 328 646 336 584 262"/><text class="small" x="636" y="356">mediation?</text>
<path class="dash" d="M565 250 C500 365 310 356 203 250"/><text class="small" x="322" y="386">rescue?</text>
<path class="dash" d="M135 250 C150 395 705 402 760 250"/><text class="small" x="340" y="430">generalization across rays, lengths, phases?</text>
</svg>
'''
    write("othello_next_experiments.svg", body)


def map_to_mechanism() -> None:
    rows = [
        ("Map", "feature X is decodable here"),
        ("Sensitivity map", "changing direction X locally affects Y"),
        ("Localization", "component C matters under intervention"),
        ("Mechanism", "representations flow through operations to produce behavior"),
    ]
    items = []
    for i, (name, note) in enumerate(rows):
        y = 112 + i * 74
        items.append(
            f'<rect class="box" x="108" y="{y}" width="604" height="48"/>'
            f'<text class="label" x="160" y="{y+29}">{name}</text>'
            f'<text class="small" x="350" y="{y+29}">{note}</text>'
        )
        if i < len(rows) - 1:
            items.append(f'<path class="line" d="M410 {y+48} L410 {y+74}"/>')
    body = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 820 460" role="img" aria-labelledby="title desc">
<title id="title">From map to mechanism</title>
<desc id="desc">A conceptual hierarchy from decodability maps to mechanistic explanations.</desc>
{STYLE}
<text class="title" x="42" y="50">The difference between a map and a mechanism</text>
<text class="small" x="42" y="74">Each level is useful. The language of the claim should match the level reached.</text>
{''.join(items)}
</svg>
'''
    write("map_to_mechanism.svg", body)


def main() -> None:
    mechanistic_workflow()
    why_othello_is_special()
    ground_truth_spectrum()
    mechanistic_roles()
    othello_next_experiments()
    map_to_mechanism()


if __name__ == "__main__":
    main()
