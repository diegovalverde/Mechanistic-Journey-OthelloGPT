# Mechanistic Journey Through Othello-GPT

## From Hidden Board States to Rule Circuits

This book follows a ten-chapter mechanistic interpretability investigation of Othello-GPT: a small GPT-2-style decoder-only transformer trained on Othello move sequences. The central puzzle is how a model trained only to predict moves can come to represent board state and use that representation when judging legal moves.

The book is for students, researchers, and engineers who want a careful path from linear probes to causal interventions, Jacobians, J-space, layer-level analysis, attribution, ablation, and candidate rule circuits. It avoids treating a probe result as the end of the story, and instead asks how information moves through the network.

The current research status is mixed by design: some findings are established, some have strong or moderate evidence, and some remain hypotheses. The book follows the real investigation and labels claims cautiously rather than turning unfinished work into a finished story. In particular, it does not claim a completed Othello legality circuit.

## Who This Is For

- Readers with basic machine learning background who want a concrete mechanistic interpretability case study.
- Students who want the math and experiments explained without losing the research details.
- Contributors who need a maintainable record of what has been tested, what is known, and what remains open.

## How to Read This Book

- Intuition-first: 1 -> 2 -> 3 -> 4 -> 7 -> 8 -> 9 -> 10.
- Full technical: read Chapters 1 through 10 in order.
- Experiment reproduction: 2 -> 4 -> 5 -> 7 -> 8 -> 9 -> [Notebook Walkthrough](appendix/notebook_walkthrough.md).
- Background-needed: use [Math Background](appendix/math_background.md) and [Transformer Background](appendix/transformer_background.md) as references when the notation or architecture becomes the bottleneck.

## What You Will Learn

- How board state can be decoded from residual-stream activations.
- Why probe accuracy alone does not prove a causal mechanism.
- How local Jacobians can approximate small residual interventions.
- How layer 7 and MLP7 emerged as important sites for legality-related evidence.
- What evidence would be needed before claiming a complete rule circuit.

## Conceptual Roadmap

This roadmap summarizes the investigation arc. It is not a proven circuit.

```mermaid
flowchart LR
  A[Move history] --> B[Hidden board representation]
  B --> C[Jacobian and J-space analysis]
  C --> D[Layer 7]
  D --> E[MLP7]
  E --> F[Candidate neurons]
  F --> G[Possible rule circuit]
```

## Current Research Status

The project supports strong evidence for board decodability, local semantic causal relevance, layer-7 capture-line enrichment, and MLP7 component importance. It supports weaker and mixed evidence at the individual-neuron rule-detector level. See the research pages for the findings snapshot, final evidence map, experiment index, open questions, and dated research log.
