# Mechanistic Journey Through Othello-GPT

## From Hidden Board States to Rule Circuits

This book follows a mechanistic interpretability investigation of Othello-GPT: a small GPT-2-style decoder-only transformer trained on Othello move sequences. The central puzzle is how a model trained only to predict moves can come to represent board state and use that representation when judging legal moves.

The book is for students, researchers, and engineers who want a careful path from linear probes to causal interventions, Jacobians, layer-level analysis, and candidate rule circuits. It avoids treating a probe result as the end of the story, and instead asks how information moves through the network.

The current research status is mixed by design: some findings are established, some have strong or moderate evidence, and some remain hypotheses. The book follows the real investigation and labels claims cautiously rather than turning unfinished work into a finished story.

## Who This Is For

- Readers with basic machine learning background who want a concrete mechanistic interpretability case study.
- Students who want the math and experiments explained without losing the research details.
- Contributors who need a maintainable record of what has been tested, what is known, and what remains open.

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

The project currently supports moderate evidence for rule-structured computation in Othello-GPT, especially around layer 7 and MLP7. It does not yet prove a complete legality circuit. See the research pages for the current findings, experiment index, open questions, and dated research log.

