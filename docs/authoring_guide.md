# Authoring Guide

This guide preserves the voice and evidence discipline for *Mechanistic Journey Through Othello-GPT*. It is a working contract for future chapters, not a style ornament.

## Audience

Write for advanced undergraduate and graduate ML students, engineers, and technically curious readers.

Assume readers know basic vectors, probability, and neural networks.

Do not assume they already know mechanistic interpretability, probes, residual streams, Jacobians, or J-space. These ideas should become necessary before they become formal.

## Tone

The voice should be clear, curious, technically rigorous, and modest about evidence. It can be conversational, but it should not become chatty. Avoid hype. Use original prose. Prefer a precise sentence over an impressive one.

The book should sound like an investigation unfolding in front of the reader. The goal is not to make Othello-GPT seem magical. The goal is to make the evidence legible.

## Central Pedagogical Principle

Intuition creates the need for the equation; the equation should not arrive before the question it answers.

When a chapter introduces notation, the reader should already feel the pressure that the notation resolves. If an equation appears, explain immediately what it says in ordinary language and why it helps.

## Recurring Chapter Arc

Most chapters should follow this logic:

```text
Question
-> intuition
-> minimal formalism
-> experiment
-> result
-> interpretation
-> limitation
-> next mystery
```

The chapter does not need literal headings with those names. The narrative should still move through that sequence. A reader should know what question we are asking, why the tool is appropriate, what was measured, what happened, what we can conclude, what we cannot conclude, and why the next chapter is necessary.

## Mathematics Conventions

Motivate variables before defining them. Build complicated equations from simpler ones. Explain every important equation immediately after it appears.

Show dimensions when dimensions clarify intuition. For example, if a probe weight has shape `[64, 3, 512]`, say what each axis means: board square, board-state class, and residual-stream direction.

Use concrete Othello examples whenever possible. A variable like \(B_t\) should connect back to a board after a sequence such as `C4 C3 D3`, not float as abstract notation.

Do not use math to create authority. Use it to remove ambiguity.

## Scientific Language

Keep these claims distinct:

- behavior
- representation
- decodability
- correlation
- local causal relevance
- component causal importance
- candidate mechanism
- mechanistic explanation

The distinctions matter. A model that predicts legal moves behaves as if it has useful board information. A probe that recovers the board shows decodability. A residual intervention that changes logits in the predicted direction gives local causal evidence. A component ablation shows component importance. None of these, by itself, establishes a complete mechanism.

Use "represents" only when supported by evidence. Use "uses" only when causal evidence supports it.

Never describe a neuron as implementing a symbolic rule unless experiments actually establish that claim. Current evidence supports candidate-neuron participation in a distributed legality computation, not a clean single-neuron Othello rule implementation.

## Evidence Flow

Scientific claims must follow the project provenance rule:

```text
executed notebook
    -> research memory
    -> book prose
```

If a claim is not in the executed notebook or research memory, write it as a hypothesis, limitation, or open question. Do not strengthen the prose beyond the evidence because the story would read better.

## Figures

Figures should be pedagogical, not decorative.

Use SVG for conceptual diagrams where possible. Use PNG only for plots, screenshots, or raster outputs when needed. Experimental figures must trace to notebook data. Conceptual diagrams should clearly distinguish hypotheses from established mechanisms.

Support light and dark mode where feasible by using transparent backgrounds and theme-tolerant colors.

Every figure should help the reader answer a question that the prose has already created.

## Pause-and-Think Boxes

Use pause-and-think boxes sparingly. They are most useful when readers can predict a result, notice a conceptual trap, or test whether they are keeping two claims separate.

Do not use them as decorative interludes.

## Exercises

Prefer conceptual reasoning and small reproducible experiments over memorization.

Good exercises ask readers to replay a move prefix, compute a board label, train or evaluate a tiny probe, compare a probe result with an intervention result, or explain what a claim does and does not establish.

## Terminology

Maintain consistent definitions for these terms:

- world model: an internal representation that tracks aspects of a latent environment useful for prediction or action.
- latent state: the hidden state of the environment that generates the observations; in Othello, the board after a move prefix.
- activation: a vector produced inside the model during a forward pass.
- residual stream: the running vector state in a Transformer to which attention and MLP components add updates.
- probe: an externally trained readout that tests whether information is decodable from activations.
- semantic direction: a direction in activation space associated with an interpretable feature, such as mine-vs-theirs for a square, under a specific operational construction.
- logit: an unnormalized score for an output token before softmax.
- causal intervention: a controlled change to an activation, component, or direction used to test whether outputs change as predicted.
- Jacobian: a local linear map of how a small activation change affects later quantities.
- J-space: the space of directions after transformation by a Jacobian, used here to reason about local effects on downstream representations or logits.

Definitions may become more precise as chapters progress, but they should not silently shift.
