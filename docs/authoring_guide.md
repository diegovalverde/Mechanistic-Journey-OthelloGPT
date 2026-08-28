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

Relational decodability is distinct from rule-sensitive causal use. A probe may recover a multi-square relation such as a directional capture predicate before the evidence shows that the model uses that probe direction to compute legality.

A later layer can be computationally important even when linear decodability does not improve. Downstream computation may transform, route, mix, or decision-align an already-decodable relation rather than making it easier for the same kind of probe to read.

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
- \(r_\text{pre}\): residual stream entering a transformer block, corresponding to `hook_resid_pre`.
- `attn_out`: the attention update written to the residual stream, corresponding to `hook_attn_out`.
- \(r_\text{mid}\): residual stream after the attention update has been added, corresponding to `hook_resid_mid`.
- `mlp_out`: the MLP update written to the residual stream, corresponding to `hook_mlp_out`.
- \(r_\text{post}\): residual stream after the full transformer block, corresponding to `hook_resid_post`.

Definitions may become more precise as chapters progress, but they should not silently shift.

Layer numbers for Othello-GPT are zero-based. The model has eight transformer blocks indexed 0 through 7. Layer 7 is the eighth and final transformer block. When ambiguity matters, write "layer 4 (zero-based block index 4)" or equivalent.

Attention patterns are hypothesis-generating evidence, not causal proof. Do not describe an attention pattern as a circuit unless causal tests establish that the relevant head output matters and carries the claimed information.

Component attribution and component ablation are different evidence types. Attribution compares current component outputs with local downstream sensitivities. Ablation changes or replaces a component and reruns the model. Do not describe them interchangeably.

For Chapter 8 component attribution, \(A_c = g^\top c\) compares a layer-7 component write with the legality-gradient direction. It is a local alignment measure, not a causal intervention.

For Chapter 8 whole-component ablation, the sign convention is `delta_legality_contrast = L_ablate - L_clean`; negative values mean the replacement reduced the selected move's legality contrast.

For Chapter 8 MLP7 neuron-group ablation, the notebook reports `legality_degradation = L_clean - L_ablate`; negative values mean the mean-replacement intervention increased the measured legality contrast. When discussing those results, focus on separation from random same-size groups unless the sign convention is explicitly explained.

High MLP7 neuron attribution or a top-k group effect supports candidate-neuron participation. It does not establish that a neuron detects a capture rule, writes legality selectively, or implements the Othello algorithm.

## Jacobian Notation

Use the following notation consistently after Chapter 4:

- \(h\): the current residual-stream activation at a specified layer and token position.
- \(\delta h\): a small additive residual-stream edit.
- \(v\): a direction in residual space, often a normalized semantic direction from a probe.
- \(\alpha\) or \(\epsilon\): a scalar perturbation strength. Prefer \(\alpha\) for semantic interventions and \(\epsilon\) for generic finite-difference checks, unless local context makes the other choice clearer.
- \(z_m(h)\): one selected output logit, usually the logit for move \(m\), as a function of the chosen residual activation and the downstream computation.
- \(z(h)\): the vector of output logits produced from the chosen residual activation.
- \(\nabla_h z_m\): the gradient of one scalar logit with respect to the residual activation, with shape `[d_model]`.
- \(J(h) = \partial z / \partial h\): the local Jacobian from residual activation to output logits, with shape `[d_vocab_out, d_model]` for logit-space analyses.
- \(Jv\) or \(J(h)v\): a Jacobian-vector product, interpreted as the first-order predicted change in all output logits from moving along direction \(v\).

Always state the layer, hook, token position, and output being differentiated when reporting an experimental Jacobian result. For Othello-GPT Chapter 4 results, the verified dimensions are `d_model = 512` and `d_vocab_out = 61`.

For hidden-state transport analyses after Chapter 5, state the source representation and the target representation explicitly. Do not silently reuse logit-space Jacobian notation when the actual function maps from one residual-stream state to a later residual-stream state. It is acceptable to write \(J_x v\) for the local image of a source-space direction \(v\), but the prose must specify the context \(x\), source hook, source position, target hook or target representation, target position, and vector dimension.

Use \(\mathbb{E}_x[J_x v]\), \(\bar{J}v\), or \(\bar{v}_J\) only when the averaging procedure has been specified. If the implementation averages JVPs rather than materializing full Jacobian matrices, say so directly.

When comparing directions, define cosine similarity as an angle-based comparison. A cosine is not a percentage of shared computation; never write or imply that a cosine of 0.62 means "62% the same."

## Final Book Principles

Preserve these principles in future revisions:

- Localization is not interpretation. Finding a layer, component, neuron, or subspace tells us where to look next; it does not by itself identify the computation.
- Causal importance does not imply monosemanticity. A component or neuron can matter for a behavior without corresponding to one clean human-readable feature.
- The mathematics often generalizes more easily than the semantic ontology. Probes, interventions, Jacobians, JVPs, attribution, ablation, mediation, and rescue can travel across domains, but the trustworthiness of the latent-variable label may not.
- Negative results constrain mechanisms and belong in the narrative. Weak layer-4 enrichment, mixed neuron selectivity, weak input-weight alignment, tiny conjunction-regression gains, and absent rescue are part of the evidence.
- Claim strength should decrease when moving beyond domains with known ground-truth latent state unless new evidence compensates for the weaker labels.
- A distributed mechanism can be interpretable without a one-neuron-one-rule decomposition. Directions, subspaces, populations, paths, and write/read geometry can be legitimate mechanistic objects.
