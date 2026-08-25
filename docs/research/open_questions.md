# Open Questions

These questions should be answered from executed experiments before becoming book claims.

## Neuron-Level Representation

- Do candidate MLP7 neurons detect individual board features such as target emptiness, adjacent opponent occupancy, or friendly terminator occupancy, or are these variables represented mainly in a distributed MLP subspace?
- Which candidate neurons, if any, have reproducible activation selectivity after stronger controls than the current matched valid-vs-invalid pairing?
- Why do some high-attribution neurons, especially neuron 399, have negative valid-vs-opponent-without-terminator selectivity while still having strong output-weight and ablation evidence?

## Relational Conjunctions

- Do any MLP7 neurons detect relational conjunctions rather than additive board features?
- Can an interaction model with better features explain more than the current tiny delta R2 values, whose maximum was 0.002291?
- Are valid-line effects suppressed by ReLU sparsity, position sampling, or the final-token measurement choice?

## Mediation and Causality

- Does a small neuron group mediate a meaningful fraction of the MLP7 legality effect, or do current top-k ablations only identify a weak tail of a broader distributed computation?
- Can activation patching rescue legality after ablating candidate neurons?
- Can candidate-neuron activation patching rescue legality after semantic board-state interventions that break capture-line evidence?
- What fraction of the MLP7 component-level ablation effect is explained by the current candidate neuron set?

## Complete Capture-Line Circuit

- Is there a complete capture-line legality circuit from board representation to attention heads to MLP7 to logits?
- Which attention heads supply the relevant inputs to MLP7?
- Do the layer-7 attention heads identified by attribution or ablation carry board-state features, ray-structure features, or position-history features?
- Can path patching separate direct residual-stream board features from attention-mediated relational features?

## Algorithmic Description

- Can the rule circuit be described algorithmically as target empty plus opponent line plus friendly terminator implies legal move?
- Does the model implement different subcircuits for short lines, long lines, and multi-direction captures?
- Are current line-length and direction effects genuinely weak, or is the present candidate-neuron analysis looking at the wrong variables or positions?
- Does the mechanism generalize across sampled positions, game phases, and all legal move families?

## Scope Beyond Othello

- Does the methodology generalize beyond Othello to other board games or symbolic environments?
- Which parts of the workflow are specific to an available ground-truth simulator and which would transfer to less structured domains?
