# Glossary

This glossary defines terms as they are used in this book.

**Ablation**: A causal intervention that removes, replaces, or suppresses an activation, component, neuron, or group and then reruns the model to measure the behavioral effect.

**Activation**: A vector or tensor produced inside the model during a forward pass, such as a residual-stream row, attention output, or MLP neuron activation.

**Attention head**: One of the eight attention mechanisms inside each Othello-GPT block. A head mixes information across token positions and writes a 512-dimensional contribution to the residual stream.

**Attribution**: A local measurement comparing a clean-run component write with a downstream sensitivity direction, often by a dot product such as \(g^\top c\). Attribution is not the same as ablation.

**Board probe**: A trained readout from residual activations to Othello board labels. The main probe predicts empty/mine/theirs labels for 64 squares from `blocks.4.hook_resid_post`.

**Bootstrap**: A resampling procedure used to estimate uncertainty in a statistic, such as the layer-7 capture-minus-control sensitivity difference.

**Capture line**: A contiguous ray of opponent discs between a candidate move square and a friendly terminator, making the move legal under Othello rules.

**Capture opponent**: An opponent disc that lies on a true capture line for a selected legal move.

**Causal intervention**: A controlled change to an internal activation or component followed by a rerun of the model to test whether outputs change.

**Component**: A named model part whose output can be inspected or intervened on, such as L7H0, L7H7, or MLP7.

**Component ablation**: A causal replacement or removal of a whole component output, followed by recomputation of a target metric. In Chapter 8, whole-component ablation uses `L_ablate - L_clean`.

**Component attribution**: A clean-run dot-product alignment between a component write and a gradient target. It ranks local relevance but does not itself change the model.

**Confidence interval**: A range produced by a statistical procedure. The book mainly uses bootstrap intervals as uncertainty summaries for dataset-level effects.

**Cosine similarity**: An angle-based comparison between two vectors, computed as a normalized dot product. A cosine of `0.617840` is not "61.7840 percent the same."

**Decodability**: The fact that information can be read from activations by an external probe. Decodability alone does not prove the model uses that information.

**Directional derivative**: The local rate of change of a scalar output when an activation moves along a chosen direction.

**Distributed representation**: Information represented across a pattern of directions, subspaces, components, or neurons rather than one clean coordinate or one monosemantic neuron.

**Evidence ladder**: The book's discipline for keeping claim strength aligned with evidence: behavior, representation, causal relevance, localization, component importance, mediation, rescue, and mechanism.

**Friendly terminator**: The friendly disc at the far end of a capture line. It terminates the ray of opponent discs and makes the selected move legal.

**Gradient**: A vector of partial derivatives of one scalar target with respect to an activation. It points in the local direction that most increases that scalar.

**Hook**: A named TransformerLens point where an activation can be cached or edited during a model run.

**Illegal empty-square move**: A move to an empty square that is currently illegal. The legality contrast uses these moves as its baseline, excluding occupied squares.

**Jacobian**: A local linear map of derivatives from many input coordinates to many output coordinates, such as residual-stream coordinates to move logits.

**Jacobian-vector product / JVP**: The product \(Jv\), giving the local transformed effect of direction \(v\) under a Jacobian without necessarily materializing the whole matrix.

**J-space**: In this book, the space of locally transformed semantic directions under a Jacobian. Some analyses use logit-space Jacobians; Chapter 5's main J-space analysis is hidden-state transport.

**Layer 7**: The zero-based final block of the eight-block Othello-GPT model. It is the eighth block in ordinary one-based counting.

**Legality contrast**: The selected legal move logit minus the mean logit of currently illegal empty-square moves. It is designed to isolate legal-vs-illegal separation better than a raw logit.

**Linear probe**: A linear readout trained on activations to predict labels, such as board-square state. It tests linear decodability.

**Logit**: An unnormalized output score for a move token before softmax.

**Matched control**: A comparison example selected to match relevant nuisance variables, such as target square, region, game phase, local structure, or occupancy.

**Mean replacement**: An ablation method that replaces an activation with an average activation from sampled examples rather than zeroing it.

**Mediation**: Evidence that an upstream effect acts through a proposed intermediate component or representation. The current notebook has mediation-like diagnostics, not a complete mediation result.

**MLP**: The position-wise feedforward sublayer inside each block. It reads the current residual row after attention, applies a 2048-dimensional GELU hidden layer, and writes back to the residual stream.

**MLP7**: The MLP in zero-based layer 7, the final block. It is the strongest tested layer-7 component under both attribution and mean-replacement ablation.

**Monosemanticity**: The property of a neuron or direction corresponding cleanly to one human-interpretable feature. The book does not establish monosemantic Othello rule neurons.

**Neuron**: One coordinate in the MLP hidden layer, usually discussed through pre-activation, post-GELU activation, input weights, and output/write direction.

**Off-manifold intervention**: An activation edit that may move the model into an internal state unlike those produced by natural game histories.

**Output/write direction**: The residual-space direction a component or neuron adds to the stream. For an MLP neuron, this is related to its row of `W_out`.

**Permutation test**: A control that shuffles labels or assignments to estimate what a statistic might look like if the proposed relationship were broken.

**Probe direction**: A direction derived from probe weights, such as a mine-vs-theirs square direction.

**Relational-condition dataset**: The Chapter 9 dataset labeling candidate moves by Othello ray structure, including valid capture, multiple capture, opponent-without-terminator, friendly-adjacent, and empty-adjacent conditions.

**Rescue**: A sufficiency-like experiment that disrupts a computation and then patches back a proposed intermediate activation to test whether behavior recovers. Rescue was not executed in the current notebook.

**Residual stream**: The running 512-dimensional vector state at each token position to which attention and MLP components add updates.

**Semantic direction**: An activation-space direction associated with an interpretable feature under a specific construction, such as a probe-derived mine-vs-theirs direction for one square.

**Superposition**: A representational regime where features are packed into overlapping directions rather than cleanly assigned to individual neurons. In this book it is a possible frame for messy neuron results, not an established explanation of Othello-GPT.

**TransformerLens**: The mechanistic interpretability library used to run Othello-GPT with named hooks, cached activations, and activation-editing interventions.

**World model**: An internal representation that tracks aspects of a latent environment useful for prediction or action. In Othello-GPT, the relevant latent environment is the hidden board state implied by the move transcript.
