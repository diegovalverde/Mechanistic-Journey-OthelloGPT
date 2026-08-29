# Math Background

This appendix collects the mathematical tools used in the book. It is not a general calculus or statistics text. Each idea is included because it appears somewhere in the Othello-GPT investigation: in probe training, residual-stream directions, Jacobian predictions, component attribution, ablations, bootstrap intervals, or validation controls.

The guiding question is always: why did we need this in the book?

## Scalars, Vectors, and Matrices

A scalar is one number. A move logit, a cosine similarity, a loss value, and a bootstrap mean are scalars. When Chapter 7 reports a layer-7 capture-vs-unrelated ratio of `2.746573`, that is a scalar summary of many positions.

A vector is an ordered list of numbers. In this book, the most important vectors live in residual-stream space. Othello-GPT has `d_model = 512`, so a residual activation at one token position is a vector with 512 coordinates:

$$
h \in \mathbb{R}^{512}.
$$

The model does not label these coordinates for us. A vector is just a point or direction in the model's internal space. A probe direction, a gradient, an attention-head output at one position, and an MLP output at one position can all be 512-dimensional vectors.

A matrix is a rectangular table of numbers. A matrix often represents many vectors at once or a linear map from one vector space to another. The logit Jacobian in Chapter 4 has shape `[61, 512]`: one row for each move-token logit and one column for each residual coordinate.

Why we needed this: almost every measurement in the book asks whether one internal vector contains, changes, or writes information relevant to another vector.

## Shape Notation

Shape notation records the axes of a tensor. Transformer code becomes much less mysterious when shapes are explicit.

For this model:

- `d_vocab = 61`: there are 61 output move tokens.
- `n_ctx = 59`: a game transcript has up to 59 modeled move positions.
- `d_model = 512`: each residual-stream row has 512 coordinates.
- `d_mlp = 2048`: each MLP layer has 2048 hidden neuron activations.
- `n_heads = 8` and `d_head = 64`: each attention layer has 8 heads, each with a 64-dimensional internal value stream.

If we cache `blocks.4.hook_resid_post` for a batch of game prefixes, the activation has shape:

```text
[batch, pos, d_model]
```

For Othello-GPT, that means:

```text
[batch, pos, 512]
```

When the board probe is reshaped as `board_probe_weight[q, c, d_model]`, its shape is:

```text
[64, 3, 512]
```

The 64 axis is board square, the 3 axis is class (`empty`, `mine`, `theirs`), and the last axis is the residual direction read by the probe.

Why we needed this: many interpretability bugs are shape bugs. The expression \(Jv\) only makes sense if the dimensions line up.

Two shape conventions recur in the prose. First, a single activation vector is often written without the batch and position axes, because the text has already selected one game and one token position. Thus `cache["blocks.4.hook_resid_post"][0, target_pos]` has shape `[512]`, and the book may call it \(h\). Second, a hook name usually names a whole tensor even when the analysis uses one row of that tensor. A claim about `blocks.7.hook_mlp_out` often means "the 512-dimensional MLP7 output at the final prefix token," not every position in the batch.

This compression is convenient, but it can hide mistakes. If a code fragment returns `[batch, pos, 512]` and a formula expects `[512]`, the missing operation is usually selecting the example and position. If a formula expects `[61]` and the code returns `[batch, pos, 61]`, the missing operation is usually selecting the final prefix row.

## Dot Products

The dot product combines two same-length vectors into one scalar:

$$
a \cdot b = \sum_i a_i b_i.
$$

If both vectors are 512-dimensional, the result is one number. Large positive values mean the vectors point in similar signed directions and have substantial length. Large negative values mean they point in opposite signed directions. Values near zero mean either weak alignment, cancellation, or short vectors.

The dot product appears in three central places.

First, a linear probe score is a dot product plus a bias:

$$
s(q,c) = W_{q,c} \cdot h + b_{q,c}.
$$

For square \(q\) and class \(c\), the probe asks whether the activation \(h\) points along the learned class direction \(W_{q,c}\).

Second, a directional derivative is a dot product between a gradient and an intervention direction:

$$
\Delta z_m \approx \alpha \nabla_h z_m \cdot v.
$$

Third, component attribution compares a component write \(c\) with a downstream gradient \(g\):

$$
A_c = g^\top c.
$$

In Chapter 8 this is a local alignment measure, not an ablation. It asks whether the component's current write is aligned with increasing the legality contrast.

Why we needed this: dot products turn high-dimensional internal geometry into scalar evidence we can rank, plot, and validate.

## Norms

A norm measures vector length. The Euclidean norm is:

$$
||v|| = \sqrt{\sum_i v_i^2}.
$$

If a probe direction has a very large norm, its dot product with another vector may be large partly because it is long, not because it is especially well aligned. That is why the notebook often normalizes directions before intervention:

$$
\hat{v} = \frac{v}{||v||}.
$$

After normalization, \(||\hat{v}|| = 1\). An intervention \(h' = h + \alpha \hat{v}\) then has a controlled step size \(\alpha\) in residual space.

Why we needed this: without norms and normalization, comparing intervention strengths across semantic directions would mix geometry with arbitrary scale.

## Cosine Similarity

Cosine similarity compares direction while mostly ignoring length:

$$
\cos(a,b) = \frac{a \cdot b}{||a||\,||b||}.
$$

The value ranges from -1 to 1. A cosine near 1 means the vectors point in nearly the same direction. A cosine near -1 means they point in nearly opposite directions. A cosine near 0 means they are close to orthogonal.

Chapter 5 reports that the local transformed G6 mine-vs-theirs direction had cosine `0.617840` with the average transformed direction over 100 sampled Othello positions. That is meaningful alignment, but it is not "62 percent the same." Cosine is an angle-based comparison, not a percentage of shared computation.

Why we needed this: J-space analysis asks whether transformed semantic directions are similar across contexts. Cosine is the natural first summary of that question.

??? question "Exercise: dot product versus cosine"
    Suppose two component writes have the same dot product with a legality gradient, but one write has ten times the norm of the other. Which one has the larger cosine with the gradient?

    The shorter write has the larger cosine, assuming the gradient norm is the same. The same dot product achieved with a shorter vector means stronger angular alignment.

## Linear Maps

A linear map is a function that preserves addition and scalar multiplication:

$$
L(a + b) = L(a) + L(b)
$$

and

$$
L(\alpha a) = \alpha L(a).
$$

Matrices implement linear maps. If \(W\) has shape `[out, in]` and \(x\) has shape `[in]`, then \(Wx\) has shape `[out]`.

Linear maps matter because probes, unembeddings, attention output projections, and MLP output weights are all linear at the point where they are applied. The full Transformer is nonlinear, but many of its local pieces are linear enough to analyze directly.

Why we needed this: a linear board probe asks whether board state can be read out by a linear map from the residual stream. That is a much stronger and simpler claim than saying a complicated nonlinear decoder can recover it.

## Matrix-Vector Multiplication

Matrix-vector multiplication is a batch of dot products. If:

```text
J shape: [61, 512]
v shape: [512]
```

then:

```text
Jv shape: [61]
```

Each output coordinate is the dot product of one row of \(J\) with \(v\). In Chapter 4, \(J\) can be the Jacobian from a residual activation to all move logits. Multiplying by a semantic direction \(v\) gives the first-order predicted change in every move logit:

$$
\Delta z \approx \alpha Jv.
$$

The concrete dimensions were:

```text
[61, 512] [512] -> [61]
```

Why we needed this: a Jacobian-vector product lets us ask how one semantic residual direction affects all possible moves without manually testing each output one at a time.

The same idea appears in the unembedding. A final residual vector with shape `[512]` is multiplied by an unembedding-like matrix to produce `[61]` move logits. A board probe similarly multiplies a `[512]` activation by many learned directions to produce scores for 64 squares and 3 states. These are different maps with different meanings, but the dimension rule is the same: the inner dimensions must match.

## Derivatives as Local Slopes

In one dimension, a derivative is a local slope:

$$
\frac{df}{dx}(x) \approx \frac{f(x+\epsilon)-f(x)}{\epsilon}
$$

for very small \(\epsilon\). If the derivative is positive, increasing \(x\) locally increases \(f(x)\). If it is negative, increasing \(x\) locally decreases \(f(x)\).

The word "local" is essential. A derivative does not promise that the same slope holds after a large movement. It says what happens for small changes around the current point.

Why we needed this: residual interventions in the book are small because Jacobian predictions are local first-order predictions.

## Partial Derivatives

When a function has many input coordinates, a partial derivative asks how the output changes when one coordinate changes and the others are held fixed:

$$
\frac{\partial f}{\partial h_i}.
$$

For a residual vector \(h \in \mathbb{R}^{512}\), there are 512 partial derivatives for one scalar output such as the E3 logit.

Why we needed this: the model activation is not one number. To understand local sensitivity, we need one slope per residual coordinate.

## Gradients

A gradient stacks all partial derivatives of one scalar output into a vector:

$$
\nabla_h z_m =
\left[
\frac{\partial z_m}{\partial h_1},
\frac{\partial z_m}{\partial h_2},
\ldots,
\frac{\partial z_m}{\partial h_{512}}
\right].
$$

The gradient points in the residual-space direction that most rapidly increases the scalar output locally. In Chapter 4, \(z_m\) was a selected move logit. In Chapter 7 and Chapter 8, the scalar target was often a legality contrast rather than a raw move logit.

Why we needed this: gradients let us turn "what would increase this move's score?" into a vector we can compare with probe directions and component writes.

## Directional Derivatives

A directional derivative asks how a scalar output changes if we move in a chosen direction \(v\):

$$
D_v z_m = \nabla_h z_m \cdot v.
$$

If \(v\) is a normalized semantic direction, this number estimates the change in the selected logit per unit step along that semantic direction. The executed notebook reported a directional derivative of `+0.030897` for the E3 logit along the normalized G6 mine-vs-theirs probe direction in the Chapter 4 setup.

Why we needed this: a probe direction becomes more than an observational handle when moving along it changes logits in the direction predicted by the local gradient.

??? question "Exercise: sign of a semantic edit"
    If \(\nabla_h z_m \cdot v = 0.03\), what does the first-order approximation predict for \(h' = h - 0.1v\)?

    It predicts \(\Delta z_m \approx -0.1 \times 0.03 = -0.003\). Moving opposite the direction locally decreases the selected logit.

## Jacobians

A Jacobian generalizes a gradient from one scalar output to many outputs. If \(z(h)\) is the vector of all 61 move logits, then:

$$
J(h) = \frac{\partial z}{\partial h}
$$

has shape `[61, 512]`. Row \(m\) is the gradient of logit \(z_m\) with respect to \(h\).

The Jacobian is a local linear map. Near the current activation:

$$
z(h+\delta h) \approx z(h) + J(h)\delta h.
$$

Why we needed this: the book often cares about a whole output distribution, not just one move. The Jacobian tells us how all move logits locally respond to a residual edit.

## Jacobian-Vector Products

A Jacobian-vector product, or JVP, computes \(Jv\) without necessarily materializing all of \(J\). This is useful when the input and output are high-dimensional. Autograd systems such as PyTorch can compute JVPs efficiently for many functions.

In Chapter 5, the main J-space experiment was not the `[61, 512]` logit Jacobian. It transported a source residual direction from `blocks.4.hook_resid_post` to a later hidden state before final normalization and unembedding. That map has a hidden-state target, so the shape is conceptually:

```text
[512, 512] [512] -> [512]
```

The prose must say which function is being differentiated. "J-space" in this book can refer to local hidden-state transport or to logit-space sensitivities, but the source hook, target representation, token position, and dimensions determine the actual map.

Why we needed this: JVPs let us follow a semantic direction through downstream computation, not merely ask whether it is decodable at the source.

## First-Order Taylor Approximation

The first-order Taylor approximation says that a differentiable function can be approximated near a point by its value plus a local linear correction:

$$
f(x+\delta) \approx f(x) + \nabla f(x)^\top \delta.
$$

For vector outputs:

$$
z(h+\delta h) \approx z(h) + J(h)\delta h.
$$

The Chapter 4 finite-difference validation tested exactly this idea. Along the normalized G6 mine-vs-theirs direction, predicted and actual E3 logit changes across alphas from `-0.1` to `0.1` had maximum absolute error `0.000067`.

Why we needed this: this is the mathematical bridge between gradients and actual residual interventions.

## Chain Rule

The chain rule tells us how derivatives compose through a sequence of functions. If:

$$
y = f(x)
$$

and

$$
z = g(y),
$$

then a small change in \(x\) affects \(z\) through both stages:

$$
\frac{dz}{dx} = \frac{dz}{dy}\frac{dy}{dx}.
$$

Transformers are long chains of embeddings, attention layers, MLPs, residual additions, normalization, and unembedding. Autograd applies the chain rule through this computation graph.

Why we needed this: when the notebook differentiates a final move logit with respect to an activation at layer 4, it is differentiating through all downstream layers. We do not have to manually derive every intermediate slope, but the meaning is chain-rule sensitivity.

The chain rule also explains why the hook location changes the question. A gradient at `blocks.4.hook_resid_post` includes the effect of layers 5, 6, 7, final normalization, and unembedding. A gradient at `blocks.7.hook_mlp_out` only includes the residual addition after MLP7 plus the final readout path. Both gradients can target the same scalar legality contrast, but they describe sensitivity at different internal interfaces.

## Mean, Variance, and Correlation

A mean summarizes a collection of numbers:

$$
\bar{x} = \frac{1}{n}\sum_i x_i.
$$

Chapter 7 uses means when comparing capture-line sensitivity against unrelated occupied controls.

Variance measures spread around the mean:

$$
\mathrm{Var}(x) = \frac{1}{n}\sum_i (x_i-\bar{x})^2.
$$

High variance means the average may hide heterogeneous cases.

Correlation measures linear association between two variables after centering and scaling. A Pearson correlation near 1 means large values of one variable tend to accompany large values of the other. A correlation near -1 means large values of one tend to accompany small values of the other. A correlation near 0 means little linear relationship.

Chapter 9 used correlations to test whether candidate-neuron activations tracked capture-line structure such as longest line or number of capture directions. Those correlations were small in the current evidence record.

Why we needed this: many notebook results are dataset summaries. Means show central tendency, variance warns about spread, and correlations test simple structural hypotheses.

A mean can be persuasive and still incomplete. In the MLP7 component ablation result, the mean absolute effect identifies MLP7 as the strongest tested component. It does not say every position depends on MLP7 in the same way. Likewise, weak mean selectivity for individual neurons does not prove those neurons never matter. It says the tested aggregate did not support a clean detector story.

## Interpreting AUROC

AUROC means area under the receiver operating characteristic curve. It summarizes how well a scoring rule separates positive examples from negative examples as the classification threshold is swept. The ROC curve plots true positive rate against false positive rate across thresholds, so AUROC is threshold-independent in a way that a single accuracy number is not.

The most useful interpretation in this book is the ranking interpretation:

```text
AUROC = probability that a randomly chosen positive example receives
a higher score than a randomly chosen negative example
```

An AUROC of `0.5` means the score is no better than random ranking. An AUROC of `1.0` means every positive example is ranked above every negative example. Values below `0.5` mean the score is systematically reversed: the same information may be present, but with the sign flipped.

Why we needed this: Chapter 7½ reports macro AUROC and hard valid-vs-no-terminator AUROC for directional capture probes. Those numbers ask whether the probe score ranks true capture-ray labels above non-capture labels. A high AUROC therefore supports a decodability claim: the relation is linearly readable from the residual representation.

That is not the same as top-1 direction accuracy. Top-1 asks whether the highest-scoring direction for one target square is truly valid. AUROC instead pools positive-negative ranking comparisons. A representation can have very high AUROC while occasionally assigning the single highest score to a wrong direction in a particular board context.

It is also not the same as calibration or precision. AUROC does not say that a score of `0.9` is a 90% probability. It does not say how many predicted positives will be correct at a chosen threshold. Those quantities depend on the score calibration, the threshold, and the base rate of positives and negatives.

So the book reads AUROC as strong evidence about separability, not as proof of mechanism. When hard AUROC rises from `0.9601` at post4 to `0.9905` at post5, the cautious interpretation is that the valid-capture versus no-friendly-terminator distinction became easier for this linear readout to rank. It does not by itself prove that MLP5 implements the Othello capture rule.

For a general reference on ROC curves, threshold tradeoffs, and AUC, see Wikipedia's [Receiver operating characteristic](https://en.wikipedia.org/wiki/Receiver_operating_characteristic).

## Bootstrap Intuition

Bootstrapping estimates uncertainty by resampling the observed data. Suppose we have sensitivity measurements over positions. A bootstrap sample draws positions with replacement until it has the same number of positions as the original set. We compute the statistic again, repeat many times, and inspect the distribution of bootstrap statistics.

The method does not create new evidence from nowhere. It asks how much the statistic would vary if the observed sample were a rough proxy for the population of positions we care about.

Why we needed this: Chapter 7 needed to show that layer-7 capture-line enrichment was not just one fragile average. The layer-7 validation reported a capture-minus-unrelated difference of `0.040162` with 95% bootstrap confidence interval `[0.035965, 0.044268]`.

## Confidence Intervals

A confidence interval is a range produced by a statistical procedure. In this book, bootstrap intervals are used as uncertainty summaries for dataset-level effects.

If a bootstrap 95% interval for a difference is far from zero, the observed effect is less likely to be a pure sampling accident under the assumptions of the resampling procedure. It still does not prove mechanism. It only strengthens the dataset-level measurement.

Why we needed this: the layer-7 capture enrichment needed uncertainty bars before it could support strong evidence.

## Permutation and Shuffle Nulls

A permutation or shuffle null breaks a proposed relationship while preserving much of the surrounding data structure. In Chapter 7, the question was whether capture-line squares were especially sensitive compared with unrelated occupied controls. A shuffled-square control asks what ratios appear when the square labels are disrupted.

The validated layer-7 result had shuffled mean ratio `1.046078` and shuffled 95th percentile `1.176336`, while the actual ratio was `2.746573`. That comparison helps show that the effect is not merely an artifact of denominator choice or generic occupied-square sensitivity.

Why we needed this: a raw effect is easier to trust when a targeted null fails to reproduce it.

## R2 and Cross-Validation Intuition

\(R^2\) measures how much variance in a target variable is explained by a model, compared with predicting the mean. An \(R^2\) near 1 is strong predictive fit. An \(R^2\) near 0 means the model is not much better than the mean. A negative cross-validated \(R^2\) means the model performed worse on held-out data than a mean predictor.

Chapter 9 used simple regression models to ask whether candidate-neuron activations looked like additive feature detectors or interaction detectors. Adding interaction features improved in-sample \(R^2\) by at most `0.002291`, and cross-validated values were near zero or negative for many neurons.

Cross-validation matters because a model can fit accidental quirks in the training sample. Holding out examples asks whether the pattern generalizes.

Why we needed this: without cross-validation, a tiny in-sample improvement could be mistaken for a relational rule detector.

## The Minimal Mathematical Toolkit

The book's core workflow can now be summarized compactly:

1. Use linear maps to test whether board state is decodable from residual vectors.
2. Convert probe weights into semantic directions.
3. Use gradients and Jacobians to predict local effects of moving along those directions.
4. Use JVPs to study how semantic directions are transformed downstream.
5. Use dot products and cosines to compare semantic directions, gradients, and component writes.
6. Use bootstraps, confidence intervals, and shuffle nulls to separate robust dataset effects from fragile examples.
7. Use ablations, matched controls, and cross-validation before interpreting localized components or neurons as mechanisms.

That is enough mathematics to read the investigation. The hard part is not the notation. The hard part is preserving what each measurement can and cannot prove.

## Common Mathematical Misreadings

Several mistakes are easy to make.

Do not read a high probe accuracy as a causal result. A probe is a map from activations to labels. It shows decodability. Causal use requires changing the activation or component and measuring the model's output.

Do not read a cosine similarity as a percentage. The Chapter 5 cosine `0.617840` is an angular comparison between transformed directions. It is not a fraction of the algorithm explained.

Do not read a Jacobian as a global simulator. A Jacobian is local. It can accurately predict small residual edits around one activation while failing for large moves or different contexts.

Do not read an attribution score as an ablation. Attribution uses the clean run and a local gradient. Ablation reruns the model after a replacement or removal. The book keeps those evidence types separate because they answer different questions.

Do not read a confidence interval as a mechanism. The layer-7 bootstrap interval strengthens the dataset-level enrichment result. It does not identify which component computes the rule.

These distinctions are mathematical, but they are also scientific. They keep the prose from turning useful measurements into claims the measurements cannot support.
