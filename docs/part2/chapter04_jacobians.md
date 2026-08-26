# Jacobians: Listening to the Network's Thoughts

For a tiny movement in any residual-space direction, can we predict the resulting change in the output?

That was the question Chapter 3 left us with.

We can already do one kind of intervention. We can take a residual activation \(h\), choose a semantic direction \(v\) from the board probe, add a small amount of that direction, and watch the move logits change:

$$
h' = h + \alpha v.
$$

This is useful, but by itself it is still clumsy. If we want to test ten directions, we can run ten interventions. If we want to test every square, every state contrast, and every output move, the procedure becomes a long list of nudges:

```text
h + epsilon v_1
h + epsilon v_2
h + epsilon v_3
...
```

The problem is not that this is impossible. The layer-4 residual stream in Othello-GPT has 512 dimensions, and the model produces 61 output logits. We can write code that nudges the model in many directions and records many changes. But conceptually, that is not what we want.

What we want is a local map:

```text
small change inside
        ->
predicted small change outside
```

The word "local" is important. We are not asking for a complete symbolic description of Othello-GPT. We are not asking for a formula that predicts the model everywhere in activation space. We are asking a more modest question: near this activation, at this layer, for this board position, how does a tiny internal movement affect the output?

That local map is the Jacobian.

## One Internal Coordinate

Before using the word "Jacobian" too heavily, start with the smallest possible version of the problem.

Suppose the model had one internal coordinate:

$$
x.
$$

And suppose we cared about one output logit:

$$
z(x).
$$

Maybe \(z\) is the score assigned to move E3. Maybe \(x\) is one coordinate of the residual stream. This is not yet realistic, but it isolates the idea.

The question is:

```text
If x moves slightly, how fast does z move?
```

The answer is the derivative:

$$
\frac{dz}{dx}.
$$

Read this as a local slope. It tells us how much \(z\) changes for a tiny change in \(x\), measured at the current value of \(x\). If the derivative is positive, increasing \(x\) locally increases \(z\). If it is negative, increasing \(x\) locally decreases \(z\). If it is near zero, small changes in \(x\) have little first-order effect on \(z\).

<figure markdown>
![Local slope of a nonlinear function](../figures/local_slope.svg)
<figcaption>
A derivative is a local slope. The nonlinear function may curve globally, but near the current point its tangent line can predict the effect of a small movement.
</figcaption>
</figure>

The figure shows the central intuition. The curve is nonlinear. Far away from the current point, a straight line would be a poor description. But if we zoom in enough around one point, the curve begins to look like its tangent line.

Globally nonlinear.

Locally approximately linear.

That is the entire chapter in miniature.

## The Tangent-Line Approximation

The derivative becomes useful when we turn it into a prediction. If \(x\) changes by a small amount \(\delta x\), the output changes approximately by:

$$
z(x + \delta x)
\approx
z(x) + \frac{dz}{dx}\delta x.
$$

Every term has a plain-language meaning:

```text
z(x)
    the current output logit

delta x
    the small internal movement

dz/dx
    the local slope at the current point

(dz/dx) delta x
    the predicted change in the logit
```

For an illustrative toy example, suppose the local slope is \(0.7\). If we move by:

$$
\delta x = 0.01,
$$

then the predicted change is:

$$
\delta z \approx 0.7 \times 0.01 = 0.007.
$$

This number is not from Othello-GPT. It is only a scalar example. Its job is to make the logic visible before the real experiment adds dimensions.

The approximation says: do not recompute the whole nonlinear function from scratch for every tiny movement. Use the local slope to predict the first-order effect.

Chapter 3 ended with a residual intervention:

$$
h' = h + \alpha v.
$$

That equation is the same idea in a larger space. We do not have one internal scalar \(x\). We have a 512-dimensional residual vector \(h\). We do not always care about one scalar output. We often care about many move logits. The derivative has to grow up with the problem.

## From One Input to 512 Inputs

Now let the hidden state be:

$$
h \in \mathbb{R}^{512}.
$$

This is the actual residual-stream dimensionality printed by the executed notebook for Othello-GPT:

```text
d_model: 512
```

For the moment, still choose one output logit. Write the logit for move \(m\) as:

$$
z_m(h).
$$

In the executed notebook, the selected move was E3, token 21. The model's favorite move on that prefix was E8, token 57, but E3 was a legal move and was chosen for analysis because it had a clear capture line: D3 and C3.

With 512 input coordinates, we can ask 512 local-slope questions:

```text
If h_1 moves slightly, how does z_m move?
If h_2 moves slightly, how does z_m move?
If h_3 moves slightly, how does z_m move?
...
If h_512 moves slightly, how does z_m move?
```

The answers form the gradient:

$$
\nabla_h z_m
=
\left[
\frac{\partial z_m}{\partial h_1},
\frac{\partial z_m}{\partial h_2},
\ldots,
\frac{\partial z_m}{\partial h_{512}}
\right].
$$

The gradient is a 512-dimensional vector. Each entry is a local slope along one raw coordinate axis of the residual stream.

This is already more useful than the scalar derivative, but raw coordinate axes are rarely what we care about in mechanistic interpretability. We usually do not believe coordinate 117, by itself, means "G6 is mine" or "C3 is an opponent disc." Learned representations are distributed. The semantic directions from Chapter 2 were not single neurons. They were directions in the full residual space, constructed from the linear board probe's weights.

So the real question is not only:

```text
How sensitive is E3's logit to h_117?
```

It is:

```text
How sensitive is E3's logit to a meaningful direction v?
```

## Directional Derivatives

Let \(v\) be a semantic direction from the probe, such as a mine-vs-theirs direction for one square. Let the actual residual edit be:

$$
\delta h = \alpha v.
$$

Here \(\alpha\) is the intervention strength. If \(v\) has been normalized, \(\alpha\) controls how far we move in that direction.

For one selected logit \(z_m\), the first-order prediction is:

$$
\delta z_m
\approx
\nabla_h z_m^\top \delta h.
$$

Substituting \(\delta h = \alpha v\):

$$
\delta z_m
\approx
\alpha \nabla_h z_m^\top v.
$$

The dot product is the key operation. Geometrically, the gradient tells us the direction in residual space that would increase the selected logit fastest, at least locally. The semantic vector tells us the direction we actually want to move. Their dot product tells us how much the semantic direction overlaps with the logit's local sensitivity.

<figure markdown>
![Gradient and semantic direction](../figures/gradient_semantic_direction.svg)
<figcaption>
A 2D cartoon of the 512-dimensional residual space. The gradient points in the direction that locally increases one selected logit fastest. A semantic direction has a positive, zero, or negative first-order effect depending on its alignment with that gradient.
</figcaption>
</figure>

This is one of the most important intuitions in the book.

If the semantic direction points mostly with the gradient, moving along it should increase the logit. If it points mostly against the gradient, moving along it should decrease the logit. If it is orthogonal to the gradient, the first-order approximation predicts little or no change.

!!! question "Pause and think"
    Suppose the semantic direction \(v\) is exactly orthogonal to the gradient \(\nabla_h z_m\). What does the first-order approximation predict?

It predicts approximately zero first-order effect:

$$
\nabla_h z_m^\top v = 0.
$$

That does not mean the feature is globally irrelevant. It means that, right here, for an infinitesimal movement in this direction, the tangent-line part of the effect is zero. Larger movements could still matter through curvature. Other board positions could have a different gradient. Another output logit could be sensitive to the same direction.

This is why "local" keeps appearing. The gradient is not a universal property of the direction. It is a property of the model at a particular activation and output.

## Many Outputs: The Jacobian

Othello-GPT does not produce one logit. The executed notebook printed:

```text
d_vocab: 61
d_vocab_out: 61
```

So the final output vector is:

$$
z(h) \in \mathbb{R}^{61}.
$$

The 61 outputs correspond to the move-token vocabulary used by this model: pass plus the 60 playable board-square tokens, excluding the four fixed starting center squares from the move vocabulary. The probe's square labels still cover all 64 board squares, but the model's next-token distribution has 61 entries.

If one scalar output gives us one gradient, then 61 scalar outputs give us 61 gradients. Stack them as rows and we get the Jacobian:

$$
J(h) = \frac{\partial z}{\partial h}.
$$

For this experiment, the shape is:

```text
J(h): [61, 512]
```

Each row is the gradient of one output logit with respect to the 512-dimensional residual state. Each column says how all 61 logits respond to one residual coordinate.

<figure markdown>
![Jacobian matrix shape](../figures/jacobian_matrix.svg)
<figcaption>
Conceptual shape of the Othello-GPT logit Jacobian at a single residual-stream activation. Each row is one move logit's gradient; each column is one residual coordinate's local effect on all logits.
</figcaption>
</figure>

Now the central equation can finally appear:

$$
z(h + \delta h)
\approx
z(h) + J(h)\delta h.
$$

This is the tangent-line approximation in vector form.

The pieces are:

```text
h
    the current hidden state

delta h
    a tiny internal edit

z(h)
    the original vector of output logits

J(h)
    the local transformation from hidden-state changes to logit changes

J(h) delta h
    the predicted change in every output logit
```

The dimensions are part of the intuition:

```text
J(h):         [61, 512]
delta h:     [512]
J(h)delta h: [61]
```

A 512-dimensional internal edit goes in. A 61-dimensional predicted output change comes out.

<figure markdown>
![Jacobian-vector product](../figures/jacobian_vector_product.svg)
<figcaption>
A Jacobian-vector product, or JVP, applies the local Jacobian to one chosen residual-space perturbation and produces a predicted change in all output logits.
</figcaption>
</figure>

This operation is called a Jacobian-vector product, usually abbreviated JVP. It is the natural object when we ask: if I push the model in this one internal direction, what output changes should I expect?

!!! info "Three objects that are easy to confuse"
    Gradient:
    one scalar output with respect to many inputs

        \nabla_h z_m
        shape [512]

    Jacobian:
    many outputs with respect to many inputs

        J = \partial z / \partial h
        shape [61, 512]

    Jacobian-vector product:
    effect of one chosen input direction on all outputs

        Jv
        shape [61]

    A gradient dot vector, \(\nabla_h z_m \cdot v\), is one scalar directional derivative.

## We Do Not Need the Whole Matrix

The full Jacobian is a useful mathematical object. It is not always the object we need to materialize in memory.

If we only care about one selected output logit, reverse-mode autodiff can give us:

$$
\nabla_h z_m.
$$

Then for any semantic direction \(v\), the directional sensitivity is:

$$
\nabla_h z_m^\top v.
$$

That is a scalar. It tells us the first-order effect on one chosen move logit.

If instead we care about one perturbation \(\delta h\) and many outputs, we can compute:

$$
J(h)\delta h
$$

without explicitly writing down all \(61 \times 512\) entries. This distinction matters much more in larger models, where activations and vocabularies can be enormous. The idea scales because autodiff systems can compute the products we need without forcing us to stare at the whole matrix.

The executed notebook used a practical version of the one-logit path. It created a zero residual edit variable, ran the downstream computation from the cached layer-4 residual stream, selected the E3 logit, and differentiated that scalar with respect to the edit variable:

```python
delta0 = torch.zeros(D_MODEL, device=DEVICE, requires_grad=True)
baseline_logits = logits_from_residual_delta(delta0)
move_logit = baseline_logits[0, MOVE_ID]
move_grad = torch.autograd.grad(move_logit, delta0)[0]
```

This code is not meant as a PyTorch tutorial. The conceptual point is that `move_grad` is the local gradient of one selected output logit with respect to one residual-stream edit.

## How the Hook Defines the Function

To compute a derivative, we have to be precise about the function being differentiated.

In the notebook, the residual activation was captured from:

```text
blocks.4.hook_resid_post
```

The analysis prefix had length 28, so both the source and target positions were the final prefix position, index 27. The cached activation at that hook is the state after layer 4 has finished writing to the residual stream at the final token. The experiment then modifies that cached residual vector and continues the model from layer 5 onward.

Conceptually:

```text
moves
  -> layers 0...4
  -> layer-4 residual stream h
  -> optional edit h + delta h
  -> layers 5...7
  -> final normalization and unembedding
  -> move logits
```

<figure markdown>
![TransformerLens hook intervention flow](../figures/hook_intervention_flow.svg)
<figcaption>
A TransformerLens hook lets us capture or edit an activation and then continue the forward pass. The hook is a research instrument, not a component the trained model uses during ordinary inference.
</figcaption>
</figure>

This hook construction matters. The derivative is not with respect to the input tokens. It is not with respect to the model weights. It is with respect to a possible additive edit at one residual-stream site.

So when we write:

$$
\frac{\partial z_m}{\partial h},
$$

the operational meaning is:

```text
How would the selected output logit change if we added a tiny vector
to the cached layer-4 residual activation at the final prefix position,
then let the rest of the model run normally?
```

That is a narrow, experimentally grounded question. It is also exactly the kind of question Chapter 3 prepared us to ask.

## The Finite-Difference Test

Autograd can produce a gradient even when our experimental setup has a mistake. We might have chosen the wrong hook. We might have indexed the wrong token position. We might have selected the wrong output token. We might have the sign convention backwards. We might be differentiating one function but intervening on another.

So the first empirical question is not yet semantic:

```text
Does the derivative machinery work?
```

The check is a finite-difference validation.

For a direction \(v\) and a small scalar \(\epsilon\), the Jacobian predicts:

$$
\Delta z_\text{pred}
=
\epsilon \nabla_h z_m^\top v.
$$

Then we actually run the edited model:

$$
\Delta z_\text{actual}
=
z_m(h + \epsilon v) - z_m(h).
$$

If the derivative and hook intervention are describing the same local function, these two numbers should be close when \(\epsilon\) is small enough.

This check verifies several things at once:

- the hook site is the intended residual stream
- the tensor indexing selects the intended source and target position
- the selected output logit is the intended move token
- the autograd gradient has the correct sign and scale
- the downstream continuation is the function we meant to differentiate
- the perturbation size is small enough for local linearization to apply

!!! question "Pause and think"
    If a Jacobian prediction is excellent for \(\epsilon = 0.001\) but poor for \(\epsilon = 10\), is that surprising?

No. A derivative is a local object. Large edits can move the activation into a region where the tangent line at the starting point is no longer a good approximation.

## Sanity Check: Gradient Direction

The executed notebook first tested the machinery along the normalized gradient direction itself.

The setup was:

```text
source layer: 4
source hook: blocks.4.hook_resid_post
source position: final prefix position, index 27
target position: final prefix position, index 27
selected output: E3, token 21
baseline selected logit: 8.940763473510742
gradient norm: 0.17336732149124146
```

The direction was:

$$
v = \frac{\nabla_h z_{E3}}{\|\nabla_h z_{E3}\|}.
$$

In this direction, the directional derivative is the gradient norm. The notebook compared the actual E3 logit change against the first-order prediction across several epsilon values:

| Epsilon | Actual delta logit | First-order predicted delta logit |
| --- | ---: | ---: |
| 0.0001 | 0.000019073 | 0.000017337 |
| 0.0003 | 0.000052452 | 0.000052010 |
| 0.001 | 0.000170710 | 0.000173370 |
| 0.003 | 0.000522610 | 0.000520100 |
| 0.01 | 0.001734700 | 0.001733700 |
| 0.03 | 0.005184200 | 0.005201000 |
| 0.1 | 0.017184000 | 0.017337000 |

This is the engineering sanity check. It does not yet say anything deep about Othello board semantics. It says that the local derivative machinery and the actual intervention machinery agree in a simple direction where we know what should happen.

That distinction is important:

```text
First:
    Does the derivative machinery work?

Later:
    Does it say something meaningful about board-state directions?
```

The table answers the first question.

## Semantic Board-State Intervention

Now we can return to the Othello problem.

Instead of nudging along the selected logit's own gradient, use a semantic board direction from the probe. The layer-4 board probe from Chapter 2 produced mine-vs-theirs directions:

$$
v_{q,\text{mine-vs-theirs}}
=
W_{q,\text{mine}} - W_{q,\text{theirs}},
$$

normalized in the notebook before use. These are not magic variables inside the model. They are operational directions defined by the trained linear probe. But they give us a way to ask a targeted question: if the residual stream is moved slightly along a board-state direction, does the selected move logit change as the local Jacobian predicts?

The notebook computed the sensitivity of the E3 logit to every square's mine-vs-theirs direction:

$$
v_q^\top \nabla_h z_{E3}.
$$

It then selected the square with the largest absolute sensitivity. In the executed section 9, that square was:

```text
G6, square index 46
```

The semantic direction was the normalized G6 mine-vs-theirs probe direction. The directional derivative was:

```text
v^T g_m = +0.030897
```

The experiment tested alphas from \(-0.1\) to \(0.1\). For each alpha, it compared:

$$
\Delta z_\text{pred}
=
\alpha v^\top g_m
$$

against:

$$
\Delta z_\text{actual}
=
z_m(h + \alpha v) - z_m(h).
$$

The measured result:

| Alpha | Predicted delta logit | Actual delta logit | Absolute error |
| --- | ---: | ---: | ---: |
| -0.1 | -0.003090 | -0.003156 | 0.000066057 |
| -0.03 | -0.000927 | -0.000932 | 0.000004844 |
| -0.01 | -0.000309 | -0.000308 | 0.000000928 |
| -0.003 | -0.000093 | -0.000095 | 0.000002678 |
| 0.003 | 0.000093 | 0.000092 | 0.000001137 |
| 0.01 | 0.000309 | 0.000308 | 0.000000928 |
| 0.03 | 0.000927 | 0.000919 | 0.000007553 |
| 0.1 | 0.003090 | 0.003023 | 0.000066504 |

The maximum absolute prediction error across these alphas was:

```text
0.000067
```

<figure markdown>
![Measured Jacobian prediction versus actual intervention](../figures/jacobian_prediction_vs_intervention.svg)
<figcaption>
Measured data from the executed notebook, section 9. The points compare Jacobian-predicted E3 logit deltas with actual E3 logit deltas after layer-4 residual interventions along the G6 mine-vs-theirs probe direction. The dashed diagonal is perfect prediction.
</figcaption>
</figure>

This is the chapter's main empirical result.

It says that, in this tested case, the local linear prediction and the actual nonlinear model output were extremely close. The largest discrepancy was about \(6.7 \times 10^{-5}\) logit units. That is tiny compared with the baseline E3 logit of about \(8.94\), and tiny compared with the largest intervention deltas in the table.

But the interpretation must stay local.

The result does not mean:

```text
Othello-GPT is a linear model.
```

It means:

```text
Near this layer-4 activation, for these perturbation sizes,
along this semantic probe direction, the first-order approximation
accurately predicted the selected output-logit change.
```

That is a narrower claim, and a stronger scientific statement.

!!! question "Pause and think"
    Why can a nonlinear Transformer have a useful Jacobian?

Because the Jacobian describes the model near one activation. A nonlinear function can curve across a large region while still being well approximated by a linear map in a small neighborhood.

## What the Small Error Does and Does Not Prove

The max error \(0.000067\) is a validation of the local linearization in the tested setup. It gives us confidence that when we compute:

$$
v^\top \nabla_h z_m,
$$

we are measuring a real first-order sensitivity of the downstream model to that residual-space direction.

It also makes the semantic-intervention story sharper. Chapter 3 argued that interventions move us beyond decodability. Chapter 4 adds that, at least locally, we can predict the magnitude and sign of those intervention effects before running the intervention.

That is a meaningful upgrade in the evidence ladder:

```text
behavior
    the model predicts legal moves

decodability
    a probe can read board state from activations

local causal intervention
    changing a semantic direction changes logits

local quantitative prediction
    the Jacobian predicts the small intervention's effect
```

This strengthens the third rung. It does not jump to the later rungs. The Jacobian does not tell us which attention head or MLP neuron implements the effect. It does not isolate a circuit. It does not prove that G6 is represented exactly as our probe direction describes. It tells us that the model's output is locally sensitive to that direction in a quantitatively predictable way.

!!! question "Pause and think"
    If \(Jv\) is large for a semantic direction, does that prove the model represents the semantic concept exactly as we describe it?

No. It proves local sensitivity to an operational direction. The direction may mix the intended semantic feature with correlated features. Interpretation still depends on controls, generalization tests, and later component analysis.

This is the right level of evidence for this stage of the book. We have not found the whole legality circuit. We have found a reliable local measuring instrument.

## Why Local Does Not Mean Universal

The tangent-line approximation has a hidden remainder. Conceptually, it says:

```text
actual change
    =
linear prediction
    +
smaller higher-order terms
```

For very small perturbations, the linear term usually dominates. As the perturbation grows, the higher-order terms can become important. If we write:

$$
z(h + \delta h)
=
z(h) + J(h)\delta h + O(\|\delta h\|^2),
$$

the final term means "terms whose size shrinks roughly like the square of the perturbation size as the perturbation becomes small." You do not need the full formalism to use the intuition. Halve a tiny perturbation, and the linear part roughly halves. The quadratic part shrinks faster. That is why derivatives become accurate in the small-movement limit.

But several things can make the approximation degrade:

- a larger alpha
- a different board position
- a different semantic direction
- a different layer
- crossing an activation boundary or normalization regime where local behavior changes
- moving along a direction with stronger curvature

In a Transformer, these issues are not exotic. The model contains attention patterns, MLP nonlinearities, layer normalization, and many interacting features. A residual edit that is tiny in one context may be large enough to change the downstream computation in another. A semantic direction that is clean for one square may be entangled for another. A move logit that is locally sensitive to a direction in one position may ignore it in a different position.

So the Jacobian is not a global map of the model. It is a local report for one point in activation space.

That local nature is a limitation, but it is also what makes the Jacobian experimentally useful. Instead of making a vague claim that a feature matters, we can ask a precise local question:

```text
At this activation, in this direction, what output change does the
network's own local geometry predict?
```

## The Geometric View

At each Othello position, the nonlinear model has a different local Jacobian.

This means each current board and move history comes with its own local geometry. The same semantic direction can point into a region of downstream sensitivity in one context and a region of relative insensitivity in another.

For example, imagine a direction associated with "G6 is mine rather than theirs." On one board, G6 might be part of a capture line that affects a legal move. On another board, G6 might be irrelevant to the next-move logits we care about. The residual direction may be similar, but the downstream effect can differ because the rest of the computation is context-dependent.

This is the entrance to the next question.

If every board position has its own Jacobian, are those local maps unrelated? Or are they variations of a shared transformation? Can we compare semantic directions after the Jacobian has transformed them? Can we average those transformed directions across examples and still learn something?

Those are not Chapter 4 claims. They are Chapter 4's exit ramp.

!!! question "Pause and think"
    Would you expect the same Jacobian at every Othello position? Why or why not?

Probably not exactly. The model's downstream computation depends on the current activation, and the activation depends on the board and move history. But if the model uses reusable computation, local Jacobians across related positions may share structure.

## The "Thoughts" Metaphor

The chapter title says "listening to the network's thoughts." That metaphor needs a warning label.

A Jacobian is not a transcript of hidden verbal reasoning. It does not translate activations into sentences. It does not tell us that the model is internally saying, "E3 is legal because D3 and C3 are opponent discs." It is narrower than that.

A Jacobian tells us how local changes to an internal numerical state affect downstream numerical quantities.

That is less romantic than mind-reading. It is also more experimentally useful. The Jacobian gives us a way to make quantitative predictions about interventions. We can test those predictions. We can compare directions. We can ask whether board-state handles have output consequences. We can later ask where in the model those consequences are transmitted or transformed.

Mechanistic interpretability benefits from metaphors only when they point back to measurements. Here the measurement is local sensitivity.

## What We Learned

The progression of this chapter was:

```text
one input, one output
    derivative as local slope

many inputs, one output
    gradient as a vector of local slopes

one chosen direction
    dot product as directional sensitivity

many outputs, many inputs
    Jacobian as a local linear map

one chosen perturbation
    Jv as predicted output-logit change

actual intervention
    finite-difference validation
```

In the executed Othello-GPT notebook, the concrete local function was defined by a TransformerLens hook at `blocks.4.hook_resid_post`, using the final token position of a 28-move prefix. The selected output was the E3 move logit, token 21, in a model with 512-dimensional residual activations and 61 output logits.

The first finite-difference sanity check showed that perturbing along the normalized E3-gradient direction produced logit changes very close to the first-order predictions. The semantic board-state check then used the normalized G6 mine-vs-theirs probe direction and found a maximum absolute prediction error of `0.000067` across the tested alphas.

This validates the Jacobian as a local measuring tool for the tested setup. It does not prove global linearity. It does not identify the full circuit. It does not remove the need for controls. It does give us a way to move from "the model changes when we nudge it" to "the model changes by about the amount its local derivative predicts."

That is a useful step.

## Try It Yourself

1. Scalar slope: suppose a local slope is \(0.4\) and \(\delta x = 0.02\). What is the predicted \(\delta z\)?
2. Gradient: given \(\nabla z = [2, -1]\) and \(v = [0, 1]\), compute \(\nabla z \cdot v\). Does moving in direction \(v\) locally increase or decrease \(z\)?
3. Geometry: describe what happens when \(v\) is parallel to the gradient, orthogonal to the gradient, and opposite the gradient.
4. Dimensions: for 61 logits and a 512-dimensional residual stream, give the shapes of \(J\), \(v\), and \(Jv\).
5. Finite difference: explain how you would empirically test whether a Jacobian prediction matches an actual residual-stream intervention.
6. Interpretation: why does accurate local linearization not imply that the model is globally linear?
7. Advanced: use PyTorch autograd or TransformerLens to compute a selected move-logit gradient with respect to a cached residual activation, then compare \(\epsilon \nabla_h z_m^\top v\) against a hook intervention \(z_m(h+\epsilon v)-z_m(h)\).

## The Next Mystery

We now know how to ask:

```text
For this board,
at this internal state,
what happens to a semantic direction as it flows downstream?
```

But the Jacobian is local. Another Othello position produces another Jacobian. A different semantic direction may be transformed differently. A different layer may change the geometry again.

So the next mystery is:

```text
Are these local maps variations of one shared transformation?
Could we average them?
Could we compare semantic directions after they are transformed?
And what would such a transformed semantic space mean?
```

That is Chapter 5: J-Space.

## References

- Executed notebook: `demos/Othello_GPT_Jacobian_Lens.ipynb`, sections `5. Sanity check: does the Jacobian predict an actual intervention?` and `9. Jacobian prediction vs actual board-state intervention`, on `diegovalverde/TransformerLens`, branch `othello-jspace-analysis`.
- Project research memory: [research log](../research/research_log.md), [experiment index](../research/experiment_index.md), [findings snapshot](../research/findings_snapshot.md), and [provenance](../research/provenance.md).
- Chapter 4 measured figure data: [jacobian_prediction_vs_intervention.json](../figures/jacobian_prediction_vs_intervention.json), generated by `scripts/generate_jacobian_validation_figure.py`.
- Li et al., [*Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task*](https://openreview.net/forum?id=DeG07_TcZvT), ICLR 2023.
- Neel Nanda, [*Actually, Othello-GPT Has A Linear Emergent World Representation*](https://www.neelnanda.io/mechanistic-interpretability/othello), 2023.
