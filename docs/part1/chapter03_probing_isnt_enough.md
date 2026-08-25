# Why Probing Isn't Enough

We trained a simple decoder.

It reconstructed the Othello board from the layer-4 residual stream with nearly 98% held-out accuracy under a strict game-level split.

It is tempting to stop there.

That would be a mistake.

The result from Chapter 2 is real evidence. It says that board state is highly linearly decodable from an internal activation. Given the move prefix, an external linear probe can recover the 64 square labels with high accuracy. The model was not shown a board image. The labels were computed by an Othello simulator. The readout was deliberately simple. This is a strong representation result.

But the probe is our computation.

It is not part of Othello-GPT. It is a trained instrument we attach after the model has already produced its activation. It demonstrates that information is accessible to an external reader. It does not demonstrate that the Transformer's downstream computation consults that information when it predicts the next move.

That distinction is the hinge of Part I.

Chapter 1 asked whether a board could exist internally. Chapter 2 showed that a board is linearly decodable from a particular residual-stream activation. This chapter asks what remains unresolved: if we can read the board from the residual stream, is Othello-GPT itself actually using that information?

The short answer is: not from probing alone.

## The Address and the Barcode

Imagine a package moving through a warehouse. On the outside of the package there is a human-readable address. There is also a machine-readable routing barcode.

If we stand beside the conveyor belt, we can read the printed address and predict the package's destination. Our prediction may be excellent. We might even build a little camera system that reads addresses with very high accuracy.

But the sorting machine may ignore the printed address entirely. It may scan only the barcode. In that case the destination is visible to us, and visible in a representation carried by the package, but the causal variable used by the machine is different.

The printed address is not useless. It is correlated with the route. It may have been produced from the same shipping database as the barcode. It may be a perfectly good representation of where the package is going. But observing that we can decode the destination from the address does not tell us which physical mark the sorting machine reads.

The Othello case has the same shape:

```text
activation
    contains board-correlated information

probe
    can read it

downstream Transformer
    may use that feature,
    another correlated feature,
    or some distributed transformation
```

The board probe is like our address reader. It tells us that the destination can be read from one part of the package. The remaining question is whether the machine itself uses that mark.

This is not a skeptical technicality. It changes what kind of evidence we need. A probe answers an observational question. A mechanism claim requires some kind of causal evidence.

!!! question "Pause and think"
    If a probe can decode the board perfectly, what is the strongest claim you can make? What claim would still require a different experiment?

## Probes Are Observational Instruments

A probe gives us a statement like:

```text
When D3 is mine, activation coordinate patterns tend to look like X.
```

That is an observational statement. It is about a statistical relationship between activations and labels.

An intervention asks a different kind of question:

```text
If I change the internal representation toward "D3 is theirs",
does the model's output respond?
```

The words are similar enough that it is easy to slide from one to the other. We should not. Prediction is not intervention. Correlation is not causal effect. Decodability is not use.

<figure markdown>
![Observation versus intervention](../figures/observation_vs_intervention.svg)
<figcaption>
Observation asks what an external readout can recover from an activation. Intervention asks whether a controlled change to an activation changes the model's later computation.
</figcaption>
</figure>

In the left side of the figure, the activation flows into a probe, and the probe produces a board label. The Transformer does not have to do anything with that label. The probe is outside the model.

In the right side, we change the activation itself, allow the rest of the Transformer to continue from that edited state, and measure whether the logits change. That experiment is still not a complete mechanism. But it is a causal test in a way that probing is not.

The key distinction is between asking:

```text
What can we read?
```

and asking:

```text
What does changing it do?
```

Both questions matter. The first one gives us candidate variables. The second begins to test whether those variables participate in the computation.

## Why a Linear Probe Can Still Mislead

Linear probes are not arbitrary black boxes. That is why Chapter 2 used one. A linear readout cannot run a new Othello simulator inside itself. It has to classify each square by taking weighted sums of residual-stream coordinates. High linear accuracy therefore means the board information is arranged in an accessible way.

But even a linear probe can mislead if we ask it the wrong question.

There are at least four possibilities after a successful probe.

First, the straightforward case: the board feature is represented and downstream computation relies on it. In this case, the probe has found a direction or subspace closely related to what later layers use. If we perturb that feature carefully, relevant output logits should change.

Second, the correlated-passenger case: board information is represented because it correlates with useful history features, but the downstream computation ignores the particular probe direction. For example, a move-history pattern might reliably imply both a board state and a legal next move. The activation may contain both. The probe can read the board, while later layers use the history cue.

Third, the redundant-representation case: several encodings contain board information. Destroying or editing one readout direction may have little effect because another representation carries the same fact. A failed intervention on one direction would not automatically prove that board information is unused. It might prove only that this direction was not the relevant bottleneck.

Fourth, the misaligned-readout case: the probe's decision boundary may combine features in a way that is useful for us but not aligned with the model's own causal basis. The probe can draw a clean separating hyperplane through activation space without that hyperplane corresponding to a variable that downstream components read as a unit.

<figure markdown>
![Decodable does not necessarily mean used](../figures/decodable_not_necessarily_used.svg)
<figcaption>
Conceptual possibilities after a successful probe. The readout direction may be causally used, correlated with what is used, redundant with another encoding, or useful for classification while misaligned with the model's own causal basis.
</figcaption>
</figure>

This is why the sentence "the board is in the model" needs care. Chapter 2 supports a precise version: board state is linearly decodable from layer-4 residual activations under the strict split used in the notebook. That is not the same as saying the model has a symbolic board variable, or that every legal-move computation reads the probe's square directions.

!!! question "Pause and think"
    If ablating one probe direction has no effect, does that prove the model does not use a board representation?

No. Redundancy, misalignment, or an incorrect intervention site could all explain the result. A negative result would still teach us something, but it would need careful interpretation.

## From a Probe to a Handle

The useful thing about the Chapter 2 probe is that it gives us more than a classification score. Its weights define candidate semantic directions.

For a square \(q\), the probe has one weight vector for "mine" and another for "theirs." Their difference is:

$$
v_{q,\text{mine-vs-theirs}}
=
W_{q,\text{mine}} - W_{q,\text{theirs}}.
$$

Until now, \(v\) was something we read along. If an activation has a larger projection in the "mine" direction than the "theirs" direction, the probe tends to classify that square as mine.

Now we can ask a new question.

What if we move along the direction?

Let \(h\) be the original residual state. Let \(v\) be a semantic direction, such as the mine-vs-theirs direction for D3. Let \(\alpha\) be an intervention strength. We can define an edited activation:

$$
h' = h + \alpha v.
$$

This equation is simple, but conceptually it is the first activation intervention in the book.

The pieces are:

```text
h
    original residual state

v
    semantic direction defined by the probe

alpha
    how far we move along that direction

h'
    edited residual state
```

If \(\alpha\) is positive, the edit moves in the direction we have operationally associated with the feature. If \(\alpha\) is negative, it moves the other way. Testing both signs is useful because a genuine directional effect should not behave like arbitrary noise.

We should describe this carefully. We are not literally changing the Othello board. We are not opening the model and flipping a named variable from "opponent" to "mine." We are adding a vector to a residual-stream activation in a direction defined by a trained probe.

That is a narrower claim, and it is the right one.

<figure markdown>
![Semantic residual-space intervention](../figures/semantic_residual_intervention.svg)
<figcaption>
Operational residual-space intervention, not a literal board edit. The true board still has D3 as "theirs"; the experiment edits the residual vector in a direction associated with D3 becoming more mine-like under the probe.
</figcaption>
</figure>

## Whispering Into the Model

Suppose the actual board says:

```text
D3 = opponent
```

The probe has a direction associated with:

```text
D3 becoming more mine-like
```

We can add a small amount of that direction to the residual stream and then let the rest of the Transformer run. Conceptually, we are asking:

```text
What happens downstream if this internal state becomes slightly more
consistent with D3 being mine?
```

The word "slightly" is doing real work. A small edit is easier to interpret than a large one. A large edit can push the activation into a strange part of space where many unrelated features change at once. A small edit gives us a more local question: near this actual activation, in this particular direction, how do later computations respond?

This is a residual edit in a direction operationally associated with the probe's D3 mine-vs-theirs distinction. It is a way of whispering into the model's internal state, not a guarantee that we have changed one clean human-readable belief.

!!! question "Pause and think"
    If changing a semantic direction changes every output logit dramatically, is that good evidence for a specific board feature?

Probably not. It may mean the intervention was too large or too nonspecific. Specific causal evidence should connect the edited feature to relevant output changes, while controls help rule out generic disruption.

## What Should We Measure?

The model ultimately produces logits. Let:

$$
z_m(h)
$$

be the logit for candidate move \(m\), given that the downstream computation starts from internal state \(h\).

After the intervention, the corresponding logit is:

$$
z_m(h + \alpha v).
$$

The finite change is:

$$
\Delta z_m
=
z_m(h + \alpha v) - z_m(h).
$$

In words: run the model from the original activation and record the logit for move \(m\). Then edit the activation by adding \(\alpha v\), run the rest of the model again, and record the same logit. The difference is the effect of that intervention on that logit.

If \(\Delta z_m\) is systematically different from zero for moves that should depend on the edited board feature, that gives causal evidence.

But causal evidence for what?

Only for this local residual edit, at this layer, at this token position, in this direction, and at this intervention strength. That qualification matters. It keeps the claim tied to the actual experiment rather than turning it into a broad statement about the whole model.

For example, if nudging a D3 mine-vs-theirs direction changes a nearby move logit in a rule-relevant direction, that supports local causal relevance of that semantic direction for that output. It does not yet tell us which attention head reads the direction, whether an MLP transforms it, whether another square representation participates, or whether the same effect generalizes across all boards.

This is how the evidence ladder works.

<figure markdown>
![Evidence ladder for mechanistic claims](../figures/evidence_ladder.svg)
<figcaption>
A pedagogical evidence hierarchy for this investigation. It is useful for keeping claims separate, not a universal formal taxonomy.
</figcaption>
</figure>

Behavior is the starting point: the model predicts legal moves. Decodability is stronger: board information can be read from an activation. Local intervention is stronger again: changing a semantic direction changes outputs. Component localization asks which pieces of the model matter. Path or mediation evidence asks how the effect travels. A mechanistic explanation should eventually describe how the computation is implemented.

No single rung gives the whole story.

## The Off-Manifold Problem

There is a caveat hidden inside the equation:

$$
h' = h + \alpha v.
$$

Mathematically, this is always a valid vector. The residual stream lives in a 512-dimensional vector space, and adding another 512-dimensional vector produces another point in that space.

But neural activations encountered during ordinary model execution do not fill the whole space evenly. They occupy some structured subset of it. They come from real move prefixes passing through real layers of this trained Transformer.

A photograph gives a useful analogy. A photograph with one million pixels lives in a million-dimensional pixel space. You can add arbitrary noise to those pixels and still get a valid array of numbers. But the result may not correspond to any plausible photograph. It may be a mathematical image that no camera would naturally produce.

Similarly, \(h + \alpha v\) is a mathematically valid residual vector, but it may not correspond to any naturally occurring Othello computation.

This is the off-manifold problem.

It does not make interventions useless. It makes experimental discipline necessary. Interventions should be small enough to remain locally interpretable. They should be compared with controls. They should be checked across multiple examples when the claim is meant to generalize. If we rely on a local analysis, the observed effects should behave approximately linearly in the small-perturbation regime.

!!! question "Pause and think"
    Why might a small intervention be more interpretable than a large one?

A small edit asks how the model behaves near an activation it actually produced. A large edit may create a state that is dominated by artifacts of the intervention rather than by the feature we intended to test.

## Semantic Validity Is Not Automatic

The direction \(v\) is defined by a classifier. That does not guarantee that adding \(v\) changes exactly one human-readable feature.

"Make D3 more mine-like" is an operational description, not a magic spell. The direction may also move along features correlated with D3's state. In Othello, those could include:

- game phase
- neighboring occupancy
- current legal moves
- move-history structure
- other square states correlated with D3

This is why controls matter.

Good controls might include unrelated square directions, opposite-sign edits, multiple intervention strengths, shuffled semantic labels, and finite-difference checks. Which controls are necessary depends on the claim. A claim about a single example can be supported by a smaller set of checks than a claim about a general circuit. A claim about causal relevance is weaker than a claim about a complete mechanism.

The important habit is to ask what else could explain the effect.

If every semantic direction changes the same move logit by about the same amount, then the result may reflect generic sensitivity to residual perturbations rather than square-specific board information. If a direction changes many unrelated logits more strongly than the intended move, the intervention may be too broad. If only one carefully chosen example works, the effect may be real but not general.

Controls do not make interpretation automatic. They narrow the space of explanations.

!!! example "From reading to intervening"

    Observational question:
    Can a probe decode D3's state?

    Interventional question:
    If we nudge the residual stream along the D3 semantic direction, does a relevant move logit change?

    Stronger mechanistic question:
    Which components transmit and transform that effect?

## What Our Experiments Support So Far

The research memory records more than the Chapter 2 probe result. In the executed notebook, semantic board-state directions were also used in local residual-space tests. Those edits produced measurable changes in move logits. More importantly, the changes were extremely predictable from a local linear approximation.

That sentence is intentionally incomplete.

We are not going to unpack the local linear approximation here. The point for this chapter is not the mathematical machinery. The point is the kind of question we have now learned to ask.

The board probe gave us a semantic handle. A residual intervention asks whether moving that handle affects outputs. The executed notebook supports the claim that board-state directions are locally causally relevant to logits in tested examples. The research memory labels this as strong evidence, while keeping the scope local: tested positions, tested directions, tested layer, and small perturbations.

That is a stronger claim than decodability. It is still not a complete mechanism.

## What Would Convince Us?

For a stronger causal claim, we would want several kinds of evidence to point in the same direction.

1. The direction has a clear semantic interpretation under a held-out probe.
2. A small intervention changes relevant outputs.
3. The opposite-sign intervention changes or reverses the effect appropriately.
4. Unrelated directions have weaker or different effects.
5. The effect is reproducible across positions.
6. Predictions hold in the local small-perturbation regime.
7. Later component or path analyses explain where the effect travels.

No single item proves a full mechanism. Together, they narrow the space of explanations.

The checklist also shows why probes remain valuable. The first item is still a probe-style result. We need readable directions before we can make targeted semantic interventions. The mistake is not using probes. The mistake is treating a probe as the end of the investigation.

## Causal Relevance Is Not Mechanism

Even if changing D3's semantic direction changes E3's logit, we still do not know the full computation.

We do not yet know which attention head reads the relevant information. We do not know whether an MLP transforms it into a rule-sensitive feature. We do not know whether the effect depends on another square representation. We do not know whether the model tests a capture ray in anything like the symbolic Othello rule. We do not know whether the effect is direct or mediated through later components. We do not know whether the same pattern generalizes across boards, phases, and move families.

So we should keep the notation honest:

```text
semantic intervention
    != complete circuit explanation
```

This distinction will matter later when we descend into layers, components, and candidate neurons. A component can be causally important without being the whole algorithm. A neuron can have a selective effect without implementing a clean symbolic rule. A direction can be locally causal without being the only representation of a board fact.

Mechanistic interpretability is not a race to rename every vector as a concept. It is a process of building evidence that a computation flows through particular representations and components in particular ways.

## Try It Yourself

1. Explain in your own words why probe accuracy is observational evidence rather than causal evidence.
2. For \(h' = h + \alpha v\), write the edited activation for \(\alpha = \epsilon\) and \(\alpha = -\epsilon\). Why are both signs useful?
3. Design an unrelated-square control for an intervention on the D3 mine-vs-theirs direction.
4. Give a scenario in which board state is linearly decodable but the chosen probe direction is not causally important.
5. Explain the off-manifold concern without using the word "manifold."
6. Advanced: implement a TransformerLens hook that adds \(\epsilon v\) to one residual-stream position and records the resulting logit deltas.

## The Next Mystery

We can perturb one semantic direction and measure one set of logit changes. That already moves us beyond observation.

But doing interventions one by one is clumsy. We would like something more general. For a tiny movement in any residual-space direction, can we predict the resulting change in the output?

If we could measure the local slope of the model around its current hidden state, we could predict the first-order consequence of a small internal nudge.

Chapter 4 introduces that local slope.

Its name is the Jacobian.

## References

- Executed notebook: `demos/Othello_GPT_Jacobian_Lens.ipynb`, sections `7. Train a linear mine / theirs / empty board probe`, `5. Sanity check: does the Jacobian predict an actual intervention?`, and `9. Jacobian prediction vs actual board-state intervention`, on `diegovalverde/TransformerLens`, branch `othello-jspace-analysis`.
- Project research memory: [provenance](../research/provenance.md), [findings snapshot](../research/findings_snapshot.md), [research log](../research/research_log.md), and [experiment index](../research/experiment_index.md).
- Li et al., [*Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task*](https://openreview.net/forum?id=DeG07_TcZvT), ICLR 2023.
- Neel Nanda, [*Actually, Othello-GPT Has A Linear Emergent World Representation*](https://www.neelnanda.io/mechanistic-interpretability/othello), 2023.
