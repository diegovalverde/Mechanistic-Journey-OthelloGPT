# Notebook Walkthrough

This appendix is a roadmap to the experimental notebook:

```text
/Users/diegovalverdegarro/workspace/projects/TransformerLens/demos/Othello_GPT_Jacobian_Lens.ipynb
```

The notebook is the primary executed source for the book's new experiments. The polished chapters are not a substitute for the notebook. If a future revision wants to strengthen a scientific claim, the provenance path is:

```text
executed notebook -> research memory -> book prose
```

This walkthrough explains what each major notebook section is asking, what it consumes, what it produces, which book chapter uses it, and how strong the evidence is. It does not add new findings.

## How to Read the Notebook

The notebook grew as an investigation. It starts by making TransformerLens usable for local Jacobian analysis, then trains a strict board probe, then shifts from raw move logits to legality contrasts, then localizes a late layer and component, and finally tests whether high-attribution MLP7 neurons behave like recognizable Othello rule detectors.

That order matters. The notebook is not a single clean pipeline where every later cell simply depends on the immediately previous result. Some sections are exploratory, some are validation sections, and some are negative controls. The book deliberately preserves this structure because it is part of the evidence discipline.

There are three broad phases:

1. Build tools: gradients, hooks, board simulator, board probe, semantic directions.
2. Establish causal geometry: Jacobian prediction, J-space transport, legality contrast, layer sweep, component attribution and ablation.
3. Characterize candidates: MLP7 neuron attribution, group ablation, relational-condition data, matched controls, input/output geometry, and end-to-end neuron tests.

The missing phase is also important: rescue. No executed rescue experiment is present in the notebook.

The notebook should be read as executable evidence plus lab notes, not as a polished API. Some section numbers repeat because later work was appended after earlier summaries. That is harmless as long as claims cite the section title and experimental role rather than relying only on numbering.

## Section Map

| Notebook section | Question | Book chapter | Inputs | Outputs | Evidence level |
| --- | --- | --- | --- | --- | --- |
| 1. Re-enable gradients only for J-space analysis | How can the inference-oriented Othello demo support autograd without changing its global style? | Chapter 4, Appendix C | Loaded `HookedTransformer`, PyTorch autograd context | Local `torch.enable_grad()` pattern | Setup |
| 2. Find a valid Othello token sequence already loaded by Neel's notebook | Which tensor in the inherited notebook namespace is a usable Othello token sequence? | Chapter 4 | Existing tensors from the original demo | `sample_input` and token-sequence diagnostics | Setup |
| 3. Define a differentiable intervention at one residual stream | How can a residual activation be treated as a local coordinate system? | Chapter 4 | Tokens, hook name, layer, source position | Hook function that injects differentiable `delta` | Method |
| 4. Move-logit gradients: a local Othello J-space | Which residual directions locally increase candidate move logits? | Chapter 4 | Differentiable residual intervention | Move-logit gradients with shape `[61, 512]` conceptually | Method |
| 5. Sanity check: does the Jacobian predict an actual intervention? | Does the gradient construction match finite residual interventions? | Chapter 4 | Selected E3 logit, normalized gradient direction | Finite-difference validation table | Strong evidence for the hook/Jacobian setup |
| 6. Compare J-space directions for all candidate moves | Which moves have similar local residual sensitivities? | Chapter 4 and 5 | Move-logit gradients | Move-gradient cosine comparisons | Exploratory |
| 7. Bridge to Othello board probes | Can probe directions be compared with move-logit gradients? | Chapter 4 | Board semantic directions and move gradients | Direction-to-move sensitivity scores | Method bridge |
| 7. Train a linear mine / theirs / empty board probe | Can board state be decoded from residual activations under a strict split? | Chapter 2 | Synthetic random-play games, simulator labels, `blocks.4.hook_resid_post` | Validation accuracy `0.9796` and class accuracies | Established |
| 8. Project move-logit J-space into board coordinates | Which board squares are locally sensitive for a selected move logit? | Chapter 4 | Probe directions, move-logit gradients, board state | 8x8 sensitivity views | Exploratory support |
| 9. Jacobian prediction vs actual board-state intervention | Do semantic probe-direction edits produce logit changes predicted by the Jacobian? | Chapter 4 | G6 mine-vs-theirs direction, E3 logit gradient, alphas | Max prediction error `0.000067` | Strong evidence |
| 10. Local J-space vs averaged J-space | Is a transformed semantic direction stable across contexts? | Chapter 5 | Layer-4 G6 mine-vs-theirs direction, 100 sampled prefixes | Local-vs-average cosine `0.617840`, JVP validation | Moderate evidence |
| 11. A legality score instead of a raw move logit | Can the target separate legality from ordinary move preference? | Chapter 7 | Current legal/illegal move sets, logits | Selected move legality contrast | Method |
| 12. Which board squares causally support legality? | Does legality sensitivity align with actual capture-line squares? | Chapter 7 | Legality-gradient target, probe directions, simulator capture lines | Capture-line sensitivity table | Example-level evidence |
| 13. Counterfactual board-state interventions | What happens when semantic residual edits approximate board changes? | Chapter 7 | Probe-derived board directions, selected move | Residual-edit effects | Local causal support, with off-manifold caveat |
| 14. Build true board counterfactuals when possible | Which single-square board changes would matter under the simulator? | Chapter 7 | Explicit simulator board state | Simulator-only counterfactual labels | Ground-truth control, not a model result |
| 15. Dataset-level legality relevance test | Does capture-line sensitivity separate from controls across positions? | Chapter 7 | Sampled prefixes, capture lines, controls | Weak early dataset-level effect | Weak evidence |
| 16. Path structure, not just square relevance | Does distance along a capture ray explain sensitivity? | Chapter 7 | Capture and noncapture ray structures | Distance-bucket summaries | Exploratory, weak |
| 17. Which layer computes legality? | Where among tested layers is capture-line enrichment strongest? | Chapter 7 | Layer probes for layers 2, 4, 6, 7 | Layer-7 ratio `2.251362` | Strong evidence |
| 19. Is the layer-7 legality enrichment real? | Does the layer-7 result survive bootstrap and shuffle controls? | Chapter 7 | Layer-7 dataset, position bootstrap, shuffled-square null | Ratio `2.746573`, CI and empirical p-value | Strong evidence |
| 20. Layer-7 component decomposition | Which layer-7 component writes align with the legality-gradient direction? | Chapter 8 | Layer-7 cached components and gradients | Component attribution ranking, MLP7 largest mean absolute attribution | Strong attribution evidence |
| 21. Causal ablation of layer-7 components | Which layer-7 component matters when replaced and rerun? | Chapter 8 | Mean replacement hooks for heads and MLP7 | MLP7 largest mean absolute ablation effect | Strong causal component evidence |
| 22. Does a candidate component specifically care about capture lines? | Does the top component effect track capture-line structure? | Chapter 8 | Capture buckets and component effects | Mixed bucket/correlation summaries | Weak/exploratory |
| 23. Attention-pattern inspection | Which attention patterns suggest possible information routing? | Chapter 8 | Attention cache for selected heads | Pattern inspection | Hypothesis-generating |
| 24. Capture-line intervention x component ablation interaction | Does component ablation reduce semantic-edit effects? | Chapter 8 and 9 | Capture-line semantic edits and component ablations | Example-level mediation-like quantities | Weak diagnostic |
| 25. What is MLP7 doing? | Which MLP7 neurons have high legality-related attribution? | Chapter 8 | MLP7 post-activations, output weights, legality gradient | Neuron attribution ranking | Moderate evidence |
| 26. MLP7 neuron ablation | Do attribution-selected neuron groups matter more than random groups? | Chapter 8 | Top-k MLP7 neurons, random controls, mean replacement | Top-k group effects separate from random groups | Moderate evidence |
| 27. Layer-7 board-direction mediation | Do transported board directions align with candidate component directions? | Chapter 9 | Transported board direction, component/neuron directions | Small displayed cosines | Weak/negative |
| 28. Final summary | What is the conservative state of layer-7 evidence? | Chapter 8 | Previous layer-7 results | Summary table | Synthesis |
| 28. Fix the candidate MLP7 legality neurons | Which candidate set is frozen for downstream analysis? | Chapter 8 and 9 | Existing MLP7 attribution ranking | Fixed top-20 neuron list | Setup |
| 29. What board conditions activate the candidate neurons? | Which relational board conditions should test rule-neuron hypotheses? | Chapter 9 | Random-play histories and simulator annotations | Condition-labeled dataset | Established dataset |
| 30. Candidate neuron activation by Othello condition | Do candidate neurons activate more for valid capture conditions? | Chapter 9 | MLP7 activations by condition | Small and mixed selectivity | Weak evidence |
| 31. Matched controls | Does valid-vs-invalid selectivity survive approximate matching? | Chapter 9 | Matched valid and invalid examples | 654 matched pairs, small mixed differences | Weak evidence |
| 32. Does neuron activation encode capture-line structure? | Do activations track line length, direction count, or flipped pieces? | Chapter 9 | Valid-condition examples and structure labels | Small correlations | Weak evidence |
| 33. Do board-state edits causally change candidate-neuron activation? | Can semantic residual edits affect candidate-neuron activations? | Chapter 9 | Semantic residual edits on the concrete example | Sparse example-level activation changes | Weak evidence |
| 34. What directions do the candidate neurons detect? | Do input weights align with board or transported directions? | Chapter 9 | MLP7 input weights, board directions, transported directions | Small input-weight cosines | Weak/negative |
| 35. Test conjunction vs linear feature detection | Do interactions explain activation beyond additive features? | Chapter 9 | Feature tables and candidate activations | Tiny delta `R2`, weak cross-validation | Weak/negative |
| 36. What do the candidate neurons write? | Are output weights aligned with legality-relevant directions? | Chapter 9 | MLP7 output weights and legality gradients | Neuron 399 strongest displayed write geometry | Moderate evidence |
| 37. End-to-end causal test of the strongest neurons | Do combined-evidence neurons matter on preferred relational examples? | Chapter 9 | Combined-evidence top-five, random and low-attribution controls | Small selected-neuron effects over 195 examples | Weak to moderate evidence |
| RESCUE: not executed / open | Can patching restore behavior after disruption? | Chapter 9 and 10 | Would require disrupted run plus patched intermediate activation | No executed result in current notebook | Not established |

## The Experimental Progression

### Board Simulator

The notebook cannot assume that the original synthetic dataset and helper library are available in the current checkout. It therefore rebuilds a small explicit Othello simulator. This simulator supplies the hidden board state after a move prefix, legal move sets, capture-line information, and relational labels.

This matters because Othello gives us ground truth. We do not have to guess whether C3 is an opponent piece on a capture line. The simulator can say so. That makes the domain unusually good for mechanistic interpretability: the latent state is hidden from the model input but available to the researcher.

### Strict Probe

The board probe section trains a linear mine/theirs/empty probe from residual activations at `blocks.4.hook_resid_post[:, -1, :]`. The split is strict at the game level, and exact-prefix overlaps are removed. This matters because a probe can otherwise exploit accidental overlap between training and validation prefixes.

The result is established board decodability: overall validation accuracy `0.9796` over 330 validation positions, with empty/mine/theirs accuracies `0.9976`, `0.9561`, and `0.9703`. The book uses this as a representation result, not a complete mechanism result.

### Semantic Directions

Once the probe is trained, differences between class weights become semantic directions. For example, a mine-vs-theirs direction for square G6 is an operational residual-space direction learned from the probe. Moving along that direction does not literally edit a simulator board. It is an internal activation edit whose semantics are inherited from the probe.

This distinction is central. A semantic direction is a handle. It needs causal validation before it can support claims about model use.

### Jacobian Sanity Check

The notebook constructs a differentiable residual edit at a chosen hook and token position. It then differentiates an output scalar with respect to that edit. The first sanity check moves along the normalized gradient direction itself and compares predicted and actual logit changes.

This verifies the plumbing: the hook is at the intended site, gradients are flowing through the intended downstream computation, and the finite intervention agrees with first-order prediction in the local regime.

### Semantic Intervention

The more Othello-specific Chapter 4 test moves along a normalized G6 mine-vs-theirs direction and measures the E3 logit. The directional derivative is positive, and actual residual interventions across alphas from `-0.1` to `0.1` match the linear prediction with maximum absolute error `0.000067`.

That is strong evidence for local causal relevance in the tested setup. It still does not prove that the probe basis is the model's native basis or that the same direction works globally.

### J-Space

The J-space section transports the same kind of semantic direction through downstream computation. The main comparison maps a layer-4 source residual direction to a final residual-stream target, before final normalization and unembedding. This is hidden-state transport, not merely a logit-space Jacobian.

The local JVP finite-difference validation reported cosine `0.999944` and relative error `0.010651`. The local transformed direction had cosine `0.617840` with the average transformed direction over 100 sampled positions. The result is moderate evidence: there is shared transformed geometry, but it is context dependent.

### Legality Contrast

Raw logits mix many things. A legal move can have a high or low raw score because of legality, strategy, move frequency, board context, or local preference among legal moves. The notebook therefore defines a legality contrast:

$$
L_m^\text{legality} =
z_m - \operatorname{mean}_{j \in I} z_j,
$$

where \(I\) is the set of currently illegal empty-square moves. The empty-square restriction matters. Occupied squares are not the right baseline for this contrast because their illegality is often trivial from occupancy alone.

### Capture-Line Analysis

Othello legality is relational. A legal move requires an empty target, a contiguous line of opponent discs, and a friendly terminator. The notebook projects legality gradients onto board-state semantic directions and compares true capture-line squares with controls.

The early dataset-level test at layer 4 was weak. That negative result matters because it prevents the story from pretending that the first aggregate analysis worked cleanly.

### Dataset Controls

The notebook repeatedly compares proposed rule-relevant sets with controls: unrelated occupied squares, unrelated empty squares, shuffled-square baselines, random neuron groups, matched invalid examples, and low-attribution controls. These controls are not decorative. They define what would count as an effect rather than a generic artifact.

For example, the layer-7 validation asks whether capture-line sensitivity exceeds unrelated occupied-square sensitivity and whether a shuffled-square null can reproduce the ratio.

### Layer Sweep

The layer sweep trains lightweight probes at layers 2, 4, 6, and 7, then compares capture-vs-unrelated sensitivity ratios. Layer 7 is strongest among the tested layers, with ratio `2.251362` and capture-minus-unrelated `0.026569`.

The phrasing "among the tested layers" is important. The notebook did not run every possible layer/probe/control combination. The result supports layer 7 as the strongest tested site for capture-line legality enrichment.

### Layer-7 Validation

The validation section reruns the layer-7 analysis with bootstrap and shuffle controls. The capture mean is `0.063157`, unrelated occupied mean is `0.022995`, ratio is `2.746573`, and difference is `0.040162`. The difference 95% confidence interval is `[0.035965, 0.044268]`; the ratio confidence interval is `[2.524081, 2.971348]`; the shuffled mean ratio is `1.046078`; the shuffled 95th percentile is `1.176336`; and the empirical permutation p-value is `0.003322`.

This is one of the book's strongest dataset-level results.

### Component Attribution

Component attribution decomposes the clean layer-7 residual update into current component writes and compares each write with a legality-gradient direction. In Chapter 8 notation:

$$
A_c = g^\top c.
$$

This is not a causal intervention. It is a local alignment score on the clean run. Over 30 positions, MLP7 had mean absolute component attribution `0.267666`, ahead of the strongest layer-7 attention heads in the displayed summary.

### Component Ablation

Ablation changes the model run. The notebook replaces component outputs with mean activations and recomputes the legality contrast. For whole-component ablation, the sign convention is:

```text
delta_legality_contrast = L_ablate - L_clean
```

MLP7 had mean signed effect `-0.105164` and mean absolute effect `0.262614`, the largest tested layer-7 component effect. This strengthens the MLP7 result because attribution and ablation are different evidence types that point to the same component.

### Semantic Mediation Diagnostic

The capture-line intervention x component ablation section asks whether ablating a candidate component reduces the effect of semantic capture-line edits. The section is useful, but the current result is example-level and mixed. It should be described as a mediation-like diagnostic, not a completed mediation result.

The later layer-7 board-direction mediation section is also weak/negative in the displayed summary: transported direction cosines with candidate directions are small. This constrains the mechanism story.

### Neuron Ranking

The MLP7 neuron attribution section ranks neurons using post-activation values and output-weight alignment with a legality-gradient direction. The fixed top-20 candidate list is:

```text
399, 1322, 1576, 366, 558, 1858, 1747, 495, 1167, 14,
1400, 272, 1673, 1953, 991, 734, 1000, 877, 125, 912
```

This is a candidate-generation step. It identifies neurons worth testing, not neurons that have already been interpreted.

### Top-N Neuron Ablation

The notebook then ablates top-k candidate groups and compares them with same-size random groups. The top-k groups separate sharply from random groups in the 30-position setting. This is moderate evidence that attribution found a causally relevant tail of MLP7 neurons.

The sign convention for the neuron-group figure is:

```text
legality_degradation = L_clean - L_ablate
```

This differs from the whole-component ablation convention. The book focuses on separation from random same-size groups rather than overinterpreting the sign.

### Relational-Condition Dataset

Chapter 9 needs to test rule-neuron hypotheses directly. The notebook builds a condition-labeled dataset from real random-play histories with categories such as valid capture, multiple capture, opponent without terminator, friendly adjacent, and empty adjacent.

The dataset sizes are established: `763` valid-capture, `515` multiple-capture, `654` opponent-without-terminator, `563` friendly-adjacent, and `467` empty-adjacent examples.

### Matched Controls

Simple valid-vs-invalid differences can be confounded by square location, game phase, local density, and related variables. The matched-control section greedily matches valid capture examples with invalid opponent-without-terminator examples by approximate structural features.

Across 654 matched pairs, activation differences remain small and mixed. That weakens the clean detector story.

### Structural Correlations

Within valid conditions, the notebook tests whether candidate activations correlate with line length, total flipped pieces, or number of capture directions. The displayed correlations are small. This weakens a simple "neuron encodes capture-line size" interpretation.

### Semantic Neuron Edits

The notebook applies semantic residual edits and measures candidate-neuron activations. Some example-level activation changes appear, especially for friendly-terminator edits on neurons such as 1322 and 125. The result is sparse and example-specific.

This supports the idea that residual board-state directions can affect MLP7 neurons, but it does not establish a dataset-level path.

### Input-Weight Geometry

Input-weight geometry asks what directions candidate neurons may read. The notebook compares MLP7 input weights with layer-7 board directions, transported layer-4 directions, capture aggregates, and legality gradients. The displayed cosines are small. This is a negative constraint on clean detector interpretations.

### Interaction Regression

A rule detector might respond to a conjunction such as "opponent adjacent and friendly terminator exists." The notebook compares additive and interaction regression models. The in-sample delta \(R^2\) is tiny, with maximum `0.002291`, and cross-validated performance is weak. This again argues against a simple single-neuron relational detector story.

### Output-Weight Geometry

Output-weight geometry asks what candidate neurons write. Here the evidence is stronger for some neurons, especially neuron 399. Its mean post-activation is `0.604659`, mean legality-gradient dot is `-0.127217`, and activation-by-legality-write score is `-0.076923`.

This supports a writer interpretation more than a clean detector interpretation. A neuron can matter because of what it writes, even if its activation is not a simple symbolic feature label.

### End-to-End Neuron Tests

The final causal test chooses strongest neurons by combined evidence and ablates them on preferred relational-condition examples. The combined-evidence top-five neurons are `[734, 1747, 1673, 125, 1167]`. Over 195 examples they averaged `-0.019919` legality degradation, compared with near-zero low-attribution controls and small random-control effects.

This is weak to moderate evidence for selective neuron participation. It is not a complete rule circuit.

### Rescue: Not Executed / Open

The notebook does not contain a rescue experiment. Existing patching is used for ablation or replacement, not for restoring behavior after a disruption. A true rescue test would disrupt a proposed computation, patch back a proposed intermediate activation or population, and show selective recovery of the relevant legality behavior.

This absence is not a footnote. It is why the book says the complete legality circuit is not established.

## Suggested Reader Paths

### Minimum Reproduction

Read and run:

1. Sections 1-5 for gradient and hook validation.
2. Section 7 for the strict board probe.
3. Section 9 for semantic Jacobian-predicted intervention.
4. Section 17 and 19 for layer-7 enrichment and validation.
5. Sections 20-21 for component attribution and ablation.

This path reproduces the main evidence backbone without the full neuron-characterization deep dive.

### Follow the Main Book

Read the notebook in the order of the chapters:

1. Board probe: section 7.
2. Jacobian validation and semantic intervention: sections 3-5 and 9.
3. J-space: section 10.
4. Legality contrast and layer sweep: sections 11-19.
5. MLP7 component work: sections 20-26.
6. Rule-circuit boundary and neuron characterization: sections 27-37.

This path matches the narrative arc from representation to causal geometry to component candidates.

### Neuron-Characterization Deep Dive

Start after the MLP7 localization is established:

1. Section 25 for attribution ranking.
2. Section 26 for top-k neuron ablation.
3. Section 28 for the fixed candidate list.
4. Sections 29-32 for relational-condition tests and matched controls.
5. Sections 33-36 for semantic edits and read/write geometry.
6. Section 37 for end-to-end neuron tests.

This path is the right one if you care about why the book rejects a clean single-neuron rule implementation.

## Troubleshooting Notes

If per-head results are missing, check that `model.set_use_attn_result(True)` was called before caching `blocks.L.attn.hook_result`.

If gradients are missing, check whether the tensor came from `run_with_cache`. Cached activations are usually detached. For Jacobian work, create a differentiable `delta`, insert it with a hook, and differentiate the downstream scalar with respect to `delta`.

If a result changes sign unexpectedly, check which metric is being reported. Whole-component ablation uses `L_ablate - L_clean`; the MLP7 neuron-group figure reports `L_clean - L_ablate`.

If a legality contrast looks strange, inspect the illegal baseline. In this book, legality contrast uses illegal empty-square moves, not all illegal tokens and not occupied-square moves.

If a cell becomes slow, reduce dataset sizes only for debugging and avoid reporting the resulting numbers as book evidence. The recorded results come from executed notebook outputs and research memory, not ad hoc runtime edits.

If a cell depends on earlier helper functions, run the notebook from the top or restart and run all. Many sections share simulator helpers, token mappings, hook helpers, cached probes, and candidate lists.

If a claim seems stronger in prose than in the notebook, prefer the notebook and research memory. In particular: rescue is not executed, the complete legality circuit is not established, and individual-neuron detector evidence is weak and mixed.

## What Not to Conclude from the Notebook

The notebook does not show that Othello-GPT contains a symbolic board object identical to the simulator board. It shows strong linear decodability and local causal relevance of probe-derived semantic directions.

The notebook does not show that layer 7 is the only layer involved in legality. It shows that layer 7 was the strongest among the tested layers for the capture-line enrichment measurement.

The notebook does not show that MLP7 alone computes legal moves. It shows that MLP7 is the strongest tested layer-7 component under attribution and mean-replacement ablation.

The notebook does not show that neuron 399, or any other candidate neuron, implements the Othello legality rule. It supports selective participation and output/write geometry for candidate populations, while input-side detector evidence remains weak and mixed.

The notebook does not show rescue or sufficiency. That is why the book ends with a distributed MLP7 population hypothesis rather than a completed legality circuit.
