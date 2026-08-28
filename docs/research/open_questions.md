# Open Questions

These questions should be answered from executed experiments before becoming stronger book claims. Completed or partially answered questions are preserved because their negative and mixed outcomes constrain future mechanisms.

## Completed / Constrained by Current Experiments

- **Layer-4 capture enrichment.** The first dataset-level layer-4 aggregate was inconclusive: mean(capture - unrelated occupied) `-0.000499`, bootstrap 95% CI `[-0.003646, 0.002784]`. This weak result motivated the later layer sweep.
- **Where capture-line enrichment appears among tested layers.** The layer sweep over layers 2, 4, 6, and 7 found layer 7 strongest, with capture-vs-unrelated ratio `2.251362` and capture-minus-unrelated `0.026569`.
- **Whether layer-7 enrichment survives validation controls.** The layer-7 capture-opponent validation found observed ratio `2.746573`, ratio 95% CI `[2.524081, 2.971348]`, shuffled 95th percentile `1.176336`, and empirical permutation p-value `0.003322`.
- **Whether directional capture relations are linearly decodable.** A linear directional capture probe recovered \(C(q,d)\) strongly on held-out data. Across `13,701` held-out valid targets, post4 top-1 true-direction accuracy was `0.9829209547` and macro AUROC was `0.9957207226`; post5 top-1 was `0.9837968032` and macro AUROC was `0.9985151889`.
- **Whether MLP5 sharpens the no-terminator distinction.** The hard valid-vs-no-terminator AUROC improved from `0.9600865639` at post4 to `0.9905437983` at post5, and the mean valid-minus-no-terminator probability gap improved from `0.2661360229` to `0.3829065047`. This constrains MLP5 as an interesting transformation site but does not establish a complete rule mechanism.
- **Which tested layer-7 component is strongest.** MLP7 ranked first under both mean absolute component attribution `0.267666` and mean absolute component-ablation effect `0.262614`.
- **Whether fixed attribution-ranked neurons are clean valid-capture detectors under current tests.** Current evidence is weak and mixed. Unpaired selectivity values are small, matched medians are `0.0`, and neuron 399 is negative under valid-vs-invalid activation comparisons.
- **Whether simple single-neuron conjunction regressions explain candidate activations.** Current interaction-regression gains are tiny, with maximum delta \(R^2\) `0.002291`.
- **Whether current candidate-neuron output geometry is informative.** Neuron 399 is better supported as a writer-like candidate than as a clean detector: mean post-activation `0.604659`, mean legality-gradient dot `-0.127217`, and activation-by-legality-write `-0.076923`.

## Neuron-Level Representation

- Do candidate MLP7 neurons detect individual board features such as target emptiness, adjacent opponent occupancy, or friendly terminator occupancy, or are these variables represented mainly in a distributed MLP subspace?
- Which candidate neurons, if any, have reproducible activation selectivity after stronger controls than the current matched valid-vs-invalid pairing?
- Is neuron 399 better understood primarily through output geometry rather than input-side semantics?
- Are valid-line effects suppressed by GELU sparsity, position sampling, the final-token measurement choice, or the condition taxonomy?

## Relational Conjunctions and Population Structure

- What computation produces the capture-predicate geometry already visible by post4?
- Why does MLP5 sharpen the no-terminator contrast?
- Does MLP6 transform or use the capture relation despite no AUROC gain?
- Is the learned directional capture relation causally aligned with the model's native computation?
- How does the upstream directional relation become layer-7 legality-aligned geometry?
- Do any MLP7 neurons detect relational conjunctions rather than additive board features under a better feature model?
- Is the important object a low-dimensional MLP7 subspace rather than individual neurons?
- Can population directions predict target-empty, opponent-line, and friendly-terminator structure on held-out positions?
- Can a subspace discovered from one set of ray directions generalize to other ray directions?
- Can a subspace discovered on short capture lines generalize to longer capture lines?
- Do multiple-capture positions use the same population geometry as single-capture positions?

## Mediation and Causality

- Does a small neuron group mediate a meaningful fraction of the MLP7 legality effect, or do current top-k ablations only identify a weak tail of a broader distributed computation?
- Can activation patching rescue legality after ablating candidate neurons?
- Can candidate-neuron or population activation patching rescue legality after semantic board-state interventions that break capture-line evidence?
- What fraction of the MLP7 component-level ablation effect is explained by the current candidate neuron set?
- Do semantic board edits lose their legality effect when MLP7 or candidate populations are disrupted in a dataset-level mediation experiment?

## Complete Capture-Line Circuit

- Is there a complete capture-line legality circuit from board representation to attention heads to MLP7 to logits?
- Which attention heads supply the relevant inputs to MLP7?
- Do the layer-7 attention heads identified by attribution or ablation carry board-state features, ray-structure features, or position-history features?
- Can path patching separate direct residual-stream board features from attention-mediated relational features?
- Does MLP7 transform information supplied by layer-7 attention, earlier residual paths, or both?

## Algorithmic Description

- Can the rule circuit be described algorithmically as target-empty plus opponent-line plus friendly-terminator implies legal move?
- Is the weak terminator signal in the layer-7 validation because terminator information is represented elsewhere, already settled earlier, or poorly captured by the square-local probe basis?
- Are current line-length and direction effects genuinely weak, or is the present candidate-neuron analysis looking at the wrong variables or positions?
- Does the mechanism generalize across sampled positions, game phases, and all legal move families?
- Do simulator-generated counterfactual board states expose cleaner causal structure than naturally sampled random-play positions?

## J-Space and Semantic Transport

- How does J-space geometry vary across semantic directions other than the tested G6 mine-vs-theirs direction?
- Which context variables explain the variation in \(J_x v\)?
- Does local-vs-average geometry change systematically by layer, game phase, capture relevance, or selected move?
- Can transported semantic directions identify population subspaces that are more stable than individual probe directions?

## Scope Beyond Othello

- Does the methodology generalize beyond Othello to other board games, symbolic environments, code models, simulators, or agents?
- Which parts of the workflow are specific to an available ground-truth simulator and which transfer to less structured domains?
- How should claim strength be calibrated when the latent variable is theory-dependent rather than externally defined?
- Can external verifiers such as interpreters, theorem checkers, parsers, or robotics simulators provide enough ground truth for Othello-style causal workflows in richer domains?
