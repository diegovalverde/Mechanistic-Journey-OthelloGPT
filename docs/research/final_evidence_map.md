# Final Evidence Map

This page records the state of the Othello-GPT investigation at book completion. It is grounded in `research_log.md`, `findings_snapshot.md`, and `experiment_index.md`.

## Established

- Othello-GPT is a small GPT-2-style decoder-only Transformer trained on Othello move sequences, not GPT-2 itself.
- The verified model dimensions are 8 transformer blocks, residual width 512, 8 attention heads per block, head width 64, MLP width 2048, output vocabulary 61, context length 59, GELU activation, and `LNPre` normalization.
- Layer numbering is zero-based. Layer 7 is the eighth and final transformer block.
- Board state is highly linearly decodable from `blocks.4.hook_resid_post[:, -1, :]` under the strict game-level split used in the executed notebook.
- The layer-4 board probe reached overall validation accuracy `0.9796` over 330 validation positions, with empty/mine/theirs accuracies `0.9976`, `0.9561`, and `0.9703`.
- The relational-condition dataset for candidate-neuron analysis contained `763` valid-capture, `515` multiple-capture, `654` opponent-without-terminator, `563` friendly-adjacent, and `467` empty-adjacent examples.
- No rescue experiment is present in the executed notebook. Existing patching is ablation or replacement, not rescue.

## Strong Evidence

- Probe-derived board semantic directions are locally causally relevant in tested examples.
- Local Jacobians accurately predicted small residual interventions in the Chapter 4 setup. The G6 mine-vs-theirs semantic intervention had maximum absolute E3 logit prediction error `0.000067` across alphas from `-0.1` to `0.1`.
- Capture-line semantic geometry is strongly enriched at layer 7 among the tested layers. In the layer sweep, layer 7 had capture-vs-unrelated ratio `2.251362` and capture-minus-unrelated `0.026569`, while layers 2, 4, and 6 were near one.
- The validated layer-7 capture-opponent enrichment was strong: capture mean `0.063157`, unrelated occupied mean `0.022995`, ratio `2.746573`, difference `0.040162`, difference 95% CI `[0.035965, 0.044268]`, ratio 95% CI `[2.524081, 2.971348]`, shuffled mean ratio `1.046078`, shuffled 95th percentile `1.176336`, and empirical permutation p-value `0.003322`.
- MLP7 is the strongest tested layer-7 component under both attribution and mean-replacement ablation.
- Over 30 positions, MLP7 had mean absolute component attribution `0.267666`, ahead of L7H0 `0.201140`, L7H2 `0.186625`, and L7H7 `0.180907`.
- Whole-component ablation over 30 positions gave MLP7 mean signed legality effect `-0.105164` and mean absolute effect `0.262614`, larger in absolute effect than L7H7 `0.109719`, L7H2 `0.094140`, and L7H0 `0.090048`.

## Moderate Evidence

- Local semantic transport is context dependent but not random in the tested J-space setup. The local transformed G6 mine-vs-theirs direction had cosine `0.617840` with the average transformed direction over 100 sampled Othello positions.
- Attribution-selected MLP7 neuron groups are much more causally important than random same-size groups under the tested mean-replacement neuron intervention.
- The fixed top-20 candidate MLP7 neurons by mean absolute attribution were `399, 1322, 1576, 366, 558, 1858, 1747, 495, 1167, 14, 1400, 272, 1673, 1953, 991, 734, 1000, 877, 125, 912`.
- Neuron 399 had the largest mean absolute MLP7 neuron attribution in the 30-position component analysis, `0.276027`.
- Top-k attribution-selected MLP7 neuron groups separated sharply from random groups. Top-1/top-2/top-5/top-10/top-20 mean legality degradations were `-0.137254`, `-0.153469`, `-0.204493`, `-0.335030`, and `-0.543530`, while same-size random means were `0.000685`, `-0.001759`, `-0.000735`, `0.002325`, and `0.012949`.
- Some candidate neuron output directions are meaningfully aligned with legality-gradient geometry. Neuron 399 had mean post-activation `0.604659`, mean legality-gradient dot `-0.127217`, mean legality-gradient cosine `-0.110732`, and activation-by-legality-write `-0.076923`.

## Weak Evidence

- Individual-neuron valid-vs-invalid selectivity is small and mixed. The largest positive unpaired standardized selectivities were neuron 1167 at `0.107989`, 734 at `0.094814`, 1747 at `0.081680`, 272 at `0.077419`, and 877 at `0.062688`; several high-attribution neurons were negative, including 399 at `-0.125210`.
- Matched valid-vs-invalid controls do not show a clean detector story. Across 654 matched pairs, the largest mean valid-minus-invalid activations were 991 at `0.069904`, 877 at `0.053469`, 272 at `0.047312`, and 1167 at `0.039280`; medians were `0.0` for all listed candidates.
- Capture-line length and direction dependence are weak in the current candidate-neuron tests. Across 1,278 valid-condition examples, the largest displayed correlations were small, including `0.040129` for longest line, `-0.070929` for number of capture directions, and `-0.071322` for total flipped pieces.
- Semantic residual edits can causally affect candidate-neuron activations, but current effects are sparse and example-specific. Friendly-terminator edits changed neuron 1322 by about `+0.019275` and `-0.019228`, and neuron 125 by about `+0.018370` and `-0.017688`.
- MLP7 input-weight geometry does not isolate clean board-feature detectors under the tested comparisons. For example, neuron 14 had mean absolute cosine `0.040991` to layer-7 capture/terminator directions and `0.023797` to controls.
- Single-neuron conjunction evidence is weak. The largest in-sample interaction-regression delta \(R^2\) was `0.002291`, and cross-validated \(R^2\) values were near zero or negative for many neurons.
- End-to-end selected-neuron causal effects are positive but small. The combined-evidence top-five neurons averaged `-0.019919` legality degradation over 195 examples, compared with `-0.000001` for low-attribution controls and `-0.003751` for random controls.

## Hypotheses

- Othello-GPT uses a board-like internal representation to support legal-move prediction, but the precise internal representation may not match the probe basis exactly.
- Some late computation transforms board-state information into capture-line-sensitive legality evidence.
- MLP7 participates in a distributed legality computation that combines board-state features with capture-line structure.
- Some candidate MLP7 neurons may be better understood by output/write geometry than by input-side detector semantics. Neuron 399 is the central current example.
- The meaningful neuron-level object may be a low-dimensional MLP7 subspace or distributed population rather than a single monosemantic neuron.
- Layer-7 attention heads may supply information that MLP7 reads, but the current evidence has not established the attention-to-MLP path.

## Open Questions

- Does a small neuron group mediate a meaningful fraction of the MLP7 legality effect, or are current top-k ablations only identifying a tail of a broader distributed computation?
- Can activation patching rescue legality after ablating candidate neurons or editing semantic board-state features?
- Which layer-7 attention heads supply board or ray information used by MLP7?
- Are important variables represented in individual neurons, a low-dimensional MLP subspace, or a distributed pattern across many neurons?
- Can the rule circuit be described algorithmically as target-empty plus opponent-line plus friendly-terminator, or is the model using correlated shortcuts?
- Do observed effects generalize across sampled positions, game phases, capture directions, line lengths, and multiple-capture positions?
- How does J-space geometry generalize across semantic directions, layers, and context families?
- Does the methodology replicate in other domains with known latent state?

## Experiments Still Needed

- Dataset-level mediation testing whether semantic board edits lose their legality effect when MLP7 or candidate populations are disrupted.
- Rescue experiments that patch back candidate population activations after semantic disruption or neuron-population ablation.
- Attention-to-MLP7 path patching to identify which heads, if any, supply relevant inputs to MLP7.
- Population-subspace discovery inside MLP7, with held-out tests against relational capture conditions.
- Cross-ray, line-length, multi-capture, and game-phase generalization tests.
- Sign-resolved semantic interventions for target-empty, opponent-line, and friendly-terminator features.
- Alternative probe bases and subspace probes to test whether the current probe directions mismatch the model's causal basis.
- Counterfactual board-state generation with the simulator to improve matched causal controls.
- Cross-model replication on independently trained Othello-GPT variants or related synthetic sequential domains.
