# Research Log

Every new experiment should be recorded here before it becomes polished book prose. Entries should preserve the question, method, result, interpretation, confidence, notebook location, and figure/table references.

## 2026-08-25 — Characterizing the Layer-7 legality computation

### Question

Can specific MLP7 neurons be connected to the relational structure of Othello legality?

More specifically: after establishing that layer 7 has unusually strong capture-line enrichment and that MLP7 is the strongest causal layer-7 component in ablation, can we identify candidate MLP7 neurons that respond to the relational pattern required by Othello legality: empty target, adjacent opponent line, and friendly terminator?

### Experiment

The executed notebook first revalidated the broader setup:

- A strict split board-state probe was trained from residual activations.
- A legality contrast was defined for a selected move by comparing the selected legal move against illegal moves.
- Capture-line sensitivity was measured across board squares and across layers.
- Layer-7 component decomposition and component ablations tested whether MLP7, attention heads, or the whole residual stream most affected the legality contrast.

The newest phase then focused on MLP7 neurons:

- Candidate MLP7 neurons were fixed from attribution rankings over 30 positions.
- Neuron-group ablations compared top-attribution neurons with random groups.
- A relational-condition dataset was built with valid captures, multiple captures, opponent-without-terminator invalid cases, friendly-adjacent cases, and empty-adjacent cases.
- Candidate-neuron activations were compared across those conditions.
- Matched controls paired valid captures with opponent-without-terminator cases while approximately controlling target square, target region, game phase, adjacent opponent-run length, and occupancy.
- Valid captures were bucketed by line length and number of capture directions.
- Residual-space semantic board edits were applied to the concrete analysis example, then candidate-neuron activations and legality contrast were measured.
- MLP7 input weights were compared to board-state and transported-board directions.
- Additive vs interaction regressions tested whether an opponent-adjacent by friendly-terminator conjunction explained candidate activations.
- MLP7 output weights were compared with legality-gradient directions and legal-vs-illegal output directions.
- Single-neuron end-to-end ablations compared a combined-evidence top-five neuron group against low-attribution and random controls.

### Result

Board-state representation and local causal validation:

- The board probe reached 0.9796 overall validation accuracy across 330 validation positions.
- Per-class validation accuracy was 0.9976 for empty, 0.9561 for mine, and 0.9703 for theirs.
- A board-state residual intervention had max absolute Jacobian prediction error 0.000067.
- Local vs averaged J-space cosine for the tested G6 mine-vs-theirs direction was 0.617840.

Layer-7 legality enrichment:

- In the layer sweep, layer 7 had capture-vs-unrelated ratio 2.251362 and capture-minus-unrelated 0.026569.
- Layers 2, 4, and 6 were much closer to baseline: ratios 1.076216, 1.098189, and 1.006662.
- The validated layer-7 enrichment test found capture mean 0.063157 and unrelated occupied mean 0.022995.
- The observed ratio was 2.746573.
- The mean difference was 0.040162 with bootstrap 95% CI [0.035965, 0.044268].
- The ratio 95% CI was [2.524081, 2.971348].
- The shuffled-square control had mean ratio 1.046078, shuffled 95th percentile 1.176336, and empirical permutation p-value 0.003322.

Layer-7 components:

- In the single example, MLP7 had legality attribution -0.560007, the largest absolute component attribution.
- Over 30 positions, MLP7 had mean legality attribution 0.126682 and mean absolute attribution 0.267666, rank 1 among layer-7 components.
- Component ablation over 30 positions gave MLP7 mean signed effect -0.105164, mean absolute effect 0.262614, median effect -0.128594, and mean selected-logit effect -0.101680.
- The next largest component-ablation absolute effects were L7H7 at 0.109719, L7H2 at 0.094140, and L7H0 at 0.090048.

Candidate MLP7 neurons:

- The fixed candidate MLP7 neurons were 399, 1322, 1576, 366, 558, 1858, 1747, 495, 1167, 14, 1400, 272, 1673, 1953, 991, 734, 1000, 877, 125, and 912.
- Neuron 399 had mean legality attribution -0.219701 and mean absolute attribution 0.276027 over 30 positions.
- The top-20 neuron group ablation had mean legality degradation -0.543530, while the same-size random mean was 0.012949.
- Top-1/top-2/top-5/top-10 selected-neuron group degradations were -0.137254, -0.153469, -0.204493, and -0.335030.
- Corresponding random means were 0.000685, -0.001759, -0.000735, and 0.002325.

Relational-condition dataset:

- The condition dataset contained 763 valid-capture examples, 515 multiple-capture examples, 654 opponent-without-terminator examples, 563 friendly-adjacent examples, and 467 empty-adjacent examples.
- The largest unpaired valid-vs-opponent-without-terminator selectivities were neuron 1167 at 0.107989, 734 at 0.094814, 1747 at 0.081680, 272 at 0.077419, and 877 at 0.062688.
- Several high-attribution neurons had negative selectivity in this test: neuron 399 at -0.125210, 1322 at -0.074643, 1673 at -0.067286, 14 at -0.052277, and 366 at -0.051922.

Matched controls:

- The matched-control test built 654 valid/invalid pairs.
- The largest mean valid-minus-invalid activations were neuron 991 at 0.069904, 877 at 0.053469, 272 at 0.047312, 1167 at 0.039280, and 1747 at 0.018654.
- Negative mean valid-minus-invalid activations included neuron 399 at -0.089261, 366 at -0.053210, 558 at -0.041524, and 1576 at -0.038735.
- The median valid-minus-invalid value was 0.0 for all listed candidate neurons.

Capture-line length and direction dependence:

- Within 1,278 valid-condition examples, correlations between candidate-neuron activation and line-structure variables were small.
- For longest capture line, the largest displayed Pearson correlations were 1167 at 0.040129 and 1322 at -0.069893.
- For number of capture directions, the largest displayed magnitudes included 1673 at -0.070929 and 991 at 0.031771.
- For total flipped pieces, the largest displayed magnitudes included 1322 at -0.071322 and 991 at 0.036569.

Semantic interventions:

- In the concrete example, semantic board-state edits changed some candidate activations, but the effects were sparse.
- Friendly-terminator edits changed neuron 1322 by +0.019275 in the natural direction and -0.019228 in the opposite direction.
- Friendly-terminator edits changed neuron 125 by +0.018370 in the natural direction and -0.017688 in the opposite direction.
- A capture-opponent opposite edit changed neuron 1322 with mean absolute delta activation 0.013741.

Input and output geometry:

- MLP7 input-weight geometry showed small cosines to board and transported-board directions. For example, neuron 14 had mean absolute cosine 0.040991 to L7 capture/terminator directions and 0.023797 to controls.
- The conjunction/interaction regression had tiny in-sample improvements. The largest delta R2 was 0.002291 for neuron 1673; 366 and 734 were 0.001217, 1747 was 0.001038, and 125 was 0.001001. Cross-validated R2 values were near zero or negative for many neurons.
- Output-weight geometry showed that neuron 399 writes most strongly in the displayed legality-gradient direction: mean post-activation 0.604659, mean legality-gradient dot -0.127217, mean legality-gradient cosine -0.110732, and activation-by-legality-write -0.076923.
- Neurons 1322 and 1576 also had comparatively large negative activation-by-write values, -0.026670 and -0.021911.

End-to-end single-neuron tests:

- The combined-evidence top-five neurons were 734, 1747, 1673, 125, and 1167.
- Low-attribution matched controls were 1819, 694, 988, 940, and 1963.
- Random controls were 12, 1346, 1386, 1664, and 1945.
- Across 195 preferred relational-condition examples, the combined-evidence top-five group averaged mean legality degradation -0.019919, with median -0.000007 and mean selected-logit change 0.026172.
- Low-attribution controls averaged -0.000001, with median 0.000000.
- Random controls averaged -0.003751, with median 0.000000.
- Individual combined-evidence neurons were mixed: neuron 1673 averaged -0.065093, neuron 125 -0.037334, neuron 734 -0.001635, neuron 1747 -0.001183, and neuron 1167 +0.005649.

### Interpretation

Observations:

- Board state is represented well enough for a linear probe to decode it under a strict split.
- Local semantic board-state interventions can affect logits in a Jacobian-predicted way.
- Layer 7 is the clearest layer for capture-line legality enrichment in this notebook.
- MLP7 is the strongest layer-7 component by both attribution and component ablation.
- A small group of MLP7 neurons has larger aggregate ablation effects than random groups.
- Candidate-neuron evidence is heterogeneous: some neurons show weak valid-vs-invalid selectivity, some show output geometry compatible with legality effects, and some show causal effects under ablation, but these signals do not all align on the same neurons.

Hypotheses:

- MLP7 likely participates in a distributed legality computation that uses board-state information and capture-line geometry.
- Candidate neurons may be pieces of this computation rather than standalone rule detectors.
- Neuron 399 is especially important by attribution and output geometry, but its negative valid-vs-invalid condition selectivity means it should not be described as a simple valid-capture detector.
- The top combined-evidence neurons may identify a weak causal subspace, but the end-to-end effects are too small and mixed to call it a complete mechanism.

What this does not show:

- It does not show that any individual neuron implements the Othello legality rule.
- It does not identify a full attention-to-MLP-to-logit circuit.
- It does not prove that the model uses the same algorithm as a symbolic Othello legal-move generator.
- It does not include a rescue or activation-patching experiment that restores legality after disrupting candidate neurons or semantic board-state features.

### Confidence

Strong evidence for layer-7 capture-line enrichment and MLP7 component-level causal importance.

Moderate evidence that a selected MLP7 neuron group has a real aggregate effect.

Weak evidence for individual-neuron relational selectivity and conjunction detection, because valid-vs-invalid effects are small, matched-control medians are 0.0, line-structure correlations are weak, and single-neuron ablations are mixed.

Open question for any claim that specific neurons implement the Othello legality rule.

### Open questions

- Which attention heads supply the board and ray information used by MLP7?
- Are the important variables represented in individual neurons, a low-dimensional MLP subspace, or a distributed pattern across many neurons?
- Can activation patching rescue legality after ablating candidate neurons or editing semantic board-state features?
- Can we identify the upstream sources of the MLP7 input directions with head-level path patching?
- Can the rule circuit be described algorithmically as target-empty plus opponent-line plus friendly-terminator, or is the model using a correlated shortcut?
- Do the observed effects generalize beyond the sampled positions and selected move families?

### Related notebook section

`demos/Othello_GPT_Jacobian_Lens.ipynb`, executed on `diegovalverde/TransformerLens`, branch `othello-jspace-analysis`.

Relevant sections:

- `17. Which layer computes legality?`
- `19. Is the layer-7 legality enrichment real?`
- `20. Layer-7 component decomposition`
- `21. Causal ablation of layer-7 components`
- `22. Does a candidate component specifically care about capture lines?`
- `25. What is MLP7 doing?`
- `26. MLP7 neuron ablation`
- `27. Layer-7 board-direction mediation`
- `28. Fix the candidate MLP7 legality neurons`
- `29. What board conditions activate the candidate neurons?`
- `30. Candidate neuron activation by Othello condition`
- `31. Matched controls`
- `32. Does neuron activation encode capture-line structure?`
- `33. Do board-state edits causally change candidate-neuron activation?`
- `34. What directions do the candidate neurons detect?`
- `35. Test conjunction vs linear feature detection`
- `36. What do the candidate neurons write?`
- `37. End-to-end causal test of the strongest neurons`

### Figures / tables

Notebook outputs to preserve or convert into stable figures/tables:

- Layer sweep table: layer, probe validation accuracy, capture-vs-unrelated ratio, capture-minus-unrelated.
- Layer-7 validation table: capture mean, unrelated occupied mean, observed ratio, bootstrap CIs, shuffled control, empirical p-value.
- Layer-7 component attribution and ablation tables.
- Top MLP7 neuron attribution table.
- Top-k neuron-group vs random ablation table.
- Relational-condition dataset summary.
- Candidate-neuron valid-vs-opponent-without-terminator selectivity table.
- Matched valid-vs-invalid control table.
- Line-length and capture-direction correlation table.
- Semantic-edit to candidate-activation table.
- MLP7 input-weight geometry summary.
- Conjunction/interaction regression table.
- MLP7 output-weight geometry summary.
- End-to-end combined-evidence, low-attribution-control, and random-control ablation summaries.
