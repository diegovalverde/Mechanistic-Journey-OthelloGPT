# Findings Snapshot

This page summarizes the current state of evidence. It should be updated only when the research log and experiment index support the change. Numerical claims here come from executed outputs in `demos/Othello_GPT_Jacobian_Lens.ipynb` on `diegovalverde/TransformerLens`, branch `othello-jspace-analysis`.

## Current Evidence

| Status | Finding |
| --- | --- |
| Established | Othello-GPT is a small GPT-2-style decoder-only transformer trained on Othello move sequences, not GPT-2 itself. |
| Established | Board state is represented in the residual stream: a strict game-level split linear probe reached 0.9796 held-out board-state accuracy over 330 validation positions, with empty/mine/theirs accuracies 0.9976/0.9561/0.9703. |
| Strong evidence | Probe directions can be used as semantic residual-space directions such as mine-vs-theirs and occupied-vs-empty; small residual interventions follow local Jacobian predictions with max absolute error 0.000067 in the tested example. |
| Moderate evidence | Local vs averaged final-residual transformed semantic directions are related but not identical; the executed notebook reports cosine similarity 0.6178 for the layer-4 G6 mine-vs-theirs direction transported to the final residual stream. |
| Strong evidence | Board representation is causally relevant to logits in local residual-space tests: semantic board-state directions change selected move logits and legality contrasts in the directions predicted by the local Jacobian. |
| Strong evidence | Capture-line geometry is selectively important at layer 7. In the layer sweep, layer 7 had capture-vs-unrelated ratio 2.2514 and capture-minus-unrelated 0.026569, while layers 2, 4, and 6 were near ratio 1.08, 1.10, and 1.01. |
| Strong evidence | The validated layer-7 capture-opponent enrichment is about 2.7466x: capture mean 0.063157 vs unrelated occupied mean 0.022995, mean difference 0.040162, bootstrap 95% CI [0.035965, 0.044268], ratio 95% CI [2.524081, 2.971348], shuffled mean ratio 1.046078, shuffled 95th percentile 1.176336, empirical permutation p=0.003322. |
| Strong evidence | Directional capture relations are linearly decodable from upstream residual states. Across 13,701 held-out valid targets, post4 top-1/top-2/top-3 true-direction accuracies were 0.982921/0.997591/0.999416; post5 values were 0.983797/0.997153/0.998978. |
| Strong evidence | The hard valid-capture versus no-friendly-terminator contrast sharpens by post5: hard AUROC rises from 0.960087 at post4 to 0.990544 at post5, and the mean valid-minus-no-terminator probability gap rises from 0.266136 to 0.382907. This is decodability evidence, not proof that MLP5 computes the rule. |
| Strong evidence | MLP7 is causally important among layer-7 components. Whole-component ablation over 30 positions gave MLP7 mean signed legality effect -0.105164 and mean absolute effect 0.262614, larger in absolute effect than L7H7 (0.109719), L7H2 (0.094140), and L7H0 (0.090048). |
| Strong evidence | MLP7 also had the largest mean absolute component attribution over the 30-position component set: 0.267666 for MLP7, followed by L7H0 0.201140, L7H2 0.186625, and L7H7 0.180907. |
| Moderate evidence | A fixed set of candidate MLP7 neurons has larger legality attribution than typical neurons. The top 20 by mean absolute attribution were 399, 1322, 1576, 366, 558, 1858, 1747, 495, 1167, 14, 1400, 272, 1673, 1953, 991, 734, 1000, 877, 125, and 912. |
| Moderate evidence | Group ablation of top MLP7 neurons has a much larger effect than random neuron groups in the 30-position ablation set: top-1/top-2/top-5/top-10/top-20 mean legality degradations were -0.137254/-0.153469/-0.204493/-0.335030/-0.543530, while same-size random means were 0.000685/-0.001759/-0.000735/0.002325/0.012949. |
| Weak evidence | Some candidate neurons distinguish valid capture conditions from opponent-runs without a terminator, but the effects are small and mixed. The largest unpaired standardized valid-vs-invalid selectivities were neuron 1167 at 0.107989, 734 at 0.094814, 1747 at 0.081680, 272 at 0.077419, and 877 at 0.062688; several high-attribution neurons were negative, including 399 at -0.125210 and 1322 at -0.074643. |
| Weak evidence | Matched valid-vs-invalid controls do not show a uniform valid-line detector across the candidate set. Across 654 matched pairs, the largest mean valid-minus-invalid activations were neuron 991 at 0.069904, 877 at 0.053469, 272 at 0.047312, and 1167 at 0.039280, while neuron 399 was -0.089261 and neuron 366 was -0.053210. Medians were 0.0 for all listed neurons. |
| Weak evidence | Capture-line length and direction dependence are weak in the current candidate-neuron tests. Across 1,278 valid-condition examples, absolute Pearson correlations with longest capture line, number of capture directions, and total flipped pieces were small; the largest shown were 0.040129 for longest line, -0.070929 for number of capture directions, and -0.071322 for total flipped. |
| Weak evidence | Semantic residual edits can change candidate-neuron activations, but the clearest effects are sparse and example-specific. In the concrete example, friendly-terminator edits changed neuron 1322 by about +0.019275 in the natural direction and -0.019228 in the opposite direction; neuron 125 changed by +0.018370 and -0.017688. |
| Weak evidence | MLP7 input-weight geometry does not yet isolate a clean board-feature detector. Candidate input-weight cosines with board and transported-board directions are small in the displayed summary; for example neuron 14 had mean absolute cosine 0.040991 to L7 capture/terminator directions and 0.023797 to controls. |
| Weak evidence | The current conjunction regression provides little evidence for a strong nonlinear relational detector in single-neuron activations. Adding interaction features improved in-sample R2 by at most 0.002291, and cross-validated R2 values were near zero or negative for many neurons. |
| Moderate evidence | MLP7 output-weight geometry identifies candidate neurons that can write in legality-relevant directions, especially neuron 399. Mean legality-gradient dot products ranged from -0.003147 to -0.127217; neuron 399 had mean post-activation 0.604659, mean legality-gradient dot -0.127217, and activation-by-write score -0.076923. |
| Weak evidence | End-to-end single-neuron causal tests on preferred relational-condition moves are selective but small. The combined-evidence top-five neurons were 734, 1747, 1673, 125, and 1167; over 195 examples their mean legality degradation was -0.019919, compared with -0.000001 for low-attribution controls and -0.003751 for random controls. |
| Hypothesis | Some MLP7 neurons participate in a distributed legality computation that combines board-state features with capture-line structure, but current evidence does not identify a complete algorithmic circuit. |
| Open question | Whether any specific MLP7 neuron implements the Othello legality rule is not established. Current experiments show selective effects and causal relevance, not a single-neuron rule implementation. |

## Claim Boundaries

| Claim | Current label | Boundary |
| --- | --- | --- |
| A. Board state is represented. | Established | Supported by held-out board-probe accuracy around 98%. |
| B. Board representation is causally relevant. | Strong evidence | Supported by Jacobian-predicted residual interventions; still local to tested positions and directions. |
| C. Capture-line geometry is selectively important. | Strong evidence | Supported by layer-7 enrichment, bootstrap CI, and shuffled controls. |
| D. Directional capture relations are linearly decodable. | Strong evidence | Supported by held-out directional capture probe metrics at post4/post5 and hard no-terminator controls. This is not causal-use evidence. |
| E. MLP7 is causally important. | Strong evidence | Supported by layer-7 component attribution and component ablations. |
| F. Specific MLP7 neurons have selective effects. | Weak to moderate evidence | Supported by attribution, group ablation, condition activation, and small end-to-end tests; effects are mixed and often small. |
| G. Specific neurons implement the Othello legality rule. | Open question | Not established by the current notebook. No finding should claim a complete rule implementation by a specific neuron. |
