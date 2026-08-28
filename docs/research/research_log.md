# Research Log

Every new experiment should be recorded here before it becomes polished book prose. Entries should preserve the question, method, result, interpretation, confidence, notebook location, and figure/table references.

## 2026-08-28 - Directional capture relations are linearly decodable

### Question

Can a linear decoder recover not merely square occupancy, but the directional Othello capture relation:

```text
C(q,d) = "target q has a valid capture ray in direction d"?
```

### Experimental setup

Notebook sections `47. Where does a capture ray become an internal feature?` and `48. Visualizing decoded capture rays` trained and visualized a linear directional probe for the binary capture predicate over target-square/direction pairs. For an empty target square \(q\) and direction \(d\), \(C(q,d)=1\) when the ray from \(q\) contains one or more opponent pieces followed by a friendly terminator.

The probe used held-out game-level test data and predicted independent binary labels for the eight Othello directions. The visualization examples were selected deterministically as the first held-out test example in each named category, before inspecting decoded probabilities.

Relevant visualization sites:

| Hook | Label |
| --- | --- |
| `blocks.4.hook_resid_post` | post4 |
| `blocks.5.hook_resid_post` | post5 |
| `blocks.6.hook_resid_mid` | mid6 |
| `blocks.6.hook_resid_post` | post6 |
| `blocks.7.hook_resid_post` | post7 |

Primary display site for the hero figures: `blocks.5.hook_resid_post`.

### Key held-out results

For all held-out valid targets, `n_targets_with_valid_direction = 13701`.

| Site | Top-1 true-direction accuracy | Top-2 | Top-3 | Macro AUROC | Hard valid-vs-no-terminator AUROC |
| --- | ---: | ---: | ---: | ---: | ---: |
| post4 | 0.9829209547 | 0.9975914167 | 0.9994161010 | 0.9957207226 | 0.9600865639 |
| post5 | 0.9837968032 | 0.9971534924 | 0.9989781768 | 0.9985151889 | 0.9905437983 |
| mid6 | 0.9825560178 | 0.9973724546 | 0.9989781768 | 0.9983874094 | 0.9896910253 |
| post6 | 0.9387635939 | 0.9935771112 | 0.9983212904 | 0.9972399046 | 0.9811240573 |
| post7 | 0.9187650536 | 0.9908765784 | 0.9972264798 | 0.9974116697 | 0.9797869301 |

The hard contrast compared valid capture directions with opponent runs that lacked a friendly terminator. The no-terminator class had `n = 27798` direction examples at each site; the valid-capture class had `n = 18529`.

| Site | Mean valid probability | Mean no-terminator probability | Mean valid minus no-terminator probability |
| --- | ---: | ---: | ---: |
| post4 | 0.9471895269 | 0.6810535039 | 0.2661360229 |
| post5 | 0.9729266445 | 0.5900201398 | 0.3829065047 |
| mid6 | 0.9724616930 | 0.5908260356 | 0.3816356573 |
| post6 | 0.9671435112 | 0.5708823211 | 0.3962611901 |
| post7 | 0.9654093235 | 0.5792380653 | 0.3861712582 |

### Suppression diagnostic

Section 48 also visualized finite edits along the learned capture direction at the primary display site. For held-out target `B2`, direction `SE`, at `blocks.5.hook_resid_post`, a negative probe-direction edit with `alpha = 2.0` changed the decoded probability for the suppressed direction from `0.9950110912` to `0.9235469103` (`delta = -0.0714641809`). The same edit changed the selected legality contrast from `11.1585502625` to `10.9237470627` (`delta = -0.2348031998`).

This is a magnitude-dependent diagnostic, not clean proof that the learned probe direction is the model's native causal coordinate. Small-alpha edits did not provide robust local causal evidence in the same sense as the earlier Jacobian finite-difference checks; larger finite edits showed stronger expected-sign effects but carry more off-manifold risk.

### Most important interpretation

The directional capture relation is already highly linearly decodable by layer 4. The clearest improvement in the difficult valid-capture versus opponent-run-without-friendly-terminator contrast occurs across MLP5: hard AUROC rises from `0.9600865639` at post4 to `0.9905437983` at post5, and the mean valid-minus-no-terminator probability gap rises from `0.2661360229` to `0.3829065047`.

The representation does not become monotonically easier to decode at every later layer. Top-1 directional accuracy declines after MLP6/7, from `0.9837968032` at post5 to `0.9387635939` at post6 and `0.9187650536` at post7, while macro AUROC remains extremely high.

### Claim boundary

This establishes strong linear decodability of a relational capture predicate. It does not establish that the learned probe direction is the model's causal basis, that MLP5 implements the complete capture rule, that MLP6 is irrelevant, that Chapter 7's layer-7 legality-sensitivity result is contradicted, or that we have found a complete legality circuit.

Chapter 7 measured capture-line semantic sensitivity to legality. Sections 47-48 measure decodability of directional capture relations. These are different quantities.

### Related notebook sections

- `47. Where does a capture ray become an internal feature?`
- `48. Visualizing decoded capture rays`

### Source and figures

- Source repository: `https://github.com/diegovalverde/TransformerLens`
- Branch: `othello-jspace-analysis`
- Exact visualization commit: `b4b529fec329dc318755c579c58af65950143323`
- Notebook: `demos/Othello_GPT_Jacobian_Lens.ipynb`
- Output directory: `demos/othello_jacobian_lens_outputs/capture_ray_visualization_20260828_193735/`
- TransformerLens notes: `docs/research/section48_capture_ray_visualization_notes.md`
- Book figure directory: `docs/figures/capture_rays/`

## 2026-08-27 - Localizing capture-line legality geometry to layer 7

### Question

Where in the network does board-state sensitivity become selectively aligned with the squares that make a legal move possible?

### Legality contrast

The executed notebook used the selected legal move's logit minus the mean logit of currently illegal empty-square move tokens:

```text
L_m = z_m - mean_{j in illegal empty-square moves} z_j
```

The illegal baseline is the actual illegal EMPTY-square move set in the current Othello position. It excludes occupied squares, non-move starting-center squares, and `pass`.

### Concrete E3 example

The analysis example selected `E3` as the target move. Its verified capture ray was:

```text
E3 target
D3 opponent
C3 opponent
B3 friendly terminator
```

Notebook section `11. A legality score instead of a raw move logit` reported:

| Quantity | Value |
| --- | ---: |
| Raw `E3` logit | 8.9408 |
| Mean illegal empty-square logit | -1.5438 |
| Mean other-legal logit | 8.9298 |
| Legality contrast | 10.4845 |
| Legal-preference contrast | 0.0110 |
| Rank among all output tokens | 4 |
| Rank among current legal moves | 4 |
| Illegal empty-square tokens in contrast | 23 |

### Layer-4 aggregate result

The first dataset-level layer-4 aggregate was negative/inconclusive. It reported mean(capture - unrelated occupied) `-0.000499`, bootstrap 95% CI `[-0.003646, 0.002784]`, mean capture/unrelated ratio `1.118791`, and mean shuffled capture/unrelated ratio `1.038956`.

### Layer sweep

The layer sweep in notebook section `17. Which layer computes legality?` trained lightweight board probes for layers `2`, `4`, `6`, and `7`, for `8` epochs each. It used the union of capture-line opponent squares and friendly terminators as the capture-support set, and unrelated occupied squares as controls.

| Layer | Probe validation accuracy | Capture-vs-unrelated ratio | Capture-minus-unrelated |
| ---: | ---: | ---: | ---: |
| 2 | 0.883759 | 1.076216 | -0.000256 |
| 4 | 0.952699 | 1.098189 | -0.000439 |
| 6 | 0.891667 | 1.006662 | -0.000596 |
| 7 | 0.850000 | 2.251362 | 0.026569 |

### Stronger Layer-7 validation

Notebook section `19. Is the layer-7 legality enrichment real?` focused the validation on capture-opponent squares versus unrelated occupied controls. It used position-level means, `1000` bootstrap samples, `300` shuffled-square repeats, and `RATIO_EPSILON = 1e-9`.

| Metric | Value |
| --- | ---: |
| Capture mean | 0.063157 |
| Unrelated occupied mean | 0.022995 |
| Difference | 0.040162 |
| Ratio | 2.746573 |
| Difference 95% CI | [0.035965, 0.044268] |
| Ratio 95% CI | [2.524081, 2.971348] |
| Shuffled-null mean ratio | 1.046078 |
| Shuffled-null 95th percentile | 1.176336 |
| Empirical permutation p | 0.003322 |

The bootstrap resampled positions rather than individual squares. The shuffled control permuted square labels inside each layer-7 sensitivity map before recomputing capture-opponent and unrelated-occupied means.

### Interpretation

Strong evidence: capture-line semantic geometry is enriched at layer 7.

Not established: responsible component, algorithm, neuron, or complete circuit.

### Related notebook sections

- `11. A legality score instead of a raw move logit`
- `12. Which board squares causally support legality?`
- `15. Dataset-level legality relevance test`
- `16. Path structure, not just square relevance`
- `17. Which layer computes legality?`
- `19. Is the layer-7 legality enrichment real?`

### Figures

- `docs/figures/chapter07_layer_sweep.json`
- `docs/figures/chapter07_layer_sweep.svg`
- `docs/figures/chapter07_layer7_validation.json`
- `docs/figures/chapter07_layer7_validation.svg`

## 2026-08-27 - Localizing the layer-7 legality effect to MLP7

### Question

Within layer 7, which component most strongly affects the selected-move legality contrast?

### Component attribution

Notebook section `20. Layer-7 component decomposition` used:

```text
A_c = dot(component output, legality gradient)
```

The individual head vectors came from `blocks.7.attn.hook_result`; MLP7 came from `blocks.7.hook_mlp_out`. Over `30` component positions, MLP7 ranked first by mean absolute attribution:

| Component | Mean signed attribution | Mean absolute attribution |
| --- | ---: | ---: |
| MLP7 | 0.126682 | 0.267666 |
| L7H0 | 0.169024 | 0.201140 |
| L7H2 | 0.157269 | 0.186625 |
| L7H7 | 0.135668 | 0.180907 |

In the concrete `E3` example, MLP7 attribution was `-0.560007`, the largest absolute component attribution.

### Component ablation

Notebook section `21. Causal ablation of layer-7 components` patched each component at the final token and reran the model. Heads were patched at `blocks.7.attn.hook_result[:, final_token, head, :]`; MLP7 was patched at `blocks.7.hook_mlp_out[:, final_token, :]`. Replacements were mean activations from the 30-position component set.

Sign convention:

```text
delta_legality_contrast = L_ablate - L_clean
```

| Component | Mean signed effect | Mean absolute effect | Median effect |
| --- | ---: | ---: | ---: |
| MLP7 | -0.105164 | 0.262614 | -0.128594 |
| L7H7 | -0.109719 | 0.109719 | -0.104066 |
| L7H2 | -0.093151 | 0.094140 | -0.093622 |
| L7H0 | -0.090048 | 0.090048 | -0.083029 |

MLP7 had the largest mean absolute ablation effect under this intervention.

### Semantic mediation diagnostic

Notebook section `24. Capture-line intervention x component ablation interaction` measured an example-level interaction:

```text
M = delta_L_normal - delta_L_component_ablated
```

For MLP7 on the concrete example, two capture-line edits had mediation-like effects `0.013071` for `C3` and `0.059675` for `D3`. Two unrelated edits had effects `-0.004507` for `G4` and `-0.008799` for `F6`. Mean over all four rows was `0.014860`. This is a small example-level diagnostic, not a dataset-level mediation distribution.

### Neuron attribution ranking

Notebook sections `25. What is MLP7 doing?` and `28. Fix the candidate MLP7 legality neurons` decomposed MLP7 into neuron writes `post_activation_j * W_out[j, :]` and ranked neurons by mean absolute legality attribution over `30` positions.

The fixed top-20 candidate MLP7 neurons were:

```text
399, 1322, 1576, 366, 558, 1858, 1747, 495, 1167, 14,
1400, 272, 1673, 1953, 991, 734, 1000, 877, 125, 912
```

Neuron 399 had mean signed legality attribution `-0.219701` and mean absolute legality attribution `0.276027`.

### Top-N vs random group ablations

Notebook section `26. MLP7 neuron ablation` patched selected MLP7 `hook_post` activations at the final token to mean post-activation baselines. It compared top-attribution groups against `25` random same-size groups for each size.

Sign convention:

```text
legality_degradation = L_clean - L_ablate
```

| Group size | Top-attribution group | Random same-size mean |
| ---: | ---: | ---: |
| 1 | -0.137254 | 0.000685 |
| 2 | -0.153469 | -0.001759 |
| 5 | -0.204493 | -0.000735 |
| 10 | -0.335030 | 0.002325 |
| 20 | -0.543530 | 0.012949 |

### Interpretation and claim boundaries

Strong evidence: MLP7 is the most important tested layer-7 component under component attribution and component ablation.

Moderate evidence: high-attribution MLP7 neuron groups identify an unusually important subpopulation compared with random groups.

Not established: MLP7 computes the legality algorithm; attention is irrelevant; neuron 399 or the top-20 group implements the Othello rule; a complete attention-to-MLP-to-logit circuit.

### Figures

- `docs/figures/layer7_component_map.svg`
- `docs/figures/mlp7_component_attribution.json`
- `docs/figures/mlp7_component_attribution.svg`
- `docs/figures/mlp7_component_ablation.json`
- `docs/figures/mlp7_component_ablation.svg`
- `docs/figures/attention_to_mlp7_hypothesis.svg`
- `docs/figures/mlp7_semantic_mediation.json`
- `docs/figures/mlp7_semantic_mediation.svg`
- `docs/figures/mlp7_neuron_group_ablation.json`
- `docs/figures/mlp7_neuron_group_ablation.svg`
- `docs/figures/evidence_ladder_mlp7.svg`

## 2026-08-26 - Local vs averaged J-space

### Question

For a fixed semantic board-state direction, is the downstream transformed direction mostly stable across Othello contexts, or does it depend strongly on the current board and move history?

Chapter 4 validated local Jacobian predictions for one source activation and one output logit. This experiment asks a different question: if we use the same source-space semantic direction at layer 4, how similar is its local downstream image \(J_x v\) to an average image across many sampled positions?

### Experiment

The executed notebook used `demos/Othello_GPT_Jacobian_Lens.ipynb` on `diegovalverde/TransformerLens`, branch `othello-jspace-analysis`, section `10. Local J-space vs averaged J-space`.

The source setup was:

- Model dimensions from the printed config: `d_model = 512`, `d_vocab = 61`, and `d_vocab_out = 61`.
- Source layer: layer 4.
- Source hook: `blocks.4.hook_resid_post`.
- Analysis prefix length: 28.
- Analysis prefix token IDs: `[20, 19, 18, 10, 2, 1, 27, 3, 41, 42, 34, 12, 4, 40, 11, 29, 43, 13, 48, 56, 33, 39, 22, 44, 24, 5, 46, 6]`.
- Source token position: final prefix position, index 27.
- Target token position: final prefix position, index 27.
- Semantic direction: G6 mine-vs-theirs, square index 46.
- Direction construction: the layer-4 board probe weight difference `W[G6, mine] - W[G6, theirs]`, normalized before JVP computation.

The target of the JVP was not the output-logit vector. It was the final residual-stream representation at the same token position, immediately before final layer normalization and unembedding. Operationally, the notebook defined a function that:

1. cloned the cached `blocks.4.hook_resid_post` activation,
2. added a source residual delta at position 27,
3. continued the model from layer 5 through the end of the transformer stack with `start_at_layer=layer + 1` and `stop_at_layer=model.cfg.n_layers`,
4. returned `final_resid[0, target_pos, :]`.

The notebook computed the local transformed direction with `torch.autograd.functional.jvp` at a zero residual delta, using the normalized G6 mine-vs-theirs direction as the tangent vector. The resulting vectors therefore had shape `[512]`.

The local JVP was validated with a central finite difference:

```text
(epsilon v) and -(epsilon v)
epsilon = 0.001
(F(h + epsilon v) - F(h - epsilon v)) / (2 epsilon)
```

For the averaged direction, the notebook sampled `100` unique Othello positions. Sampling used `JSPACE_AVG_PREFIX_MIN_LEN = 12`, `JSPACE_AVG_PREFIX_MAX_LEN = 45`, and `JSPACE_AVG_RANDOM_SEED = PROBE_RANDOM_SEED + 1`. For each sampled position, the code generated a random legal Othello game, selected one prefix length uniformly from the valid range, skipped duplicate prefixes and positions with no legal moves, and recorded the legal move that maximized `(num_flipped, num_capture_lines, -token_id)`. The chosen move metadata was used to ensure nontrivial legal positions, but the averaged transformed direction was the average of `J_i v` for the same fixed G6 semantic source direction:

```text
J_avg_v = mean_i J_i v
```

The notebook did not materialize or average a full `512 x 512` Jacobian matrix.

### Result

The local JVP finite-difference validation reported:

- Local JVP finite-difference cosine: `0.999944`.
- Local JVP finite-difference relative error: `0.010651`.

The local-vs-average table reported:

| Quantity | Value |
| --- | ---: |
| source-space derivative `v^T grad z_m` | `0.030897` |
| final-readout effect of local `J_local_v` | `0.030897` |
| final-readout effect of averaged `J_avg_v` | `0.018023` |
| `||J_local_v||` | `1.496970` |
| `||J_avg_v||` | `0.819020` |
| `cos(J_local_v, J_avg_v)` | `0.617840` |

The sampled J-space positions output was:

```text
Sampled J-space positions (independent games, one prefix each): 100
Sampled prefix lengths (min / mean / max): 12 / 29.14 / 45
First five sampled moves: [(30, 'F3', 5), (29, 'A3', 3), (28, 'G4', 5), (16, 'B4', 4), (16, 'D1', 5)]
```

### Interpretation

The local JVP validation supports that the computed JVP is the correct first-order downstream displacement for the tested source hook, token position, target final-residual representation, and semantic direction.

The local-vs-average cosine of `0.617840` suggests that the local transformed G6 mine-vs-theirs direction has a substantial shared component with the average transformed direction, but it is not close to context-independent. This should not be read as "62% the same." Cosine similarity is an angle-based comparison, not a percentage of shared computation. The result is best interpreted as moderate evidence for shared transformed geometry plus substantial context-dependent variation.

This experiment does not show that every board-state direction behaves this way, that all layers share the same J-space geometry, or that the averaged transformed direction is a complete account of the model's legality computation.

### Confidence

Moderate evidence for context-dependent but nonrandom transformed geometry in the tested G6 mine-vs-theirs direction.

### Related notebook section

- `10. Local J-space vs averaged J-space`

### Figures / tables

- `docs/figures/jspace_jvp_validation.json`
- `docs/figures/jspace_jvp_validation.svg`
- `docs/figures/local_vs_average_jspace.svg`

## 2026-08-26 - Chapter 4 Jacobian local-linearization details

### Question

Do the local derivatives computed at a TransformerLens residual hook quantitatively predict the effect of actually perturbing Othello-GPT's residual stream?

### Experiment

The executed notebook used `demos/Othello_GPT_Jacobian_Lens.ipynb` on `diegovalverde/TransformerLens`, branch `othello-jspace-analysis`.

The setup was:

- Model dimensions from the printed config: `d_model = 512`, `d_vocab = 61`, and `d_vocab_out = 61`.
- Source layer: layer 4.
- Source hook: `blocks.4.hook_resid_post`.
- Analysis prefix length: 28.
- Analysis prefix token IDs: `[20, 19, 18, 10, 2, 1, 27, 3, 41, 42, 34, 12, 4, 40, 11, 29, 43, 13, 48, 56, 33, 39, 22, 44, 24, 5, 46, 6]`.
- Source and target token position: final prefix position, index 27.
- Legal next moves under the inferred token mapping: `E3`, `B4`, `B5`, `C7`, `F7`, `G7`, `H7`, `C8`, `E8`.
- Selected move logit: token 21, `E3`.
- Model favorite move on the same prefix: token 57, `E8`.
- Selected move baseline logit: `8.940763473510742`.
- Selected move capture line: `D3`, `C3`.
- Gradient norm for the selected E3 logit with respect to the layer-4 residual edit variable: `0.17336732149124146`.

Section 5 first tested the derivative machinery along the normalized selected-logit gradient direction. It compared actual logit deltas from hook interventions against first-order predictions for epsilons `0.0001`, `0.0003`, `0.001`, `0.003`, `0.01`, `0.03`, and `0.1`.

Section 9 then tested a semantic board-state intervention. It used the trained layer-4 board probe, selected the square with largest absolute mine-vs-theirs sensitivity for the E3 logit, and perturbed along that normalized probe direction.

### Result

Section 5 finite-difference sanity-check values:

| Epsilon | Actual delta logit | First-order predicted delta logit |
| --- | ---: | ---: |
| 0.0001 | 0.000019073 | 0.000017337 |
| 0.0003 | 0.000052452 | 0.000052010 |
| 0.001 | 0.000170710 | 0.000173370 |
| 0.003 | 0.000522610 | 0.000520100 |
| 0.01 | 0.001734700 | 0.001733700 |
| 0.03 | 0.005184200 | 0.005201000 |
| 0.1 | 0.017184000 | 0.017337000 |

Section 9 semantic board-state intervention details:

- Selected intervention square: `G6`, square index 46.
- Direction: normalized G6 mine-vs-theirs probe direction.
- Directional derivative `v^T g_m`: `+0.030897`.
- Tested alphas: `-0.1`, `-0.03`, `-0.01`, `-0.003`, `0.003`, `0.01`, `0.03`, `0.1`.

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

The maximum absolute prediction error across the semantic intervention alphas was `0.000067`.

### Interpretation

These checks support the local use of the Jacobian at the tested layer, position, direction, and perturbation sizes. They verify that the hook site, selected token position, selected output logit, autograd gradient, sign convention, and downstream continuation are mutually consistent.

They do not show that Othello-GPT is globally linear, that the same approximation is equally accurate at other board positions, or that the selected G6 probe direction is a complete semantic variable used by the model. The claim is local: near this layer-4 activation, these small residual edits had output-logit effects that were accurately predicted by the first-order approximation.

### Confidence

Strong evidence for local first-order prediction in the tested examples.

### Related notebook sections

- `5. Sanity check: does the Jacobian predict an actual intervention?`
- `9. Jacobian prediction vs actual board-state intervention`

### Figures / tables

- `docs/figures/jacobian_prediction_vs_intervention.json`
- `docs/figures/jacobian_prediction_vs_intervention.svg`

## 2026-08-25 — Characterizing the Layer-7 legality computation

### Question

Can specific MLP7 neurons be connected to the relational structure of Othello legality?

More specifically: after establishing that layer 7 has unusually strong capture-line enrichment and that MLP7 is the strongest causal layer-7 component in ablation, can we identify candidate MLP7 neurons that respond to the relational pattern required by Othello legality: empty target, adjacent opponent line, and friendly terminator?

### Experiment

The executed notebook first revalidated the broader setup:

- A strict split board-state probe was trained from residual activations.
- A legality contrast was defined for a selected move by comparing the selected legal move against currently illegal empty-square moves.
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
