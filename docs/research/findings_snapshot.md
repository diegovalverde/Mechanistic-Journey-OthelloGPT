# Findings Snapshot

This page summarizes the current state of evidence. It should be updated only when the research log and experiment index support the change.

| Status | Finding |
| --- | --- |
| Established | Othello-GPT is a small GPT-2-style decoder-only transformer trained on Othello move sequences, not GPT-2 itself. |
| Established | Board state is linearly decodable from the residual stream. |
| Established | In our reproduction, held-out board-state probe accuracy was about 98% with a strict game-level split. |
| Strong evidence | Probe directions can be used as semantic residual-space directions such as mine-vs-theirs and occupied-vs-empty. |
| Strong evidence | Local Jacobian predictions closely match small residual interventions. |
| Moderate evidence | Local vs averaged transformed semantic directions showed cosine similarity around 0.62. |
| Strong evidence | Legality-focused sensitivity showed much stronger capture-line enrichment in layer 7 than earlier layers. |
| Strong evidence | Validated layer-7 capture-vs-unrelated enrichment was about 2.75x. |
| Strong evidence | Position-level bootstrap difference confidence interval was clearly above zero. |
| Strong evidence | Shuffled-square control was close to 1, with empirical p around 0.003. |
| Strong evidence | MLP7 was the strongest whole layer-7 component in legality ablations. |
| Moderate evidence | A selected small group of MLP7 neurons had substantially larger legality effects than random neuron groups. |
| Moderate evidence | Current evidence supports moderate evidence for rule-structured computation, but does not yet prove a complete legality circuit. |
| Hypothesis | Some MLP7 neurons may participate in detecting relational capture conditions like empty target plus opponent line plus friendly terminator implies legal move. |

