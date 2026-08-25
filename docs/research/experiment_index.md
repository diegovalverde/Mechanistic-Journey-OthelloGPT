# Experiment Index

| Experiment ID | Question | Method | Main result | Confidence | Notebook section | Book chapter | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| linear-board-probe | Can board state be decoded from residual activations? | Linear probe | About 98% held-out accuracy with strict game-level split | Established | TODO | Chapter 2 | Done, needs polished writeup |
| jacobian-finite-difference-validation | Do local Jacobians predict small residual interventions? | Finite-difference validation | Local Jacobian predictions closely matched small interventions | Strong evidence | TODO | Chapter 4 | Done, needs documentation |
| local-vs-averaged-jspace | How stable are local transformed semantic directions? | Compare local and averaged J-space directions | Cosine similarity around 0.62 | Moderate evidence | TODO | Chapter 5 | Done, needs interpretation |
| legality-contrast | Which activations are sensitive to legality-relevant contrasts? | Legality-focused contrast analysis | Layer 7 showed notable sensitivity | Strong evidence | TODO | Chapter 7 | Done, needs writeup |
| capture-line-enrichment | Is sensitivity enriched on capture-line squares? | Capture-vs-unrelated square comparison | Layer 7 capture-line enrichment was strong | Strong evidence | TODO | Chapter 7 | Done, needs figures |
| layer-sweep | Where across layers does enrichment appear? | Layer sweep | Layer 7 was stronger than earlier layers | Strong evidence | TODO | Chapter 7 | Done, needs documentation |
| layer7-bootstrap-shuffle-validation | Does layer-7 enrichment survive validation controls? | Bootstrap and shuffled-square control | About 2.75x enrichment, bootstrap CI above zero, shuffled control near 1 with empirical p around 0.003 | Strong evidence | TODO | Chapter 7 | Done, needs methods detail |
| layer7-component-ablation | Which layer-7 component matters most for legality? | Component ablation | MLP7 was strongest whole layer-7 component | Strong evidence | TODO | Chapter 8 | Done, needs writeup |
| mlp7-neuron-ablation | Do selected MLP7 neurons matter more than random groups? | Neuron-group ablation | Selected small group had substantially larger legality effects than random groups | Moderate evidence | TODO | Chapter 8 | Done, needs controls |
| semantic-mediation | Can semantic board-state interventions explain legality effects? | Semantic mediation analysis | TODO | TODO | TODO | Chapter 9 | Placeholder |
| mlp7-neuron-characterization | What rules, if any, do candidate MLP7 neurons implement? | Neuron characterization | TODO | TODO | TODO | Chapter 9 | Placeholder |

