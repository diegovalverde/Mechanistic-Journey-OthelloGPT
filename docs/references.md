# References

This page collects the main sources behind the book and explains why each one matters. It is intentionally selective. The book does not need a long bibliography; it needs clear provenance for the claims it makes.

## Othello-GPT and World Representations

**Li et al. (2023).** Kenneth Li, Aspen K. Hopkins, David Bau, Fernanda Viegas, Hanspeter Pfister, and Martin Wattenberg. [*Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task*](https://mlanthology.org/iclr/2023/li2023iclr-emergent/). International Conference on Learning Representations, 2023.

Why it matters for this book: this is the original Othello-GPT world-representation paper and the source of the controlled synthetic-task framing.

**Othello World repository.** Li et al.'s public [Othello World repository](https://github.com/likenneth/othello_world).

Why it matters for this book: this repository is the public source associated with the original Othello-GPT work and provides historical grounding for the model, data, and probing setup.

**Nanda (2023).** Neel Nanda. [*Actually, Othello-GPT Has A Linear Emergent World Representation*](https://www.neelnanda.io/mechanistic-interpretability/othello). 2023.

Why it matters for this book: this work motivates the linear mine/theirs board representation, probe-direction interventions, and the use of TransformerLens-style tools for Othello-GPT.

## Tools and Mechanistic Interpretability Methods

**Nanda and Bloom (2022).** Neel Nanda and Joseph Bloom. [*TransformerLens*](https://transformerlensorg.github.io/TransformerLens/content/citation.html). 2022.

Why it matters for this book: TransformerLens supplies the `HookedTransformer`, hook names, activation cache, and intervention APIs used by the executed notebook.

**Elhage et al. (2021).** Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. [*A Mathematical Framework for Transformer Circuits*](https://transformer-circuits.pub/2021/framework/index.html). Transformer Circuits Thread, 2021.

Why it matters for this book: this source provides the broader circuits vocabulary for residual streams, paths, attention, and decomposing Transformer computation into mechanistic parts.

**Elhage et al. (2022).** Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, Roger Grosse, Sam McCandlish, Jared Kaplan, Dario Amodei, Martin Wattenberg, and Christopher Olah. [*Toy Models of Superposition*](https://transformer-circuits.pub/2022/toy_model/index.html). Transformer Circuits Thread, 2022.

Why it matters for this book: superposition is one reason the book treats messy neuron-level evidence cautiously and does not require one-neuron-one-feature explanations.

**Gurnee et al. (2026).** Wes Gurnee, Nicholas Sofroniew, Adam Pearce, Mateusz Piotrowski, Isaac Kauvar, Runjin Chen, Anna Soligo, Paul Bogdan, Euan Ong, Rowan Wang, T. Ben Thompson, David Abrahams, Subhash Kantamneni, Emmanuel Ameisen, Joshua Batson, and Jack Lindsey. [*Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html). Transformer Circuits Thread, 2026.

Why it matters for this book: this is the broader Jacobian Lens and J-space reference that inspired the book's hidden-state transport framing, while the book's Othello experiments remain separate and more limited.

**Anthropic Jacobian Lens repository (2026).** Anthropic. [*jacobian-lens*](https://github.com/anthropics/jacobian-lens). 2026.

Why it matters for this book: this is the public reference implementation for fitting and applying averaged Jacobian Lens transports; the book adapts the local geometric idea to Othello-GPT rather than claiming to reproduce the full global-workspace pipeline.

**PyTorch autograd documentation.** PyTorch contributors. [Automatic differentiation package: `torch.autograd`](https://pytorch.org/docs/stable/autograd.html).

Why it matters for this book: the executed notebook uses PyTorch autograd to compute gradients and Jacobian-vector products for residual interventions.

**Receiver operating characteristic.** Wikipedia contributors. [*Receiver operating characteristic*](https://en.wikipedia.org/wiki/Receiver_operating_characteristic).

Why it matters for this book: this page provides the background definition of ROC curves and AUROC used when interpreting directional-capture probe results.

## Project Research Memory

The following pages are internal project records. They are not external citations, but they are the authoritative source for what this book currently claims from its own executed experiments.

- [Research provenance](research/provenance.md): records the rule that new scientific claims must flow from executed notebook to research memory to book prose.
- [Model architecture](research/model_architecture.md): records verified Othello-GPT dimensions, block structure, and hook names.
- [Research log](research/research_log.md): preserves dated experimental notes and measured outputs.
- [Findings snapshot](research/findings_snapshot.md): summarizes current claims and evidence labels.
- [Experiment index](research/experiment_index.md): maps notebook sections to questions, methods, results, and book chapters.
- [Final evidence map](research/final_evidence_map.md): states the final evidence boundary after the ten narrative chapters.
- [Open questions](research/open_questions.md): records what remains untested or unresolved.
- [Chapter summaries](research/chapter_summaries.md): summarizes the completed narrative chapters for future maintenance.
