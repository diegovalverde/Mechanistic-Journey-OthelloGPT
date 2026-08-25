# Research Provenance

This book is a research record. Scientific claims in the book must trace back to primary experimental or source materials, not to polished prose.

## Source Lineage

The current research record draws from three sources:

1. Li et al.'s original Othello-GPT / Othello World work.
2. Neel Nanda's TransformerLens reproduction and interpretability work.
3. Our executed Othello-GPT Jacobian Lens experiments.

The current experimental source repository is:

```text
diegovalverde/TransformerLens
```

The current experimental branch is:

```text
othello-jspace-analysis
```

The source notebook is:

```text
demos/Othello_GPT_Jacobian_Lens.ipynb
```

Claims about new experiments should be checked against executed notebook outputs on that branch.

## Workflow

Scientific claims should flow in this direction:

```text
executed notebook
    -> research_log.md
    -> findings_snapshot.md / experiment_index.md
    -> book prose
```

The book must never reverse this direction.

Book prose is not an authoritative source for experimental claims. If polished prose conflicts with the executed notebook, research log, findings snapshot, or experiment index, update the prose or the research memory only after returning to the executed experimental source.

## Practical Rule

Before adding or strengthening a scientific claim:

- Locate the executed notebook section that produced the result.
- Record the result in `research_log.md` with the relevant numerical output.
- Add or update the corresponding row in `experiment_index.md`.
- Update `findings_snapshot.md` with a conservative evidence label.
- Only then write or revise book prose.

If the notebook does not contain an executed result, the claim remains a hypothesis or open question.
