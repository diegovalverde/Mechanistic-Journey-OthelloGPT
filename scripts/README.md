# Scripts

Reproducibility scripts belong here.

Scripts should document inputs, outputs, environment assumptions, and the research-log or notebook section they support.

## Experimental Figure Scripts

Run scripts from the repository root unless a script says otherwise.

```bash
python3 scripts/generate_probe_accuracy_figure.py
```

This regenerates `docs/figures/board_probe_accuracy.svg` from measured values recorded in the executed TransformerLens notebook:

```text
diegovalverde/TransformerLens
branch: othello-jspace-analysis
notebook: demos/Othello_GPT_Jacobian_Lens.ipynb
section: 7. Train a linear mine / theirs / empty board probe
```

The script intentionally stores the measured values as constants rather than rerunning the notebook. That keeps the book figure reproducible from the research memory while preserving the provenance chain:

```text
executed notebook -> research memory -> book figure
```
