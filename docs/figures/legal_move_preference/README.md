# Legal-Move Preference Figures

These PNGs are measured experimental figures, not conceptual illustrations.

They were copied from the executed TransformerLens notebook output:

- Source repository: `https://github.com/diegovalverde/TransformerLens`
- Branch: `othello-jspace-analysis`
- Commit: `cd6f523`
- Notebook: `demos/Othello_GPT_Jacobian_Lens.ipynb`
- Notebook section: `52. Does Layer 7 linearly encode preference among legal moves?`
- Output directory: `demos/othello_jacobian_lens_outputs/l7_legal_move_preference_20260830_223054/`

Files:

- `l7_preference_probe_site_progression.png`: site progression for legal-vs-legal preference probe metrics from post5 through post7.
- `l7_preference_probe_example.png`: held-out large-gap board example showing final legal logits and probe ranking at pre7, mid7, and post7.
- `l7_preference_neartie_example.png`: held-out near-tie board example showing final legal logits and probe ranking at pre7, mid7, and post7.

In the board-example figures, the `pre7` column can also be read as `post6/pre7`: Section 52 measured a maximum absolute residual difference of `0` between `blocks.6.hook_resid_post` and `blocks.7.hook_resid_pre` at the final prefix position.
