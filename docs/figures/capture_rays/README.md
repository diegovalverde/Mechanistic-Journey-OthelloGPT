# Capture Ray Figure Provenance

These PNGs are measured experimental figures, not conceptual illustrations.

## Source

- Source repository: `https://github.com/diegovalverde/TransformerLens`
- Source branch: `othello-jspace-analysis`
- Exact visualization commit: `b4b529fec329dc318755c579c58af65950143323`
- Exact legal-mask reconstruction commit: `97ecdbc`
- Exact direct-legality reconstruction commit: `c6e32d6`
- Notebook: `demos/Othello_GPT_Jacobian_Lens.ipynb`
- Notebook sections: `47. Where does a capture ray become an internal feature?` and `48. Visualizing decoded capture rays`
- Source directory: `demos/othello_jacobian_lens_outputs/capture_ray_visualization_20260828_193735/`
- Primary display site: `blocks.5.hook_resid_post`

Legal-mask reconstruction figures were copied from:

- Notebook section: `49. Can decoded capture rays reconstruct the legal-move mask?`
- Source directory: `demos/othello_jacobian_lens_outputs/legal_mask_reconstruction_20260829_204725/`
- Best validation-selected site: `blocks.7.hook_resid_post`

Direct legal-square reconstruction figures were generated from:

- Notebook section: `50. Does MLP6 make legal-square identity linearly explicit?`
- Source directory: `demos/othello_jacobian_lens_outputs/direct_legality_probe_20260830_011103/`
- Primary MLP6 comparison: `blocks.6.hook_resid_mid` to `blocks.6.hook_resid_post`
- Display site: `blocks.6.hook_resid_post`

## Copied Files

- `hero_example_01_board_and_compass.png`
- `hero_example_01_site_progression.png`
- `hero_example_02_board_and_compass.png`
- `hero_example_02_site_progression.png`
- `hero_example_03_board_and_compass.png`
- `hero_example_03_site_progression.png`
- `hero_example_04_board_and_compass.png`
- `hero_example_04_site_progression.png`
- `hero_example_05_board_and_compass.png`
- `fullboard_heatmap_example_01.png`
- `hard_nearmiss_site_summary.png`
- `site_metric_summary.png`
- `capture_direction_suppression_example_01.png`
- `legal_mask_10_board_side_by_side.png`
- `direct_legality_post6_10_board_reconstructions.png`

The files were copied unchanged from the source directory.
