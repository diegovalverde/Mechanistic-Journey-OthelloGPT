# Othello-GPT Model Architecture

This page is a durable technical reference for the model architecture used in the book. It records source-level facts, not experimental results.

## Source

| Field | Value |
| --- | --- |
| Repository | `diegovalverde/TransformerLens` |
| Local checkout | `/Users/diegovalverdegarro/workspace/projects/TransformerLens` |
| Branch | `othello-jspace-analysis` |
| Original notebook | `demos/Othello_GPT.ipynb` |
| Experimental notebook | `demos/Othello_GPT_Jacobian_Lens.ipynb` |
| Config source | `HookedTransformerConfig(...)` construction in the original notebook |

The original notebook constructs the model with `HookedTransformerConfig` and then instantiates `HookedTransformer(cfg)`. The experimental notebook prints the same configuration before the Jacobian and J-space analyses.

## Model Dimensions

| Quantity | Verified value |
| --- | ---: |
| `n_layers` | 8 |
| `d_model` | 512 |
| `n_heads` | 8 |
| `d_head` | 64 |
| `d_mlp` | 2048 |
| `d_vocab` | 61 |
| `d_vocab_out` | 61 |
| `n_ctx` | 59 |
| `act_fn` | `gelu` |
| `normalization_type` | `LNPre` |

Layer numbering in TransformerLens is zero-based. The eight transformer blocks are indexed:

```text
0, 1, 2, 3, 4, 5, 6, 7
```

Therefore, "layer 7" means the eighth and final transformer block. "Layer 4" means zero-based block index 4, not the fourth block in ordinary one-based prose.

## Block Structure

For this model, the relevant TransformerLens block structure is the standard sequential attention-then-MLP path:

```text
hook_resid_pre
attention input normalization
attention
hook_attn_out
residual add
hook_resid_mid
MLP input normalization
MLP
hook_mlp_out
residual add
hook_resid_post
```

Schematic equations for one block \(l\):

$$
a_l = \mathrm{Attention}_l(\mathrm{LNPre}_1(r_l))
$$

$$
r_l^\text{mid} = r_l + a_l
$$

$$
m_l = \mathrm{MLP}_l(\mathrm{LNPre}_2(r_l^\text{mid}))
$$

$$
r_{l+1} = r_l^\text{mid} + m_l
$$

`LNPre` in this TransformerLens implementation performs the centering and scaling parts of LayerNorm without learned LayerNorm weights or biases. The source code comment says the LayerNorm weights have been folded, so the normalization modules only need center and scale operations. The final normalization path similarly uses `LayerNormPre` for `normalization_type="LNPre"` before unembedding.

The ordinary MLP uses TransformerLens's orientation:

| Parameter | Shape |
| --- | --- |
| `W_in` | `[d_model, d_mlp] = [512, 2048]` |
| `b_in` | `[d_mlp] = [2048]` |
| `W_out` | `[d_mlp, d_model] = [2048, 512]` |
| `b_out` | `[d_model] = [512]` |

For one token position:

$$
p = x W_\text{in} + b_\text{in}
$$

$$
g = \mathrm{GELU}(p)
$$

$$
m = g W_\text{out} + b_\text{out}
$$

## TransformerLens Hook Points Used in This Project

For this Othello-GPT model, each of the following block-level hook tensors has shape:

```text
[batch, pos, 512]
```

| Hook name | What it represents | Where used in the investigation |
| --- | --- | --- |
| `hook_resid_pre` | Residual stream entering a transformer block, before attention input normalization. | Architecture reference point for layer-wise residual analyses and additive identity checks. |
| `hook_attn_out` | The 512-dimensional attention update before it is added to the residual stream. | Component-output analysis, attribution, and attention-head decomposition. |
| `hook_resid_mid` | Residual stream after the attention output has been added. | Verifying the identity `resid_mid = resid_pre + attn_out`; MLP input-side analyses. |
| `hook_mlp_out` | The 512-dimensional MLP update before it is added to the residual stream. | MLP component attribution, ablation, and neuron-level decomposition. |
| `hook_resid_post` | Residual stream after the full block. | Chapter 2 board probe at `blocks.4.hook_resid_post`; Chapter 4 residual interventions; Chapter 5 source hook for J-space transport. |

Additional lower-level hooks exist for attention internals and MLP pre/post activations, but the table above covers the durable block-level measurement points used throughout the book.

## Final Normalization and Unembedding

After the last transformer block, the model applies final normalization and then unembedding to produce output logits over the 61-token move vocabulary. In the Chapter 5 J-space experiment, the hidden-state target was the final residual stream immediately before this final normalization and unembedding path. In logit-space experiments, the downstream function includes the final normalization and unembedding.

## Evidence Boundary

These architecture facts are verified from source. They should not be described as experimental discoveries. Experimental claims about board decodability, Jacobian validation, legality sensitivity, layer sweeps, component importance, or neuron behavior must continue to flow through:

```text
executed notebook
    -> research memory
    -> book prose
```
