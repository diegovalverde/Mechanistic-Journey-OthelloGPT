# TransformerLens

This appendix is a practical guide to the TransformerLens concepts used in the project. It is not a complete TransformerLens manual. It explains the parts needed to read `demos/Othello_GPT_Jacobian_Lens.ipynb`, reproduce the book's hook-based experiments, and avoid the most common mistakes when caching, editing, or differentiating through Othello-GPT.

The sibling source repository used by the project is [diegovalverde/TransformerLens](https://github.com/diegovalverde/TransformerLens).

The experimental notebook is:

```text
demos/Othello_GPT_Jacobian_Lens.ipynb
```

The important model fact is that this is a `HookedTransformer`-style Othello model with 8 blocks, residual width 512, 8 heads per block, MLP width 2048, output vocabulary 61, context length 59, GELU MLPs, and `LNPre` normalization.

## HookedTransformer

`HookedTransformer` is the TransformerLens class used by the Othello notebooks. It behaves like a PyTorch model, but its internals contain named `HookPoint`s. A hook point is a place where you can observe or modify an activation during a forward pass.

The original Othello notebook constructs a model from a `HookedTransformerConfig`, converts weights into TransformerLens format, and loads them into the model. The Jacobian notebook then reuses that model. The important point for the book is not how to train the model, but how to access and intervene on its internal activations.

A typical forward pass returns logits:

```python
logits = model(tokens)
```

For Othello-GPT, `tokens` is an integer tensor of move-token IDs. For a single prefix, the shape is usually:

```text
[1, pos]
```

and the logits have shape:

```text
[1, pos, 61]
```

The final prefix position is the row most analyses use, because that is where the model predicts the next move.

## `run_with_cache`

`run_with_cache` runs the model and saves activations at hook points:

```python
logits, cache = model.run_with_cache(
    tokens,
    names_filter=["blocks.4.hook_resid_post"],
)

h = cache["blocks.4.hook_resid_post"]
```

The returned cache maps hook names to tensors. For the residual hook above, the tensor shape is:

```text
[batch, pos, 512]
```

You can cache all hook points by leaving `names_filter=None`, but this is often wasteful. The Othello notebook usually requests the few hooks needed for a section.

The cache stores detached activations by default. That is appropriate for most inspection and plotting. If you want gradients, you need to be more careful, as described below.

`names_filter` can be a single hook name, a list of hook names, or a predicate over hook names. A predicate is useful when exploring, but a list is easier to audit in book-supporting code because it documents exactly which activations the section depends on. For example, a component-decomposition cell should show the layer-7 residual, attention, MLP, and per-head hooks it needs.

The cache can also remove the batch dimension for single-example runs, but the Othello notebook often keeps the batch axis explicit. Keeping `[batch, pos, ...]` visible makes it harder to confuse a final-prefix row with a whole sequence tensor.

## Hook Names

TransformerLens hook names are strings that identify module locations. The project uses block-level hook names of the form:

```text
blocks.L.hook_resid_pre
blocks.L.attn.hook_result
blocks.L.hook_attn_out
blocks.L.hook_resid_mid
blocks.L.mlp.hook_pre
blocks.L.mlp.hook_post
blocks.L.hook_mlp_out
blocks.L.hook_resid_post
```

`L` is the zero-based layer index. For Othello-GPT, valid block indices are `0` through `7`. Therefore:

```text
blocks.7.hook_mlp_out
```

means the MLP output of the eighth and final block.

The hook name is part of the scientific claim. A result at `blocks.4.hook_resid_post` is not the same as a result at `blocks.7.hook_resid_post`.

## Activation Shapes

For this model, the most important shapes are:

| Hook | Shape | Meaning |
| --- | --- | --- |
| `blocks.L.hook_resid_pre` | `[batch, pos, 512]` | residual stream entering block `L` |
| `blocks.L.attn.hook_result` | `[batch, pos, 8, 512]` | per-head residual-space attention writes |
| `blocks.L.hook_attn_out` | `[batch, pos, 512]` | summed attention update written by the block |
| `blocks.L.hook_resid_mid` | `[batch, pos, 512]` | residual stream after attention has been added |
| `blocks.L.mlp.hook_pre` | `[batch, pos, 2048]` | MLP neuron pre-activations |
| `blocks.L.mlp.hook_post` | `[batch, pos, 2048]` | MLP neuron post-GELU activations |
| `blocks.L.hook_mlp_out` | `[batch, pos, 512]` | MLP update written to the residual stream |
| `blocks.L.hook_resid_post` | `[batch, pos, 512]` | residual stream after the full block |

The component-analysis sections call:

```python
model.set_use_attn_result(True)
```

before using `blocks.L.attn.hook_result`. Per-head result tensors are memory-expensive, so TransformerLens does not always compute them by default.

## Capturing Activations

A small cache is often enough:

```python
names = [
    "blocks.7.hook_resid_pre",
    "blocks.7.attn.hook_result",
    "blocks.7.hook_attn_out",
    "blocks.7.hook_resid_mid",
    "blocks.7.mlp.hook_pre",
    "blocks.7.mlp.hook_post",
    "blocks.7.hook_mlp_out",
    "blocks.7.hook_resid_post",
]

_, cache = model.run_with_cache(tokens, names_filter=names)
target_pos = tokens.shape[1] - 1
mlp_post = cache["blocks.7.mlp.hook_post"][0, target_pos]
```

That last tensor has shape `[2048]`. It is the post-GELU activation vector for MLP7 at the final prefix token.

For residual hooks, selecting `[0, target_pos]` gives a 512-dimensional vector. For per-head results, selecting `[0, target_pos]` gives an `[8, 512]` tensor: one residual-space write per head.

A cache should be treated as a snapshot of one model run. If a later hook changes an activation, old cached tensors do not update. For ablation and intervention experiments, compute the clean metric from the clean run and the edited metric from the edited run, then compare them explicitly.

## Editing Activations with Hooks

A forward hook is a function that receives the activation tensor and a hook object. It returns the activation to use for the rest of the forward pass. To add a residual direction:

```python
def add_direction_hook(act, hook):
    act = act.clone()
    act[:, target_pos, :] += alpha * direction
    return act

edited_logits = model.run_with_hooks(
    tokens,
    fwd_hooks=[("blocks.4.hook_resid_post", add_direction_hook)],
)
```

The clone avoids modifying a tensor in place in a way that can confuse PyTorch's autograd or accidentally affect other references. In many notebook experiments, the direction is a normalized semantic probe direction.

For mean replacement, the hook overwrites part of a tensor with a replacement value:

```python
def replace_mlp_out(act, hook):
    act = act.clone()
    act[:, target_pos, :] = mean_mlp_out
    return act
```

This kind of intervention is an ablation or replacement. It is not automatically a rescue. A rescue experiment would first disrupt a computation and then patch back a proposed intermediate activation to see whether behavior selectively recovers. The executed Othello notebook does not contain such a rescue result.

For neuron-level mean replacement, the edited tensor is usually `blocks.7.mlp.hook_post`, because that is the 2048-dimensional post-GELU neuron activation vector:

```python
def replace_selected_neurons(act, hook):
    act = act.clone()
    act[:, target_pos, neuron_ids] = mean_post[neuron_ids]
    return act
```

This changes selected neuron activations before they are multiplied by `W_out` and written back to the residual stream. It is different from replacing `hook_mlp_out`, which changes the already-combined 512-dimensional MLP output.

## Rerunning Downstream

When a hook edits `blocks.4.hook_resid_post`, all later blocks see the edited activation. That is why the experiment is causal rather than merely observational. The model is rerun downstream from the edited internal state and produces new logits.

The Jacobian notebook also uses `start_at_layer` and `stop_at_layer` patterns in places where it wants to expose a hidden state as an input to a downstream function. The conceptual structure is:

```python
base_final_resid = model(tokens, stop_at_layer=model.cfg.n_layers)

logits = model(
    final_resid[:, None, :],
    start_at_layer=model.cfg.n_layers,
    return_type="logits",
)
```

The exact tensor rank depends on the local helper. The important idea is that TransformerLens can run a prefix of the model, manipulate or differentiate a hidden state, and then run the downstream readout path.

## Gradients and Autograd

Neel Nanda's original Othello demo disables gradients because it is inference-oriented. The Jacobian notebook keeps that design intact and re-enables gradients only where needed:

```python
with torch.enable_grad():
    delta = torch.zeros(D_MODEL, device=device, requires_grad=True)
    logits = run_with_residual_delta(tokens, delta)
    score = logits[0, target_pos, move_id]
    grad = torch.autograd.grad(score, delta)[0]
```

This pattern treats `delta` as a differentiable residual edit. The gradient of a scalar score with respect to `delta` is the same local sensitivity as the gradient with respect to the chosen residual activation.

The notebook uses this for raw move logits and for legality contrasts. The target scalar must be stated because it defines the gradient. A gradient of the E3 raw logit is not the same object as a gradient of the E3 legality contrast.

A simplified legality-gradient pattern is:

```python
with torch.enable_grad():
    delta = torch.zeros(D_MODEL, device=device, requires_grad=True)
    logits = run_with_residual_delta(tokens, delta)
    selected = logits[0, target_pos, move_id]
    illegal_mean = logits[0, target_pos, illegal_empty_ids].mean()
    legality = selected - illegal_mean
    grad = torch.autograd.grad(legality, delta)[0]
```

The output `grad` has shape `[512]`. It is a sensitivity direction for the selected scalar contrast at the selected hook and position.

## Re-Enabling Gradients Carefully

There are two common pitfalls.

First, a cached tensor is usually detached. If you take `cache["blocks.4.hook_resid_post"]` and then ask for gradients through it, you will not get the computation graph you expect. The notebook instead creates a fresh differentiable variable, inserts it with a hook, and differentiates the final scalar with respect to that variable.

Second, backward hooks and cached gradients require hook lifetime care. TransformerLens removes temporary hooks at the end of `run_with_hooks` by default. That is good for forward editing. If you add backward hooks and then call `loss.backward()` later, you may need `reset_hooks_end=False` so the backward hooks remain active. The Othello notebook mostly avoids this by using `torch.autograd.grad` inside a local `torch.enable_grad()` block.

Third, avoid wrapping the differentiable region in `torch.no_grad()`. It is common for inference notebooks to use no-grad globally for speed. The Jacobian notebook's first section exists because local derivative work needs an explicit escape hatch from that inference style.

## Jacobian-Vector Products in Practice

The notebook uses JVPs when it wants the effect of one direction under a downstream hidden-state map. A simplified pattern is:

```python
def downstream_final_resid(source_resid):
    return run_downstream_from_source(tokens, source_resid)

base_source = get_source_resid(tokens)
direction = normalized_probe_direction

base_out, transported = torch.autograd.functional.jvp(
    downstream_final_resid,
    (base_source,),
    (direction,),
)
```

The transported vector is the local image of the source direction under the downstream computation. In Chapter 5, this is why the book describes hidden-state transport rather than only a logit-space Jacobian. The source and target are both residual-space objects, and the final readout is a separate check.

Finite differences are still useful:

```python
plus = downstream_final_resid(base_source + eps * direction)
minus = downstream_final_resid(base_source - eps * direction)
finite_diff = (plus - minus) / (2 * eps)
```

Comparing the finite difference with the JVP catches hook-location and graph-construction errors.

## Per-Head Result Tensors

For component attribution, we need individual head writes, not only the total attention output. The per-head tensor at:

```text
blocks.7.attn.hook_result
```

has shape:

```text
[batch, pos, 8, 512]
```

At one target position:

```python
head_results = cache["blocks.7.attn.hook_result"][0, target_pos]
```

has shape:

```text
[8, 512]
```

Each row is one head's residual-space contribution after the output projection into `d_model`. This is why the book can rank L7H0, L7H1, and so on by dot product with a legality-gradient direction.

The total `hook_attn_out` is still useful. It is the block's full attention write, while `hook_result` lets us decompose that write into heads.

Per-head tensors are easy to misread. `hook_result` is not the attention pattern. The attention pattern says which source positions were weighted. The result tensor says what each head wrote into residual space after values were mixed and projected. The book's component attribution uses result vectors because they live in the same 512-dimensional residual space as the legality gradient.

## Residual Hooks

The residual hooks mark the main sequence of states inside a block:

```text
resid_pre -> attention -> resid_mid -> MLP -> resid_post
```

For Othello-GPT's ordinary sequential blocks:

```text
resid_mid = resid_pre + attn_out
resid_post = resid_mid + mlp_out
```

These identities are pedagogically useful and experimentally useful. If a cached component output has the expected shape, it can be compared to a downstream gradient. If the residual addition identity fails unexpectedly, the hook site or model configuration may not be what the code assumes.

## MLP Hooks

The MLP has three hook sites that matter for the book:

```text
blocks.L.mlp.hook_pre
blocks.L.mlp.hook_post
blocks.L.hook_mlp_out
```

`hook_pre` is the 2048-dimensional vector before GELU. `hook_post` is the 2048-dimensional vector after GELU. `hook_mlp_out` is the 512-dimensional vector written back to the residual stream.

For neuron \(k\), a simplified contribution to the MLP output is:

$$
\text{post}_k W_\text{out}[k,:].
$$

This is why Chapter 9 can separate detector-like questions from writer-like questions. `hook_post` asks whether a neuron activates under certain board conditions. `W_out[k, :]` and `hook_mlp_out` ask what direction the neuron can write when active.

For input-weight geometry, the relevant parameter is the neuron's input direction in `W_in`. For output-weight geometry, the relevant parameter is the neuron's row of `W_out`. These are not interchangeable. A neuron can have weak alignment with simple board-state directions on the input side while its output direction still aligns with legality gradients.

## Hook Hygiene

Hooks are global model state while installed. TransformerLens provides `run_with_hooks` and `run_with_cache` so hooks are usually cleaned up at the end of a call. This is safer than manually attaching hooks and forgetting to remove them.

When debugging, use small, named hook lists and reset hooks between unusual experiments. If two interventions appear to interact when they should not, stale hooks are one possible cause. Another is that a cached tensor from a previous run is being compared with logits from a later run. The cleanest pattern is to compute clean and edited metrics in the same helper function and return a small dictionary of scalar results.

The book's evidence depends on this bookkeeping. A hook experiment is only interpretable when the reader can identify the hook name, selected position, edited tensor slice, replacement value, and target metric.

Small ambiguities become large interpretation errors.

## Minimal Reproducible Snippets

For book-supporting work, prefer tiny snippets that expose one idea over large copied notebook cells. A good snippet should include the hook name, selected position, tensor shape, and target scalar:

```python
target_pos = tokens.shape[1] - 1
hook_name = "blocks.4.hook_resid_post"
direction = direction / direction.norm()
```

That is enough context for a reader to know where the intervention occurs. The full notebook can hold data loading, plotting, and batching details. The appendix should teach the interface without duplicating pages of exploratory code.

## Attribution Versus Ablation in Code

Attribution is usually a calculation on cached clean activations:

```python
component_attr = torch.dot(component_write, legality_gradient)
```

No model behavior has changed. The result says the current component write is locally aligned or anti-aligned with the scalar target.

Ablation changes the model run:

```python
ablated_logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, replacement_hook)])
```

Then the notebook compares the clean and ablated legality contrast. For whole-component ablation, the sign convention is:

```text
delta_legality_contrast = L_ablate - L_clean
```

For the MLP7 neuron-group ablation figure, the notebook reports:

```text
legality_degradation = L_clean - L_ablate
```

Those are different signs. Keep the convention visible when interpreting results.

## Minimal Project Checklist

When reading or adapting the notebook, check these items before trusting a result:

1. Which hook is used?
2. Which token position is selected?
3. Is the target a raw logit, a legality contrast, or a hidden-state target?
4. Are gradients enabled only for the differentiable calculation?
5. Was `model.set_use_attn_result(True)` called before per-head result analysis?
6. Is the tensor shape consistent with the intended component?
7. Is the run observational caching, attribution, ablation, intervention, mediation-like comparison, or rescue?

Most mistakes in this project are not deep mathematical mistakes. They are usually one of these bookkeeping errors.
