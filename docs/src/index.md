# Tempest

Tempest compiles Julia tensor expressions into explicit derivatives, fused kernel
regions, and planned storage, with optional CUDA graph replay. It is a compiler
prototype with functional optimizer updates, planned state donation, runtime
integer indices, and counter-based dropout. Symbolic shapes are not implemented.

```julia
using Tempest

loss(x, w) = sum(abs2, tanh.(w' * x))
x = randn(Float32, 32, 64)
w = randn(Float32, 32, 16)
ex = prepare(loss, x, w)
loss_buffer, dx, dw = execute!(ex)
explain(ex)
```

`examples/gpt.jl` trains a small GPT this way: grouped-query attention through
the built-in `Tempest.attention` verb, runtime token indices, and one captured
graph per step.

## Where to read

- The **Guide** is for writing models: the [usage guide](usage.md) covers
  CPU and GPU preparation, capture, supported array operations, examples and
  limitations; [training state](state.md) covers functional updates, selected
  gradients and persistent optimizer buffers; [semantic verbs](verbs.md),
  [reverse rules](reverse-rules.md) and [scalar rules](scalar-rules.md) cover
  connecting custom kernels and derivatives.
- The **Compiler** pages are for changing Tempest or writing implementations.
  [Architecture](architecture.md) is the current description of the whole:
  the pipeline, what fuses and why, the cost model, numerics, and the open
  directions. [Implementations](implementations.md), [graph regions](graph-regions.md)
  and [proposed regions](proposed-regions.md) document the provider protocol;
  [layouts](layouts.md) and the [source guide](internals.md) the internals.
- The [API reference](api.md) lists documented functions and types. Provider
  and verb APIs are public but not exported.
- [History](history.md) records what was decided and when. The **Reports**
  are dated investigations with their measurements; their plans are superseded
  by [Architecture](architecture.md#Open-directions).
