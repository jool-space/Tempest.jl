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

The [usage guide](usage.md) covers CPU/GPU preparation, capture, supported array
operations, examples, and limitations. [Semantic verbs](verbs.md) describes how
to connect custom kernels. [Training state](state.md) covers functional updates,
selected gradients, and persistent optimizer buffers. [Implementation selection](implementations.md) covers
eligibility, construction, and fallback. The [source guide](internals.md) explains
the compiler stages and ownership invariants.

See the [API reference](api.md) for exported functions and types. Provider and
verb APIs are public but not exported; the [Implementations](implementations.md)
page describes them.

## Where things stand

Tempest owns the semantic tensor graph, differentiation, functional state
updates, region boundaries, storage lifetimes, and CUDA graph capture.
Implementations realize regions behind one `support`/`build_implementation`
protocol: generated CPU and CUDA kernels, cuBLAS, user
kernel pairs with private saved tensors, and cuTile-compiled
[graph regions](graph-regions.md) selected before storage planning.

`examples/gpt.jl` trains a small GPT this way: grouped-query attention through
the built-in `Tempest.attention` verb (fused cuTile kernels under TF32), runtime
token indices, and one captured graph per step. The [transformer report](transformer-step.md) compares it with
the same model under Reactant/XLA. `Tempest.profile(ex)` ranks a prepared
schedule's calls by device time.

The [compiler direction](compiler-direction.md) page records the architectural
conclusions and the next experiments. Dated investigations, measurements, and
review decisions are collected under Reports; their "next step" language
describes the state at that date, not the current plan.
