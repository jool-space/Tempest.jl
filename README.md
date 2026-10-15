# Tempest.jl

Tempest compiles Julia tensor expressions into explicit forward/backward graphs,
fused kernel regions, and reusable storage. Prepared schedules combine vendor
kernels with generated kernels and can replay through CUDA graphs. `explain`
shows the execution decisions.

**Status:** a compiler prototype with tested CPU and CUDA execution. It currently
supports explicit gradients and functional optimizer updates with planned state
donation, runtime integer indices, and counter-based dropout. Shapes are fixed
during tracing. `examples/gpt.jl` trains a small GPT as one captured CUDA graph,
faster than the same model under Reactant/XLA on the same GPU. The lead comes
from graph capture and a fused attention kernel rather than from better
generated code; see the [transformer report](docs/src/transformer-step.md).

## A prepared loss and gradient

```julia
using Tempest

loss(x, w) = sum(abs2, tanh.(w' * x))
x = randn(Float32, 32, 64)
w = randn(Float32, 32, 16)

program = trace(loss, x, w)
display(differentiate(program))
ex = prepare(program, x, w)
loss_buffer, dx, dw = execute!(ex)
explain(ex)
```

Core Tempest depends only on LinearAlgebra. For GPU execution, load CUDACore and
cuBLAS and prepare with device arrays:

```julia
using CUDACore
import cuBLAS
# Optional: import cuTile

ex = prepare(program, CuArray(x), CuArray(w))
capture!(ex)
bind!(ex, 2x, w)
loss_buffer, dx, dw = replay!(ex)
```

Inputs are copied into owned stable storage. Outputs are borrowed arrays,
overwritten by subsequent execution, and may alias other results. Floating
scalar inputs can change during replay without recapture. Executables are not
safe for concurrent use.

## What composes

Base broadcasting, matrix products, axis sums, reshape/dropdims, views, gathers,
and permutations work with symbolic arrays. Typed `AbstractArray` helpers can
compose these operations. Tuple/NamedTuple inputs preserve their structure through
selected gradients and functional updates. Built-in verbs cover batched matrix
products, a stable `logsumexp`, scaled dot-product attention (fused cuTile kernels
under TF32), runtime column gathers and dropout. Custom semantic verbs can retain an opaque kernel
boundary or expand a reference implementation. With Statistics loaded, `mean`,
`var` and `std` work too. Loading cuTile also enables graph fusion of supported
pointwise work and dependent reductions over leading, trailing or all axes,
selected before storage planning and kept only when estimated not to lose to
the kernels it displaces. See [graph regions](docs/src/graph-regions.md) for
the current scope and custom Julia callable contract.

General scalar indexing, data-dependent host branches, arbitrary mutation,
symbolic shapes and mixed-precision training are not
implemented. See the [usage guide](docs/src/usage.md) for exact boundaries.

## Examples and tests

```sh
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --project=examples examples/adam.jl
julia --project=examples examples/normalized_mlp.jl
julia --project=examples examples/semantic_kernels.jl
julia --project=examples examples/gpt.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

GPU tests run when `CUDACore.functional()` is true. Tests cover finite-difference
gradients, alias lifetimes, storage reuse, and repeated replay with rebound inputs.
Speed guard rails, which compare default plans with the CUDA kernels by time,
run on an idle GPU with `julia --project=test test/runtests.jl benchmarks`.

## Read more

- [Architecture](docs/src/architecture.md): the compiler as it stands, and its open directions
- [Functional updates and training state](docs/src/state.md)
- [Array and layout support](docs/src/layouts.md)
- [Optional shared scalar rules](docs/src/scalar-rules.md)
- [Semantic verbs and custom kernels](docs/src/verbs.md)
- [Implementation selection](docs/src/implementations.md)
- [Compiler source guide](docs/src/internals.md)
- [History](docs/src/history.md) of decisions, and dated reports: [a transformer training step](docs/src/transformer-step.md),
  [cuTile issues found along the way](docs/src/cutile-issues.md), the [outside review](docs/src/outside-review.md)
- [Contributing](CONTRIBUTING.md)
