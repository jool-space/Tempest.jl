# Using Tempest

Julia training, minus the eager tax. **A working compiler experiment, not a complete training system.**

Tempest traces tensor expressions, constructs an explicit backward graph, fuses
scalar expressions into kernel regions, and assigns storage by last use. The
prepared schedule runs through vendor GEMMs and generated kernels, with optional
CUDA graph replay. Model code contains neither allocation spaces nor launch logic.

[Source guide](internals.md) describes the current compiler stages and invariants.
[Contributing](https://github.com/jool-space/Tempest.jl/blob/main/CONTRIBUTING.md) describes expectations for maintainable changes.

[Working design notes](design-notes.md) are hypotheses, not a specification.
[First experiment report](first-experiment.md) records observations and measurements.
[Array frontend report](array-frontend.md) describes the next experiment.

## A forward/backward executable

```julia
using Tempest
using CUDACore
import cuBLAS
# Optional: import cuTile

f(x, w) = sum(abs2, tanh.(w' * x))
x, w = randn(Float32, 32, 64), randn(Float32, 32, 16)

p = trace(f, x, w)
display(differentiate(p))             # inspect forward AND backward operations
loss, (dx, dw) = value_and_gradient(p, x, w)  # allocating CPU reference

ex = prepare(p, CuArray(x), CuArray(w))  # infer CUDA placement
# prepare(p, x, w) runs the same region/storage plan on CPU
execute!(ex)
loss_buffer, dx_buffer, dw_buffer = outputs(ex)
explain(ex)                          # regions, providers, slots, lifetimes, bytes

capture!(ex)                        # warmup, then capture; owns graph and storage
bind!(ex, 2x, w)                     # copies into stable input buffers, outside capture
replay!(ex)
println(Array(outputs(ex)[1])[])     # explicit device-to-host read
```

Outputs are **borrowed, read-only results**, overwritten by subsequent execution.
They may alias each other or owned input storage; copy results you need to retain
or mutate. By default there is one gradient tree for every positional argument; `wrt` selects
a subset in the requested order. The executable owns independent input
copies; changing the original arrays does not change replay until `bind!` runs.
It is single-stream/single-execution infrastructure, not safe for concurrent use.

For updates and persistent optimizer buffers, see [Training state](state.md).

## Optional GPU integration

Core Tempest depends only on LinearAlgebra. Tracing, differentiation, planning,
and CPU execution require no CUDA packages. Install `CUDACore` and `cuBLAS` in
an application environment and load both to enable the CUDA extension. Loading
`cuTile` additionally enables tiled implementations of eligible regions. Loading
`cuDNN` enables the opt-in `CuDNNKernels()`, see below.

`prepare(f, args...)` infers placement from all array arguments, including wrappers
such as views and transposes. Floating scalars do not choose a device; scalar-only
programs default to CPU. Mixed host/device inputs or inputs on different devices
require an explicit target. Unknown array storage also requires an explicit target.

Use `target=CPUTarget()` or `target=CUDATarget()` for intentional input copies.
CUDA preparation and execution require the intended device/context to be current;
`CUDATarget(device=dev)` checks it explicitly. Owned buffers stay on that device.

`CUDATarget(kernels=:auto)` chooses cuTile where eligible if its extension is loaded,
and CUDA kernels elsewhere. `kernels=:cuda` uses no cuTile kernels, including the
reduction finalizer. `kernels=:cutile` requires cuTile to be loaded and uses CUDA
for mapped indexing and general axis reductions. Choices are fixed at preparation;
loading a provider later does not change existing executables. Compilation errors
propagate without retrying another provider.

## Additional implementations

Pass implementations to `prepare` to try them before the defaults:

```julia
ex = prepare(f, args...; implementations = (MyImplementation(),))
```

Tempest ships one opt-in implementation: `CuDNNKernels()`, available once
`cuDNN` is loaded alongside `CUDACore`. It runs attention, forward and backward,
on cuDNN's SDPA engines. Those engines take Float16 or BFloat16 operands only.
Float32 attention needs an explicit
`CuDNNKernels(float32_cast = Core.BFloat16)` or `Float16`; otherwise
cuDNN declines it and cuTile or the reference runs it. See
[cuDNN](implementations.md#cuDNN) for the precision table, rejections and
measurements.

`support` returns `Supported()` or `Unsupported(reason)`; construction can also
reject a region. `explain(ex)` reports selections, rejection reasons, and owned
workspace. Unexpected compiler errors propagate.

See [implementation contracts and fusion](implementations.md) for the
extension interface, current fusion policy, and the distinction between semantic
verbs and opaque kernel implementations. Semantic calls can also select custom
implementations or expand their references before region planning.

## Semantic verbs and forward-only programs

`trace` accepts symbolic values and nested tuple/named-tuple results.
`prepare_forward` preserves that structure without constructing gradients:

```julia
ex = prepare_forward(x -> (; y = sin.(x), total = sum(x)), randn(Float32, 32))
result = execute!(ex)
result.total[]
```

`prepare` remains the scalar-loss training convenience. For custom tensor verbs,
a symbolic method calls `record_call`; `output_spec` describes its results,
`reference` supplies a decomposition, and `verb_pullback` supplies an explicit
backward recipe. Implementations receive a `CallRequest` and can provide custom
forward/backward launches writing preassigned buffers. No model-level
implementation keyword or Pol allocation wrapper is required.

Calls without an available implementation expand into their reference before
fusion and storage planning. Selected calls remain opaque boundaries and receive
dense inputs, with copies for mapped views/transposes where needed. Multi-output
calls get distinct destinations, including outputs unused by the surrounding model.
The reference and pullback are explicit contracts, not inferred from kernel code.

See [semantic verb contracts](verbs.md) and the runnable
[custom CUDA training example](https://github.com/jool-space/Tempest.jl/blob/main/examples/semantic_kernels.jl). Automatic larger-region
matching, implementation-private saved state, and arbitrary mutation remain future
work.

## Ordinary array interfaces, symbolic values

Tracing executes the Julia function on `TracedArray{T,N} <: AbstractArray{T,N}`
stand-ins. Typed `AbstractMatrix`/`AbstractVector` helpers and concrete metadata
queries (`size`, `axes`, `ndims`, `eltype`) work without materializing data:

```julia
function rmsnorm(x::AbstractMatrix{T}, gain::AbstractVector{T}; eps = T(1e-4)) where {T}
    r = inv.(sqrt.(sum(abs2, x; dims = 1) / size(x, 1) .+ eps))
    x .* r .* gain
end
```

This is currently an ordinary composition of tensor operations, **not a tuned
normalization primitive**. Julia's `BroadcastStyle` machinery preserves lazy
broadcast trees, including singleton axes and arbitrary-arity scalar functions.

A full `sum(x)` produces `TracedScalar{T} <: Real`; `sum(x; dims=1)` remains an
array. A genuine zero-dimensional array remains distinct from a scalar, with
`x[]` explicitly producing a symbolic number. Both use one-element buffers in
the prepared executor. Floating scalar arguments are runtime inputs: `bind!` can
change their device-resident contents without recapturing. They dispatch as `Real`,
not as concrete `Float32`/`Float64` values.

Other scalar indexing, data-dependent host comparisons, `similar`, and mutation
are refused rather than falling through to scalar loops. Views, slices, permutations, and reshapes carry composed index mappings. Generated
kernels can read these mappings directly; a GEMM or returned result that requires
native storage gets an explicit materialization in the plan. AbstractArray does
not mean arbitrary Julia is supported.

## Views and layout composition

```julia
function cropped_loss(x::AbstractArray, weights)
    y = reshape(@view(x[1:2, :, :]), 2, :)
    sum(abs2, y .* weights)
end

x = randn(Float32, 3, 4, 2)
weights = randn(Float32, 2, 8)
ex = prepare(cropped_loss, x, weights)
execute!(ex)
explain(ex)
```

The crop and reshape above become one affine read from the original input buffer.
Flattening across a storage gap keeps a composed mapping instead of pretending
that the data are contiguous. Permutations, nested views, and gathers can compose
with those reshapes. Supported structural indices are integers, colons, integer
ranges, integer vectors, and individual `CartesianIndex` values. Index vectors are
snapshotted during tracing; changing their contents requires preparing a new graph.
CUDA uploads those snapshots to owned buffers outside capture.

`view(x, i, j)` returns a symbolic zero-dimensional array; `[]` extracts its scalar.
Direct scalar indexing of nonzero-dimensional symbolic arrays remains refused to
prevent generic array algorithms from unrolling a graph element by element.
Views and slices inside tracing produce `TracedArray` values, so helpers should
accept `AbstractArray` rather than require the concrete `SubArray` type. Mutation
remains unsupported; slice copy identity is not exposed by the functional graph.

External `SubArray` inputs still copy into stable dense input buffers. Their
returned gradients are with respect to the view arguments, not their host parents.
See [layout design](layouts.md) for the representation and current boundaries.

## Runtime integer indices

Integer arrays can be traced arguments. They carry no gradient and are used as
data-dependent indices; rebinding them does not re-prepare the program:

```julia
function loss(p, tokens, targets)
    x = p.embedding[:, tokens]                       # (hidden, size(tokens)...)
    logits = p.head' * reshape(x, size(x, 1), :)
    sum(Tempest.logsumexp(logits) .- Tempest.take_rows(logits, vec(targets)))
end
```

`table[:, indices]` gathers matrix columns; its gradient accumulates into the
table in index order, so repeated indices sum deterministically.
`Tempest.take_rows(x, rows)` picks one row per column. Out-of-range indices throw
on the CPU and read as NaN on CUDA.

## Randomness

`Tempest.dropout(x, seed; rate, stream = 0)` zeroes elements with probability
`rate` and rescales the rest. Its mask is a counter hash of the first element of
the integer array `seed`, the per-site `stream` number, and each element's
index. It is identical on the CPU and CUDA and is redrawn, not stored, for the
gradient. Randomness is functional: a training step advances the seed like
any other state, for example `(; params, seed = seed .+ Int32(1))` with the
seed donated.

## Generic scalar expressions

A scalar function needs no new tensor opcode or named GPU kernel. Supply its
scalar partials explicitly; Tempest constructs tensor derivatives and compiles
the expressions into the surrounding kernels:

```julia
ripple(x) = sin(x) * tanh(x)

function Tempest.scalar_partials(::typeof(ripple))
    (x -> cos(x) * tanh(x) + sin(x) * (one(x) - tanh(x)) * (one(x) + tanh(x)),)
end

ex = prepare((x, w) -> sum(abs2, ripple.(w' * x)), x, w; target = CUDATarget())
```

Functions must be pure, infer a concrete real scalar return type, and be
compilable by the selected GPU backend. GPU captures must be isbits. This is
**not automatic differentiation of arbitrary scalar Julia**. Built-in recipes
cover basic arithmetic, tanh, abs2, sin/cos, exp/log, sqrt/inv, powers, identity,
and floating casts. There is no ChainRules integration yet; `scalar_partials`
remains a provisional scalar-only hook, not the settled tensor-verb interface.
Tanh's backward consumes the forward output rather than recomputing tanh.

## What currently executes

- Concrete shapes; one-based real floating arrays, integer index arrays, and
  floating scalar inputs, with
  general forward results or a scalar-number loss for automatic differentiation.
- Matrix multiplication, transpose/real adjoint, reshape, dropdims, and axis permutations.
- `view`/`@view`, array slicing, and fixed integer-vector gathers. Ranges can have
  positive or negative steps; singleton and empty selections are supported.
- Layout composition with affine simplification when strides permit, and a general
  composed mapping otherwise. View pullbacks accumulate into the parent, including
  repeated indices; CUDA uses deterministic source-order accumulation.
- Lazy scalar broadcasts with Julia's leading-axis singleton expansion; unary
  `map`; full and axis sums, products, maxima and minima, including `sum(f,x)`,
  `prod(f,x)` and `reduce`/`mapreduce` with any associative operator (`init` is
  required beyond `+`, `*`, `max` and `min`), and user folds through `Tempest.fold`.
- Explicit reverse operations, broadcast cotangent reductions back to each
  operand's shape, repeated-use accumulation, and zeros for unused inputs.
- Dead-work elimination and conservative scalar-region fusion. GEMMs remain
  library calls; this does **not** generate custom GEMM engines.
- Exact-size/dtype slot reuse with alias-aware last-use accounting. Returned values
  stay live; shared reduction scratch is separate. `reuse=false` is a checkable baseline.
- CPU interpretation of the same schedule; optional CUDA kernels and cuBLAS, with
  optional cuTile implementations and opt-in cuDNN attention.
- `CUDATarget(kernels=:cuda)` explicitly selects generated CUDA scalar kernels instead.
  Mapped broadcast/layout reads use CUDA indexing. Dense leading-axis matrix
  reductions have a cuTile schedule; other axis reductions use a general CUDA
  schedule. Each provider has its own full-reduction finalizer.
- Stable input/output/workspace addresses and repeatedly tested graph replay.
  Capture rejects allocation/free nodes rather than assuming a mutating API is safe.

The cuBLAS handle, device alpha/beta scalars, and workspace are owned by the
executable. It uses `CUBLAS_DEFAULT_MATH`, independent of ambient CUDA
fast-math settings; no TF32 policy is silently selected.
`CUDATarget(matmul_precision=:tf32)` explicitly allows TensorFloat-32 tensor-core
products for Float32 matrices, including batched products. Workspace defaults
to 4 MiB and is configurable with `CUDATarget(blas_workspace_bytes=...)`. Memory reports separate
value slots, input copies, shared implementation workspace, and cuBLAS coefficients.
GPU gather indices and repeated-index accumulation schedules are prepared once,
owned by the runtime, and reported separately as layout indices. Host allocations
and driver/library metadata are not included in those byte counts.

CUDA scalar kernels require bits element types; cuBLAS GEMM currently requires
matching Float32 or Float64 operands. Float32/Float64 paths are covered by the
GPU regression suite. Cotangent storage follows each logical primal's dtype. This is a provisional
policy, **not mixed-precision or MXFP8 training support**.

## Run the composed examples

```sh
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --project=examples examples/gated_mlp.jl
julia --project=examples examples/normalized_mlp.jl
```

This runs a residual gated MLP with a custom SiLU scalar function, compares all
input/weight gradients with the CPU reference, captures it, explains its memory
plan, and times captured versus uncaptured execution of **the same schedule**.
It computes gradients for inputs and targets too; no optimizer update is included.
These timings are not comparisons with PyTorch or claims of SOTA performance.

`normalized_mlp.jl` adds typed RMSNorm composition, vector bias, and a runtime
scalar epsilon. It validates the full backward and replays with different epsilon
values while preserving buffer addresses. No normalization-specific engine is used.

## Known boundaries

No symbolic shapes, Boolean masks, arbitrary mutation, random distributions
beyond dropout, activation checkpointing, distributed execution, or general
autotuner yet. Runtime indices cover column gathers and row picks.
Functional optimizer updates support proven in-place donation and grouping of
independent elementwise updates; other updates retain staged commits.
Pol allocator integration is not yet implemented. The planner directly owns reusable arrays. Neither the node representation nor the
public API is settled.

Two cuTile/toolchain issues were encountered on GB10 with cuTile 1.0 and Julia
1.12.7: an indexed gather/scatter off-by-one with dynamically constructed indices,
and a mixed scalar/tile FMA verifier failure for some scalar expressions. Dense
regions use tiled loads/stores; transposed reads use CUDA scalar indexing. Some
derivatives are factored to avoid the FMA issue. Regions containing Float64
literals use CUDA because cuTile scalar/tile promotion can otherwise narrow
intermediate arithmetic. `examples/cutile_diagnostics.jl`
contains standalone reproducers. Compilation failures are not silently hidden;
`CUDATarget(kernels=:cuda)` is an explicit alternative for scalar expressions.

## Tests

```sh
# Run CPU tests and GPU tests when CUDA is functional
julia --project -e 'using Pkg; Pkg.test()'

# Optionally run both GPU provider configurations in the test environment,
# or a subset by name prefix (`--list` shows the names)
julia --project=test -e 'using Pkg; Pkg.instantiate()'
julia --project=test test/runtests.jl
TEMPEST_TEST_CUTILE=false julia --project=test test/runtests.jl
julia --project=test test/runtests.jl cuda/ verbs/
```

Tests cover finite-difference gradients, the explicit backward graph, custom scalar
functions, typed array helpers, lifetimes/reuse, broadcast axes, reshaped/transposed
reads, views/gathers and their finite-difference pullbacks, scalar versus
zero-dimensional inputs, reduction tails, and repeated CUDA
replay with changed inputs and stable pointers. GPU test dependencies are declared in `test/Project.toml`. GPU tests run when
`CUDACore.functional()` is true and otherwise log a skip. cuTile is enabled by
default; `TEMPEST_TEST_CUTILE=false` exercises the CUDA-only provider path and
skips `test/cutile/`. A fresh process checks that core use loads no GPU package. GPU execution has been tested on NVIDIA GB10;
other architectures and Julia versions need validation. The manual **GPU validation**
workflow runs both provider configurations on supplied CUDA runner labels; it needs
an available GPU runner. Regular CI uses the same functional-CUDA gate.
