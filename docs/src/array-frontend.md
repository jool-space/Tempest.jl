> This report describes the earlier array-frontend experiment. Views, slices, and
> permutations are now supported through the [layout layer](layouts.md).

# Second experiment: an array frontend that composes

This is a report of an experiment, not a specification or a frozen API. It follows
`first-experiment.md`; that document records the earlier, more restricted state.

## The frontend is now an array/number interface

`TracedArray{T,N}` is an `AbstractArray{T,N}`. Shape and axes are concrete metadata;
its elements are not host values. `TracedScalar{T}` is a `Real`, distinct from a
zero-dimensional traced array. The logical node records that distinction even
though both can occupy one element of physical storage.

`trace` runs ordinary Julia on those stand-ins. AbstractMatrix/AbstractVector
methods dispatch normally, but methods requiring a concrete CuArray or Float32
scalar do not automatically become traceable. Scalar iteration, allocation through
`similar`, mutation, and data-dependent scalar comparisons fail instead of
silently constructing one node per array element. The exception is `x[]` for a
zero-dimensional array: that explicitly introduces a symbolic scalar alias.

Julia's BroadcastStyle machinery now captures lazy broadcast trees. The old
handful of unary/binary `broadcasted` overloads is gone. Custom scalar callables
can have multiple arguments; the derivative hook must provide all their partials.
Built-in product recipes handle arbitrary arity without dividing by operands, so
zero-valued factors do not invalidate the product rule.

This remains a tensor-language subset, not a promise that every AbstractArray
algorithm is supported. General slicing/views, mutation, and arbitrary host control
flow remain outside it. The tests check metadata behavior and method ambiguities
as well as numerical behavior.

## Shapes have executable consequences

Broadcasting follows Julia's leading-axis convention, including singleton and
missing trailing axes. Backward sums expanded dimensions back to each operand's
shape, then restores its rank and scalar/array kind. A column bias, row bias,
vector gain, runtime scalar, and zero-dimensional array have distinct logical
contracts, not a single ad hoc 'scalar broadcasting' case.

Full sums return numbers. `sum(...; dims=...)` preserves rank and reduced singleton
axes. Unary map and sum-map reductions compose through the same node vocabulary.
Reshape and dropdims are represented as aliases when the layout permits it.
Reshaping a noncontiguous transpose requires an explicit materialization node;
it is not mislabeled as a free stride change. That applies to generated backward
work too.

The reference interpreter explicitly expands broadcast outputs to the node's shape
and skips unreachable backward work. A derivative for a constant exponent, for
example, must not evaluate an unused logarithm at negative inputs. The prepared
scheduler continues to account for underlying alias storage when computing last
use and assigning reusable slots.

## Kernel choices are still intentionally simple

The same expression trees have several concrete execution paths:

- dense pointwise/full reductions: generated cuTile;
- leading-axis matrix reductions with dense/scalar operands: a cuTile column
  schedule, looping over tiles and masking after the scalar expression;
- other axis reductions and mapped broadcast/transpose indexing: generated CUDA;
- matrix multiplication: the existing owned-handle/workspace cuBLAS path.

The general CUDA axis reduction uses one block per output, a strided traversal of
reduced coordinates, and a fixed shared-memory reduction tree. This establishes
correct execution, not a competitive general normalization engine. Provider
selection is explicit in `explain`; compilation errors are not hidden by an
unannounced fallback. `target=CUDATarget(kernels=:cuda)` remains an explicit alternative.

## A normal model helper now works

`examples/normalized_mlp.jl` defines an ordinary typed RMSNorm helper:

```julia
function rmsnorm(x::AbstractMatrix{T}, gain::AbstractVector{T}; eps = T(1e-4)) where {T}
    r = inv.(sqrt.(sum(abs2, x; dims = 1) / size(x, 1) .+ eps))
    x .* r .* gain
end
```

It composes that helper with a gated MLP, a residual, vector bias, and a scalar
loss. There is no normalization-specific node or kernel: it is a deliberately
visible decomposition, useful as a baseline for later specialized implementations.

Epsilon is a floating scalar argument to the loss function. `bind!` writes its new
value into stable device storage, and repeated replay sees the update without
recapture. All argument gradients—including the epsilon derivative—are checked
against the CPU reference. This is not an optimizer step.

On the tested GB10 configuration (D=64, I=96, N=97, Float32), the example produced
32 regions and 43 captured kernel nodes. Value storage was 411,420 bytes versus
610,468 without reuse. Separately it owned 123,908 bytes of input storage, 100 bytes
of reduction scratch, 4 MiB of cuBLAS workspace, and 8 bytes of coefficients. These
are observations from this implementation, not performance targets or process-wide
memory measurements. Run the example to inspect the current decisions.

## Still open

ChainRules is not integrated. The scalar partial hook and tensor differentiation
recipes remain experimental. The typed array frontend should make it possible to
test symbolic execution of suitable rules, but opaque allocating pullbacks do not
by themselves expose lifetimes or saved-state contracts to the planner.

Structured parameter trees, selected gradients, specialized forward/backward
pairs, Stiletto region lowering, Pol storage integration, checkpointing, and mixed
precision remain future work. This experiment gives those questions a more useful
baseline; it does not decide their interfaces in advance.
