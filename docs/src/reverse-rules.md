# Implementation-specific reverse rules

An implementation can choose its forward/backward pair without changing the
model's call signature. This is useful when an optimized forward saves statistics
that are absent from the public mathematical result.

The public verb still declares `output_spec`, `reference`, and `verb_pullback`.
Those describe its mathematics and reference derivative. A provider optionally
implements `reverse_rule(provider, request::CallRequest)` and returns a
`ReverseRule` or an `Unsupported` reason.

For example, a normalization adapter can define:

```julia
function norm_rule(x, weight; eps)
    saved = norm_forward(x, weight; eps)  # private semantic call: (y, rstd)
    saved.y, dy -> norm_backward(dy, x, weight, saved.rstd)
end

using Tempest: CallRequest, Unsupported, ReverseRule

function Tempest.reverse_rule(::MyKernels, request::CallRequest)
    request.call.f === rmsnorm || return Unsupported("not RMSNorm")
    request.target isa CUDATarget || return Unsupported("requires CUDA")
    all(s -> s.dtype === Float32, request.inputs) || return Unsupported("Float32 only")
    ReverseRule(norm_rule)
end
```

This is a symbolic recipe, not a runtime AD closure. Tempest invokes it during
preparation, after the target and input metadata are known. The pullback is called
with symbolic cotangents; its captured `rstd` becomes an explicit graph input to
backward. The compiled executable contains ordinary launches and buffers.

`norm_forward` and `norm_backward` use the existing semantic-call interface.
Their implementations receive concrete destination and input arrays through
`build_implementation`. Launch wrappers may require `CuArray`, and kernels may
require their own concrete device array types. Neither sees a Tempest tracer.

The model remains ordinary Julia:

```julia
loss(x, residual, weight) = sum(abs2, rmsnorm(x .+ residual, weight))
ex = prepare(loss, x, residual, weight; implementations = (MyKernels(),))
```

## Lifetime and fallback contracts

Saved tensors are outputs of the private forward call. They receive storage slots
and remain live until their last consumer, including across other layers' forward
calls. They are not temporary workspaces. A backward implementation can separately
request `WorkspaceSpec`s for partial reductions; those buffers can be reused by
subsequent launches.

Selection applies to a complete pair. Every semantic call introduced by the rule
must have a supported implementation. If any of those calls rejects construction,
Tempest discards that specialization, tries the next reverse rule, and finally
uses the original forward plus reference pullback. Storage planning is repeated;
value IDs and donation decisions from the rejected graph are not reused. Compiler
exceptions propagate rather than becoming capability rejections.

The provider can choose a different saved representation, or recompute values
instead. A later rule receives the original public call, not the rejected rule's
private outputs. Existing independent forward/backward semantic implementations
continue to work when no specialized reverse rule is selected.

`explain` reports the selected rule and pair rejection reasons. Its ordinary call
regions and storage slots include the private saved outputs.

## Representation

Differentiation traces the reference `verb_pullback` into an explicit recipe
stored in a `CallPullback` boundary. Reference execution evaluates that recipe.
Preparation either splices its operations into the main graph or replaces the
forward and pullback together. This preserves the opportunity to select a rule
when preparing a pre-differentiated program, including one extended by
`with_update`; target choice does not have to precede differentiation.

The boundary is an internal compiler representation. It is not a new public verb
that kernel libraries must implement. Public output shapes and scalar/array
identity are checked, as are pullback shapes and trace ownership. Numerical
agreement with the reference remains the implementation author's responsibility.

## Executable example

`examples/kernel_composition/rmsnorm.jl` defines the public reference, its ordinary
derivative, and private saved-statistic calls. `rmsnorm_cuda_reference.jl` contains
a small deterministic CUDA library with no Tempest dependency, followed by its
adapter. Both its launch wrappers and device kernels retain concrete CUDA array
signatures. These kernels validate the execution contract; they are not tuned
normalization kernels. The example converts epsilon to the input element type in
both the reference and kernel wrapper, so static keyword precision does not
silently change the mathematical output contract.

Run the complete captured Adam example:

```sh
julia --project=examples examples/kernel_composition/train.jl
```

The regular GPU suite checks partial tiles, changed-input capture replay, dtype
fallback, and Adam through two residual normalization layers. CPU checks compare
the reference derivative with finite differences and exercise construction
fallback, overlapping saved lifetimes, and contract errors.

### Existing archived kernels: compatibility probe

`rmsnorm_cuda.jl` adapts the archived `tile-kernels` RMSNorm forward/backward
wrappers. It loads only their source files, without modifying that repository or
adding a dependency to Tempest's core. The archive's backward uses grouped
partial weight gradients and locks. The adapter assigns one group per column to
avoid lock contention and fix accumulation order. This spends a
feature-by-column temporary buffer; it is not a memory-optimal schedule.

**This external adapter is not currently validated.** On the installed cuTile
1.0.0, preparation fails in `promote_scalar_type` with a `UnionAll.parameters`
error on Julia 1.12 and 1.13. An isolated guard experiment exposed a further
`throw_undef_if_not`/Tile IR failure; that compiler patch was not retained. These
errors propagate; the adapter does not claim successful reference fallback.

Reproduce the kernel failure independently of Tempest:

```sh
julia --project=examples examples/kernel_composition/reproduce_archived.jl /path/to/archive/tile-kernels
```

When a compatible compiler/kernel version is available, enable the same GPU
contract tests for the archived adapter:

```sh
TEMPEST_RMS_SOURCE=/path/to/archive/tile-kernels julia --project=test test/runtests.jl
```

This optional test is expected to fail on the currently recorded environment.
It is separate from the always-enabled CUDA reference-kernel coverage.

## Current bounds

- The ordinary reference pullback is still required. Its `nothing` positions
  determine which inputs have cotangents; specialization cannot add gradients to
  inputs declared nondifferentiable there.
- Saved values use the existing floating `TensorSpec` contract. Integer masks,
  RNG state, opaque descriptors, and asynchronous resource events need additional
  contracts; they are not disguised as floating tensors or scratch buffers.
- Forward-only preparation does not invoke reverse rules.
- Rule recipes may contain array arithmetic and semantic calls. This is not a
  cost model or a promise to fuse an opaque kernel with its neighbors.
- Higher-order differentiation through a selected implementation is not provided.

## Validation recorded on 20 September 2026

On GB10, the complete CPU/CUDA/cuTile/Stiletto suite passed after integrating the
new rule boundary. Final focused checks passed 66 CPU assertions and 125 CUDA
assertions, including captured strided views and Float64 capability fallback.
The CPU reference derivatives were checked against finite differences. The
standalone example completed five changed-batch Adam updates, comparing parameters,
moments, step counter, and loss with reference execution. Its plan contains two
private forward/backward pairs and 13 direct state writes, with no staged commits.

Compute Sanitizer memcheck, racecheck, and synccheck each completed successfully
on the new CUDA reference kernels at 7×5, 259×3, and 1024×8. They reported zero
memory/synchronization errors and zero shared-memory hazards. Documentation and
doctests also passed. These are correctness checks, not performance measurements
or a claim about the archived kernel's correctness.

The archived RMSNorm source used in the failing compatibility probe has SHA-256
`8e23f4c05bd8421e84595cd2a124ccb02eae6a246fa3adf4497969b96f813aaf`.
No edits were made to that source or to the shared cuTile installation.
