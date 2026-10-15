# Functional training steps

An explicit backward graph can be followed by an ordinary Julia update. Parameters,
optimizer moments, and the step counter are inputs and outputs of that graph.
They remain ordinary values during tracing; the planner assigns safe in-place
writes and explicit commit copies.

## Compose a backward graph and an update

`trace` accepts positional inputs containing nested tuples and NamedTuples.
Array and floating-scalar leaves keep their shapes and types during replay.
`differentiate(program; wrt=(1,))` returns only the gradient of argument 1,
preserving its structure. The selection order determines gradient order.

```julia
params = (weight = randn(Float32, 2, 3), bias = zeros(Float32, 2, 1))
batch = (x = randn(Float32, 3, 8), y = randn(Float32, 2, 8))
loss(p, b) = sum(abs2, p.weight * b.x .+ p.bias .- b.y)

gradients = differentiate(trace(loss, params, batch); wrt = (1,))
step = with_update(gradients) do loss, gradients, p, b
    next_params = map((p, g) -> p .- 0.001f0 .* g, p, only(gradients))
    (next_params, loss)
end

ex = prepare(step, params, batch; donate = (1,))
next_params, loss_buffer = execute!(ex)
bind_argument!(ex, 2, batch) # replace only the batch; keep current parameters
execute!(ex)
```

`with_update` appends traced arithmetic to the explicit backward graph. Its callback
receives `(loss, gradients, original_arguments...)`; return any supported result
tree. The compiler plans the combined graph, including forward, backward, and
update arithmetic. No special optimizer verb is required. The runnable
[Adam example](https://github.com/jool-space/Tempest.jl/blob/main/examples/adam.jl)
includes moments, bias correction, and a runtime floating step counter.

## Donation and ownership

`donate=(1, 2)` means top-level output group 1 replaces owned input group 1, and
output group 2 replaces owned input group 2. Tuple positions also determine this
mapping for NamedTuple outputs. Matching groups must have identical nested
structure, leaf shapes, element types, and scalar-versus-array identity.

Preparation copies caller inputs into owned buffers. Donation transfers the next
state into those buffers; it does not take ownership of the caller's arrays.
Donated leaves must not alias any other input leaf. Represent tied parameters once
in the input tree and reuse that symbolic value in the model.

The planner can write an elementwise result directly into its donated input when
all later reads use the new value and every read of the overwritten input uses
the same output lane. Returned old-state aliases, cross-lane gathers, and competing
destinations prevent this reuse. An implementation must explicitly accept an
`InPlaceRequest`; otherwise preparation keeps the staged path. CPU and CUDA scalar
kernels support this contract. Vendor and custom implementations retain their
existing distinct-output contract until they opt in.

Staged results commit after the regions finish. Old-state aliases are snapshotted
before any commit, preserving swaps and observations. `explain(ex)` reports direct
writes, remaining commits, snapshot storage, and grouped launches.
`inplace=false` disables direct writes for comparison with the staged baseline.

## Independent array updates

Preparation groups consecutive independent elementwise regions with matching
expression types, up to 128 arrays per launch. It checks physical storage aliases
as well as logical independence, so ordinary temporary-buffer reuse cannot create
a race between group members. `GroupedRequest` passes through the same ordered
support/build interface. If no implementation accepts a group, the individual
regions are selected normally.

CUDA uses device descriptor tables for outputs, inputs, and scalar expressions.
Array lengths and literal values may differ within a group; a group does not
create one kernel argument per array. The descriptor tables and their referenced
arrays remain owned by the executable. CPU groups execute ordinary scalar loops.
`grouped=false` disables grouping for inspection and timing comparisons.

This is bounded grouping of existing neighboring regions, not global scheduling
or an Adam-specific recognizer. It does not group reductions, GEMMs, opaque
calls, or differently typed scalar expressions. Very different member sizes can
waste CUDA blocks in the rectangular launch grid.

The [many-array example](https://github.com/jool-space/Tempest.jl/blob/main/examples/many_array_adam.jl)
measures the optimizer slice with supplied gradients. It validates repeated Adam
updates and reports tracing/preparation time, value storage, launches, commits,
and warmed replay time. Its results do not measure a complete model's training
throughput. See the [measurement report](many-array-updates.md) for results and
cold-compilation limitations.

Outputs are borrowed arrays, including 0D buffers for scalar leaves. Donated
outputs point at persistent input storage. Copy any observation you need to keep.
`bind!` resets every input group; `bind_argument!` replaces only the named positional
argument. Validation of a group finishes before its first buffer is written.

## Capture and state

With CUDACore and cuBLAS loaded, prepare on CUDA arrays or choose `CUDATarget()`.
`capture!(ex)` saves donated state outside capture, warms the schedule, captures
it, and restores the saved values even if warmup throws. Recapture preserves the
current state rather than resetting it to the initial examples. Other result
buffers may contain warmup observations.

Each `execute!` or `replay!` advances the state once. Buffer addresses remain fixed;
state commit copies are part of the captured graph. Tests compare repeated Adam
steps against independent eager model derivatives, change batches without
recapture, and verify parameters, moments, the counter, and pointer stability.

Integer arrays can be state, such as a dropout seed advanced by `seed .+ Int32(1)`;
`Tempest.dropout` redraws its mask from that seed in the backward pass.
Arbitrary Julia mutation, dynamic shapes, and mixed-precision optimizer policy
need further contracts. State restoration covers
declared donated buffers, not external effects in custom launches.
