# Layout mappings and pullbacks

Tempest separates a value's logical shape from the mapping of its linear indices
to a storage root. This follows the direction explored by Rewrap's `rewrite`
branch: describe a transformation before choosing a physical representation.
Tempest currently implements its own small mapping algebra; Rewrap is not a
package dependency and its unfinished layout implementation is not vendored.

## Representation

- `AffineMap` records logical dimensions, element strides, and a zero-based offset.
- `Selection` records checked, fixed structural indices relative to parent axes.
  Integer vectors are private snapshots, not live references to captured arrays.
- `ComposedMap` retains a parent mapping and a local mapping when the composition
  cannot be reduced to an affine map.

All maps evaluate a one-based logical linear index to a one-based storage index.
Reshape preserves linear order. The simplifier splits and merges contiguous stride
chunks, allowing singleton axes without imposing spurious stride constraints.
It never flattens across a gap merely because element counts match.

For a `(3,4,2)` parent, the map for
`reshape(view(x,1:2,:,:),2,8)` has dimensions `(2,8)` and strides `(1,3)`.
This is equivalent to `view(reshape(x,3,8),1:2,:)`. Flattening that cropped view
still reads the sequence `1,2,4,5,...`; it is not a contiguous parent slice.
An actual `Base.OneTo(2)` index is checked and cropped to length two, independently
of the parent extent. Cropped permutations retain their offsets and strides.

## Planning and execution

The frontend records logical operations without inserting a copy for every nested
wrapper. Planning composes layouts up to each materialized storage root. Mapped
reads feed CPU loops or generated CUDA kernels. Dense regions remain eligible for
cuTile. A GEMM or borrowed output that cannot be represented as dense storage or a
simple matrix transpose receives a scheduled materialization. This is deliberately
a conservative consumer policy, not an autotuned layout cost model.

Layout aliases extend the lifetime of their storage root. Materializations are
ordinary steps with their own slots, and slot reuse never overlaps a consuming
step. `explain` reports layout aliases, materialized layout values, and owned CUDA
index-buffer bytes.

External views are still copied into the executable's stable dense input buffers.
This layer does not change input binding or expose mutation through aliases.

## Differentiation

Reshape and permutation pullbacks use the inverse presentation. A view or slice
pullback creates a zero parent cotangent and adds each selected cotangent at its
parent index. Contributions from overlapping views are then ordinary graph adds.

CPU execution accumulates in source order. CUDA uses a zero-fill followed by
conflict-free writes for injective selections. For repeated indices, preparation
builds source-order segments per destination; a CUDA thread sums each segment in
that order. There are no floating-point atomics. Segment and gather buffers are
owned by the runtime, uploaded before capture, and reused during replay.

The allocating reference interpreter uses Julia's native views, permutations,
and `LinearIndices`, independently of the generated mapping evaluator. Tests
compare position-weighted results and finite-difference gradients, with and
without slot reuse, then exercise GPU replay with changed input contents.

## Boundaries

Supported indices are fixed integers, colons, integer ranges (including negative
steps), host integer vectors, and individual `CartesianIndex` values. Empty
selections are supported even though external example arrays must be nonempty.
Boolean masks, integer index matrices, runtime index inputs, and data-dependent
output sizes remain outside this stage. Direct scalar indexing of nonzero-rank
symbolic arrays remains refused; zero-dimensional views can use `[]`.

Wrappers constructed inside a traced function are represented as `TracedArray`,
not concrete `SubArray` or `PermutedDimsArray` objects. Generic `AbstractArray`
helpers work for supported operations; concrete-wrapper dispatch is not promised.
Since mutation is refused, view and slice nodes share the same read mapping;
model-visible mutable copy/alias semantics need a separate state design.

General function outputs and forward-only compilation are now supported; see
[verbs.md](verbs.md). Callable specialization caches and mutation/control-flow
support remain subsequent work.
