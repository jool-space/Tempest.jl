# Graph regions

Tempest can select implementations for connected pieces of ordinary array code
before assigning layouts and storage. A `GraphRegion` retains original operations
and value IDs, including pointwise work, constants, several reductions, and
multiple escaping outputs. Providers decide whether they can realize it.

This selection is enabled by default in `prepare` and `prepare_forward`. Set
`fuse_regions=false` to retain the existing schedule for comparison. Without a
provider that supports `GraphRequest`, the search is skipped. CPU defaults have
no graph provider; loading cuTile enables its CUDA reduction schedules.

## Ordinary array code

```julia
using Tempest, CUDACore
import cuBLAS, cuTile

function residual_norm(x, residual, weight, epsilon)
    h = x .+ residual
    mean_square = sum(abs2, h; dims = 1) ./ Float32(size(h, 1))
    rstd = inv.(sqrt.(mean_square .+ epsilon))
    (y = h .* rstd .* weight, residual = h, rstd = rstd)
end

x, residual = (CuArray(randn(Float32, 259, 7)) for _ in 1:2)
weight = CuArray(ones(Float32, 259))
ex = prepare_forward(residual_norm, x, residual, weight, 0.03f0)
result = execute!(ex)
explain(ex)

baseline = prepare_forward(residual_norm, x, residual, weight, 0.03f0;
                           fuse_regions = false)
```

Returned residuals and statistics are explicit graph outputs and must remain
available even if an implementation could keep them inside a kernel.
The same boundary calculation includes values used by backward or by a view
outside the region. Shape and dtype specialization happens during preparation;
`bind!` can change input contents, including the scalar epsilon, without changing
that signature.

## Implementation interface

A provider participates by defining the existing
`support(provider, request::GraphRequest)` and `build_implementation` methods.
There is no additional registry. User providers precede target defaults for a
particular request. A provider implementing only `RegionRequest` does not compete
for `GraphRequest`: graph selection can claim work before decomposition reaches
that provider. Use `fuse_regions=false` when comparing such implementations.
Comparing providers across different region boundaries needs broader selection
policy; candidate ordering alone does not solve it.

`request.region` contains `program`, `members`, `inputs`, and `outputs`. Those ID
vectors refer to the original execution graph. `request.inputs` and
`request.outputs` contain corresponding ordered tuples of `TensorSpec`s. Support
runs before storage planning and must not inspect input contents.

Construction receives `(provider, request, runtime, outputs, inputs)`, where both
buffer collections are ordered tuples of dense arrays with their logical
shapes. Scalars use zero-dimensional arrays. Inputs are read-only and do not
alias outputs. Every escaping output must be written completely. Preserve each
node's element conversion and accumulation dtype; parallel sums may associate
terms differently. Return `BuiltImplementation`, `ImplementationPlan`, or
`Unsupported(reason)` through the [usual implementation protocol](implementations.md).

Views, reshapes, matrix products, scatter, and opaque semantic calls remain
outside the primitive region. An implementation that fuses across a matrix
product proposes that region itself; see [proposed regions](proposed-regions.md). Their results can be boundary inputs. Layout
planning materializes dense presentations where the selected provider needs
them, while preserving required aliases and lifetimes outside the region.
Graph regions currently have no direct in-place donation contract; functional
state updates use staged commits after their old-state reads finish.

## Recomputation is an explicit promise

The cuTile provider may reevaluate pointwise producers after a reduction.
`recomputable(f)` is the permission to do that. Unknown functions return false;
a derivative rule or successful device compilation does not imply purity.

```julia
struct ExponentialScale
    gain::Float32
end
(f::ExponentialScale)(x) = exp(f.gain * x)
Tempest.recomputable(::ExponentialScale) = true
```

That declaration promises deterministic, side-effect-free evaluation using only
explicit operands and immutable scalar captures. Arrays must enter as graph
inputs; hiding a tensor or pointer inside a callable defeats dependency and
lifetime planning. A provider may still reject a recomputable function whose
body its compiler cannot support. This promise neither supplies a derivative
nor guarantees cuTile compilation.

## Selection and fallback

Implementation-proposed regions are selected first, and growth treats their
members as claimed. The current selector grows connected candidates from
reductions, bounded to 64 members.
It checks that selected regions remain jointly acyclic and first offers the
complete structural candidate. If that is declined, it starts from a supported
anchor and asks providers about each proposed addition. Declined neighbors stay
outside that growth attempt, so an unsupported activation or reduction need not
cut off other supported branches. Private literals enter with their consumers.
Support verdicts are cached for each exact candidate during selection.

The initial complete-candidate check matters: a provider may accept a full formula
without accepting its fragments. This remains bounded greedy search. Seed and
neighbor order can affect the result, and combinations that become supported
only when several nodes enter together can still be missed.

Metadata rejection permits alternative smaller candidates. If construction returns
`Unsupported`, preparation excludes that exact member set and replans; shared
nodes remain eligible for smaller regions. Every selected region that fails to
build in one pass is excluded before the single replan. Unexpected exceptions propagate.
`explain` reports selected members, logical inputs, escaping outputs, and
rejection reasons. If a semantic reverse-rule pair is rolled back, region IDs
from the discarded graph are cleared before selection resumes.

## Native cuTile scope

The provider handles nonempty Float32 domains of rank zero, one, or two. Each
region uses one reduction domain and one set of axes. Singleton broadcasts,
scalar inputs, irregular tile tails, several independent or dependent folds
(sums, products, maxima, logsumexp, and zero-size user folds with Float32 or
Int32 state, among them `reduce` by a singleton operator), and multiple escaping
outputs are supported.

| Axes | Execution | Shared workspace |
| --- | --- | --- |
| First axis | One block per column; a column of at most 4096 rows stays resident in one tile, longer columns stream 1024 rows at a time | None |
| Second axis | One block per row tile, streaming eight columns at a time | None |
| All axes | Parallel partial sums (unobserved state tuples for other folds), then a finishing kernel for each dependency phase; a final kernel writes escaping outputs | Partial sums or states, and completed statistics |

Second-axis tiles hold up to 32 rows. Full-reduction tiles contain 256 elements,
with their shape chosen from the domain. A first-axis reduction over fewer than
32 streamed columns splits each column across blocks; its tiles shorten, down to
256 rows, until the grid holds 128 blocks. The streamed tile height and the split
thresholds were measured on GB10 (see
[reduction domains](reduction-domains.md#Taller-streamed-tiles,-14-October-2026));
the other tile shapes are fixed policies, not tuned choices. Independent sums share a phase; dependent sums start a later
phase. A resident column loads and computes every value once and shares it
across phases and outputs; a streamed schedule retains completed statistics in
the block and recomputes permitted pointwise producers in each sweep.

Kernels receive each array as a separate parameter, and cuTile gives each its
own alias class, so stores need not wait for loads from other arrays. Without
that ordering `tileiras` allocates 255 registers for resident kernels, which then
run several times slower; they compile with an occupancy hint of 4, which limits
registers instead. Resident kernels previously took one tuple argument, whose
arrays shared an alias class, until cuTile.jl#318 gave them separate classes.
See the [cuTile issues](cutile-issues.md) report for the measurements. Full reductions
use workspace declared through `ImplementationPlan`, which the planner can
reuse across successive regions. Every invocation overwrites the workspace it
reads, including captured replay.

Computed contributions are masked before reduction: zero-padding loads alone
would be incorrect for functions such as `exp` or `log`. Singleton outputs have
one writer even when the grid spans an axis absent from their logical shape.
Each original pointwise conversion remains explicit. Custom scalar callables
are broadcast over flattened tiles at the call boundary, working around a
multi-axis scalar-broadcast shape bug that cuTile 1.0.0 and current `main`
share; logical indexing and reduction scheduling remain two-dimensional.

`explain` and `Tempest.profile` print the chosen schedule of each cuTile graph
call on one line: the reduced axes and domain, the tile, whether a column stays
resident (and how many columns a block packs) or is streamed (and across how
many blocks it is split), the launch grid, dependency phases, and kernel count.

```text
call 1: cuTile reductions over axis 1 of 128×1024: resident tile 128×4, 4 columns per block, grid 256; 2 phases, 1 kernel
call 1: cuTile reductions over axis 1 of 8193×5: streamed tile 256×1, split 32 ways, grid 5×32; 2 phases, 5 kernels
```

For full and split reductions the grid is that of the partial-sum launches.
Scheduled calls and captured kernels are different counts: a full-reduction or
split region launches several kernels.

Regions with different reduction axes remain separate. Rank-three domains,
mixed precision, layout operations inside a region, contractions, and opaque
calls remain outside this provider. The frontend currently rejects empty array
inputs. Grid and indexing limits are checked during support. Unknown scalar
functions still need the recomputation promise and a compilable body.

Coverage does not establish profitability. In particular, a small full reduction
pays for three launches even when a dedicated kernel could do it in one; the
selector does not yet compare costs. The [reduction-domain experiment](reduction-domains.md)
records correctness checks, MLP coverage, and matched independent measurements.
