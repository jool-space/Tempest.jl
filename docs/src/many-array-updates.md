# Many-array update experiment

!!! note "Historical report"
    Written 7 September 2026. It records the state, measurements and plans of that
    date; the current design is described in [Architecture](architecture.md),
    whose open directions replace the plans here.

The optimizer benchmark uses 100 or 1,000 Float32 parameter arrays. Their lengths
cycle through 17, 63, 127, 513, and 1,025. It takes gradients as inputs and executes
ordinary Adam arithmetic, including both moments and the step counter. This
isolates update composition; it does not measure model forward/backward throughput.

```sh
julia --project=examples examples/many_array_adam.jl 1000
```

To compare both policies in one process:

```julia
include("examples/many_array_adam.jl")
for count in (100, 1000), optimized in (false, true)
    benchmark_many_arrays(count; inplace = optimized, grouped = optimized)
end
```

Each configuration checks three state transitions against the eager Julia update.
Timing resets the initial state, warms replay, then takes the best of five batches
of 100 replays followed by synchronization. Binding, capture, and validation are
excluded from replay timing. Measurements below were taken on GB10 on 7 September
2026, on a shared machine.

| Arrays | Policy | Launches | Commit copies | Value temporaries | Descriptor tables | Replay |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 100 | Staged, ungrouped | 301 | 301 | 418,804 B | 0 B | 1.640 ms |
| 100 | Direct, grouped | 4 | 0 | 0 B | 74,400 B | 0.177 ms |
| 1,000 | Staged, ungrouped | 3,001 | 3,001 | 4,188,004 B | 0 B | 16.903 ms |
| 1,000 | Direct, grouped | 25 | 0 | 0 B | 744,000 B | 1.687 ms |

Owned input storage is unchanged: 558,404 B and 5,584,004 B respectively, including
parameters, moments, gradients, and the scalar counter. Zero value temporaries
does not mean zero auxiliary memory: grouped kernels retain descriptor tables,
and capture temporarily saves donated state for restoration.

## What changed

The graph remains functional. Last-use and same-lane read proofs allow the planner
to assign updated values directly to owned inputs. Native implementations accept
this aliasing explicitly through `InPlaceRequest`; other implementations retain
staged fallback. No Adam operation or derivative rule was added.

Consecutive independent elementwise regions with matching expression types are
grouped through `GroupedRequest`. CUDA descriptor tables hold the arrays and
expressions; group size does not become the kernel's argument count. Groups are
capped at 128 arrays. This example has one counter launch and three update groups
at 100 arrays, or one counter launch plus 24 groups at 1,000 arrays.

The tests separately cover cross-lane reads, old-state observations, overlapping
storage, repeated destinations, unavailable grouped engines, recapture, changed
inputs, and fallback to implementations without an aliasing contract.

## Preparation remains a limitation

In the comparison process, the first 1,000-array trace took 77.45 seconds and
staged preparation took 52.56 seconds. The subsequent optimized configuration
reused compilation: tracing took 91 ms and preparation took 2.94 seconds. Those
are different cache conditions and do not demonstrate faster cold preparation.

An earlier benchmark using `map` directly over a flat 1,000-element tuple triggered
a Julia inference stack overflow; its measurements were discarded. The benchmark
now iterates vectors inside its ordinary Julia update and returns the original
tuple structure. Tempest also bounds specialization in host tree traversal and
preparation. Wide user-level tuple code can still be expensive to compile; this
experiment has not solved that general Julia compilation problem.

Grouping currently follows schedule adjacency rather than a global cost model.
It does not group reductions, GEMMs, opaque calls, or different expression types.
Uneven lengths can waste blocks in the rectangular CUDA launch. The replay results
establish an improvement for this optimizer slice, not universal scaling or a
comparison with another training framework.
