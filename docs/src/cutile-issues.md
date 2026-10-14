# cuTile issues found by Tempest

23 September 2026. cuTile 1.0.0, CUDA_Tile_jll `tileiras` as resolved in the
workspace manifest, GB10 (sm_121), Julia 1.13. Two separate problems, both
reached through the kernel entry the graph-region provider generates, and a
third found by an older reproducer. The repro scripts are in
`examples/cutile_issues/` and `examples/cutile_issues/scalar_derivative.jl`.

## 1. Tuple argument read from two regions: invalid bytecode (fixed locally)

A kernel that reads an array passed inside a tuple argument from two sibling
regions, such as two loops, fails in `tileiras`:

```
error: 'cuda_tile.make_tensor_view' op operand #0 must be 0D tile of Pointer
type values, but got '!cuda_tile.tile<f32>'
```

`tuple_two_regions.jl` reproduces it in 20 lines. The same kernel with separate
array parameters compiles.

**Cause.** `wrap_for` (`src/compiler/intrinsics/views.jl`) caches the
`AssumeOp`-wrapped value of each source value for the whole kernel. Kernel
arguments are normally wrapped at entry by `apply_arg_assume_predicates!`, but
that function skips arguments that are not themselves a `TileArray`
(`argT <: TileArray || continue`), so an array inside a tuple is wrapped by its
first consumer. When that consumer sits inside a region (the first loop's
guard), the cached value is scoped to that region. The second loop reuses it,
and in the bytecode it names a different value.

**Fix.** Branch `tempest-fixes` of `~/.julia/dev/cuTile` (commit `b3bb121`, on
v1.0.0). Branch `tempest-fixes-main` carries the same commit cherry-picked
cleanly onto `origin/main` (`485137d`), which still has the bug.

- `wrap_for` records the block that emitted each wrap and reuses it only from
  that block or its descendants. This alone fixes the repro, and also covers
  values that are not kernel arguments.
- `apply_arg_assume_predicates!` descends through tuple and struct fields, so
  nested arrays get the same entry wrap as top-level ones.
- Tests: two FileCheck tests in `test/codegen/assume.jl` and a device test in
  `test/device/control_flow.jl`. All three fail on v1.0.0 and pass with the
  fix. The full cuTile suite otherwise passes except `array construction` in
  `codegen/operations` (a bytecode-version FileCheck case) and `device/random`
  (worker terminated), which fail identically on unmodified v1.0.0 here.

**To do:** push `tempest-fixes-main` to a fork and open a PR against
JuliaGPU/cuTile.jl; it carries this fix and the fma fix below as separate
commits, each with its explanation and tests.

Tempest does not depend on the fix, because its streamed kernels use separate
parameters for performance (below). The workspace currently develops the local
checkout; `examples/Project.toml` has an uncommitted `[sources]` entry for it.
`Pkg.free("cuTile")` from the examples project returns to the registered
release.

## 2. `tileiras` register allocation depends on cross-array token ordering

`store_ordering_registers.jl` loads a column of `x`, reduces it, and stores a
scaled column into `y`:

| Entry | Registers | Time (1024 × 1024) |
| --- | ---: | ---: |
| Separate `x`, `y` parameters | 255 | 82.5 μs |
| One `(y, x)` tuple parameter | 25 | 26.1 μs |

cuTile gives each separate array parameter its own alias class, so the store's
input token does not include the load's result token. A tuple parameter puts
both arrays in the universe class. The Tile IR differs only in that
`join_tokens`. Without the reduction, a load-scale-store kernel uses 22
registers either way. The load already precedes the store through the data
dependence, so the extra token edge costs no parallelism in this kernel.

The effect reverses for loops. Tempest's streamed column schedule sweeps rows
twice: once for the statistics, once for the outputs. There, the tuple entry
makes each sweep's loads wait for the previous stores, which serializes the
pipeline:

| Tempest schedule, residual RMSNorm | Separate parameters | Tuple | Separate + stores wait for all prior loads |
| --- | ---: | ---: | ---: |
| Resident, 1024 × 1024 (cold) | 117.9 μs | 86.7 μs | 86.8 μs |
| Resident, 128 × 4096 (cold) | 327.3 μs | 94.4 μs | 94.5 μs |
| Streamed, 8192 × 32 (cold) | 54.4 μs | 253.1 μs | 129.4 μs |

The last column is an experiment in cuTile's token-order pass (a store also
joins the `LAST_OP` token of every other alias class; loads unchanged;
`examples/cutile_issues/stores_after_all.patch`, enabled by
`CUTILE_EXPERIMENT=stores_after_all`). It
recovers the resident kernels completely but still slows the streamed one
2.4×. The streamed kernel uses 255 registers with separate parameters and
30 with a tuple, yet is fastest with 255: occupancy is not what limits it.

This looks like a `tileiras` scheduling problem rather than a cuTile bug:
removing a token edge that the data dependence already implies should not
raise register allocation from 25 to 255. Next steps, in order of value:

1. Report `store_ordering_registers.jl` to NVIDIA (the `tileiras` team, or the
   cuTile Python project, which has the same token-order design), with the two
   Tile IR listings (`SHOW_TILE_IR=true`).
2. Ask cuTile maintainers whether Julia-side control is wanted: an alias hint
   per argument, or an option for stores to join the tokens of the loads their
   value depends on.

Tempest kept this split until cuTile.jl#318, merged 24 September, gave arrays
inside tuples their own alias classes: on cuTile main the tuple entry also uses
255 registers. Since 14 October, every kernel takes separate parameters, and
resident kernels compile with an occupancy hint of 4, which limits them to 64
registers and restores their earlier times
(`ext/cuTileExt/graph/build.jl`).

## 3. FMA fusion with a scalar addend: invalid bytecode (fixed locally)

`examples/cutile_issues/scalar_derivative.jl` broadcasts `v -> 1 + 3v * v` over a
tile. cuTile's FMA fusion rewrites `addf(mulf(x, y), z)` into `fma(x, y, z)`.
Here `z` is the constant `1`, a 0-D tile that `addf` would have broadcast, but
the `fma` emitter passed it through, and `tileiras` rejected the module:

```
error: 'cuda_tile.fma' op failed to verify that all of {lhs, rhs, acc, result}
have same type
```

The emitter now broadcasts all three operands to a common shape, as the
binary operations do (`93de217` on `tempest-fixes`, `1bca3d2` on
`tempest-fixes-main`). A device test in `test/device/math.jl` fails without the
change and passes with it. The example passes with the local checkout; with
the registered release it still fails.
