# cuTile issues found by Tempest

First written 23 September 2026 against cuTile 1.0.0 on GB10 (sm_121), Julia
1.13. Rechecked on 15 October 2026 against cuTile `main` (`d2fafa3`, which
includes cuTile.jl#315, #316, #318 and #326). The one reproducer still
failing is in `examples/cutile_issues/`.

| Issue | Status on cuTile main |
| --- | --- |
| 1. Tuple argument read from two regions | fixed by cuTile.jl#315 |
| 2. Register allocation and token order | probe no longer slow (cuTile.jl#318 and later) |
| 3. FMA with a scalar addend | fixed by cuTile.jl#316 |
| 4. Scalars broadcast to multi-axis tiles | fixed by cuTile.jl#326 |
| 5. Dynamic gather off-by-one | no longer reproduces |
| 6. Eager `mapreduce` counts padded lanes | still reproduces (`padding_probe.jl`) |
| 7. `log1p`, `expm1`, `asin`, `acos`, one-argument `atan` | still excluded; fixes on an unreleased cuTile branch |

The probes for issues 1 to 5 were deleted once they passed; this page keeps
what they showed.

## 1. Tuple argument read from two regions: invalid bytecode (fixed upstream)

A kernel that reads an array passed inside a tuple argument from two sibling
regions, such as two loops, fails in `tileiras`:

```
error: 'cuda_tile.make_tensor_view' op operand #0 must be 0D tile of Pointer
type values, but got '!cuda_tile.tile<f32>'
```

A 20-line probe reproduced it; the same kernel with separate array parameters
compiled.

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

**Upstream.** cuTile.jl#315 ("Keep AssumeOp wraps within the region that
emits them") fixed it, and #318 gave arrays inside tuple and struct arguments
their own alias classes. On cuTile main the probe compiles and is correct, as
do two older probes of the same failure through streamed loops (a tuple entry
against a flat one, and a two-pass column normalization with wrapped
arguments).

## 2. `tileiras` register allocation depends on cross-array token ordering (no longer slow)

The probe loaded a column of `x`, reduced it, and stored a scaled column into
`y`:

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
a patch enabled by `CUTILE_EXPERIMENT=stores_after_all`). It
recovers the resident kernels completely but still slows the streamed one
2.4×. The streamed kernel uses 255 registers with separate parameters and
30 with a tuple, yet is fastest with 255: occupancy is not what limits it.

This looks like a `tileiras` scheduling problem rather than a cuTile bug:
removing a token edge that the data dependence already implies should not
raise register allocation from 25 to 255. Next steps, in order of value:

1. Report the probe to NVIDIA (the `tileiras` team, or the
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

On cuTile main of 15 October both entries of the probe still use 255
registers, but both run in 21.5 μs, faster than either did before, so the
probe and the patch were deleted. Tempest's occupancy hints were measured on
its own kernels and have not been remeasured.

## 3. FMA fusion with a scalar addend: invalid bytecode (fixed upstream)

The probe broadcast `v -> 1 + 3v * v` over a tile. cuTile's FMA fusion rewrites `addf(mulf(x, y), z)` into `fma(x, y, z)`.
Here `z` is the constant `1`, a 0-D tile that `addf` would have broadcast, but
the `fma` emitter passed it through, and `tileiras` rejected the module:

```
error: 'cuda_tile.fma' op failed to verify that all of {lhs, rhs, acc, result}
have same type
```

The emitter now broadcasts all three operands to a common shape, as the
binary operations do (`93de217` on `tempest-fixes`, `1bca3d2` on
`tempest-fixes-main`). cuTile.jl#316 ("Broadcast 0-D operands of fma and
select") made the same change upstream, and the probe passes on main.
Tempest's tanh derivative keeps its factored form `(1 - y)(1 + y)`, now for
accuracy: as `|y| → 1`, `1 - y` is exact while `1 - y²` cancels the rounding
of `y²`.

## 4. Scalars broadcast to multi-axis tiles: reversed dimensions (fixed upstream)

A scalar callable containing constants, such as `ifelse(x > zero(x), x,
zero(x))`, broadcast over a 64 × 4 tile failed to compile: cuTile compared it
against a 4 × 64 tile. `sign` failed the same way over the two-axis tiles of
column reductions. Tempest flattened every scalar call to one axis and
reshaped the result back, and excluded `sign` from cuTile regions.
cuTile.jl#326 ("Fix dimension order of scalars broadcast to
multi-dimensional tiles") fixed it; both workarounds are removed, and
`test/cutile/graph_regions.jl` covers `sign` and selects of constants over
two-axis tiles.

## 5. Dynamic gather off-by-one (no longer reproduces)

A `ct.gather` with indices built from `ct.bid` and `ct.arange` read shifted
lanes on cuTile 1.0. On main it reads every lane correctly. Tempest's scalar
regions on cuTile still read dense storage with tile loads only; transposed
and gathered reads stay on CUDA, since a gather path would be new code rather
than a re-enabled one.

## 6. Eager `mapreduce` counts padded lanes (open)

`examples/cutile_issues/padding_probe.jl` sums `exp` over the columns of a
zero matrix with cuTile's eager `mapreduce` on `Tiled` arrays. With 8 rows it
is correct; with 259 rows every column sums to 512, the padded tile size,
instead of 259: padded lanes contribute `exp(0)`. Tempest's own kernels mask
computed contributions before reducing and are not affected.

## 7. Functions without a tile lowering (open)

`log1p`, `expm1`, `asin` and `acos` fail to compile over tiles, comparing a
0-D tile with a 256-lane one, and one-argument `atan` needs a `copysign`
without a Tile IR equivalent. Their fixes are on a cuTile branch that is not
yet released, so cuTile regions decline them (`ext/cuTileExt/scalar.jl`) and
CUDA runs them.
