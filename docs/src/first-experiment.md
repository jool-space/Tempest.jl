# First executable experiment

This is an implementation report, not a new specification. It records what was
actually built and tested after the initial design conversation.

## What exists

The tiny `sum(abs2, tanh.(w' * x))` example now has an explicit differentiated
tensor program. Its CPU reference interpreter is separate from the prepared
executor. Preparation identifies materialization boundaries, builds scalar
expression trees, and assigns exact-size/dtype storage slots using last use.
Transpose aliases resolve to their underlying storage for lifetime accounting.

The CUDA executor combines ordinary cuBLAS GEMMs with generated cuTile scalar
regions and two-stage full reductions. A CUDA scalar expression lowering handles
transposed indexing and is also selectable explicitly. These lowerings share the
same scalar expression vocabulary. There is no per-verb registry or tuning cache.

Custom pure unary/binary scalar functions can participate with explicit scalar
partial rules. This is not generic scalar AD. A few built-in mathematical recipes
avoid retaining arrays just to compute constant derivatives, and tanh's derivative
uses its forward result. The planner remains conservative around shared values.

The residual gated-MLP example exercises three forward GEMMs, their backward
GEMMs, a custom SiLU function, gating, a residual, and a scalar loss. All positional
argument gradients are computed, including the input and target. There is no
optimizer or actual multi-step parameter update in this experiment.

## Capture exposed real resource contracts

A warm `mul!` was not enough to guarantee allocation-free capture. The cuBLAS
path created device scalar parameters, and library-internal workspace could also
appear as captured memory nodes. Preparation now owns a separate cuBLAS handle,
alpha/beta device scalars, and explicit workspace. This avoids modifying a shared
task handle's precision/workspace policy. The handle uses CUBLAS_DEFAULT_MATH.

`capture!` warms the same schedule and rejects graphs containing allocation/free
nodes. Tests replay repeatedly with different copied input contents, compare all
gradients, and check that value-buffer addresses do not change. The graph and its
buffers remain owned by the prepared executable.

This establishes capture for the current pure forward/backward schedule—not for
optimizer state, RNG, quantization refresh, concurrent streams, or arbitrary
library calls.

## Toolchain observations

On GB10, Julia 1.12.7, and cuTile 1.0.0, dynamically formed indices in a
gather/scatter path showed an off-by-one. Dense tile loads/stores avoid this path;
transposed reads lower through CUDA scalar indexing. Some scalar expressions also
triggered a mixed-shape FMA verifier error. Factored derivative expressions worked.
Standalone reproducers are in `examples/cutile_issues/diagnostics.jl` so these workarounds
can be re-examined rather than fossilized into architecture.

Backend compile errors remain visible. An explicit `scalar_backend=:cuda` permits
trying the same expression schedule through CUDA scalar kernels instead.

## One measurement, not a performance claim

A run of `examples/gated_mlp.jl` on the GB10, with D=128, I=256, N=192 and Float32,
produced 19 scheduled regions and 29 captured kernel nodes. That difference is
intentional: a library region can launch multiple kernels.

The value-slot plan used 1,867,784 bytes versus 2,457,608 without reuse. Separately,
it owned 589,824 bytes of input copies, 384 bytes of reduction scratch, 4 MiB of
cuBLAS workspace, and 8 bytes of cuBLAS coefficients. These are buffer payload
counts, not process-wide memory or host-allocation measurements.

In that run, device-event intervals for repeated submissions were approximately
123 μs per uncaptured prepared forward/backward and 58 μs for replaying the same
captured schedule. Preparation, binding, and transfers were excluded in both
cases. This comparison demonstrates reduced submission overhead; it says nothing
about competitiveness with PyTorch, state-of-the-art kernels, or complete training.
Rerun the script rather than treating these numbers as a target or guarantee.

## What this suggests doing next

The shared computation/storage distinction has survived more than the original
single expression. It is now possible to inspect backward, change scalar lowering,
and disable reuse without rewriting model code.

Still missing are important training questions: selecting which inputs need
gradients, structured parameters, axis reductions and broadcasting for norms and
biases, specialized forward/backward operation pairs, checkpointing, mixed
precision, and explicit effects. Stiletto regional lowering and Pol storage
integration remain experiments to undertake, not foundations already delivered.

The current source is intentionally small and direct. Its Symbol-tagged nodes,
concrete shape rules, provider branches, and storage allocation strategy are not
proposed as permanent interfaces.
