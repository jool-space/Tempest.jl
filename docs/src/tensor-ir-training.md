# TensorIR training experiment

This research integration connects Tempest's semantic calls and paired reverse
rules to NVIDIA TensorIR's public C API. It asks whether an external compiler can
supply fused forward and backward regions while Tempest retains tracing,
differentiation, saved tensors, storage planning, and CUDA graph capture.

The provider compiles explicitly outlined Julia array computations. Residual
RMSNorm is the training fixture, not a pattern built into the provider. All code
lives under `examples/research`; importing Tempest adds no TensorIR, LLVM, or
Python dependency. This is not automatic graph partitioning or a general
training backend.

## How the pieces fit

| File under `examples/research/` | Responsibility |
|:--|:--|
| `tensor_ir_export.jl` | Export a bounded Tempest `TensorProgram` to TensorIR MLIR, preserving shared values and Julia broadcast axes. |
| `tensor_ir_runtime/` | Build a small C bridge around the upstream compiler/runtime API. |
| `tensor_ir_runtime.jl` | Bind exact CUDA buffer signatures, retain program ownership, and launch on the caller's stream. |
| `tensor_ir_backend.jl` | Define generic `GraphCall` regions and the optional `TensorIRImplementation` candidate. |
| `tensor_ir_normalization.jl` | Supply residual RMSNorm's reference, derivative, and saved-state reverse rule. |
| `tensor_ir_training.jl` | Exercise the integration and compare its numerical results with independent references. |
| `tensor_ir_benchmark.jl` | Compare captured replay with the native CUDA reference schedule. |

A `GraphCall(name, reference, output_specification)` provides ordinary Julia code
and an output specification. The name labels the graph; the provider traces the
reference and compiles its operations without recognizing that name. Region
boundaries are supplied explicitly. Tempest continues to schedule the remaining
operations and can use its existing implementations around these regions.

The normalization fixture outlines three computations:

1. Forward: form `h = x + residual`, compute inverse RMS over the leading axis,
   and return the weighted result together with `h` and inverse RMS.
2. Input backward: use the incoming gradient, weight, and saved forward values
   to compute `dh`. The two residual inputs share this mathematical gradient.
3. Weight backward: reduce over the batch axis and remove the singleton axis
   to recover the weight vector's shape.

The reference derivative recomputes the statistic. The specialized reverse rule
uses the saved forward statistic instead. This exercises a forward/backward
pair and its saved-state contract, rather than substituting only a forward
kernel. Saved values are regular graph results whose lifetimes are planned by
Tempest.

## Using the experimental provider

After building the bridge, an outlined forward graph can be prepared directly:

```julia
include("examples/research/tensor_ir_backend.jl")
using .TensorIRBackend, Tempest, CUDACore
import cuBLAS

compiler = Compiler("/path/to/libtempest_tensor_ir_runtime.so")
region = GraphCall(
    :example_region,
    x -> sin.(x) .+ sum(x .* x; dims = 1),
    Tempest.TensorSpec,
)
x = CuArray(randn(Float32, 64, 8))
ex = prepare_forward(region, x;
                     implementations = (TensorIRImplementation(compiler),))
capture!(ex)
y = replay!(ex)  # borrowed output, owned by ex
```

For the training fixture, include `tensor_ir_normalization.jl`, use
`residual_rmsnorm` in the loss, and pass `NormalizationImplementation(compiler)`
to `prepare`. That wrapper supplies the saved-state reverse rule and delegates
its three private graphs to the same generic provider. `GraphCall` itself does
not yet have a generic differentiation rule.

## Current acceptance boundary

The exported region must have dense, positive-rank Float32 array inputs and
outputs with nonempty dimensions. Runtime binding checks argument count, exact
dtypes, shapes, column-major strides, and CUDA context. TensorIR's v0.1.4 runtime
support check is a stub, so forwarding that check alone is insufficient.

Supported scalar expressions include basic arithmetic, negation, selected
unary functions, squares/cubes, Float32 identity conversions, and variadic
addition/multiplication. Integer literals must be exactly representable as
Float32. Sum reductions retain singleton axes. Julia's leading-axis broadcasting
becomes explicit rank padding followed by TensorIR broadcasting.

Reshape may only insert or remove singleton axes. TensorIR defines general
reshape in logical row-major order; column-major buffer strides do not change
that definition. For example, `(H, 1)` to `(H,)` is supported, while flattening an
arbitrary matrix needs a separate index transformation and is rejected here.
Views, transposes, opaque operations inside the outlined region, mixed
precision, runtime scalar inputs, scalar outputs, and repeated output IDs are
outside this slice. The exporter raises a specific `UnsupportedExport` for these
cases; it does not swallow unrelated Julia errors.

The provider additionally requires power-of-two extents on reduced axes. This
is a conservative boundary around observed v0.1.4 lowering/assembly failures,
not a claim that every other extent is inherently impossible or that every
power-of-two graph compiles. Known eligibility failures return `Unsupported`,
allowing Tempest to reject a specialized pair and expand its reference.
Unexpected compiler or runtime failures remain visible in this research slice;
they are not silently converted into fallback execution.

## Build and run

The pinned source is
[NVIDIA TensorIR v0.1.4](https://github.com/NVIDIA/tensor-ir/tree/v0.1.4), commit
`664d385996371db2a8de67b76f82fc89e2470109`. Its
[dependency pins](https://github.com/NVIDIA/tensor-ir/blob/v0.1.4/cmake/TensorIRDependencyPins.cmake)
select a compatible compiler pair:

| Dependency | Pinned revision |
|:--|:--|
| LLVM | `57109befac92811d2253109242ca6fa69c961fb2` |
| CUDA Tile | `af2417041cc939b87ef56d92cfdcf61737c5457e` |
| DLPack | `6ea9b3eb64c881f614cd4537f95f0e125a35555c` |
| FlatBuffers | `v24.3.25` |

Use a C++17 compiler, CMake, Make, and a discoverable CUDA toolkit. The build
script downloads the pinned sources as needed, disables Python bindings, and
builds the bridge. It accepts one to eight parallel jobs; the default is eight.
From the Tempest checkout:

```sh
git clone --branch v0.1.4 --depth 1 https://github.com/NVIDIA/tensor-ir.git \
    /tmp/tempest-nvidia-tensor-ir

bash examples/research/tensor_ir_runtime/build.sh \
    /tmp/tempest-nvidia-tensor-ir \
    /tmp/tempest-tensor-ir-pinned-build 8
```

An existing matching source/build directory can be reused. The resulting
library is
`/tmp/tempest-tensor-ir-pinned-build/tempest/libtempest_tensor_ir_runtime.so`.
The CMake attachment does not patch upstream source. LLVM/MLIR symbols from
static dependencies are hidden, and Julia loads the bridge with local symbol
visibility.

A compatible `tileiras` must already be on the Julia process's `PATH` when
compilation runs. The earlier GB10 experiment used the installed CUDA Tile
assembler **13.4.59**. The bridge does not download an assembler or alter the
environment. Point `PATH` at the directory containing your compatible binary:

```sh
PATH=/path/to/tileiras/bin:$PATH \
    julia --project=examples examples/research/tensor_ir_training.jl \
    /tmp/tempest-tensor-ir-pinned-build/tempest/libtempest_tensor_ir_runtime.so
```

The Julia example environment must already be instantiated. To check the
exporter independently of CUDA and the C++ build:

```sh
julia --project=. examples/research/tensor_ir_export.jl --test
```

## Launch and capture ownership

The bridge delegates kernel symbol selection, argument packing, grid
calculation, and launch metadata to the upstream runtime. It does not infer an
ABI from a cubin name or assume a fixed RMSNorm launch shape. The compiler target
comes from the current CUDA device; the bridge requests an architecture-specific
cubin, including the appropriate target suffix for GB10.

Compilation, assembly, initialization, and workspace preparation finish before
capture. Julia owns the input, output, and workspace arrays. A bound launch
retains those arrays and the compiled program, and uses the current Julia CUDA
stream. Pointer access must participate in CUDACore's memory/stream tracking,
including capture tracking; a cached raw address alone does not establish
cross-stream dependencies.

The executable retains the program for the lifetime of any captured graph using
it. Replaying a captured graph bypasses the Julia/C bridge launch call, while
those retained resources remain alive. Explicitly destroying a program while a
launch or captured graph still uses it violates the ownership contract.

The upstream workspace query currently returns zero. The provider forwards the
query and has a declared-workspace path, but this does not demonstrate that a
future TensorIR implementation requiring scratch storage is already validated.

## Measured integration results

On 21 September 2026, the Julia 1.13 / GB10 setup passed **345 integration
assertions** and **36 exporter assertions**. The GPU environment was CUDA toolkit
13.0.88, driver 580.159.03, and Tile IR assembler 13.4.59.

The integration suite covers:

- A Float64 directional finite-difference check of each input's reference derivative.
- All three saved forward outputs, with destinations initially filled with NaN.
- A separately outlined `sin(x) + sum(x*x; dims=1)` graph, without a normalization pattern.
- Loss and every gradient element at `64 × 8` and `256 × 32`, with and without storage reuse.
- Execution on another CUDA stream, capture, changed and zero inputs, forced garbage collection,
  replay, and recapture. Owned buffer addresses remain fixed and replay does not compile again.
- Reference fallback for non-power-of-two reductions and Float64 inputs. A `64 × 7`
  case rejects only weight backward and still rolls back the entire specialized pair.
- Invalid binding count, shape, and dtype; malformed MLIR remains a visible compiler error.

Forward checks use elementwise `rtol=3e-5, atol=3e-6`; GPU gradients use elementwise
`rtol=3e-4, atol=3e-6` and require finite results. The finite-difference comparison
uses `rtol=1e-7, atol=1e-9`. TensorIR compiles three regions per training executable:
forward, input backward, and weight backward. Saved arrays remain ordinary planned
values; their retention works with Tempest's existing reuse planner.

### Initial timing comparison

Run the separate, optional comparison with the same library and assembler:

```sh
PATH=/path/to/tileiras/bin:$PATH \
    julia --project=examples examples/research/tensor_ir_benchmark.jl \
    /tmp/tempest-tensor-ir-pinned-build/tempest/libtempest_tensor_ir_runtime.so
```

Each row computes the normalized squared loss and all three gradients, with
storage reuse enabled. Both paths passed another **138 numerical and provider-selection assertions**
in total, including `4096 × 128`. Times exclude compilation and capture. They
are median CUDA-event elapsed times from seven batches of 200 graph replays,
after warmup, divided by the replay count. Host submission gaps can contribute;
this is an end-to-end replay measurement, not a kernel throughput estimate.

| Shape | Native CUDA replay | TensorIR replay | Captured kernel nodes, native → TensorIR | Planned value bytes, native → TensorIR |
|:--|--:|--:|:--|:--|
| `64 × 8` | 171.36 μs | 22.61 μs | 14 → 7 | 8,520 → 6,440 |
| `256 × 32` | 264.42 μs | 34.84 μs | 14 → 7 | 132,360 → 99,464 |
| `4096 × 128` | 10,066.32 μs | 930.08 μs | 14 → 7 | 8,406,024 → 6,308,360 |

These are measurements against Tempest's current generic CUDA implementation,
not optimized handwritten normalization or cuDNN kernels. Native cuTile was not
loaded. Planned value bytes exclude owned input copies and compiler/driver
resources. The improvement is encouraging, but neither absolute performance nor
coverage is yet a production claim. The remaining CUDA loss/pullback regions
also leave fusion opportunities outside the three explicitly outlined graphs.

## Next boundary to test

The subsequent [compiler direction note](compiler-direction.md) keeps this as an
optional realization alongside native Julia/cuTile. It distinguishes the common
region contract from the separate task of generating a tile schedule.

The experiment establishes that the existing semantic calls, saved-state reverse
rules, storage planner, and capture machinery can host an external graph compiler.
It does not settle region selection. The next experiment should extract graph
regions from ordinary traced operations, preserving explicit external inputs,
multiple outputs, layout maps, and dtype transitions. Try a candidate region,
report eligibility, and retain decomposed execution where it cannot be compiled.

Centered LayerNorm and stable softmax should then exercise sequential reductions,
followed by irregular extents, mapped inputs, and longer reductions. Mixed
precision, numerical-policy control, compilation caching, and scheduling costs
still need explicit designs. Promoting this adapter to a package extension before
those contracts are understood would make the current research restrictions too
easy to mistake for Tempest's intended semantics.

See [the scheduler investigation](scheduler-investigation.md) for earlier
standalone compiler/forward execution probes and [the compiler landscape](compiler-landscape.md)
for the broader design context.
