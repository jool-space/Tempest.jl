# TensorIR runtime bridge

This research bridge lets Julia compile an exported TensorIR graph and launch
the resulting program on a caller-owned CUDA stream. The upstream compiler
supplies the kernel symbol, argument packing, and grid calculation. Julia owns
the input, output, and workspace arrays.

The bridge wraps TensorIR's public C API. Its header is [runtime.h](runtime.h).
It does not depend on the compiler CLI's artifact metadata file, whose contents
are insufficient to reconstruct the launch contract.

## Build

The tested source is NVIDIA/tensor-ir v0.1.4, commit
`664d385996371db2a8de67b76f82fc89e2470109`. That revision fetches its compatible
LLVM and CUDA Tile sources when `TENSOR_IR_DOWNLOAD_LLVM=ON`.

```sh
bash examples/research/tensor_ir_runtime/build.sh \
    /tmp/tempest-nvidia-tensor-ir \
    /tmp/tempest-tensor-ir-pinned-build
```

The script accepts an optional third argument for one to eight build jobs.
It configures a Release build without Python bindings or tests, then builds
`BUILD_DIRECTORY/tempest/libtempest_tensor_ir_runtime.so`. An existing pinned
build is reused. CUDA's toolkit must be discoverable by CMake.

`attach.cmake` uses CMake's named project hook to add the bridge target after
upstream defines its libraries. No upstream source is patched. The resulting
shared library hides all statically linked LLVM/MLIR symbols and exports only
the seven `tempest_tensor_ir_*` entry points. Load it with `RTLD_LOCAL`.

Compilation requests an architecture-specific cubin. A compatible `tileiras`
must already be on the process `PATH`; the bridge does not change environment
variables. The validated GB10 setup used compute capability `121` and the
existing Julia artifact's CUDA Tile assembler 13.4.59.

## Ownership and execution

1. Compile the MLIR text. Optional tile sizes override upstream selection;
   an optional path records the lowered CUDA Tile IR.
2. Initialize the program while its target CUDA device/context is current.
3. Validate the concrete Julia arrays against the exported trace. Build
   `TempestTensorIRTensor` descriptors for graph inputs followed by outputs.
4. Query workspace and retain any caller-owned allocation with the program.
5. Launch on the current Julia CUDA stream. The launch is asynchronous.
6. Destroy the program after all launches and captured graphs that use it
   have finished and been released.

Tensor shapes and strides are measured in elements. Descriptor metadata only
needs to remain valid during the bridge call; the arrays themselves must remain
alive until GPU execution finishes. The bridge does not allocate or donate
device tensor storage. It creates temporary host argument descriptors for each
call. CUDA graph replay subsequently bypasses those host launch calls.

Every status function returns zero on success and minus one on failure.
Compilation returns a null handle on failure. Read `tempest_tensor_ir_last_error`
immediately after a failure; it returns a borrowed thread-local string including
the failing stage and upstream diagnostics.

## Boundaries verified in v0.1.4

The upstream `checkSupport` implementation is currently a success-returning
stub, and `queryWorkspaceSize` currently returns zero. Forwarding those APIs
does not establish that arbitrary arguments are supported. The Julia caller
must check argument count, dtype, exact shapes and strides, and device/context.

Initialization must complete before warmup and graph capture. Compilation,
assembler invocation, module loading, and lazy initialization do not belong
inside capture. Keep the program and arrays reachable for the captured graph's
entire lifetime.

The initial bridge check exercised invalid-source diagnostics, compilation of
the actual exported H256 softmax graph, runtime initialization, and destruction.
The enclosing Julia example is responsible for independent numerical checks,
changed inputs, non-default streams, and captured execution.
