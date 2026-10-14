#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 TENSOR_IR_SOURCE BUILD_DIRECTORY [JOBS]" >&2
    exit 2
fi

tensor_ir_source=$(realpath "$1")
tensor_ir_build=$(realpath -m "$2")
build_jobs=${3:-8}
bridge_source=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if [[ ! $build_jobs =~ ^[1-8]$ ]]; then
    echo "JOBS must be between 1 and 8." >&2
    exit 2
fi

cmake -S "$tensor_ir_source" -B "$tensor_ir_build" -G 'Unix Makefiles' \
    -DTENSOR_IR_DOWNLOAD_LLVM=ON \
    -DTENSOR_IR_ENABLE_BINDINGS_PYTHON=OFF \
    -DTENSOR_IR_INCLUDE_TESTS=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_INCLUDE_BENCHMARKS=OFF \
    -DLLVM_ENABLE_ASSERTIONS=OFF \
    "-DCMAKE_PROJECT_TensorIRCompiler_INCLUDE=$bridge_source/attach.cmake"

cmake --build "$tensor_ir_build" \
    --target tempest_tensor_ir_runtime --parallel "$build_jobs"

echo "$tensor_ir_build/tempest/libtempest_tensor_ir_runtime.so"
