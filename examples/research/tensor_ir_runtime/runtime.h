#ifndef TEMPEST_TENSOR_IR_RUNTIME_H
#define TEMPEST_TENSOR_IR_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#if defined(__GNUC__)
#define TEMPEST_TENSOR_IR_API __attribute__((visibility("default")))
#else
#define TEMPEST_TENSOR_IR_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Borrowed storage. Shapes and strides use element counts, not byte counts. */
typedef struct TempestTensorIRTensor {
    void *data;
    int32_t rank;
    const int64_t *shape;
    const int64_t *strides;
} TempestTensorIRTensor;

/*
 * Compile one exported graph. Returns an owning handle, or NULL on failure.
 * A zero tile_count requests upstream tile selection. dump_path is optional.
 * Compilation uses an architecture-specific cubin and the default bytecode
 * version of the pinned compiler. tileiras must already be on the process PATH.
 */
TEMPEST_TENSOR_IR_API void *tempest_tensor_ir_compile(
    const char *source, size_t source_length, int32_t compute_capability,
    const int32_t *tile_sizes, size_t tile_count, const char *dump_path);

/* All status-returning functions return zero on success and -1 on failure. */
TEMPEST_TENSOR_IR_API int tempest_tensor_ir_initialize(void *program);

/*
 * Arguments are graph inputs followed by graph outputs, in declared order.
 * v0.1.4's upstream support check is a stub: the caller must validate the
 * exact argument count, dtype, shape, strides and device against its trace.
 */
TEMPEST_TENSOR_IR_API int tempest_tensor_ir_check_support(
    void *program, const TempestTensorIRTensor *tensors, size_t tensor_count,
    int32_t *supported);

TEMPEST_TENSOR_IR_API int tempest_tensor_ir_workspace_size(
    void *program, const TempestTensorIRTensor *tensors, size_t tensor_count,
    size_t *workspace_bytes);

/*
 * Launch asynchronously on the caller's CUDA stream. The compiler owns the
 * kernel symbol, argument packing and grid calculation. No storage is donated
 * or allocated by this bridge. Initialize before warmup or CUDA graph capture.
 */
TEMPEST_TENSOR_IR_API int tempest_tensor_ir_launch(
    void *program, const TempestTensorIRTensor *tensors, size_t tensor_count,
    void *workspace, size_t workspace_bytes, void *stream);

/* Call only after all executions/captured graphs using this program are done. */
TEMPEST_TENSOR_IR_API void tempest_tensor_ir_destroy(void *program);

/* Borrowed thread-local message, valid until the next bridge operation. */
TEMPEST_TENSOR_IR_API const char *tempest_tensor_ir_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
