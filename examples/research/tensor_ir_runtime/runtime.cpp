#include "runtime.h"

#include "mlir-c/Diagnostics.h"
#include "mlir-c/IR.h"
#include "tensor_ir-c/TensorIR.h"

#include <cstring>
#include <exception>
#include <limits>
#include <string>
#include <vector>

namespace {

thread_local std::string last_error;

void append_message(MlirStringRef message, void *destination) {
    auto &text = *static_cast<std::string *>(destination);
    text.append(message.data, message.length);
}

void record_error(MlirStringRef message, void *) {
    if (!last_error.empty()) {
        last_error += '\n';
    }
    append_message(message, &last_error);
}

MlirLogicalResult record_diagnostic(MlirDiagnostic diagnostic, void *) {
    if (!last_error.empty()) {
        last_error += '\n';
    }
    mlirDiagnosticPrint(diagnostic, append_message, &last_error);
    return mlirLogicalResultSuccess();
}

int fail(const char *stage, const char *message) {
    last_error = std::string(stage) + ": " + message;
    return -1;
}

int finish(MlirLogicalResult result, const char *stage) {
    if (mlirLogicalResultIsSuccess(result)) {
        last_error.clear();
        return 0;
    }
    if (last_error.empty()) {
        last_error = "upstream operation failed without a diagnostic";
    }
    last_error = std::string(stage) + ": " + last_error;
    return -1;
}

template <typename Action>
int checked(const char *stage, Action action) {
    last_error.clear();
    try {
        return action();
    } catch (const std::exception &error) {
        return fail(stage, error.what());
    } catch (...) {
        return fail(stage, "unexpected C++ exception");
    }
}

struct ParsedModule {
    MlirContext context = mlirContextCreate();
    MlirModule module = {nullptr};

    ParsedModule() = default;
    ParsedModule(const ParsedModule &) = delete;
    ParsedModule &operator=(const ParsedModule &) = delete;

    ~ParsedModule() {
        if (!mlirModuleIsNull(module)) {
            mlirModuleDestroy(module);
        }
        mlirContextDestroy(context);
    }
};

bool pack_arguments(const TempestTensorIRTensor *tensors, size_t tensor_count,
                    std::vector<MlirTensorIRArgument> &arguments,
                    const char *stage) {
    if (tensor_count > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        fail(stage, "argument count exceeds the upstream runtime limit");
        return false;
    }
    if (tensor_count && !tensors) {
        fail(stage, "argument array is null");
        return false;
    }
    arguments.reserve(tensor_count);
    for (size_t index = 0; index < tensor_count; ++index) {
        const auto &tensor = tensors[index];
        if (!tensor.data || tensor.rank < 0 ||
            (tensor.rank && (!tensor.shape || !tensor.strides))) {
            fail(stage, "invalid tensor pointer, rank, or metadata");
            return false;
        }
        MlirTensorIRArgument argument = {};
        argument.kind = MlirTensorIRArgumentKindTensor;
        argument.value.tensor = {
            tensor.data, tensor.rank,
            const_cast<int64_t *>(tensor.shape),
            const_cast<int64_t *>(tensor.strides)};
        arguments.push_back(argument);
    }
    return true;
}

template <typename Action>
int with_arguments(const char *stage, void *program,
                   const TempestTensorIRTensor *tensors, size_t tensor_count,
                   Action action) {
    return checked(stage, [&] {
        if (!program) {
            return fail(stage, "program handle is null");
        }
        std::vector<MlirTensorIRArgument> arguments;
        if (!pack_arguments(tensors, tensor_count, arguments, stage)) {
            return -1;
        }
        MlirTensorIRPackedArgs packed = {arguments.data(), arguments.size()};
        return finish(action(MlirTensorIRProgram{program}, packed), stage);
    });
}

} // namespace

void *tempest_tensor_ir_compile(const char *source, size_t source_length,
                                int32_t compute_capability,
                                const int32_t *tile_sizes, size_t tile_count,
                                const char *dump_path) {
    void *program = nullptr;
    const char *stage = "compile";
    checked(stage, [&] {
        if (!source || !source_length) {
            return fail(stage, "MLIR source is empty");
        }
        if (tile_count && !tile_sizes) {
            return fail(stage, "tile sizes are null");
        }

        ParsedModule parsed;
        // Keep diagnostics on the calling thread. Each compilation owns its
        // context; independent callers do not share mutable compiler state here.
        mlirContextEnableMultithreading(parsed.context, false);
        mlirContextAttachDiagnosticHandler(parsed.context, record_diagnostic,
                                           nullptr, nullptr);
        MlirDialectRegistry registry = mlirDialectRegistryCreate();
        mlirTensorIRRegisterAllDialects(registry);
        mlirContextAppendDialectRegistry(parsed.context, registry);
        mlirDialectRegistryDestroy(registry);

        parsed.module = mlirModuleCreateParse(
            parsed.context, mlirStringRefCreate(source, source_length));
        if (mlirModuleIsNull(parsed.module)) {
            return finish(mlirLogicalResultFailure(), "parse");
        }

        MlirTensorIRCudaTileCompileOptions options = {};
        options.computeCapability = compute_capability;
        options.archPortability = MlirTensorIRArchPortabilityArchConditional;
        options.ctaCount = 1;
        options.warpCount = 4;
        options.tileSizes = tile_sizes;
        options.numTileSizes = tile_count;
        options.codegenStrategy = MlirTensorIRCudaTileCodegenStrategyLayoutPropagation;
        options.maxTileCandidates = 1;
        options.bytecodeVersion = mlirTensorIRGetDefaultBytecodeVersion();
        options.artifactKind = MlirTensorIRCudaTileArtifactKindCubin;
        if (dump_path && *dump_path) {
            options.dumpCudaTileIRPath = mlirStringRefCreate(dump_path, std::strlen(dump_path));
        }

        MlirTensorIRProgram compiled = mlirTensorIRProgramCompile(
            parsed.module, options, record_error, nullptr);
        if (!compiled.ptr) {
            return finish(mlirLogicalResultFailure(), stage);
        }
        program = compiled.ptr;
        last_error.clear();
        return 0;
    });
    return program;
}

int tempest_tensor_ir_initialize(void *program) {
    return checked("initialize", [&] {
        return finish(mlirTensorIRProgramInitialize(
                          MlirTensorIRProgram{program}, record_error, nullptr),
                      "initialize");
    });
}

int tempest_tensor_ir_check_support(void *program,
                                    const TempestTensorIRTensor *tensors,
                                    size_t tensor_count, int32_t *supported) {
    return with_arguments("support", program, tensors, tensor_count,
                          [&](MlirTensorIRProgram handle, MlirTensorIRPackedArgs args) {
        if (!supported) {
            record_error(mlirStringRefCreateFromCString("support output pointer is null"), nullptr);
            return mlirLogicalResultFailure();
        }
        bool result = false;
        MlirLogicalResult status = mlirTensorIRProgramCheckSupport(
            handle, args, &result, record_error, nullptr);
        if (mlirLogicalResultIsSuccess(status)) {
            *supported = result ? 1 : 0;
        }
        return status;
    });
}

int tempest_tensor_ir_workspace_size(void *program,
                                     const TempestTensorIRTensor *tensors,
                                     size_t tensor_count, size_t *workspace_bytes) {
    return with_arguments("workspace", program, tensors, tensor_count,
                          [&](MlirTensorIRProgram handle, MlirTensorIRPackedArgs args) {
        return mlirTensorIRProgramQueryWorkspaceSize(
            handle, args, workspace_bytes, record_error, nullptr);
    });
}

int tempest_tensor_ir_launch(void *program, const TempestTensorIRTensor *tensors,
                             size_t tensor_count, void *workspace,
                             size_t workspace_bytes, void *stream) {
    return with_arguments("launch", program, tensors, tensor_count,
                          [&](MlirTensorIRProgram handle, MlirTensorIRPackedArgs args) {
        return mlirTensorIRProgramLaunch(handle, args, workspace, workspace_bytes,
                                         stream, record_error, nullptr);
    });
}

void tempest_tensor_ir_destroy(void *program) {
    checked("destroy", [&] {
        mlirTensorIRProgramDelete(MlirTensorIRProgram{program});
        return 0;
    });
}

const char *tempest_tensor_ir_last_error(void) {
    return last_error.c_str();
}
