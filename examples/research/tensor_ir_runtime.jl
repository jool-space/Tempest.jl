# Included inside TensorIRBackend. The C bridge owns upstream MLIR/runtime
# details; Julia owns the exact buffer contract and CUDA resource lifetime.

struct TensorIRFailure <: Exception
    stage::Symbol
    message::String
end
Base.showerror(io::IO, err::TensorIRFailure) =
    print(io, "TensorIR ", err.stage, ": ", err.message)

struct RuntimeLibrary
    handle::Ptr{Cvoid}
    functions::NamedTuple
end

function RuntimeLibrary(path::AbstractString)
    isfile(path) || throw(ArgumentError("TensorIR runtime library does not exist: $path"))
    handle = Libdl.dlopen(abspath(path), Libdl.RTLD_NOW | Libdl.RTLD_LOCAL)
    names = (:compile, :initialize, :check_support, :workspace_size, :launch, :destroy, :last_error)
    functions = NamedTuple{names}(map(names) do name
        Libdl.dlsym(handle, Symbol("tempest_tensor_ir_", name))
    end)
    # Keep the shared library loaded for the process. Program finalizers call
    # into it, and Julia does not promise an order for independent finalizers.
    RuntimeLibrary(handle, functions)
end

"""An explicitly configured experimental compiler; no downloads or global registration."""
mutable struct Compiler
    library::RuntimeLibrary
    compilations::Int
end
Compiler(path::AbstractString) = Compiler(RuntimeLibrary(path), 0)

function failure(library, stage)
    message = ccall(library.functions.last_error, Cstring, ())
    TensorIRFailure(stage, message == C_NULL ? "no diagnostic supplied" : unsafe_string(message))
end
check_status(library, code, stage) = code == 0 ? nothing : throw(failure(library, stage))

mutable struct CompiledProgram
    library::RuntimeLibrary
    handle::Ptr{Cvoid}
    context::CUDACore.CuContext
end

function release!(program::CompiledProgram)
    program.handle == C_NULL && return
    if CUDACore.isvalid(program.context)
        CUDACore.context!(program.context) do
            ccall(program.library.functions.destroy, Cvoid, (Ptr{Cvoid},), program.handle)
        end
    end
    program.handle = C_NULL
    nothing
end

function compile_program(compiler::Compiler, source::String)
    CUDACore.functional() || throw(ArgumentError("TensorIR requires functional CUDA"))
    capability = CUDACore.capability(CUDACore.device())
    sm = Int32(10 * capability.major + capability.minor)
    library = compiler.library
    handle = GC.@preserve source begin
        ccall(library.functions.compile, Ptr{Cvoid},
              (Ptr{UInt8}, Csize_t, Int32, Ptr{Int32}, Csize_t, Cstring),
              pointer(source), sizeof(source), sm, C_NULL, 0, C_NULL)
    end
    handle == C_NULL && throw(failure(library, :compile))
    compiler.compilations += 1
    program = CompiledProgram(library, handle, CUDACore.context())
    finalizer(release!, program)
    try
        status = ccall(library.functions.initialize, Cint, (Ptr{Cvoid},), handle)
        check_status(library, status, :initialize)
    catch
        release!(program)
        rethrow()
    end
    program
end

# Matches TempestTensorIRTensor in tensor_ir_runtime/runtime.h, including
# natural C padding between rank and shape. Shapes/strides count elements.
struct TensorView
    data::Ptr{Cvoid}
    rank::Int32
    shape::Ptr{Int64}
    strides::Ptr{Int64}
end

struct BoundArguments{A}
    arrays::A
    shapes::Vector{Vector{Int64}}
    strides::Vector{Vector{Int64}}
    views::Vector{TensorView}
end

# The C ABI uses void pointers for device addresses. Going through UInt is an
# intentional address conversion, not a claim that device memory is host memory.
device_address(array) = Ptr{Cvoid}(UInt(pointer(array)))

dense_strides(dims) = ntuple(i -> prod(dims[1:(i - 1)]), length(dims))

function bind_arguments(program, arrays::Tuple, specs)
    length(arrays) == length(specs) || throw(ArgumentError("TensorIR argument count differs"))
    CUDACore.context() == program.context || throw(ArgumentError("TensorIR context differs"))
    for (array, spec) in zip(arrays, specs)
        array isa CuArray{Float32} || throw(ArgumentError("TensorIR expects Float32 CuArrays"))
        spec.dtype === Float32 && !spec.scalar ||
            throw(ArgumentError("unsupported TensorIR specification"))
        size(array) == spec.dims || throw(DimensionMismatch("TensorIR buffer shape differs"))
        strides(array) == dense_strides(spec.dims) ||
            throw(ArgumentError("TensorIR expects dense column-major storage"))
        CUDACore.context(array) == program.context ||
            throw(ArgumentError("TensorIR buffer context differs"))
    end
    shapes = [Int64[size(array)...] for array in arrays]
    element_strides = [Int64[Base.strides(array)...] for array in arrays]
    views = [TensorView(device_address(array), Int32(ndims(array)),
                        pointer(shape), pointer(stride))
             for (array, shape, stride) in zip(arrays, shapes, element_strides)]
    arguments = BoundArguments(arrays, shapes, element_strides, views)
    supported = Ref{Int32}(0)
    status = GC.@preserve arguments begin
        ccall(program.library.functions.check_support, Cint,
              (Ptr{Cvoid}, Ptr{TensorView}, Csize_t, Ref{Int32}),
              program.handle, arguments.views, length(arguments.views), supported)
    end
    check_status(program.library, status, :support)
    supported[] == 1 || throw(TensorIRFailure(:support, "runtime rejected bound arguments"))
    arguments
end

function workspace_size(program, arguments)
    bytes = Ref{Csize_t}(0)
    status = GC.@preserve arguments begin
        ccall(program.library.functions.workspace_size, Cint,
              (Ptr{Cvoid}, Ptr{TensorView}, Csize_t, Ref{Csize_t}),
              program.handle, arguments.views, length(arguments.views), bytes)
    end
    check_status(program.library, status, :workspace)
    Int(bytes[])
end

struct BoundLaunch{A, W}
    program::CompiledProgram
    arguments::A
    workspace::W
    workspace_bytes::Int
end

function (launch::BoundLaunch)()
    program = launch.program
    program.handle != C_NULL || throw(ArgumentError("TensorIR program has been released"))
    CUDACore.context() == program.context || throw(ArgumentError("TensorIR context differs"))
    # pointer(array) also establishes CUDACore stream dependencies and marks
    # memory used during capture. Cached addresses alone would skip that work.
    for (array, view) in zip(launch.arguments.arrays, launch.arguments.views)
        device_address(array) == view.data ||
            throw(ArgumentError("TensorIR buffer address changed after binding"))
    end
    stream = CUDACore.stream()
    workspace_pointer = launch.workspace_bytes == 0 ? C_NULL : device_address(launch.workspace)
    status = GC.@preserve launch stream begin
        ccall(program.library.functions.launch, Cint,
              (Ptr{Cvoid}, Ptr{TensorView}, Csize_t, Ptr{Cvoid}, Csize_t, Ptr{Cvoid}),
              program.handle, launch.arguments.views, length(launch.arguments.views),
              workspace_pointer, launch.workspace_bytes,
              Base.unsafe_convert(CUDACore.CUstream, stream))
    end
    check_status(program.library, status, :launch)
    nothing
end
