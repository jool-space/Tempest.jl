# Experimental provider, loaded explicitly. Production Tempest has no compiler
# binary dependency; see docs/src/tensor-ir-training.md for the build and scope.
if !isdefined(@__MODULE__, :TensorIRExport)
    include("tensor_ir_export.jl")
end

module TensorIRBackend

using Tempest
using Tempest:
    Supported, Unsupported, BuiltImplementation, ImplementationPlan, WorkspaceSpec, reference, record_call
using CUDACore
using Libdl
using ..TensorIRExport

const D = Tempest

export Compiler, GraphCall, TensorIRImplementation
include("tensor_ir_runtime.jl")

"""
    GraphCall(name, reference, output_specification)

An explicitly outlined array computation. The reference is ordinary Julia;
output_specification returns TensorSpec leaves without computing array values.
The compiler sees its traced graph, not the name. This experiment does not yet
choose region boundaries automatically or differentiate GraphCall itself.
"""
struct GraphCall{F, S}
    name::Symbol
    reference::F
    specification::S
end

(call::GraphCall)(args...; kwargs...) = call.reference(args...; kwargs...)
(call::GraphCall)(x::D.TracedArray, args...; kwargs...) = record_call(call, x, args...; kwargs...)
D.reference(call::GraphCall, args...; kwargs...) = call.reference(args...; kwargs...)
D.output_spec(call::GraphCall, args...; kwargs...) = call.specification(args...; kwargs...)
Base.show(io::IO, call::GraphCall) = print(io, call.name)

struct TensorIRImplementation <: D.Implementation
    compiler::Compiler
end
D.implementation_name(::TensorIRImplementation) = :tensor_ir

function region_program(request)
    nodes = D.Node[]
    placeholders = map(request.inputs) do spec
        D.emit(nodes, D.Argument(), (), spec.dims, spec.dtype; scalar = spec.scalar)
    end
    call = request.call
    f = (args...) -> call.f.reference(args...; call.kwargs...)
    program = trace(f, placeholders...)
    actual = D.tree_map(id -> D.value(program.nodes, id), program.output)
    D.check_verb_result(call, actual)
    program
end

function export_region(request)
    program = region_program(request)
    # Conservative v0.1.4 boundary. Nested/projected reductions at non-power-of-
    # two extents can pass lowering and still emit invalid TileIR. Decline those
    # candidates before build so Tempest can retain its reference schedule.
    for node in program.nodes
        node.op isa D.SumReduction || continue
        dims = program.nodes[only(node.inputs)].dims
        axes = node.op.axes isa Colon ? eachindex(dims) : node.op.axes
        all(axis -> ispow2(dims[axis]), axes) ||
            throw(TensorIRExport.UnsupportedExport(
                "this TensorIR slice requires power-of-two reduction extents"))
    end
    TensorIRExport.export_mlir(program; name = string(request.call.f.name))
end

function D.support(::TensorIRImplementation, request::D.CallRequest)
    request.call.f isa GraphCall || return Unsupported("requires an outlined graph")
    request.target isa CUDATarget || return Unsupported("requires a CUDA target")
    all(s -> s.dtype === Float32 && !s.scalar && !isempty(s.dims), request.inputs) ||
        return Unsupported("this TensorIR slice requires Float32 array inputs")
    try
        export_region(request)
    catch err
        err isa TensorIRExport.UnsupportedExport || rethrow()
        return Unsupported(sprint(showerror, err))
    end
    Supported()
end

function D.build_implementation(implementation::TensorIRImplementation,
                                request::D.CallRequest, runtime, outputs, inputs)
    source = export_region(request)
    program = compile_program(implementation.compiler, source)
    arrays = (inputs..., D.tree_leaves(outputs)...)
    specs = (request.inputs..., D.tree_leaves(request.call.specs)...)
    arguments = bind_arguments(program, arrays, specs)
    bytes = workspace_size(program, arguments)
    if bytes == 0
        launch = BoundLaunch(program, arguments, nothing, 0)
        return BuiltImplementation(launch, (program, arguments), 0)
    end
    ImplementationPlan((WorkspaceSpec(UInt8, bytes),)) do buffers
        workspace = only(buffers)
        launch = BoundLaunch(program, arguments, workspace, bytes)
        BuiltImplementation(launch, (program, arguments, workspace), 0)
    end
end

end # module
